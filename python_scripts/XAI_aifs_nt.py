import aiohttp
_orig_init = aiohttp.ClientSession.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("trust_env", True)
    _orig_init(self, *args, **kwargs)
aiohttp.ClientSession.__init__ = _patched_init

import os
import shutil
import tempfile
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.checkpoint import checkpoint
from captum.attr import IntegratedGradients, NoiseTunnel

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS


def run_ig_aifs(
    init_dt,
    target_dt=datetime(2016, 8, 9, 6),
    n_steps=5,
    nt_samples=5,
    noise_frac=0.05,
):
    """Compute SmoothGrad-IG attribution (IG wrapped in NoiseTunnel) for the
    single forecast step whose valid time equals `target_dt`, for a model
    initialized at `init_dt`.

    nt_samples noisy copies of the input are each run through a full
    n_steps-point IG integration, so total cost scales as
    ~nt_samples * n_steps forward/backward passes through the rollout.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lead_hours_total = (target_dt - init_dt).total_seconds() / 3600
    if lead_hours_total <= 0 or lead_hours_total % 6 != 0:
        raise ValueError(
            f"target_dt {target_dt} is not reachable from init_dt {init_dt} "
            f"in whole 6h steps (got {lead_hours_total}h)."
        )
    nsteps = int(lead_hours_total // 6)
    target_step = nsteps - 1  # 0-indexed: the final rollout step IS the target
    print(
        f"Init: {init_dt} -> target valid time {target_dt} ({nsteps} steps, "
        f"target_step={target_step}), nt_samples={nt_samples}, n_steps={n_steps} "
        f"(~{nt_samples * n_steps} rollout evals)"
    )

    # Fresh cache per call, deleted at the end regardless of success/failure --
    # fetched ERA5 input data never accumulates across init times.
    tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
    os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache

    try:
        package = AIFS.load_default_package()
        model = AIFS.load_model(package).to(device)

        ic = model.input_coords()
        data = ARCO()
        x, coords = fetch_data(
            source=data,
            time=[init_dt],
            variable=ic["variable"],
            lead_time=ic["lead_time"],
            device=device,
        )

        x = x.unsqueeze(0)
        coords = OrderedDict([("batch", np.array([0]))] + list(coords.items()))

        lat_bounds = (58.5, 63.0)
        lon_bounds = (4.5, 9.0)

        model_native_lat = model.latitudes.detach().flatten().cpu().numpy()
        model_native_lon = model.longitudes.detach().flatten().cpu().numpy()
        native_node_mask = np.where(
            (model_native_lat >= lat_bounds[0]) & (model_native_lat <= lat_bounds[1]) &
            (model_native_lon >= lon_bounds[0]) & (model_native_lon <= lon_bounds[1])
        )[0]
        tp_full_idx = model.VARIABLES.index("tp06")

        class AIFSPrecipRegionWrapperNative(nn.Module):
            def __init__(self, model, coords, nsteps, native_node_mask, tp_full_idx):
                super().__init__()
                self.model = model
                self.coords = coords
                self.nsteps = nsteps
                self.node_mask = torch.as_tensor(native_node_mask, device=next(model.parameters()).device)
                self.tp_full_idx = tp_full_idx

            def forward(self, x_native0):
                m = self.model
                coords_t = self.coords.copy()
                x_native = x_native0
                step = 1
                region_avg = None

                for _ in range(self.nsteps):
                    def _step(x_in, coords_t=coords_t, step=step):
                        out, coords_out = m._forward(x_in, coords_t, step=step)
                        return out

                    out = checkpoint(_step, x_native, use_reentrant=False)

                    # Only the last step (== target_step) is kept; intermediate
                    # steps are still run (autoregressive dependency) but not stored.
                    precip_native = out[:, 1, :, self.tp_full_idx]
                    region_avg = precip_native.index_select(-1, self.node_mask).mean(dim=-1)

                    coords_t = coords_t.copy()
                    coords_t["lead_time"] = coords_t["lead_time"] + m.output_coords(m.input_coords())["lead_time"]
                    x_native = m._update_input(out, coords_t)
                    step += 1

                return region_avg.unsqueeze(1)  # (batch, 1) -- single target column

        with torch.no_grad():
            x_native0 = model._prepare_input(x, coords)

        for ctx_cls in (torch.no_grad, torch.inference_mode):
            ctx_cls.__enter__ = lambda self: None
            ctx_cls.__exit__ = lambda self, *args: None

        x_ig = x_native0.clone().requires_grad_(True)
        baseline = torch.zeros_like(x_ig)

        # Relative noise scale: fixed absolute stdevs would be meaningless
        # given the input spans temperature (~10^2 K), geopotential (~10^4-10^5
        # m^2/s^2), and precip (~10^-3-10^-2 m) simultaneously. This uses a
        # single global scale (not per-variable) -- see caveat above.
        stdevs = float(noise_frac * x_ig.detach().std().item())

        wrapper = AIFSPrecipRegionWrapperNative(model, coords, nsteps, native_node_mask, tp_full_idx).to(device).eval()
        ig = IntegratedGradients(wrapper)
        nt = NoiseTunnel(ig)

        attr = nt.attribute(
            x_ig,
            baselines=baseline,
            target=0,
            nt_type="smoothgrad",
            nt_samples=nt_samples,
            nt_samples_batch_size=1,
            stdevs=stdevs,
            n_steps=n_steps,
            internal_batch_size=1,
        )
        attr = attr.detach().cpu()
        torch.cuda.empty_cache()
        print(f"target_step {target_step}: sum={attr.sum().item():.4e}")

        attr_stack = attr.numpy().squeeze(0)  # (input_lead_time, node, variable)

        lead_time_hours = (model.input_coords()["lead_time"] / np.timedelta64(1, "h")).astype(int)
        node_lat = model.latitudes.detach().flatten().cpu().numpy()
        node_lon = model.longitudes.detach().flatten().cpu().numpy()
        var_names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]

        ds = xr.Dataset(
            {"attribution": (["input_lead_time", "node", "variable"], attr_stack)},
            coords={
                "input_lead_time": lead_time_hours,
                "node": np.arange(len(node_lat)),
                "node_lat": ("node", node_lat),
                "node_lon": ("node", node_lon),
                "variable": var_names,
            },
            attrs={
                "description": "SmoothGrad-IG (NoiseTunnel-wrapped Integrated Gradients) attribution "
                                "of AIFS-predicted west Norway precipitation w.r.t. native-grid initial conditions",
                "target_region": "Vestlandet, Norway",
                "init_time": init_dt.isoformat(),
                "target_valid_time": target_dt.isoformat(),
                "forecast_lead_hours": nsteps * 6,
                "ig_n_steps": n_steps,
                "nt_samples": nt_samples,
                "nt_stdevs": stdevs,
                "nt_noise_frac": noise_frac,
            },
        )
        out_dir = "/cluster/projects/nn12107k/robin/earth2studio/aifs_attributions_vestlandet"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/aifs_ig_attributions_vestlandet_init{init_dt:%Y-%m-%dT%H}_target{target_dt:%Y-%m-%dT%H}.nc"
        ds.to_netcdf(out_path)
        print(f"Saved {out_path}")

    finally:
        shutil.rmtree(tmp_data_cache, ignore_errors=True)
        print(f"Cleaned up temporary input data cache: {tmp_data_cache}")


if __name__ == "__main__":
    target_dt = datetime(2016, 8, 9, 6)
    init_start = datetime(2016, 8, 4, 0)
    n_inits = int((target_dt - init_start).total_seconds() // (6 * 3600))  # inits with >=1 lead step
    init_times = [init_start + timedelta(hours=6 * i) for i in range(n_inits)]

    for init_dt in init_times:
        run_ig_aifs(init_dt, target_dt=target_dt)