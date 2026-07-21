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
from captum.attr import IntegratedGradients
import torch

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS

import sys
sys.path.append('/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/src')

from cyclone_rainfall_module.helpers.aiohttpfix import fix_aiohttp
from cyclone_rainfall_module.data.climatology import build_climatology_baseline
from cyclone_rainfall_module.models.wrappers import AIFSPrecipRegionWrapperNativeAllPredictions, build_day_groups
from cyclone_rainfall_module.attributions.integratedgradients import get_IG_attribution

fix_aiohttp()  # Patch aiohttp to trust environment variables for proxy settings

device = "cuda" if torch.cuda.is_available() else "cpu"

target_day = '2016-08-08'
max_lead_time_hours = 48




def compute_ig_attributions_for_target_day(target_day, max_lead_time_hours):
    first_init_dt = datetime.strptime(target_day, '%Y-%m-%d') - timedelta(hours = max_lead_time_hours)
    hours_until_target_day = ((datetime.strptime(target_day, '%Y-%m-%d') - first_init_dt).total_seconds() / 3600)
    list_init_dt = [first_init_dt + timedelta(hours=6 * i) for i in range(int(hours_until_target_day // 6))]  # 2016-08-07 12, 18, 2016-08-08 00, 06
    print(f"Computing IG attributions for target day {target_day} with max lead time {max_lead_time_hours}h (first init: {first_init_dt})")
    print(f"Number of attributions computed: {len(list_init_dt)}")

    package = AIFS.load_default_package()
    model = AIFS.load_model(package).to(device)
    ic = model.input_coords()
    data = ARCO()

    all_ds_attr = []
    for init_dt in list_init_dt:
        ds_attr = get_IG_attribution(model, init_dt, target_day, data, device, ic)
        all_ds_attr.append(ds_attr)

    return all_ds_attr


list_ds_attr = compute_ig_attributions_for_target_day(target_day, max_lead_time_hours=max_lead_time_hours)  

# def run_ig_aifs(init_dt, target_dt=datetime(2016, 8, 9, 6)):
#     """Compute IG attribution for the single forecast step whose valid time
#     equals `target_dt`, for a model initialized at `init_dt`."""

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     lead_hours_total = (target_dt - init_dt).total_seconds() / 3600
#     if lead_hours_total <= 0 or lead_hours_total % 6 != 0:
#         raise ValueError(
#             f"target_dt {target_dt} is not reachable from init_dt {init_dt} "
#             f"in whole 6h steps (got {lead_hours_total}h)."
#         )
#     nsteps = int(lead_hours_total // 6)
#     target_step = nsteps - 1  # 0-indexed: the final rollout step IS the target
#     print(f"Init: {init_dt} -> target valid time {target_dt} ({nsteps} steps, target_step={target_step})")

#     # Fresh cache per call, deleted at the end regardless of success/failure --
#     # fetched ERA5 input data never accumulates across init times.
#     tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
#     os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache

#     try:
#         package = AIFS.load_default_package()
#         model = AIFS.load_model(package).to(device)

#         ic = model.input_coords()
#         data = ARCO()
#         x, coords = fetch_data(
#             source=data,
#             time=[init_dt],
#             variable=ic["variable"],
#             lead_time=ic["lead_time"],
#             device=device,
#         )

#         # Build climatology baseline from the RAW (un-batched) x/coords
#         x_clim, coords_clim = build_climatology_baseline(model, init_dt, device, x, coords)

#         # Only now add the batch dim to the real input
#         x = x.unsqueeze(0)
#         coords = OrderedDict([("batch", np.array([0]))] + list(coords.items()))

#         lat_bounds = (58.5, 63.0)
#         lon_bounds = (4.5, 9.0)

#         model_native_lat = model.latitudes.detach().flatten().cpu().numpy()
#         model_native_lon = model.longitudes.detach().flatten().cpu().numpy()
#         native_node_mask = np.where(
#             (model_native_lat >= lat_bounds[0]) & (model_native_lat <= lat_bounds[1]) &
#             (model_native_lon >= lon_bounds[0]) & (model_native_lon <= lon_bounds[1])
#         )[0]
#         tp_full_idx = model.VARIABLES.index("tp06")


#         class AIFSPrecipRegionWrapperNative(nn.Module):
#             def __init__(self, model, coords, nsteps, native_node_mask, tp_full_idx):
#                 super().__init__()
#                 self.model = model
#                 self.coords = coords
#                 self.nsteps = nsteps
#                 self.node_mask = torch.as_tensor(native_node_mask, device=next(model.parameters()).device)
#                 self.tp_full_idx = tp_full_idx

#             def forward(self, x_native0):
#                 m = self.model
#                 coords_t = self.coords.copy()
#                 x_native = x_native0
#                 step = 1
#                 region_avg = None

#                 for _ in range(self.nsteps):
#                     def _step(x_in, coords_t=coords_t, step=step):
#                         out, coords_out = m._forward(x_in, coords_t, step=step)
#                         return out

#                     out = checkpoint(_step, x_native, use_reentrant=False)

#                     # Only the last step (== target_step) is kept; intermediate
#                     # steps are still run (autoregressive dependency) but not stored.
#                     precip_native = out[:, 1, :, self.tp_full_idx]
#                     region_avg = precip_native.index_select(-1, self.node_mask).mean(dim=-1)

#                     coords_t = coords_t.copy()
#                     coords_t["lead_time"] = coords_t["lead_time"] + m.output_coords(m.input_coords())["lead_time"]
#                     x_native = m._update_input(out, coords_t)
#                     step += 1

#                 return region_avg.unsqueeze(1)  # (batch, 1) -- single target column

        
#         with torch.no_grad():
#             x_native0 = model._prepare_input(x, coords)
#             baseline_native = model._prepare_input(x_clim, coords_clim)  # x_clim already has batch dim (added inside the function)

#         for ctx_cls in (torch.no_grad, torch.inference_mode):
#             ctx_cls.__enter__ = lambda self: None
#             ctx_cls.__exit__ = lambda self, *args: None

#         x_ig = x_native0.clone().requires_grad_(True)


#         baseline = torch.zeros_like(x_ig)

#         wrapper = AIFSPrecipRegionWrapperNative(model, coords, nsteps, native_node_mask, tp_full_idx).to(device).eval()
#         ig = IntegratedGradients(wrapper)

#         attr = ig.attribute(x_ig, baselines=baseline_native, target=0, n_steps=20, internal_batch_size=1)
#         attr = attr.detach().cpu()
#         torch.cuda.empty_cache()
#         print(f"target_step {target_step}: sum={attr.sum().item():.4e}")

        # attr_stack = attr.numpy().squeeze(0)  # (input_lead_time, node, variable)
        # lead_time_hours = (model.input_coords()["lead_time"] / np.timedelta64(1, "h")).astype(int)
        # node_lat = model.latitudes.detach().flatten().cpu().numpy()
        # node_lon = model.longitudes.detach().flatten().cpu().numpy()
        # var_names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]

        # ds = xr.Dataset(
        #     {"attribution": (["input_lead_time", "node", "variable"], attr_stack)},
        #     coords={
        #         "input_lead_time": lead_time_hours,
        #         "node": np.arange(len(node_lat)),
        #         "node_lat": ("node", node_lat),
        #         "node_lon": ("node", node_lon),
        #         "variable": var_names,
        #     },
        #     attrs={
        #         "description": "Integrated Gradients attribution of AIFS-predicted west Norway "
        #                         "precipitation w.r.t. native-grid initial conditions",
        #         "target_region": "Vestlandet, Norway",
        #         "init_time": init_dt.isoformat(),
        #         "target_day": target_dt.date(),
        #         "forecast_lead_hours": (nsteps-3) * 6,
        #     },
        # )
#         out_dir = f"/cluster/projects/nn12107k/robin/earth2studio/aifs_attributions_vestlandet/{target_dt:%Y%m%d%H}".format(target_dt=target_dt)
#         os.makedirs(out_dir, exist_ok=True)
#         out_path = f"{out_dir}/aifs_ig_attributions_vestlandet_init{init_dt:%Y-%m-%dT%H}_target{target_dt:%Y-%m-%dT%H}.nc"
#         ds.to_netcdf(out_path)
#         print(f"Saved {out_path}")

#     finally:
#         shutil.rmtree(tmp_data_cache, ignore_errors=True)
#         print(f"Cleaned up temporary input data cache: {tmp_data_cache}")


# if __name__ == "__main__":
#     target_dt = datetime(2005, 9, 14, 6) #2005-09-23 00
#     init_start = datetime(2005, 9, 8, 0)
#     n_inits = int((target_dt - init_start).total_seconds() // (6 * 3600))  # inits with >=1 lead step
#     init_times = [init_start + timedelta(hours=6 * i) for i in range(n_inits)]

#     for init_dt in init_times:
#         run_ig_aifs(init_dt, target_dt=target_dt)