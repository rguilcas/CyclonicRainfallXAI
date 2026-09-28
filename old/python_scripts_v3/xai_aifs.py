"""
Integrated-Gradients attribution of AIFS forecasts over a region and target window.
Attributions are saved on the 0.25 deg lat-lon input grid only, together with the
prediction, baseline prediction and completeness diagnostics.
--------------------------------
"""

import os

# These must be set BEFORE earth2studio is imported.
os.environ.setdefault("EARTH2STUDIO_CACHE", "/cluster/projects/nn12107k/robin/earth2studio_cache")
os.environ.setdefault("EARTH2STUDIO_DISABLE_MSC", "1")
os.environ.setdefault("EARTH2STUDIO_DATA_CACHE", "/cluster/projects/nn12107k/robin/e2s_data_cache")

import argparse
import contextlib
import sys
import time
from tqdm import tqdm
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.checkpoint import checkpoint

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS

sys.path.append("/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/src")
from cyclone_rainfall_module.helpers.aiohttpfix import fix_aiohttp
from cyclone_rainfall_module.data.climatology import build_climatology_baseline
from cyclone_rainfall_module.helpers.region_mask import load_region_mask

fix_aiohttp()  # proxy settings for ARCO / model weights on Olivia

device = "cuda" if torch.cuda.is_available() else "cpu"
STEP_HOURS = 6
OUT_ROOT = "/cluster/projects/nn12107k/robin/xai_aifs_v3"
DEFAULT_REGION_GEOJSON = "/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/aux/rainfall_regions.geojson"

# 6-hourly accumulations are in metres; attribute them in mm (x1000) so that
# gradients stay well inside fp16 range under the model's autocast.
ACCUMULATED_VARS = {"tp", "cp", "sf", "ro"}
RADIATION_VARS = {"ssrd", "strd"}


def default_scale(var):
    if var in ACCUMULATED_VARS:
        return 1000.0, "mm"
    if var in RADIATION_VARS:
        return 1e-6, "MJ m-2"
    return 1.0, "native"


# ----------------------------------------------------------------------------
# Gradients through anemoi's internal no_grad / inference_mode
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def grad_through_no_grad():
    """anemoi's predict_step runs inside torch.no_grad / inference_mode.
    Make those context managers no-ops only while this block is active."""
    saved = {cls: (cls.__enter__, cls.__exit__) for cls in (torch.no_grad, torch.inference_mode)}
    try:
        for cls in saved:
            cls.__enter__ = lambda self: None
            cls.__exit__ = lambda self, *args: None
        with torch.enable_grad():
            yield
    finally:
        for cls, (enter, exit_) in saved.items():
            cls.__enter__, cls.__exit__ = enter, exit_


# ----------------------------------------------------------------------------
# Model wrapper: initial state -> region/window aggregate for several variables
# ----------------------------------------------------------------------------
class AIFSRegionTarget(nn.Module):
    """Rolls AIFS out for n_steps and returns, for each target variable, the
    space- and time-aggregated value over the region nodes and window steps.

    Step i (0-based) of the rollout is valid at init + (i + 1) * 6 h.
    Output shape: (batch, n_targets), in scaled units (see default_scale).
    """

    def __init__(self, model, coords, n_steps, window_steps, node_idx, target_vars,
                 time_operation="sum", space_operation="mean", use_checkpoint=True):
        super().__init__()
        self.model = model
        self.coords = coords
        self.n_steps = n_steps
        self.window_steps = set(window_steps)
        self.time_operation = time_operation
        self.space_operation = space_operation
        self.use_checkpoint = use_checkpoint

        name_to_index = model.model.data_indices.data.output.name_to_index
        missing = [v for v in target_vars if v not in name_to_index]
        if missing:
            raise ValueError(f"Not AIFS outputs: {missing}. Available: {sorted(name_to_index)}")
        self.target_vars = list(target_vars)
        dev = next(model.parameters()).device
        self.target_idx = torch.tensor([name_to_index[v] for v in target_vars], device=dev)
        self.node_idx = torch.as_tensor(node_idx, dtype=torch.long, device=dev)
        self.scale = torch.tensor([default_scale(v)[0] for v in target_vars], device=dev)
        self.dt = model.output_coords(model.input_coords())["lead_time"]

    def _step(self, x, coords, step):
        out, _ = self.model._forward(x, coords, step=step)
        return out

    def forward(self, x):
        coords = self.coords.copy()
        per_step = []
        for i in range(self.n_steps):
            if self.use_checkpoint and torch.is_grad_enabled():
                out = checkpoint(self._step, x, coords, i + 1, use_reentrant=False)
            else:
                out = self._step(x, coords, i + 1)

            if i in self.window_steps:
                vals = out[:, 1][:, self.node_idx][:, :, self.target_idx].float()  # (B, nodes, T)
                per_step.append(getattr(torch, self.space_operation)(vals, dim=1))  # (B, T)

            if i < self.n_steps - 1:
                coords = coords.copy()
                coords["lead_time"] = coords["lead_time"] + self.dt
                x = self.model._update_input(out, coords)

        series = torch.stack(per_step, dim=1)  # (B, n_window, T)
        agg = getattr(torch, self.time_operation)(series, dim=1)
        if isinstance(agg, tuple):  # torch.max / torch.min return (values, indices)
            agg = agg[0]
        return agg * self.scale


# ----------------------------------------------------------------------------
# Integrated Gradients (Gauss-Legendre, batched path, several targets per forward)
# ----------------------------------------------------------------------------
def integrated_gradients(f, x, baseline, n_points=16, batch_size=1, log=True):
    """Returns (attributions, integrated_gradients), both of shape
    (n_targets, *x.shape[1:]). x and baseline have shape (1, ...).
    attributions = (x - baseline) * integrated_gradients."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    alphas = torch.tensor((nodes + 1) / 2, dtype=x.dtype, device=x.device)
    weights = torch.tensor(weights / 2, dtype=torch.float32, device=x.device)
    bshape = (-1,) + (1,) * (x.dim() - 1)

    diff = (x - baseline).detach()
    grad_sum = None
    with tqdm(total=n_points, desc="    IG path points", unit="pt", disable=not log) as bar:
        for start in range(0, n_points, batch_size):
            a = alphas[start:start + batch_size].view(bshape)
            w = weights[start:start + batch_size].view(bshape)
            xa = (baseline.detach() + a * diff).requires_grad_(True)
            out = f(xa)  # (k, T)
            n_targets = out.shape[1]
            if grad_sum is None:
                grad_sum = torch.zeros((n_targets, *x.shape[1:]), dtype=torch.float32, device=x.device)
            for t in range(n_targets):
                (g,) = torch.autograd.grad(out[:, t].sum(), xa, retain_graph=t < n_targets - 1)
                grad_sum[t] += (g.float() * w).sum(dim=0)
            del out, xa
            bar.update(len(range(start, min(start + batch_size, n_points))))
    return grad_sum * diff[0].float(), grad_sum


def native_to_latlon_attribution(model, integrated_grad, x_ll, b_ll, chunk=32):
    """Exact IG attributions on the 0.25 deg lat-lon input grid.

    The native input is a linear map of the lat-lon input, x_native = M @ x_ll
    (model.interpolation_matrix). By the chain rule the lat-lon gradient is
    M^T @ grad_native, and attr_ll = (x_ll - b_ll) * grad_ll. The sum of attr_ll
    equals the sum of the native attributions, so completeness is preserved.

    Only prognostic input variables live on the lat-lon grid; forcings and
    invariants are generated internally and have zero attribution anyway.

    integrated_grad: (T, 2, N_native, C_in)   x_ll, b_ll: (1, 1, 2, V, nlat, nlon)
    Returns: (T, 2, V, nlat, nlon) float32 tensor.
    """
    M = model.interpolation_matrix
    Mt = (M.to_sparse_coo() if M.is_sparse else M.to_sparse()).t().coalesce()

    input_full = model.input_full_ids.cpu().numpy()
    prognostic = model.input_ids.cpu().numpy()
    chan = [int(np.where(input_full == pid)[0][0]) for pid in prognostic]

    g = integrated_grad[..., chan]  # (T, L, N, V)
    T, L, N, V = g.shape
    nlat, nlon = x_ll.shape[-2:]
    g2 = g.permute(2, 0, 1, 3).reshape(N, T * L * V)

    out = torch.empty((T * L * V, nlat * nlon), dtype=torch.float32, device=g.device)
    for i in range(0, T * L * V, chunk):
        cols = g2[:, i:i + chunk].to(dtype=Mt.dtype)
        out[i:i + chunk] = torch.sparse.mm(Mt, cols).t().float()
    out = out.reshape(T, L, V, nlat, nlon)

    diff = (x_ll - b_ll)[0, 0].float()  # (L, V, nlat, nlon)
    return out * diff.unsqueeze(0)


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_inputs(model, data, init_time):
    ic = model.input_coords()
    x, coords = fetch_data(source=data, time=[init_time], variable=ic["variable"],
                           lead_time=ic["lead_time"], device=device)
    x_clim, coords_clim = build_climatology_baseline(model, init_time, device, x, coords)
    x = x.unsqueeze(0)
    with torch.no_grad():
        x_native = model._prepare_input(x, coords)
        b_native = model._prepare_input(x_clim, coords_clim)

    # Sanity check: forcing and invariant channels should be identical in input and
    # baseline, so their attributions are exactly zero.
    same = ((x_native - b_native).abs().amax(dim=(0, 1, 2)) == 0).cpu().numpy()
    var_names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]
    print(f"  Channels identical in input and baseline: {[v for v, s in zip(var_names, same) if s]}")
    return x_native.detach(), b_native.detach(), coords, x, x_clim


def region_nodes(model, region_name, geojson):
    lat = model.latitudes.detach().flatten().cpu().numpy()
    lon = model.longitudes.detach().flatten().cpu().numpy()
    idx = load_region_mask(geojson, region_name, lat, lon)
    if len(idx) == 0:
        raise ValueError(f"Region {region_name!r} contains no AIFS grid nodes")
    print(f"{region_name}: {len(idx)} of {len(lat)} native nodes")
    return idx


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run(args):
    model = AIFS.load_model(AIFS.load_default_package()).to(device).eval()
    data = ARCO()
    node_idx = region_nodes(model, args.region_name, args.region_geojson_path)

    window = args.target_time_end - args.target_time_start
    if window % timedelta(hours=STEP_HOURS):
        raise ValueError("Target window must be a multiple of 6 h")
    n_window = window // timedelta(hours=STEP_HOURS) + 1

    lead_times = args.lead_times or list(range(args.shortest_lead_time_timesteps,
                                               args.longest_lead_time_timesteps + 1))
    lead_times = sorted(set(l for l in lead_times if l >= 1))

    tag = (f"{args.region_name.replace(' ', '_')}_{'-'.join(args.target_vars)}_{args.time_operation}"
           f"_target{args.target_time_start:%Y%m%dT%H}-{args.target_time_end:%Y%m%dT%H}")
    dir_out = os.path.join(OUT_ROOT, tag)
    os.makedirs(dir_out, exist_ok=True)

    input_lead_hours = (model.input_coords()["lead_time"] / np.timedelta64(1, "h")).astype(int)
    input_vars = list(model.input_coords()["variable"])
    units = [default_scale(v)[1] for v in args.target_vars]

    predictions = []
    for lead in lead_times:
        # lead = number of 6 h steps from init to the FIRST step valid in the window
        init_time = args.target_time_start - timedelta(hours=STEP_HOURS * lead)
        n_steps = lead + n_window - 1
        window_steps = list(range(lead - 1, n_steps))
        file_out = os.path.join(dir_out, f"ig_{tag}_lead{lead:02d}_latlon.nc")
        if not args.predictions_only and os.path.exists(file_out) and not args.overwrite:
            print(f"Lead {lead}: exists, skipping ({file_out})")
            continue

        print(f"Lead {lead}: init {init_time:%Y-%m-%dT%H}, {n_steps} steps, "
              f"window valid {args.target_time_start:%Y-%m-%dT%H} to {args.target_time_end:%Y-%m-%dT%H}")
        t0 = time.time()
        x_native, b_native, coords, x_ll, b_ll = load_inputs(model, data, init_time)

        f = AIFSRegionTarget(model, coords, n_steps, window_steps, node_idx, args.target_vars,
                             time_operation=args.time_operation,
                             use_checkpoint=not args.no_checkpoint).eval()

        with torch.no_grad():
            fx = f(x_native)[0].cpu().numpy()
        if args.predictions_only:
            predictions.append((init_time, lead, fx))
            print(f"  prediction {dict(zip(args.target_vars, fx))}")
            continue

        with torch.no_grad():
            fb = f(b_native)[0].cpu().numpy()

        with grad_through_no_grad():
            attr, igrad = integrated_gradients(f, x_native, b_native, n_points=args.ig_steps,
                                               batch_size=args.ig_batch_size)

        # Only the per-target sums of the native attributions are kept (for the check).
        attr_sum_native = attr.flatten(1).double().sum(dim=1).cpu().numpy()  # (T,)
        del attr

        with torch.no_grad():
            attr_ll = native_to_latlon_attribution(model, igrad, x_ll, b_ll).cpu().numpy()
        del igrad
        torch.cuda.empty_cache()
        attr_sum_ll = attr_ll.reshape(len(args.target_vars), -1).sum(axis=1, dtype=np.float64)

        completeness = []
        for t, v in enumerate(args.target_vars):
            delta = fx[t] - fb[t]
            err = 100 * (attr_sum_ll[t] - delta) / delta if delta != 0 else np.nan
            completeness.append(err)
            print(f"  {v}: f(x)={fx[t]:.4g}, f(b)={fb[t]:.4g}, sum(attr) native "
                  f"{attr_sum_native[t]:.4g}, lat-lon {attr_sum_ll[t]:.4g}, "
                  f"completeness error {err:.2f}%")

        ds = xr.Dataset(
            {
                "attribution": (["target", "input_lead_time", "variable", "lat", "lon"],
                                attr_ll.astype(np.float32)),
                "prediction": (["target"], fx),
                "baseline_prediction": (["target"], fb),
                "attribution_sum": (["target"], attr_sum_ll),
                "attribution_sum_native": (["target"], attr_sum_native),
                "completeness_error_pct": (["target"], np.array(completeness)),
            },
            coords={
                "target": args.target_vars,
                "target_units": ("target", units),
                "input_lead_time": input_lead_hours,
                "variable": input_vars,
                "lat": coords["lat"],
                "lon": coords["lon"],
            },
            attrs={
                "description": "Gauss-Legendre Integrated Gradients of AIFS region/window aggregate "
                               "w.r.t. initial conditions, climatological baseline. Attributions on "
                               "the 0.25 deg input grid (exact transform of native-grid IG via the "
                               "interpolation matrix).",
                "target_region": args.region_name,
                "init_time": init_time.isoformat(),
                "lead_steps_to_window_start": lead,
                "rollout_steps": n_steps,
                "target_time_start": args.target_time_start.isoformat(),
                "target_time_end": args.target_time_end.isoformat(),
                "time_operation": args.time_operation,
                "ig_points": args.ig_steps,
                "note": "Accumulated targets: each step's value covers the 6 h ending at its valid time. "
                        "Completeness error uses the lat-lon attribution sum; forcings and invariants "
                        "are not on the lat-lon grid (their attribution is zero).",
            },
        )
        ds.to_netcdf(file_out, encoding={"attribution": {"zlib": True, "complevel": 1}})
        print(f"  saved {file_out}  ({time.time() - t0:.0f} s for this lead)")
        del attr_ll, ds

    if args.predictions_only and predictions:
        file_out = os.path.join(dir_out, f"predictions_{tag}.nc")
        ds = xr.Dataset(
            {"prediction": (["init_time", "target"], np.stack([p[2] for p in predictions]))},
            coords={"init_time": [p[0] for p in predictions],
                    "lead_steps": ("init_time", [p[1] for p in predictions]),
                    "target": args.target_vars,
                    "target_units": ("target", units)},
            attrs={"target_region": args.region_name, "time_operation": args.time_operation,
                   "target_time_start": args.target_time_start.isoformat(),
                   "target_time_end": args.target_time_end.isoformat()},
        )
        ds.to_netcdf(file_out)
        print(f"Saved predictions to {file_out}")


def parse_args():
    p = argparse.ArgumentParser(description="AIFS Integrated Gradients for a region and target window.")
    p.add_argument("--region-geojson-path", default=DEFAULT_REGION_GEOJSON)
    p.add_argument("--region-name", default="California heatwave above38 2019-06-11T00")
    p.add_argument("--target-vars", nargs="+", default=["2t"],
                   help="One or more AIFS outputs, e.g. tp ro. They share the forward rollouts.")
    p.add_argument("--target-time-start", type=datetime.fromisoformat, default=datetime(2019, 6, 11, 0))
    p.add_argument("--target-time-end", type=datetime.fromisoformat, default=None)
    p.add_argument("--lead-times", type=int, nargs="+", default=None,
                   help="Explicit lead times in 6 h steps to the window start, e.g. 1 2 4 8 12. "
                        "Overrides --shortest/--longest.")
    p.add_argument("--longest-lead-time-timesteps", type=int, default=24)
    p.add_argument("--shortest-lead-time-timesteps", type=int, default=1)
    p.add_argument("--ig-steps", type=int, default=16, help="Gauss-Legendre points along the IG path.")
    p.add_argument("--ig-batch-size", type=int, default=1, help="Path points per forward pass (memory permitting).")
    p.add_argument("--no-checkpoint", action="store_true", help="Disable activation checkpointing (faster, more memory).")
    p.add_argument("--time-operation", default="sum", choices=["mean", "sum", "max", "min"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--predictions-only", action="store_true")
    args = p.parse_args()

    if args.target_time_end is None:
        args.target_time_end = args.target_time_start
    if args.target_time_end < args.target_time_start:
        p.error("--target-time-end must be >= --target-time-start")
    if args.shortest_lead_time_timesteps > args.longest_lead_time_timesteps:
        p.error("--shortest-lead-time-timesteps must be <= --longest-lead-time-timesteps")
    return args


if __name__ == "__main__":
    run(parse_args())