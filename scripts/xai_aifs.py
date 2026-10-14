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


BASE_DIR = "/cluster/home/rguilcas/code/AIFS-XAI"
sys.path.append(f"{BASE_DIR}/src")

from xai_aifs_module.data.climatology import build_climatology_baseline
from xai_aifs_module.models.wrappers import AIFSRegionTarget, default_scale, ACCUMULATED_VARS, RADIATION_VARS
from xai_aifs_module.helpers import grad_through_no_grad, fix_aiohttp
from xai_aifs_module.data.grids import native_to_latlon_attribution,region_nodes
from xai_aifs_module.attributions.integratedgradients import integrated_gradients


fix_aiohttp()  # proxy settings for ARCO / model weights on Olivia

device = "cuda" if torch.cuda.is_available() else "cpu"
STEP_HOURS = 6
OUT_ROOT = "/cluster/projects/nn12107k/robin/xai_aifs_v3"
DEFAULT_REGION_GEOJSON = f"{BASE_DIR}/aux/rainfall_regions.geojson"


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

# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def shrink_attribution(ds, sum_lead=True, coarsen=None, mass=0.99, crop_box=False, pad_deg=5):
    a = ds["attribution"]
    if sum_lead:
        ds["share_attributions_at_time_t0"] = abs(a.isel(input_lead_time=-1)).sum(["variable", "lat", "lon"]) / abs(a).sum(["input_lead_time", "variable", "lat", "lon"])
        a = a.sum("input_lead_time")
    # if coarsen:  # e.g. 4 -> 1 deg; sum keeps completeness
    #     a = a.coarsen(lat=coarsen, lon=coarsen, boundary="pad").sum()

    # one mask from |attr| summed over everything except space
    other = [d for d in a.dims if d not in ("lat", "lon")]
    mag = abs(a).sum(other)
    v = np.sort(mag.values.ravel())[::-1]
    cum = np.cumsum(v) / v.sum()
    thr = v[min(np.searchsorted(cum, mass), len(v) - 1)]
    mask = mag >= thr
    a = a.where(mask, 0.0)

    if crop_box:  # optional; assumes the signal doesn't straddle the lon seam
        lat_ok = mask.any("lon"); lon_ok = mask.any("lat")
        lats = a.lat[lat_ok]; lons = a.lon[lon_ok]
        a = a.sel(lat=slice(float(lats.max()) + pad_deg, float(lats.min()) - pad_deg)
                  if a.lat[0] > a.lat[-1] else
                  slice(float(lats.min()) - pad_deg, float(lats.max()) + pad_deg),
                  lon=slice(float(lons.min()) - pad_deg, float(lons.max()) + pad_deg))

    ds = ds.drop_vars("attribution").drop_dims([d for d in ("input_lead_time",) if sum_lead], errors="ignore")
    ds["attribution"] = a.astype(np.float32)
    # keep track of what the masking threw away (sum over space+variable per target)
    ds["attribution_sum_kept"] = a.sum([d for d in a.dims if d != "target"])
    ds.attrs.update(mask_mass_fraction=mass, coarsen_factor=coarsen or 1,
                    summed_input_lead_time=int(sum_lead))
    return ds


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
                                               batch_size=args.ig_batch_size, grad_scale=args.grad_scale)

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
        ds = shrink_attribution(ds, sum_lead=True, coarsen=4, mass=0.99)
        enc = {"attribution": {"zlib": True,
                               "complevel": 4, 
                               "shuffle": True,
                               "significant_digits": 3}}   # quantization, needs netCDF4 >= 1.6
        ds.to_netcdf(file_out, encoding=enc)
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
    p.add_argument("--grad-scale", type=float, default=1.0, help="Scale factor for gradients.")
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