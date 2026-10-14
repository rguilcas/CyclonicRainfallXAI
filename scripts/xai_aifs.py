"""
Integrated-Gradients attribution of AIFS forecasts over a region and target window.

For each lead time, attributions of the region/window aggregate with respect to the
initial conditions are saved on the 0.25 deg lat-lon input grid, together with the
prediction, the baseline prediction and completeness diagnostics.

Example:
    python scripts/xai_aifs.py --region-name "California heatwave above38 2019-06-11T00" \
        --target-vars 2t --target-time-start 2019-06-11T00 --lead-times 1 4 8 --ig-steps 30
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Allow running without `pip install -e .` (e.g. inside the container).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_aifs_module import config  # noqa: E402

config.set_earth2studio_env()  # must happen before earth2studio is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from earth2studio.data import ARCO  # noqa: E402
from earth2studio.models.px import AIFS  # noqa: E402

from xai_aifs_module.attributions.integratedgradients import integrated_gradients  # noqa: E402
from xai_aifs_module.data.grids import native_to_latlon_attribution  # noqa: E402
from xai_aifs_module.data.inputs import load_inputs, region_nodes  # noqa: E402
from xai_aifs_module.helpers import fix_aiohttp, grad_through_no_grad  # noqa: E402
from xai_aifs_module.models.wrappers import AIFSRegionTarget  # noqa: E402
from xai_aifs_module.outputs import shrink_attribution  # noqa: E402

STEP = timedelta(hours=config.STEP_HOURS)


# ----------------------------------------------------------------------------
# Set-up helpers
# ----------------------------------------------------------------------------
def window_length(args):
    """Number of 6-hourly steps in the (inclusive) target window."""
    window = args.target_time_end - args.target_time_start
    if window % STEP:
        raise ValueError(f"Target window must be a multiple of {config.STEP_HOURS} h")
    return window // STEP + 1


def lead_schedule(args):
    """Lead times, in 6 h steps from init to the first step of the window."""
    leads = args.lead_times or range(args.shortest_lead_time_timesteps,
                                     args.longest_lead_time_timesteps + 1)
    return sorted({lead for lead in leads if lead >= 1})


def run_tag(args):
    return (f"{args.region_name.replace(' ', '_')}_{'-'.join(args.target_vars)}_{args.time_operation}"
            f"_target{args.target_time_start:%Y%m%dT%H}-{args.target_time_end:%Y%m%dT%H}")


def completeness_errors(fx, fb, attr_sum):
    """Percent error of sum(attributions) relative to f(x) - f(baseline), per target."""
    delta = fx - fb
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(delta != 0, 100 * (attr_sum - delta) / delta, np.nan)


# ----------------------------------------------------------------------------
# One lead time
# ----------------------------------------------------------------------------
def attribute_lead(model, data, node_idx, lead, n_window, args, device):
    """Run IG for one lead time. Returns (fx, None) with --predictions-only,
    otherwise (fx, xr.Dataset)."""
    init_time = args.target_time_start - lead * STEP
    n_steps = lead + n_window - 1
    window_steps = list(range(lead - 1, n_steps))
    print(f"Lead {lead}: init {init_time:%Y-%m-%dT%H}, {n_steps} steps, window valid "
          f"{args.target_time_start:%Y-%m-%dT%H} to {args.target_time_end:%Y-%m-%dT%H}")

    x_native, b_native, coords, x_ll, b_ll = load_inputs(model, data, init_time, device)
    f = AIFSRegionTarget(model, coords, n_steps, window_steps, node_idx, args.target_vars,
                         time_operation=args.time_operation,
                         space_operation=args.space_operation,
                         use_checkpoint=not args.no_checkpoint).eval()

    with torch.no_grad():
        fx = f(x_native)[0].cpu().numpy()
    if args.predictions_only:
        print(f"  prediction {dict(zip(args.target_vars, fx))}")
        return init_time, fx, None

    with torch.no_grad():
        fb = f(b_native)[0].cpu().numpy()

    with grad_through_no_grad():
        attr, igrad = integrated_gradients(f, x_native, b_native, n_points=args.ig_steps,
                                           batch_size=args.ig_batch_size, grad_scale=args.grad_scale)
    # Keep only the per-target sums of the native attributions (for the check).
    attr_sum_native = attr.flatten(1).double().sum(dim=1).cpu().numpy()
    del attr

    with torch.no_grad():
        attr_ll = native_to_latlon_attribution(model, igrad, x_ll, b_ll).cpu().numpy()
    del igrad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attr_sum_ll = attr_ll.reshape(len(args.target_vars), -1).sum(axis=1, dtype=np.float64)
    errors = completeness_errors(fx, fb, attr_sum_ll)
    for t, v in enumerate(args.target_vars):
        print(f"  {v}: f(x)={fx[t]:.4g}, f(b)={fb[t]:.4g}, sum(attr) native {attr_sum_native[t]:.4g}, "
              f"lat-lon {attr_sum_ll[t]:.4g}, completeness error {errors[t]:.2f}%")

    ds = attribution_dataset(model, args, coords, init_time, lead, n_steps,
                             attr_ll, fx, fb, attr_sum_ll, attr_sum_native, errors)
    return init_time, fx, ds


def attribution_dataset(model, args, coords, init_time, lead, n_steps,
                        attr_ll, fx, fb, attr_sum_ll, attr_sum_native, errors):
    ic = model.input_coords()
    return xr.Dataset(
        {
            "attribution": (["target", "input_lead_time", "variable", "lat", "lon"],
                            attr_ll.astype(np.float32)),
            "prediction": (["target"], fx),
            "baseline_prediction": (["target"], fb),
            "attribution_sum": (["target"], attr_sum_ll),
            "attribution_sum_native": (["target"], attr_sum_native),
            "completeness_error_pct": (["target"], errors),
        },
        coords={
            "target": args.target_vars,
            "target_units": ("target", [config.default_scale(v)[1] for v in args.target_vars]),
            "input_lead_time": (ic["lead_time"] / np.timedelta64(1, "h")).astype(int),
            "variable": list(ic["variable"]),
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
            "space_operation": args.space_operation,
            "ig_points": args.ig_steps,
            "grad_scale": args.grad_scale,
            "note": "Accumulated targets: each step's value covers the 6 h ending at its valid time. "
                    "Completeness error uses the lat-lon attribution sum; forcings and invariants "
                    "are not on the lat-lon grid (their attribution is zero).",
        },
    )


def save_predictions(predictions, args, file_out):
    ds = xr.Dataset(
        {"prediction": (["init_time", "target"], np.stack([p[2] for p in predictions]))},
        coords={"init_time": [p[0] for p in predictions],
                "lead_steps": ("init_time", [p[1] for p in predictions]),
                "target": args.target_vars,
                "target_units": ("target", [config.default_scale(v)[1] for v in args.target_vars])},
        attrs={"target_region": args.region_name,
               "time_operation": args.time_operation,
               "space_operation": args.space_operation,
               "target_time_start": args.target_time_start.isoformat(),
               "target_time_end": args.target_time_end.isoformat()},
    )
    ds.to_netcdf(file_out)
    print(f"Saved predictions to {file_out}")


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run(args):
    fix_aiohttp()  # proxy settings for ARCO / model weights on Olivia
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_window = window_length(args)
    leads = lead_schedule(args)
    tag = run_tag(args)
    dir_out = Path(args.out_root) / tag
    dir_out.mkdir(parents=True, exist_ok=True)

    model = AIFS.load_model(AIFS.load_default_package()).to(device).eval()
    data = ARCO()
    node_idx = region_nodes(model, args.region_name, args.region_geojson_path)

    predictions = []
    for lead in leads:
        file_out = dir_out / f"ig_{tag}_lead{lead:02d}_latlon.nc"
        if not args.predictions_only and file_out.exists() and not args.overwrite:
            print(f"Lead {lead}: exists, skipping ({file_out})")
            continue

        t0 = time.time()
        init_time, fx, ds = attribute_lead(model, data, node_idx, lead, n_window, args, device)
        if ds is None:
            predictions.append((init_time, lead, fx))
            continue
        if not args.full_output:
            ds = shrink_attribution(ds, sum_lead=not args.keep_input_lead_times,
                                    mass=args.mask_mass, crop_box=args.crop_box)
        encoding = {"attribution": {"zlib": True, "complevel": 4, "shuffle": True,
                                    "significant_digits": 3}}  # quantisation; needs netCDF4 >= 1.6
        ds.to_netcdf(file_out, encoding=encoding)
        print(f"  saved {file_out}  ({time.time() - t0:.0f} s for this lead)")

    if predictions:
        save_predictions(predictions, args, dir_out / f"predictions_{tag}.nc")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AIFS Integrated Gradients for a region and target window.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--region-geojson-path", default=str(config.DEFAULT_REGION_GEOJSON))
    p.add_argument("--region-name", default="California heatwave above38 2019-06-11T00",
                   help="properties.index of the polygon in the geojson file.")
    p.add_argument("--target-vars", nargs="+", default=["2t"],
                   help="One or more AIFS outputs, e.g. tp ro. They share the forward rollouts.")
    p.add_argument("--target-time-start", type=datetime.fromisoformat, default=datetime(2019, 6, 11, 0))
    p.add_argument("--target-time-end", type=datetime.fromisoformat, default=None,
                   help="Inclusive end of the window (defaults to --target-time-start).")
    p.add_argument("--lead-times", type=int, nargs="+", default=None,
                   help="Explicit lead times in 6 h steps to the window start, e.g. 1 2 4 8 12. "
                        "Overrides --shortest/--longest.")
    p.add_argument("--shortest-lead-time-timesteps", type=int, default=1)
    p.add_argument("--longest-lead-time-timesteps", type=int, default=24)
    p.add_argument("--ig-steps", type=int, default=16, help="Gauss-Legendre points along the IG path.")
    p.add_argument("--ig-batch-size", type=int, default=1, help="Path points per forward pass (memory permitting).")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable activation checkpointing (faster, more memory).")
    p.add_argument("--time-operation", default="sum", choices=["mean", "sum", "max", "min"])
    p.add_argument("--space-operation", default="mean", choices=["mean", "sum", "max", "min"])
    p.add_argument("--grad-scale", type=float, default=1.0,
                   help="Multiply the target before backprop (and divide gradients after) to avoid "
                        "fp16 underflow, e.g. 1000 for 2t.")
    p.add_argument("--mask-mass", type=float, default=0.99,
                   help="Keep the grid points carrying this fraction of total |attribution|; 1 keeps all.")
    p.add_argument("--keep-input-lead-times", action="store_true",
                   help="Save the -6 h and 0 h inputs separately instead of summing them.")
    p.add_argument("--crop-box", action="store_true",
                   help="Crop the saved field to the box around the kept points.")
    p.add_argument("--full-output", action="store_true",
                   help="Save the raw attribution field without any masking or summing.")
    p.add_argument("--out-root", default=str(config.OUT_ROOT))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--predictions-only", action="store_true",
                   help="Only run the forecasts (no IG) and save one predictions file.")
    args = p.parse_args(argv)

    if args.target_time_end is None:
        args.target_time_end = args.target_time_start
    if args.target_time_end < args.target_time_start:
        p.error("--target-time-end must be >= --target-time-start")
    if args.shortest_lead_time_timesteps > args.longest_lead_time_timesteps:
        p.error("--shortest-lead-time-timesteps must be <= --longest-lead-time-timesteps")
    if not 0 < args.mask_mass <= 1:
        p.error("--mask-mass must be in (0, 1]")
    if args.ig_steps < 1 or args.ig_batch_size < 1:
        p.error("--ig-steps and --ig-batch-size must be >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
