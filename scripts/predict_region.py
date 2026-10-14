"""
AIFS forecasts of a region/window aggregate from a range of initialisation times,
compared with ERA5 over the same region (no baseline, no attribution).

The target window is fixed (e.g. day D0, 00-18 UTC). AIFS is initialised at every
lead time between --min-lead and --max-lead before the window start (D0-6 h,
D0-12 h, ...), and all these forecasts run together in batches of --batch-size
initial states. The target variables are aggregated over the region nodes
(--space-operation) and window steps (--time-operation), as in xai_aifs.py.
ERA5 for the window is interpolated to the same native nodes and aggregated the
same way, giving the observed value to compare against.

Lead times are counted in 6 h steps from init to the window start; --lead-stride 4
gives daily inits (D0-1, D0-2, ...).

Example (Central Valley, daily max of 2 m temperature, inits 1 to 6 days ahead):
    python scripts/predict_region.py --region-name "California Central Valley" \
        --region-geojson-path aux/california_shapes.geojson --target-vars 2t \
        --target-time-start 2013-06-29T00 --target-time-end 2013-06-29T18 \
        --time-operation max --min-lead 4 --max-lead 24 --lead-stride 4
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_aifs_module import config  # noqa: E402

config.set_earth2studio_env()  # must happen before earth2studio is imported

import torch  # noqa: E402
import xarray as xr  # noqa: E402
from earth2studio.data.utils import fetch_data  # noqa: E402
from earth2studio.models.px import AIFS  # noqa: E402

from xai_aifs_module.data.inputs import region_nodes  # noqa: E402
from xai_aifs_module.data.truth import era5_region_series  # noqa: E402
from xai_aifs_module.helpers import era5_source, fix_aiohttp  # noqa: E402
from xai_aifs_module.models.rollout import batched_region_rollout  # noqa: E402
from xai_aifs_module.models.wrappers import _reduce  # noqa: E402

STEP = timedelta(hours=config.STEP_HOURS)
REDUCTIONS = ["mean", "sum", "max", "min"]


def prepare_init(model, data, init_time, device):
    """Native-grid initial state for one init time (1, 2, nodes, channels)."""
    ic = model.input_coords()
    x_ll, coords = fetch_data(source=data, time=[init_time], variable=ic["variable"],
                              lead_time=ic["lead_time"], device=device)
    with torch.no_grad():
        return model._prepare_input(x_ll.unsqueeze(0), coords), coords


def forecast_series(model, data, node_idx, leads, n_window, args, device):
    """Regional series over the window for every lead: (n_leads, n_window, n_targets)."""
    out = {}
    # Longest leads first so each batch holds similar rollout lengths.
    ordered = sorted(leads, reverse=True)
    for start in range(0, len(ordered), args.batch_size):
        chunk = ordered[start:start + args.batch_size]
        n_steps = max(chunk) + n_window - 1
        t0 = time.time()
        inits = [args.target_time_start - lead * STEP for lead in chunk]
        prepared = [prepare_init(model, data, t, device) for t in inits]
        series = batched_region_rollout(model, [p[0] for p in prepared], [p[1] for p in prepared],
                                        n_steps, node_idx, args.target_vars,
                                        space_operation=args.space_operation)  # (B, n_steps, T)
        for b, lead in enumerate(chunk):
            # Step i is valid at init + (i + 1) * 6 h, so the window starts at step lead - 1.
            out[lead] = series[b, lead - 1:lead - 1 + n_window].cpu()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Leads {chunk} ({n_steps} steps, inits {inits[-1]:%Y-%m-%dT%H} to "
              f"{inits[0]:%Y-%m-%dT%H}): {time.time() - t0:.0f} s")
    return torch.stack([out[lead] for lead in leads])


def run(args):
    fix_aiohttp()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    window = args.target_time_end - args.target_time_start
    if window % STEP:
        raise ValueError(f"Target window must be a multiple of {config.STEP_HOURS} h")
    n_window = window // STEP + 1
    valid_times = [args.target_time_start + i * STEP for i in range(n_window)]
    leads = list(range(args.min_lead, args.max_lead + 1, args.lead_stride))

    tag = (f"{args.region_name.replace(' ', '_')}_{'-'.join(args.target_vars)}"
           f"_{args.space_operation}-{args.time_operation}"
           f"_target{args.target_time_start:%Y%m%dT%H}-{args.target_time_end:%Y%m%dT%H}")
    file_out = Path(args.out_root) / "predictions" / f"predictions_{tag}.nc"
    if file_out.exists() and not args.overwrite:
        print(f"Exists, skipping: {file_out}")
        return
    file_out.parent.mkdir(parents=True, exist_ok=True)

    model = AIFS.load_model(AIFS.load_default_package()).to(device).eval()
    data = era5_source(keep_cache=args.keep_era5_cache)
    node_idx = region_nodes(model, args.region_name, args.region_geojson_path)
    scale = torch.tensor([config.default_scale(v)[0] for v in args.target_vars])

    # ERA5 over the same nodes and window, aggregated the same way.
    truth_series = era5_region_series(model, data, valid_times, node_idx, args.target_vars,
                                      device, space_operation=args.space_operation).cpu() * scale
    truth = _reduce(truth_series, args.time_operation, dim=0)

    fc_series = forecast_series(model, data, node_idx, leads, n_window, args, device) * scale
    fc = _reduce(fc_series, args.time_operation, dim=1)  # (n_leads, T)

    for t, v in enumerate(args.target_vars):
        print(f"{v}: ERA5 {truth[t]:.3f} | " + ", ".join(
            f"L{lead}: {fc[i, t]:.3f}" for i, lead in enumerate(leads)))

    units = [config.default_scale(v)[1] for v in args.target_vars]
    ds = xr.Dataset(
        {
            "prediction": (["lead_steps", "target"], fc.numpy()),
            "truth": (["target"], truth.numpy()),
            "error": (["lead_steps", "target"], (fc - truth).numpy()),
            "prediction_series": (["lead_steps", "valid_time", "target"], fc_series.numpy()),
            "truth_series": (["valid_time", "target"], truth_series.numpy()),
        },
        coords={
            "lead_steps": leads,
            "lead_hours": ("lead_steps", [lead * config.STEP_HOURS for lead in leads]),
            "init_time": ("lead_steps", [args.target_time_start - lead * STEP for lead in leads]),
            "valid_time": valid_times,
            "target": args.target_vars,
            "target_units": ("target", units),
        },
        attrs={
            "description": "AIFS forecasts aggregated over a region and target window from several "
                           "init times, with ERA5 aggregated the same way (no baseline, no attribution).",
            "target_region": args.region_name,
            "region_geojson": str(args.region_geojson_path),
            "space_operation": args.space_operation,
            "time_operation": args.time_operation,
            "target_time_start": args.target_time_start.isoformat(),
            "target_time_end": args.target_time_end.isoformat(),
            "note": "lead_steps = number of 6 h steps from init to the window start. ERA5 is "
                    "interpolated to the AIFS native grid with the model's interpolation matrix. "
                    "Accumulated variables cover the 6 h ending at each valid time.",
        },
    )
    ds.to_netcdf(file_out)
    print(f"Saved {file_out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AIFS region/window forecasts from many init times, with ERA5.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--region-geojson-path", default=str(config.DEFAULT_REGION_GEOJSON))
    p.add_argument("--region-name", default="California heatwave above38 2019-06-11T00",
                   help="properties.index of the polygon in the geojson file.")
    p.add_argument("--target-vars", nargs="+", default=["2t"])
    p.add_argument("--target-time-start", type=datetime.fromisoformat, default=datetime(2019, 6, 11, 0))
    p.add_argument("--target-time-end", type=datetime.fromisoformat, default=None,
                   help="Inclusive end of the window (defaults to --target-time-start).")
    p.add_argument("--min-lead", type=int, default=1,
                   help="Shortest lead, in 6 h steps from init to the window start.")
    p.add_argument("--max-lead", type=int, default=24,
                   help="Longest lead, in 6 h steps from init to the window start.")
    p.add_argument("--lead-stride", type=int, default=1,
                   help="Spacing between leads in 6 h steps (4 = one init per day).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Init times run together per forward pass (lower if GPU memory runs out).")
    p.add_argument("--time-operation", default="sum", choices=REDUCTIONS)
    p.add_argument("--space-operation", default="mean", choices=REDUCTIONS)
    p.add_argument("--keep-era5-cache", action="store_true",
                   help="Keep downloaded ERA5 fields on disk (by default they are deleted after use).")
    p.add_argument("--out-root", default=str(config.OUT_ROOT))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if args.target_time_end is None:
        args.target_time_end = args.target_time_start
    if args.target_time_end < args.target_time_start:
        p.error("--target-time-end must be >= --target-time-start")
    if not 1 <= args.min_lead <= args.max_lead:
        p.error("need 1 <= --min-lead <= --max-lead")
    if args.lead_stride < 1 or args.batch_size < 1:
        p.error("--lead-stride and --batch-size must be >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
