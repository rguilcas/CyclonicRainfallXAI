"""
AIFS forecasts of a region/window aggregate, without the climatological baseline
or any attribution.

For every lead time, AIFS is initialised from ERA5 (ARCO) and rolled out to the end
of the target window; the target variables are aggregated over the region nodes
(--space-operation) and window steps (--time-operation), exactly as in xai_aifs.py.
All lead times are saved in one netCDF file.

Example:
    python scripts/predict_region.py --region-name "California Central Valley" \
        --region-geojson-path aux/california_shapes.geojson --target-vars 2t \
        --target-time-start 2013-06-29T00 --target-time-end 2013-06-29T18 \
        --time-operation max --lead-times 4 8 12 16 20 24
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_aifs_module import config  # noqa: E402

config.set_earth2studio_env()  # must happen before earth2studio is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
from earth2studio.data import ARCO  # noqa: E402
from earth2studio.data.utils import fetch_data  # noqa: E402
from earth2studio.models.px import AIFS  # noqa: E402

from xai_aifs_module.data.inputs import region_nodes  # noqa: E402
from xai_aifs_module.helpers import fix_aiohttp  # noqa: E402
from xai_aifs_module.models.wrappers import AIFSRegionTarget  # noqa: E402

STEP = timedelta(hours=config.STEP_HOURS)
REDUCTIONS = ["mean", "sum", "max", "min"]


def predict(model, data, node_idx, init_time, n_steps, window_steps, args, device):
    """Aggregated forecast (n_targets,) for one initialisation."""
    ic = model.input_coords()
    x_ll, coords = fetch_data(source=data, time=[init_time], variable=ic["variable"],
                              lead_time=ic["lead_time"], device=device)
    f = AIFSRegionTarget(model, coords, n_steps, window_steps, node_idx, args.target_vars,
                         time_operation=args.time_operation,
                         space_operation=args.space_operation,
                         use_checkpoint=False).eval()
    with torch.no_grad():
        x_native = model._prepare_input(x_ll.unsqueeze(0), coords)
        return f(x_native)[0].cpu().numpy()


def run(args):
    fix_aiohttp()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    window = args.target_time_end - args.target_time_start
    if window % STEP:
        raise ValueError(f"Target window must be a multiple of {config.STEP_HOURS} h")
    n_window = window // STEP + 1
    leads = sorted({lead for lead in (args.lead_times or range(args.shortest_lead_time_timesteps,
                                                               args.longest_lead_time_timesteps + 1))
                    if lead >= 1})

    tag = (f"{args.region_name.replace(' ', '_')}_{'-'.join(args.target_vars)}"
           f"_{args.space_operation}-{args.time_operation}"
           f"_target{args.target_time_start:%Y%m%dT%H}-{args.target_time_end:%Y%m%dT%H}")
    file_out = Path(args.out_root) / "predictions" / f"predictions_{tag}.nc"
    if file_out.exists() and not args.overwrite:
        print(f"Exists, skipping: {file_out}")
        return
    file_out.parent.mkdir(parents=True, exist_ok=True)

    model = AIFS.load_model(AIFS.load_default_package()).to(device).eval()
    data = ARCO()
    node_idx = region_nodes(model, args.region_name, args.region_geojson_path)

    init_times, values = [], []
    for lead in leads:
        t0 = time.time()
        init_time = args.target_time_start - lead * STEP
        n_steps = lead + n_window - 1
        fx = predict(model, data, node_idx, init_time, n_steps,
                     list(range(lead - 1, n_steps)), args, device)
        init_times.append(init_time)
        values.append(fx)
        print(f"Lead {lead:2d} (init {init_time:%Y-%m-%dT%H}): "
              f"{dict(zip(args.target_vars, np.round(fx, 3)))}  [{time.time() - t0:.0f} s]")

    units = [config.default_scale(v)[1] for v in args.target_vars]
    ds = xr.Dataset(
        {"prediction": (["lead_steps", "target"], np.stack(values))},
        coords={"lead_steps": leads,
                "lead_hours": ("lead_steps", [lead * config.STEP_HOURS for lead in leads]),
                "init_time": ("lead_steps", init_times),
                "target": args.target_vars,
                "target_units": ("target", units)},
        attrs={"description": "AIFS forecast aggregated over a region and target window "
                              "(no baseline, no attribution).",
               "target_region": args.region_name,
               "region_geojson": str(args.region_geojson_path),
               "space_operation": args.space_operation,
               "time_operation": args.time_operation,
               "target_time_start": args.target_time_start.isoformat(),
               "target_time_end": args.target_time_end.isoformat(),
               "note": "lead_steps = number of 6 h steps from init to the window start."},
    )
    ds.to_netcdf(file_out)
    print(f"Saved {file_out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AIFS region/window forecasts without baseline or attribution.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--region-geojson-path", default=str(config.DEFAULT_REGION_GEOJSON))
    p.add_argument("--region-name", default="California heatwave above38 2019-06-11T00",
                   help="properties.index of the polygon in the geojson file.")
    p.add_argument("--target-vars", nargs="+", default=["2t"])
    p.add_argument("--target-time-start", type=datetime.fromisoformat, default=datetime(2019, 6, 11, 0))
    p.add_argument("--target-time-end", type=datetime.fromisoformat, default=None,
                   help="Inclusive end of the window (defaults to --target-time-start).")
    p.add_argument("--lead-times", type=int, nargs="+", default=None,
                   help="Lead times in 6 h steps to the window start. Overrides --shortest/--longest.")
    p.add_argument("--shortest-lead-time-timesteps", type=int, default=1)
    p.add_argument("--longest-lead-time-timesteps", type=int, default=24)
    p.add_argument("--time-operation", default="sum", choices=REDUCTIONS)
    p.add_argument("--space-operation", default="mean", choices=REDUCTIONS)
    p.add_argument("--out-root", default=str(config.OUT_ROOT))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if args.target_time_end is None:
        args.target_time_end = args.target_time_start
    if args.target_time_end < args.target_time_start:
        p.error("--target-time-end must be >= --target-time-start")
    if args.shortest_lead_time_timesteps > args.longest_lead_time_timesteps:
        p.error("--shortest-lead-time-timesteps must be <= --longest-lead-time-timesteps")
    return args


if __name__ == "__main__":
    run(parse_args())
