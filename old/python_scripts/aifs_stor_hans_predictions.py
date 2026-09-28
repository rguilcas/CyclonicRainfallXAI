import aiohttp
_orig_init = aiohttp.ClientSession.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("trust_env", True)
    _orig_init(self, *args, **kwargs)
aiohttp.ClientSession.__init__ = _patched_init

import os
import shutil
import tempfile
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS


def run_all_predictions(init_start, init_end, out_path,
                         lat_bounds=(57.8, 63.5), lon_bounds=(4.0, 12.5),
                         nsteps=24):
    """Run AIFS precipitation forecasts for every 6-hourly init time between
    init_start and init_end (inclusive), and save ONE combined file with
    dims (init_time, valid_time, lat, lon). Cells where a given init's 6-day
    rollout doesn't reach a particular valid_time are left as NaN.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_times = []
    t = init_start
    while t <= init_end:
        init_times.append(t)
        t += timedelta(hours=6)
    print(f"Running {len(init_times)} inits, every 6h from {init_start} to {init_end} "
          f"({nsteps} steps / {nsteps * 6}h rollout each)")

    package = AIFS.load_default_package()
    model = AIFS.load_model(package).to(device).eval()
    ic = model.input_coords()
    data = ARCO()

    out_template = model.output_coords(ic)
    lat, lon = out_template["lat"], out_template["lon"]
    lat_idx = np.where((lat >= lat_bounds[0]) & (lat <= lat_bounds[1]))[0]
    lon_idx = np.where((lon >= lon_bounds[0]) & (lon <= lon_bounds[1]))[0]
    lat_sel, lon_sel = lat[lat_idx], lon[lon_idx]

    # First pass: run every init, keep each one's (valid_time, lat, lon)
    # array, and collect the union of every valid_time seen across all inits.
    per_init_data = {}
    all_valid_times = set()

    for init_dt in init_times:
        tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
        os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache
        try:
            x, coords = fetch_data(
                source=data, time=[init_dt], variable=ic["variable"],
                lead_time=ic["lead_time"], device=device,
            )

            tp_records = []
            valid_times = []
            with torch.no_grad():
                for i, (out, out_coords) in enumerate(model.create_iterator(x, coords)):
                    if i == 0:
                        continue  # step 0 is just the echoed initial condition
                    if i > nsteps:
                        break
                    tp_idx = list(out_coords["variable"]).index("tp06")
                    tp_region = out[..., tp_idx, :, :][..., lat_idx, :][..., :, lon_idx]
                    tp_records.append(tp_region.squeeze().cpu().numpy())
                    lead_hours = int(out_coords["lead_time"][0] / np.timedelta64(1, "h"))
                    valid_times.append(np.datetime64(init_dt + timedelta(hours=lead_hours)))

            per_init_data[init_dt] = (np.array(valid_times), np.stack(tp_records, axis=0))
            all_valid_times.update(valid_times)
            print(f"  Ran init {init_dt}")
        finally:
            shutil.rmtree(tmp_data_cache, ignore_errors=True)

    # Second pass: build the combined (init_time, valid_time, lat, lon) array,
    # NaN-filled wherever a given init doesn't reach that valid_time.
    all_valid_times = np.array(sorted(all_valid_times))
    valid_time_index = {vt: i for i, vt in enumerate(all_valid_times)}

    n_init, n_valid = len(init_times), len(all_valid_times)
    n_lat, n_lon = len(lat_sel), len(lon_sel)
    tp_full = np.full((n_init, n_valid, n_lat, n_lon), np.nan, dtype=np.float32)

    for i, init_dt in enumerate(init_times):
        valid_times, tp_array = per_init_data[init_dt]
        for j, vt in enumerate(valid_times):
            tp_full[i, valid_time_index[vt]] = tp_array[j]

    ds = xr.Dataset(
        {"tp06": (["init_time", "valid_time", "lat", "lon"], tp_full)},
        coords={
            "init_time": np.array(init_times, dtype="datetime64[ns]"),
            "valid_time": all_valid_times,
            "lat": lat_sel,
            "lon": lon_sel,
        },
        attrs={
            "model": "AIFS",
            "event": "Storm Hans (Norway, 6-10 Aug 2023)",
            "region": "Norway south of Trondheim",
        },
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved combined forecast dataset to {out_path}")


if __name__ == '__main__':
    run_all_predictions(
        init_start=datetime(2023, 7, 31, 0),
        init_end=datetime(2023, 8, 9, 18),
        out_path="/cluster/projects/nn12107k/robin/earth2studio/aifs_prediction_hans/aifs_south_norway_hans_combined.nc",
    )