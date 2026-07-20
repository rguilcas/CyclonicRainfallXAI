import aiohttp
_orig_init = aiohttp.ClientSession.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("trust_env", True)
    _orig_init(self, *args, **kwargs)
aiohttp.ClientSession.__init__ = _patched_init

import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import torch
import xarray as xr

os.makedirs("outputs", exist_ok=True)

# Model checkpoints/invariants: persistent, reused across runs.
os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
# Fetched input data (ERA5/GFS grids): temporary, deleted at the end so it
# doesn't accumulate on disk across runs.
tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS


def make_prediction(day):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lat_bounds = (58.5, 63.0)
    lon_bounds = (4.5, 9.0)
    nsteps = 24  # 24 x 6h = 6 days

    try:
        package = AIFS.load_default_package()
        model = AIFS.load_model(package).to(device).eval()

        ic = model.input_coords()
        data = ARCO()
        x, coords = fetch_data(
            source=data,
            time=[datetime.strptime(day, "%Y-%m-%d")],
            variable=ic["variable"],
            lead_time=ic["lead_time"],
            device=device,
        )

        out_template = model.output_coords(ic)
        lat, lon = out_template["lat"], out_template["lon"]
        lat_idx = np.where((lat >= lat_bounds[0]) & (lat <= lat_bounds[1]))[0]
        lon_idx = np.where((lon >= lon_bounds[0]) & (lon <= lon_bounds[1]))[0]

        tp_records = []
        lead_hours = []

        with torch.no_grad():
            for i, (out, out_coords) in enumerate(model.create_iterator(x, coords)):
                if i == 0:
                    continue  # step 0 is just the echoed initial condition, no forecast yet
                if i > nsteps:
                    break

                tp_idx = list(out_coords["variable"]).index("tp06")
                tp_region = out[..., tp_idx, :, :][..., lat_idx, :][..., :, lon_idx]
                tp_records.append(tp_region.squeeze().cpu().numpy())
                lead_hours.append(int(out_coords["lead_time"][0] / np.timedelta64(1, "h")))

        tp_array = np.stack(tp_records, axis=0)  # (nsteps, lat_west_norway, lon_west_norway)

        ds = xr.Dataset(
            {"tp06": (["lead_hour", "lat", "lon"], tp_array)},
            coords={"lead_hour": lead_hours, "lat": lat[lat_idx], "lon": lon[lon_idx]},
            attrs={"init_time": day, "model": "AIFS"},
        )
        out_dir = "/cluster/projects/nn12107k/robin/earth2studio/aifs_attributions_vestlandet"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/aifs_precip_west_norway_prediction_{day}.nc"
        ds.to_netcdf(out_path)
        print(f"Saved west Norway precipitation to {out_path}")

    finally:
        shutil.rmtree(tmp_data_cache, ignore_errors=True)
        print(f"Cleaned up temporary input data cache: {tmp_data_cache}")

if __name__=='__main__':
    make_prediction("1979-11-23")
    make_prediction("1979-11-22")
    make_prediction("1979-11-21")
    make_prediction("1979-11-20")
    make_prediction("1979-11-19")
    make_prediction("1979-11-18")
    make_prediction("1979-11-17")
    make_prediction("1979-11-16")
    make_prediction("1979-11-15")
    make_prediction("1979-11-14")
        