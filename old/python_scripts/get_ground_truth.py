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
import xarray as xr

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO

lat_bounds=(57.8, 63.5)
lon_bounds=(4.0, 12.5)

start =  datetime(2023, 7, 30, 00)
end =  datetime(2023, 8, 15, 18)
times = []
t = start
while t <= end:
    times.append(t)
    t += timedelta(hours=6)

try:
    data = ARCO()
    # "tp06" triggers ARCO's built-in 6-hour backward accumulation of the
    # underlying hourly ERA5 total_precipitation field -- the exact same
    # convention as AIFS's own tp06 output, so it's directly comparable.
    da = data(times, ["tp06"])

    lat = da["lat"].values
    lon = da["lon"].values
    lat_idx = np.where((lat >= lat_bounds[0]) & (lat <= lat_bounds[1]))[0]
    lon_idx = np.where((lon >= lon_bounds[0]) & (lon <= lon_bounds[1]))[0]

    da_region = da.isel(variable=0).isel(lat=lat_idx, lon=lon_idx)
    region_mean = da_region.mean(dim=["lat", "lon"])

    ds = xr.Dataset(
        {
            "tp06": (["time", "lat", "lon"], da_region.values),
            "tp06_region_mean": (["time"], region_mean.values),
        },
        coords={"time": times, "lat": lat[lat_idx], "lon": lon[lon_idx]},
        attrs={
            "description": "ARCO/ERA5 ground truth 6-hour accumulated total "
                            "precipitation (tp06) over west Norway (Vestlandet)",
            "units": "m (accumulated over preceding 6h)",
        },
    )
    out_dir = f"/cluster/projects/nn12107k/robin/earth2studio/aifs_prediction_hans/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "arco_ground_truth_tp06_vestlandet_2023-07-30_to_2023-08-15.nc")
    ds.to_netcdf(out_path)
    print(f"Saved {out_path}")
    print(ds["tp06_region_mean"].to_series())

finally:
    shutil.rmtree(tmp_data_cache, ignore_errors=True)
    print(f"Cleaned up temporary input data cache: {tmp_data_cache}")