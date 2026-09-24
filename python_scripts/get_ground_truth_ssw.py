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


start = datetime(2018, 2, 1, 0)
end = datetime(2018, 2, 15, 18)
times = []
t = start
while t <= end:
    times.append(t)
    t += timedelta(hours=6)

try:
    data = ARCO()
    # u50 is an instantaneous field (zonal wind at 50 hPa at the analysis
    # time) -- not a 6h-accumulated quantity like tp06.
    da = data(times, ["u50"])

    da_u50 = da.isel(variable=0)

    # Label-based selection via an explicit boolean mask, not slice() --
    # ERA5/ARCO latitude is stored descending (90 -> -90), and slice(59, 61)
    # silently returns an empty result on a descending coordinate. Selecting
    # by an array of matching labels works regardless of sort order.
    lat_mask = (da_u50["lat"] >= 59) & (da_u50["lat"] <= 61)
    da_region = da_u50.sel(lat=da_u50["lat"][lat_mask])
    if da_region.sizes["lat"] == 0:
        raise ValueError("No latitudes matched 59-61N -- check da.lat.values for the actual range/order")

    region_mean = da_region.mean(dim=["lat", "lon"])

    ds = xr.Dataset(
        {
            "u50": (["time", "lat", "lon"], da_region.values),
            "u50_region_mean": (["time"], region_mean.values),
        },
        coords={"time": times, "lat": da_region["lat"].values, "lon": da_region["lon"].values},
        attrs={
            "description": "ARCO/ERA5 ground truth instantaneous zonal wind (u50, 50 hPa) "
                            "averaged over 59-61N, all longitudes",
            "units": "m/s (instantaneous)",
        },
    )
    out_dir = "/cluster/projects/nn12107k/robin/earth2studio/aifs_prediction_ssw/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "arco_ground_truth_u50_60N_2018-02-01_to_2018-02-15.nc")
    ds.to_netcdf(out_path)
    print(f"Saved {out_path}")
    print(ds["u50_region_mean"].to_series())

finally:
    shutil.rmtree(tmp_data_cache, ignore_errors=True)
    print(f"Cleaned up temporary input data cache: {tmp_data_cache}")