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

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
# Fetched input data (ERA5/GFS grids): temporary, deleted at the end so it
# doesn't accumulate on disk across runs.
tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS


init_dt = datetime(2023,1,4, 18)
nsteps = 3






device = "cuda" if torch.cuda.is_available() else "cpu"
package = AIFS.load_default_package()
model = AIFS.load_model(package).to(device).eval()




ic = model.input_coords()
data = ARCO()
x, coords = fetch_data(
    source=data,
    time=[init_dt],
    variable=ic["variable"],
    lead_time=ic["lead_time"],
    device=device,
)

TARGET_VARIABLE = 'u50'
with torch.no_grad():
    for i, (out, out_coords) in enumerate(model.create_iterator(x, coords)):

        if i == 0:
            continue  # step 0 is just the echoed initial condition, no forecast yet
        if i > nsteps:
            break
        u_idx = list(out_coords["variable"]).index(TARGET_VARIABLE)
        print(out.shape)
        # # select the 60N band, then average over lat-band AND all lon -> zonal mean
        # u_band = out[..., u_idx, :, :][..., lat_idx, :]
        # u_zonal_mean = u_band.mean(dim=(-2, -1))
        # u_records.append(u_zonal_mean.squeeze().cpu().numpy())
        # lead_hours.append(int(out_coords["lead_time"][0] / np.timedelta64(1, "h")))
