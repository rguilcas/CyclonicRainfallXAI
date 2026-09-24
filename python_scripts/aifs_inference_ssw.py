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
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS

# --- SSW diagnostic settings ------------------------------------------------
# Canonical SSW definition (WMO / Charlton & Polvani 2007): zonal-mean zonal
# wind at 10 hPa, 60N reverses from westerly to easterly in winter.
#
# AIFS's pressure levels are 50/100/150/200/250/300/400/500/600/700/850/925/
# 1000 hPa (+ 10m surface) -- there is NO 10 hPa level. 50 hPa (the model's
# highest/coldest level) is the closest available proxy, but it's ~20 km vs.
# 10 hPa's ~32 km: the reversal signal there is weaker, lags the true 10 hPa
# reversal by days, and won't line up with published SSW wind thresholds.
# Treat this as "does the model see a vortex-weakening signal propagate down
# into the lower stratosphere", not a literal reproduction of the WMO
# diagnostic. Change TARGET_LEVEL_HPA if a stratosphere-extended AIFS
# variant with a true 10 hPa level becomes available.
TARGET_LEVEL_HPA = 50
TARGET_VARIABLE = f"u{TARGET_LEVEL_HPA}"
TARGET_LAT = 60.0
LAT_HALF_WIDTH = 1.0  # average over a +/-1 deg band around 60N (grid-resolution safety margin)


def make_prediction(day):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nsteps = 48  # 24 x 6h = 6 days; raise this if you want to track an SSW further into its evolution

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

        # Fail fast (before burning GPU time on the rollout) if the target
        # variable isn't actually in this model's output, and tell you what
        # u-wind levels ARE available.
        available_vars = list(out_template["variable"])
        if TARGET_VARIABLE not in available_vars:
            u_levels = sorted(
                int(v[1:]) for v in available_vars if v.startswith("u") and v[1:].isdigit()
            )
            raise ValueError(
                f"{TARGET_VARIABLE!r} not in AIFS output variables. "
                f"Available u-wind pressure levels (hPa): {u_levels}"
            )

        lat, lon = out_template["lat"], out_template["lon"]
        lat_idx = np.where(
            (lat >= TARGET_LAT - LAT_HALF_WIDTH) & (lat <= TARGET_LAT + LAT_HALF_WIDTH)
        )[0]
        if lat_idx.size == 0:
            raise ValueError(f"No grid latitudes found within {LAT_HALF_WIDTH} deg of {TARGET_LAT}N")

        u_records = []
        lead_hours = []

        with torch.no_grad():
            for i, (out, out_coords) in enumerate(model.create_iterator(x, coords)):
                if i == 0:
                    continue  # step 0 is just the echoed initial condition, no forecast yet
                if i > nsteps:
                    break

                u_idx = list(out_coords["variable"]).index(TARGET_VARIABLE)
                # select the 60N band, then average over lat-band AND all lon -> zonal mean
                u_band = out[..., u_idx, :, :][..., lat_idx, :]
                u_zonal_mean = u_band.mean(dim=(-2, -1))
                u_records.append(u_zonal_mean.squeeze().cpu().numpy())
                lead_hours.append(int(out_coords["lead_time"][0] / np.timedelta64(1, "h")))

        u_array = np.stack(u_records, axis=0)  # (nsteps,)

        ds = xr.Dataset(
            {"u_zonal_mean": (["lead_hour"], u_array)},
            coords={"lead_hour": lead_hours},
            attrs={
                "init_time": day,
                "model": "AIFS",
                "variable": TARGET_VARIABLE,
                "target_level_hpa": TARGET_LEVEL_HPA,
                "target_lat": TARGET_LAT,
                "lat_half_width_deg": LAT_HALF_WIDTH,
                "note": "50 hPa used as nearest available proxy for the canonical 10 hPa SSW level -- see script header comment",
            },
        )
        out_dir = "/cluster/projects/nn12107k/robin/earth2studio/aifs_prediction_ssw"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/aifs_u{TARGET_LEVEL_HPA}_60N_prediction_{day}.nc"
        ds.to_netcdf(out_path)
        print(f"Saved zonal-mean u{TARGET_LEVEL_HPA} @ {TARGET_LAT}N to {out_path}")

    finally:
        shutil.rmtree(tmp_data_cache, ignore_errors=True)
        print(f"Cleaned up temporary input data cache: {tmp_data_cache}")


if __name__ == "__main__":
    # Placeholder dates carried over from your original script -- for actual
    # SSW work you'll want initializations bracketing known major SSW central
    # dates, e.g. 2009-01-24, 2013-01-06, 2018-02-12, 2019-01-02, 2021-01-05,
    # 2023-02-16.
    
    make_prediction("2018-02-01")
    make_prediction("2018-02-02")
    make_prediction("2018-02-03")
    make_prediction("2018-02-04")
    make_prediction("2018-02-05")
    make_prediction("2018-02-06")
    make_prediction("2018-02-07")
    make_prediction("2018-02-08")
    make_prediction("2018-02-09")
    make_prediction("2018-02-10")
    make_prediction("2018-02-11")
    make_prediction("2018-02-12")
    make_prediction("2018-02-13")
    