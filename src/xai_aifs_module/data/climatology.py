"""Climatological baseline for Integrated Gradients.

Variables covered by the WeatherBench2 climatology are taken from it; the rest
come from a monthly, hour-of-day ERA5 climatology downloaded beforehand
(see old/python_scripts/download_era5_climatology.py).
"""

from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import xarray as xr
from earth2studio.data import WB2Climatology
from earth2studio.data.utils import fetch_data
from earth2studio.lexicon.wb2 import WB2ClimatetologyLexicon

from xai_aifs_module.config import ERA5_CLIMATOLOGY_DIR, WB2_CLIMATOLOGY_STORE


@lru_cache(maxsize=4)
def _open_era5_climatology(month):
    """Open the ERA5 climatology file for a month (file name includes the year range)."""
    matches = sorted(ERA5_CLIMATOLOGY_DIR.glob(f"era5_climatology_month{month:02d}_*.nc"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ERA5 climatology file for month {month:02d} in "
            f"{ERA5_CLIMATOLOGY_DIR}, found {[m.name for m in matches]}"
        )
    return xr.open_dataset(matches[0])


def _align_to_grid(ds, target_lat, target_lon):
    """Put a dataset on the pipeline's lat/lon grid, whatever its coordinate names
    (lat/latitude) or longitude convention (-180/180 or 0/360)."""
    ds = ds.rename({
        "lat" if "lat" in ds.coords else "latitude": "lat",
        "lon" if "lon" in ds.coords else "longitude": "lon",
    })
    if ds.lon.max() > 180 and target_lon.max() <= 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    elif ds.lon.max() <= 180 and target_lon.max() > 180:
        ds = ds.assign_coords(lon=ds.lon % 360).sortby("lon")
    return ds.interp(lat=target_lat, lon=target_lon, method="nearest")


def _era5_field(var, valid_time, lat, lon):
    """ERA5 climatology of `var` for the month and nearest 6-hourly hour of `valid_time`."""
    ds = _open_era5_climatology(valid_time.month)
    hour = min((0, 6, 12, 18), key=lambda h: abs(h - valid_time.hour))
    return _align_to_grid(ds[var].sel(hour_of_day=hour), lat, lon).values


def build_climatology_baseline(model, init_time, device):
    """Return (x_clim, coords_clim) with the same layout as the model input.

    Each input lead time (e.g. -6 h and 0 h) gets the climatology of its own
    valid time, for both the WB2 and the ERA5 variables.
    """
    ic = model.input_coords()
    all_vars = list(ic["variable"])
    covered = [v for v in all_vars if v in WB2ClimatetologyLexicon.VOCAB]
    missing = [v for v in all_vars if v not in WB2ClimatetologyLexicon.VOCAB]
    if missing:
        print(f"  No WB2 climatology for {len(missing)} vars, using ERA5 climatology: {missing}")

    clim = WB2Climatology(climatology_zarr_store=WB2_CLIMATOLOGY_STORE)
    x_cov, coords_cov = fetch_data(
        source=clim, time=[init_time], variable=covered,
        lead_time=ic["lead_time"], device=device,
    )  # (time, lead, var, lat, lon)

    x_clim = torch.zeros((*x_cov.shape[:2], len(all_vars), *x_cov.shape[3:]),
                         device=device, dtype=x_cov.dtype)
    for i, v in enumerate(all_vars):
        if v in covered:
            x_clim[:, :, i] = x_cov[:, :, covered.index(v)]
            continue
        for j, lead in enumerate(ic["lead_time"]):
            valid_time = pd.Timestamp(init_time) + pd.Timedelta(lead)
            field = _era5_field(v, valid_time, coords_cov["lat"], coords_cov["lon"])
            x_clim[:, j, i] = torch.as_tensor(field, device=device, dtype=x_clim.dtype)

    x_clim = x_clim.unsqueeze(0)  # add batch dim
    coords_clim = OrderedDict([("batch", np.array([0]))] + list(coords_cov.items()))
    coords_clim["variable"] = np.array(all_vars)
    return x_clim, coords_clim

