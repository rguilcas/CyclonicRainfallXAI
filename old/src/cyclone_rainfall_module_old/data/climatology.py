from earth2studio.data import WB2Climatology
from earth2studio.data.utils import fetch_data
from earth2studio.lexicon.wb2 import WB2ClimatetologyLexicon
import torch
from collections import OrderedDict
import numpy as np

import glob
import os


import xarray as xr

ERA5_CLIMATOLOGY_DIR = "/cluster/projects/nn12107k/robin/era5_climatology"


def _align_era5_to_grid(ds, target_lat, target_lon):
    """Align an ERA5 climatology dataset (lat/lon or latitude/longitude coords,
    either -180/180 or 0/360 longitude convention) onto the exact lat/lon grid
    used elsewhere in the pipeline (from coords_cov)."""
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    ds = ds.rename({lat_name: "lat", lon_name: "lon"})
    if ds.lon.max() > 180 and target_lon.max() <= 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    elif ds.lon.max() <= 180 and target_lon.max() > 180:
        ds = ds.assign_coords(lon=ds.lon % 360).sortby("lon")
    return ds.interp(lat=target_lat, lon=target_lon, method="nearest")



def build_climatology_baseline(model, init_dt, device, x_actual, coords_actual):
    ic = model.input_coords()
    all_vars = list(ic["variable"])
    covered = [v for v in all_vars if v in WB2ClimatetologyLexicon.VOCAB]
    missing = [v for v in all_vars if v not in WB2ClimatetologyLexicon.VOCAB]
    if missing:
        print(f"No WB2 climatology for {len(missing)} vars, using ERA5 monthly climatology instead: {missing}")

    clim = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr")
    x_cov, coords_cov = fetch_data(
        source=clim, time=[init_dt], variable=covered,
        lead_time=ic["lead_time"], device=device,
    )

    # ERA5 monthly climatology for the variables WB2 doesn't cover -- filename now
    # includes the year range, so glob for whichever range was actually downloaded
    era5_clim_pattern = os.path.join(ERA5_CLIMATOLOGY_DIR, f"era5_climatology_month{init_dt.month:02d}_*.nc")
    matches = glob.glob(era5_clim_pattern)
    if not matches:
        raise FileNotFoundError(f"No ERA5 climatology file found matching {era5_clim_pattern}")
    if len(matches) > 1:
        raise ValueError(f"Multiple ERA5 climatology files match {era5_clim_pattern}, expected exactly one: {matches}")
    era5_clim_path = matches[0]

    ds_era5 = xr.open_dataset(era5_clim_path)
    nearest_hour = min([0, 6, 12, 18], key=lambda h: abs(h - init_dt.hour))
    ds_era5_hour = ds_era5.sel(hour_of_day=nearest_hour)
    ds_era5_hour = _align_era5_to_grid(ds_era5_hour, coords_cov["lat"], coords_cov["lon"])

    x_clim = torch.zeros(
        (*x_cov.shape[:2], len(all_vars), *x_cov.shape[3:]),
        device=device, dtype=x_cov.dtype,
    )
    for i, v in enumerate(all_vars):
        if v in covered:
            x_clim[..., i, :, :] = x_cov[..., covered.index(v), :, :]
        else:
            era5_field = torch.as_tensor(
                ds_era5_hour[v].values, device=device, dtype=x_clim.dtype
            )
            x_clim[..., i, :, :] = era5_field

    x_clim = x_clim.unsqueeze(0)
    coords_clim = OrderedDict([("batch", np.array([0]))] + list(coords_cov.items()))
    coords_clim["variable"] = np.array(all_vars)
    return x_clim, coords_clim


def build_climatology_baseline_old2(model, init_dt, device, x_actual, coords_actual):
    ic = model.input_coords()
    all_vars = list(ic["variable"])
    covered = [v for v in all_vars if v in WB2ClimatetologyLexicon.VOCAB]
    missing = [v for v in all_vars if v not in WB2ClimatetologyLexicon.VOCAB]

    land_only_vars = {"swvl1", "swvl2", "stl1", "stl2"}   # meaningless over ocean
    zonal_mean_vars = {"d2m", "skt"}                       # strong lat gradient, no in-vocab proxy
    tcw_proxy = {"tcw": "tcwv"}                             # tcw ~= tcwv (+ condensate), tcwv is covered

    if missing:
        for v in missing:
            if v in land_only_vars:
                method = "ocean=actual value (zero attribution), land=land-only mean"
            elif v in tcw_proxy:
                method = f"climatological {tcw_proxy[v]} as proxy"
            elif v in zonal_mean_vars:
                method = "zonal (latitude-band) mean of sample"
            else:
                method = "global spatial mean of sample"
            print(f"  No climatology for {v}: baseline = {method}")

    # Fetch covered vars plus any proxy variables needed for the missing ones (e.g. tcwv for tcw)
    fetch_vars = list(dict.fromkeys(covered + list(tcw_proxy.values())))
    clim = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr")
    x_cov, coords_cov = fetch_data(
        source=clim, time=[init_dt], variable=fetch_vars,
        lead_time=ic["lead_time"], device=device,
    )

    lsm = model.invariants[0].to(device)   # (721, 1440), same grid as x_cov/x_actual
    land_mask = lsm > 0.5

    def zonal_mean(field):
        # (..., lat, lon) -> (..., lat, 1), broadcasts back over lon: per-latitude average,
        # preserves the equator-to-pole structure instead of collapsing to one global scalar.
        return field.mean(dim=-1, keepdim=True)

    x_clim = torch.zeros(
        (*x_cov.shape[:2], len(all_vars), *x_cov.shape[3:]),
        device=device, dtype=x_cov.dtype,
    )
    for i, v in enumerate(all_vars):
        if v in covered:
            x_clim[..., i, :, :] = x_cov[..., fetch_vars.index(v), :, :]
        elif v in tcw_proxy:
            x_clim[..., i, :, :] = x_cov[..., fetch_vars.index(tcw_proxy[v]), :, :]
        else:
            j = list(coords_actual["variable"]).index(v)
            actual_field = x_actual[..., j, :, :]  # (..., lat, lon)
            if v in land_only_vars:
                land_mean = actual_field[..., land_mask].mean(dim=-1, keepdim=True)
                fill = actual_field.clone()
                fill[..., land_mask] = land_mean.expand_as(fill[..., land_mask])
                x_clim[..., i, :, :] = fill
            elif v in zonal_mean_vars:
                x_clim[..., i, :, :] = zonal_mean(actual_field).expand(x_clim[..., i, :, :].shape)
            else:
                sample_mean = actual_field.mean(dim=(-2, -1), keepdim=True)
                x_clim[..., i, :, :] = sample_mean.expand(x_clim[..., i, :, :].shape)

    x_clim = x_clim.unsqueeze(0)
    coords_clim = OrderedDict([("batch", np.array([0]))] + list(coords_cov.items()))
    coords_clim["variable"] = np.array(all_vars)
    return x_clim, coords_clim

def build_climatology_baseline_old(model, init_dt, device, x_actual, coords_actual):
    ic = model.input_coords()
    all_vars = list(ic["variable"])
    covered = [v for v in all_vars if v in WB2ClimatetologyLexicon.VOCAB]
    missing = [v for v in all_vars if v not in WB2ClimatetologyLexicon.VOCAB]
    if missing:
        print(f"No climatology for {len(missing)} vars, using sample spatial mean: {missing}")

    clim = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr")
    x_cov, coords_cov = fetch_data(
        source=clim, time=[init_dt], variable=covered,
        lead_time=ic["lead_time"], device=device,
    )

    x_clim = torch.zeros(
        (*x_cov.shape[:2], len(all_vars), *x_cov.shape[3:]),
        device=device, dtype=x_cov.dtype,
    )
    for i, v in enumerate(all_vars):
        if v in covered:
            x_clim[..., i, :, :] = x_cov[..., covered.index(v), :, :]
        else:
            j = list(coords_actual["variable"]).index(v)
            sample_mean = x_actual[..., j, :, :].mean(dim=(-2, -1), keepdim=True)
            x_clim[..., i, :, :] = sample_mean.expand(x_clim[..., i, :, :].shape)

    x_clim = x_clim.unsqueeze(0)
    coords_clim = OrderedDict([("batch", np.array([0]))] + list(coords_cov.items()))
    coords_clim["variable"] = np.array(all_vars)
    return x_clim, coords_clim