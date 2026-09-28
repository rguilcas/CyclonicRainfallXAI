"""
Download an ERA5 climatology for the 7 AIFS variables not covered by
WB2Climatology: 4 land-only soil variables (plain monthly mean -- no real
diurnal cycle) and 3 variables with a genuine diurnal cycle (climatology
resolved by hour of day).

Requires a free CDS account + API key: https://cds.climate.copernicus.eu/how-to-api
(creates ~/.cdsapirc with your key)
"""
import os
import cdsapi
import xarray as xr

# no meaningful diurnal cycle -- plain monthly mean keeps the download small
MONTHLY_VARS = {
    "swvl1": "volumetric_soil_water_layer_1",
    "swvl2": "volumetric_soil_water_layer_2",
    "stl1": "soil_temperature_level_1",
    "stl2": "soil_temperature_level_2",
}

# real diurnal cycle -- climatology resolved by hour of day
HOURLY_VARS = {
    "d2m": "2m_dewpoint_temperature",
    "skt": "skin_temperature",
    "tcw": "total_column_water",
}

HOURS = ["00:00", "06:00", "12:00", "18:00"]


def _download_and_average(client, variables, product_type, time, years, month_str, raw_path):
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": [product_type],
            "variable": list(variables.values()),
            "year": years,
            "month": [month_str],
            "time": time,
            "data_format": "netcdf",
        },
    ).download(raw_path)

    ds_raw = xr.open_dataset(raw_path)
    time_dim = "valid_time" if "valid_time" in ds_raw.dims else "time"

    if product_type == "monthly_averaged_reanalysis":
        ds_clim = ds_raw.mean(time_dim)
    else:
        # by-hour-of-day request: time_dim spans (year x hour) -- group by hour
        # of day and average across years, keeping the hour axis
        hour_of_day = ds_raw[time_dim].dt.hour
        ds_clim = ds_raw.groupby(hour_of_day).mean(time_dim)
        ds_clim = ds_clim.rename({"hour": "hour_of_day"})

    rename_map = {long: short for short, long in variables.items() if long in ds_clim.data_vars}
    return ds_clim.rename(rename_map)


def download_era5_climatology(months, year_start=1990, year_end=2019,
                               out_dir="/cluster/projects/nn12107k/robin/era5_climatology"):
    """Build an ERA5 climatology for all 7 missing AIFS variables, one file per
    month: era5_climatology_month{MM}.nc. Every variable ends up with a
    `hour_of_day` dimension (00/06/12/18 UTC) -- soil variables are simply
    broadcast across it so the file has one consistent shape.
    """
    os.makedirs(out_dir, exist_ok=True)
    client = cdsapi.Client()
    years = [str(y) for y in range(year_start, year_end + 1)]

    for month in months:
        month_str = f"{month:02d}"
        clim_path = os.path.join(out_dir, f"era5_climatology_month{month_str}_{year_start}-{year_end}.nc")
        if os.path.exists(clim_path):
            print(f"{clim_path} already exists, skipping")
            continue

        print(f"[{month_str}] downloading monthly-mean soil variables...")
        raw_monthly = os.path.join(out_dir, f"_raw_monthly_{month_str}.nc")
        ds_monthly = _download_and_average(
            client, MONTHLY_VARS, "monthly_averaged_reanalysis", ["00:00"],
            years, month_str, raw_monthly,
        )
        ds_monthly = ds_monthly.expand_dims(hour_of_day=[0, 6, 12, 18])

        print(f"[{month_str}] downloading by-hour-of-day variables (d2m, skt, tcw)...")
        raw_hourly = os.path.join(out_dir, f"_raw_hourly_{month_str}.nc")
        ds_hourly = _download_and_average(
            client, HOURLY_VARS, "monthly_averaged_reanalysis_by_hour_of_day", HOURS,
            years, month_str, raw_hourly,
        )
        ds_hourly = ds_hourly.assign_coords(hour_of_day=[0, 6, 12, 18])

        ds_clim = xr.merge([ds_monthly, ds_hourly])
        ds_clim.to_netcdf(clim_path)
        os.remove(raw_monthly)
        os.remove(raw_hourly)
        print(f"[{month_str}] saved {clim_path}")


if __name__ == "__main__":
    download_era5_climatology(months=[1,2,3,4,5,6,7,8, 9,10,11,12], year_start=1990, year_end=2019)