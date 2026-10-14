import os, time
import xarray as xr, pandas as pd
from dask.distributed import Client, LocalCluster

OUT = "/cluster/projects/nn12107k/robin/data/ERA5_2023-2026"
URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def clean(ds):
    # drop encodings inherited from the cloud store (chunking/compression), which can break writing
    for v in ds.variables:
        ds[v].encoding = {}
    return ds


def main():
    os.makedirs(f"{OUT}/daily", exist_ok=True)
    ds = xr.open_zarr(URL, chunks={}, storage_options=dict(token="anon"))

    # --- orography, once ---
    oro_file = f"{OUT}/era5_orography_0p75.nc"
    if not os.path.exists(oro_file):
        oro = ds["geopotential_at_surface"]
        if "time" in oro.dims:
            oro = oro.isel(time=0)
        oro = (oro.sel(latitude=slice(85, 15))
                  .coarsen(latitude=3, longitude=3, boundary="trim").mean() / 9.81).rename("oro")
        oro.attrs["units"] = "m"
        clean(oro.load().to_dataset()).to_netcdf(oro_file)
        print("orography done", flush=True)

    # --- one file per day, skipping days already on disk ---
    days = [d for y in (2023, 2024, 2025)
              for d in pd.date_range(f"{y}-11-20", f"{y+1}-03-10")]
    for i, day in enumerate(days, 1):
        fn = f"{OUT}/daily/era5_feat_{day:%Y%m%d}_0p75.nc"
        if os.path.exists(fn):
            continue
        t0 = time.time()
        sub = ds.sel(time=pd.date_range(day, periods=4, freq="6h"), latitude=slice(85, 15))
        out = xr.Dataset({
            "t850":  sub["temperature"].sel(level=850),
            "q850":  sub["specific_humidity"].sel(level=850),
            "msl":   sub["mean_sea_level_pressure"],
            "ivt_u": sub["vertical_integral_of_eastward_water_vapour_flux"],
            "ivt_v": sub["vertical_integral_of_northward_water_vapour_flux"],
        }).drop_vars("level", errors="ignore")
        out = out.coarsen(latitude=3, longitude=3, boundary="trim").mean().load()
        clean(out).to_netcdf(fn + ".tmp")
        os.replace(fn + ".tmp", fn)          # file only appears once fully written
        print(f"[{i}/{len(days)}] {day:%Y-%m-%d}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=2, threads_per_worker=8, memory_limit="8GB")
    client = Client(cluster)
    print(client.dashboard_link, flush=True)
    main()