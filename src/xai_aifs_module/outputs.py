"""Post-processing of attribution datasets before saving."""

import numpy as np


def shrink_attribution(ds, sum_lead=True, mass=0.99, crop_box=False, pad_deg=5):
    """Reduce the size of an attribution dataset.

    1. Optionally sum attributions over the input lead times (-6 h and 0 h),
       recording the share of |attribution| carried by the 0 h input.
    2. Keep the smallest set of grid points that carries ``mass`` of the total
       |attribution| (summed over all non-spatial dims); zero the rest.
    3. Optionally crop to the bounding box of the kept points, plus ``pad_deg``.

    Completeness is checked before shrinking; ``attribution_sum_kept`` records
    the sum that remains after masking so the loss can be checked per target.
    """
    a = ds["attribution"]
    if sum_lead:
        total = abs(a).sum(["input_lead_time", "variable", "lat", "lon"])
        latest = abs(a.isel(input_lead_time=-1)).sum(["variable", "lat", "lon"])
        ds["share_attributions_at_time_t0"] = latest / total
        a = a.sum("input_lead_time")

    # One spatial mask from |attr| summed over every non-spatial dimension.
    other = [d for d in a.dims if d not in ("lat", "lon")]
    mag = abs(a).sum(other)
    if 0 < mass < 1 and float(mag.sum()) > 0:
        v = np.sort(mag.values.ravel())[::-1]
        cum = np.cumsum(v) / v.sum()
        thr = v[min(np.searchsorted(cum, mass), len(v) - 1)]
        mask = mag >= thr
        a = a.where(mask, 0.0)
    else:
        mask = mag > 0

    if crop_box and bool(mask.any()):  # assumes the signal doesn't straddle the lon seam
        lats = a.lat[mask.any("lon")]
        lons = a.lon[mask.any("lat")]
        lat_lo, lat_hi = float(lats.min()) - pad_deg, float(lats.max()) + pad_deg
        lat_slice = slice(lat_hi, lat_lo) if a.lat[0] > a.lat[-1] else slice(lat_lo, lat_hi)
        a = a.sel(lat=lat_slice, lon=slice(float(lons.min()) - pad_deg, float(lons.max()) + pad_deg))

    ds = ds.drop_vars("attribution")
    if sum_lead:
        ds = ds.drop_dims("input_lead_time", errors="ignore")
    ds["attribution"] = a.astype(np.float32)
    ds["attribution_sum_kept"] = a.sum([d for d in a.dims if d != "target"])
    ds.attrs.update(mask_mass_fraction=mass, summed_input_lead_time=int(sum_lead),
                    cropped=int(crop_box))
    return ds
