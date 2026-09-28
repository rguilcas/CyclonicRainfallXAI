import os
import shutil
import tempfile
from collections import OrderedDict
from datetime import datetime, timedelta
import argparse

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.checkpoint import checkpoint
from captum.attr import IntegratedGradients

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS

import sys
sys.path.append('/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/src')

from cyclone_rainfall_module.helpers.aiohttpfix import fix_aiohttp
from cyclone_rainfall_module.data.climatology import build_climatology_baseline
from cyclone_rainfall_module.models.wrappers import AIFSPrecipRegionWrapperNativeAllPredictions, build_day_groups
from cyclone_rainfall_module.attributions.integratedgradients import get_IG_attribution_v2

fix_aiohttp()  # Patch aiohttp to trust environment variables for proxy settings

device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_REGION_GEOJSON = "/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/aux/rainfall_regions.geojson"


def parse_datetime(s):
    """Accepts e.g. 2023-08-08T12:00 or 2023-08-08 12:00."""
    return datetime.fromisoformat(s.replace(" ", "T"))


def compute_ig_attributions_for_target_window(target_start, target_end, max_lead_time_hours,
                                               region_name, region_geojson_path,
                                               overwrite=False):
    """Run get_IG_attribution_v2 across every 6-hourly init from
    (target_start - max_lead_time_hours) up to (but not including) target_start,
    for the accumulation window [target_start, target_end] and the named region.
    """
    window_tag = f"{target_start:%Y%m%dT%H}-{target_end:%Y%m%dT%H}"
    region_tag = region_name.replace(" ", "_")
    save_dir = f"/cluster/projects/nn12107k/robin/xai_aifs_v2/{region_tag}_{window_tag}/"
    os.makedirs(save_dir, exist_ok=True)

    if max_lead_time_hours % 6 != 0:
        raise ValueError(f"max_lead_time_hours must be a multiple of 6, got {max_lead_time_hours}")

    first_init_dt = target_start - timedelta(hours=max_lead_time_hours)
    hours_until_target = (target_start - first_init_dt).total_seconds() / 3600
    list_init_dt = [first_init_dt + timedelta(hours=6 * i) for i in range(int(hours_until_target // 6))]
    print(f"Computing IG attributions for region '{region_name}', target window "
          f"[{target_start}, {target_end}], max lead time {max_lead_time_hours}h (first init: {first_init_dt})")
    print(f"Number of attributions computed: {len(list_init_dt)}")

    package = AIFS.load_default_package()
    model = AIFS.load_model(package).to(device)
    ic = model.input_coords()
    data = ARCO()

    for init_dt in list_init_dt:
        out_path = os.path.join(
            save_dir,
            f"ig_attribution_{region_tag}_target{window_tag}_init{init_dt.strftime('%Y%m%dT%H')}.nc",
        )
        if os.path.exists(out_path):
            if overwrite:
                print(f"Overwriting init {init_dt} (already computed attributions in {out_path})")
            else:
                print(f"Skipping init {init_dt} (already computed attributions in {out_path})")
                continue
        ds_attr = get_IG_attribution_v2(
            model, init_dt, target_start, target_end,
            region_name, region_geojson_path,
            data, device, ic,
        )
        ds_attr.to_netcdf(out_path)
        print(f"Saved IG attribution for init {init_dt} to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AIFS IG attribution across 6-hourly init times leading up to a target period."
    )
    parser.add_argument(
        "--target-start", type=parse_datetime, required=True,
        help="Start of the target accumulation window, e.g. 2023-08-08T00:00.",
    )
    parser.add_argument(
        "--target-end", type=parse_datetime, required=True,
        help="End of the target accumulation window (inclusive), e.g. 2023-08-08T18:00. "
             "Use the same value as --target-start for a single 6h step.",
    )
    parser.add_argument(
        "--maxleadtime", type=int, required=True,
        help="How many hours before target_start to start initializing runs (must be a multiple of 6).",
    )
    parser.add_argument(
        "--region-name", type=str, default="West Norway",
        help="Feature 'index' value to look up in --region-geojson.",
    )
    parser.add_argument(
        "--region-geojson", type=str, default=DEFAULT_REGION_GEOJSON,
        help="Path to the region geojson file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing attribution files.",
    )

    args = parser.parse_args()

    if args.maxleadtime % 6 != 0:
        parser.error("--maxleadtime must be a multiple of 6")
    if args.target_start > args.target_end:
        parser.error("--target-start must be <= --target-end")

    compute_ig_attributions_for_target_window(
        args.target_start, args.target_end, max_lead_time_hours=args.maxleadtime,
        region_name=args.region_name, region_geojson_path=args.region_geojson,
        overwrite=args.overwrite,
    )