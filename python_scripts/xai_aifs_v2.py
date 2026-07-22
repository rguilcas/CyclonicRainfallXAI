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
import torch

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
from cyclone_rainfall_module.attributions.integratedgradients import get_IG_attribution, get_IG_NT_attribution

fix_aiohttp()  # Patch aiohttp to trust environment variables for proxy settings

device = "cuda" if torch.cuda.is_available() else "cpu"

# target_day = '2016-08-08'
# max_lead_time_hours = 6




def compute_ig_attributions_for_target_day(target_day, max_lead_time_hours, overwrite=False):
    """
    pass
    """
    save_dir = f"/cluster/projects/nn12107k/robin/xai_aifs_v2/{target_day}/"
    os.makedirs(save_dir, exist_ok=True)
    if max_lead_time_hours%6 != 0:
        raise ValueError(f"max_lead_time_hours must be a multiple of 6, got {max_lead_time_hours}")
    first_init_dt = datetime.strptime(target_day, '%Y-%m-%d') - timedelta(hours = max_lead_time_hours)
    hours_until_target_day = ((datetime.strptime(target_day, '%Y-%m-%d') - first_init_dt).total_seconds() / 3600)
    list_init_dt = [first_init_dt + timedelta(hours=6 * i) for i in range(int(hours_until_target_day // 6))]  # 2016-08-07 12, 18, 2016-08-08 00, 06
    print(f"Computing IG attributions for target day {target_day} with max lead time {max_lead_time_hours}h (first init: {first_init_dt})")
    print(f"Number of attributions computed: {len(list_init_dt)}")

    package = AIFS.load_default_package()
    model = AIFS.load_model(package).to(device)
    ic = model.input_coords()
    data = ARCO()

    for init_dt in list_init_dt:
        out_file = os.path.join(save_dir, f"ig_nt_attribution_target{target_day}_init{init_dt.strftime('%Y%m%dT%H')}.nc")
        if out_file and os.path.exists(out_file):
            if overwrite:
                print(f"Overwriting init {init_dt} (already computed attributions in {out_file})")
            else:
                print(f"Skipping init {init_dt} (already computed attributions in {out_file})")
                continue
        ds_attr = get_IG_NT_attribution(model, init_dt, target_day, data, device, ic)
        ds_attr.to_netcdf(os.path.join(save_dir, f"ig_nt_attribution_target{target_day}_init{init_dt.strftime('%Y%m%dT%H')}.nc"))
        print(f"Saved IG NT attribution for init {init_dt} to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AIFS IG attribution across 6-hourly init times leading up to a target day."
    )    
    parser.add_argument(
        "--target-day", type=str, required=True,
        help="Target valid day, e.g. 2005-09-14 (defaults to 06:00 UTC), or 2005-09-14T06:00 for a specific hour.",
    )
    parser.add_argument(
        "--maxleadtime", type=int, required=True,
        help="How many hours before target_day to start initializing runs (must be a multiple of 6).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing attribution files.",
    )
    
    args = parser.parse_args()

    if args.maxleadtime % 6 != 0:
        parser.error("--maxleadtime must be a multiple of 6")
    
    compute_ig_attributions_for_target_day(args.target_day, max_lead_time_hours=args.maxleadtime, overwrite=args.overwrite)