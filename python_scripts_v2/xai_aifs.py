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
import contextlib
import os


from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS

os.environ["EARTH2STUDIO_CACHE"] = "/cluster/projects/nn12107k/robin/earth2studio_cache"
os.environ["EARTH2STUDIO_DISABLE_MSC"] = "1"

torch.manual_seed(123)
np.random.seed(123)

tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache

import sys
sys.path.append('/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/src')

from cyclone_rainfall_module.helpers.aiohttpfix import fix_aiohttp
from cyclone_rainfall_module.data.climatology import build_climatology_baseline
from cyclone_rainfall_module.models.wrappers import AIFSPrecipRegionWrapperNativeAllPredictions, build_day_groups, AIFSWrapper
from cyclone_rainfall_module.attributions.integratedgradients import get_IG_attribution_v2
from cyclone_rainfall_module.helpers.region_mask import load_region_mask

fix_aiohttp()  # Patch aiohttp to trust environment variables for proxy settings

device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_REGION_GEOJSON = "/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/aux/rainfall_regions.geojson"


def get_region_node_mask(model, region_name, region_geojson_path):

    model_native_lat = model.latitudes.detach().flatten().cpu().numpy()
    model_native_lon = model.longitudes.detach().flatten().cpu().numpy()

    native_node_mask = load_region_mask(region_geojson_path, region_name, model_native_lat, model_native_lon)

    print(f"{region_name} native node mask: {len(native_node_mask)} nodes selected out of {len(model_native_lat)} total nodes")

    return native_node_mask



def build_day_groups(init_dt, nsteps, boundary_hour=0):
    """Group the model's 6-hourly steps (step k valid at init_dt + k*6h, k=1..nsteps)
    into fixed calendar-day buckets. boundary_hour=0 -> standard UTC day (00-24 UTC).
    boundary_hour=6 -> 06-06 UTC "hydrological day". Returns a list of lists of
    0-indexed step positions (index 0 == first step), matching the order forward()
    accumulates them in.
    """
    valid_times = [init_dt + timedelta(hours=6 * step) for step in range(1, nsteps + 1)]
    groups = {}
    for i, vt in enumerate(valid_times):
        day_key = (vt - timedelta(hours=boundary_hour)).date()
        groups.setdefault(day_key, []).append(i)
    return [groups[k] for k in sorted(groups)]


def compute_attributions(model, data,
                         native_node_mask,
                         target_time_start, 
                         target_time_end,
                         prediction_lead_time_timesteps,
                         delta_t_hours=6, target_var='tp',
                         ig_step=10, ig_internal_batch_size=1,
                         time_operation='sum',
                         predictions_only=False):
    
    tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
    os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache
    
    # Load AIFS data
    try:
        ic = model.input_coords()
        delta_time = target_time_end - target_time_start
        delta_timestep = int(delta_time/timedelta(hours=delta_t_hours))
        true_lead_time = prediction_lead_time_timesteps + delta_timestep
        lead_time_hours = timedelta(hours=delta_t_hours*prediction_lead_time_timesteps)
        true_lead_time_hours = timedelta(hours=delta_t_hours*true_lead_time)
        init_time = target_time_end - lead_time_hours
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                x, coords = fetch_data(
                    source=data,
                    time=[init_time],
                    variable=ic["variable"],
                    lead_time=ic["lead_time"],
                    device=device,
                )
        x = x.unsqueeze(0)

        with torch.no_grad():
            x_native0 = model._prepare_input(x, coords)

        for ctx_cls in (torch.no_grad, torch.inference_mode):
            ctx_cls.__enter__ = lambda self: None
            ctx_cls.__exit__ = lambda self, *args: None

        wrapper = AIFSWrapper(model, 
                              n_steps=true_lead_time, 
                              coords=coords, 
                              target_var=target_var, 
                              return_timeseries=False, 
                              node_id=native_node_mask,
                              time_operation=time_operation ,
                              return_last_n_steps=delta_timestep+1, 
                              progress_bar=False).to(device).eval()
        
        prediction = wrapper(x_native0).detach().cpu().numpy().squeeze(0)
        if predictions_only:
            return xr.DataArray([prediction], dims=['init_time'], coords={'init_time': [init_time]},name='predictions', attrs={'description': f"AIFS {target_var} prediction for target time {target_time}"})

        x_clim, coords_clim = build_climatology_baseline(model, init_time, device, x, coords)
        with torch.no_grad():
            baseline_native = model._prepare_input(x_clim, coords_clim)
        baseline_native = baseline_native.to(device).requires_grad_(True)
        ig = IntegratedGradients(wrapper)

        x_ig = x_native0.clone().requires_grad_(True)
        print("  Computing Integrated Gradients attribution...")
        attr, conv = ig.attribute(x_ig, baselines=baseline_native, n_steps=ig_step, internal_batch_size=ig_internal_batch_size, return_convergence_delta=True)
        attr = attr.detach().cpu()
        torch.cuda.empty_cache()
        print(f"Target time: {target_time}, prediction time: {init_time} - sum={attr.sum().item():.4e}")
        
        attr_stack = attr.numpy().squeeze(0)  # (input_lead_time, node, variable)
        lead_time_hours = (model.input_coords()["lead_time"] / np.timedelta64(1, "h")).astype(int)
        node_lat = model.latitudes.detach().flatten().cpu().numpy()
        node_lon = model.longitudes.detach().flatten().cpu().numpy()
        var_names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]

        baseline_prediction = wrapper(baseline_native).detach().cpu().numpy().squeeze(0)[0]
        # print(f"Prediction for target window [{target_start}, {target_end}]: {prediction}")
        # print(f"Baseline prediction + Sum of attributions: {baseline_prediction + attr_stack.sum():.4e}")
        print(f"Relative completeness error: {100 * (prediction - (baseline_prediction + attr_stack.sum())) / prediction:.2f}%")

        ds = xr.Dataset(
            {"attribution": (["input_lead_time", "node", "variable"], attr_stack)},
            coords={
                "input_lead_time": lead_time_hours,
                "node": np.arange(len(node_lat)),
                "node_lat": ("node", node_lat),
                "node_lon": ("node", node_lon),
                "variable": var_names,
            },
            attrs={
                "description": f"Integrated Gradients attribution of AIFS-predicted {region_name} "
                                "precipitation w.r.t. native-grid initial conditions",
                "target_region": region_name,
                "init_time": init_time.isoformat(),
                "target_time": target_time.isoformat(),
            },
        )
        ds['prediction'] = prediction
        ds['baseline_prediction'] = baseline_prediction
    finally:
        def _log_rmtree_error(func, path, exc_info):
            print(f"WARNING: failed to remove {path}: {exc_info[1]}")
        shutil.rmtree(tmp_data_cache, onerror=_log_rmtree_error)
        if os.path.exists(tmp_data_cache):
            print(f"WARNING: {tmp_data_cache} still exists after cleanup attempt")
        else:
            print(f"Cleaned up temporary input data cache: {tmp_data_cache}")

    return ds
    



def AIFS_XAI_pipeline(region_name,
                      region_geojson_path,
                      target_time_start, 
                      target_time_end,  
                      longest_lead_time_timesteps,
                      shortest_lead_time_timesteps=1,
                      target_var='tp',
                      time_operation='mean',
                      overwrite=False, predictions_only=False,
                      ig_step=10,
                      delta_t_hours=6,
                      ):
    # load AIFS model weights
    package = AIFS.load_default_package()
    model = AIFS.load_model(package).to(device)

    # Load input data for the model
    data = ARCO()
    native_node_mask = get_region_node_mask(model, region_name, region_geojson_path)

    dir_out = f"/cluster/projects/nn12107k/robin/xai_aifs_v2/{region_name.replace(' ', '_')}__{target_time:%Y%m%dT%H}/"
    os.makedirs(dir_out, exist_ok=True)
    delta_time = target_time_end - target_time_start
    delta_timestep = int(delta_time/timedelta(hours=delta_t_hours))
        
    if predictions_only:
        file_out = os.path.join(dir_out, f"predictions_{region_name.replace(' ', '_')}_target{target_time:%Y%m%dT%H}.nc")
        if os.path.exists(file_out) and not overwrite:
                print(f"Skipping lead time {lead_time} (already computed attributions in {file_out})")
                return
        all_predictions = []

        for lead_time in range(shortest_lead_time_timesteps, longest_lead_time_timesteps + 1):
            prediction = compute_attributions(model, data, native_node_mask,
                                    target_time_start, target_time_end, 
                                    lead_time, predictions_only=True,
                                    target_var=target_var, ig_step=ig_step,
                                    time_operation=time_operation)
            all_predictions.append(prediction)
        ds_predictions = xr.concat(all_predictions, dim='init_time').sortby('init_time')
        ds_predictions.to_netcdf(file_out)
        print(f"Saved predictions for lead time {longest_lead_time_timesteps}-{shortest_lead_time_timesteps} to {file_out}")
    else:
        for lead_time in range(shortest_lead_time_timesteps, longest_lead_time_timesteps + 1):
            file_out = os.path.join(dir_out, f"ig_attribution_{region_name.replace(' ', '_')}_target{target_time:%Y%m%dT%H}_lead{lead_time}.nc")
            if os.path.exists(file_out) and not overwrite:
                print(f"Skipping lead time {lead_time} (already computed attributions in {file_out})")
                continue
            ds_attr = compute_attributions(model, data, native_node_mask,
                                        target_time_start, target_time_end, 
                                        lead_time, predictions_only=False,
                                        target_var=target_var, ig_step=ig_step)
            ds_attr.to_netcdf(file_out)
            print(f"Saved IG attribution for lead time {lead_time} to {file_out}")
    # x_native0 = model._prepare_input(x, coords)

# coords = OrderedDict([("batch", np.array([0]))] + list(coords.items()))


if __name__=='__main__':
    region_geojson_path = "/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/aux/rainfall_regions.geojson"
    region_name = "California heatwave above38 2019-06-11T00"
    target_var = '2t'
    target_time = datetime(2019, 6, 11, 0, 0)
    target_time_start = datetime(2019, 6, 11, 0, 0)
    target_time_end = datetime(2019, 6, 11, 18, 0)
    return_n_times = 4
    longest_lead_time_timesteps = 12
    shortest_lead_time_timesteps = 1
    overwrite = False
    predictions_only = False
    ig_step = 30
    time_operation='mean'
    AIFS_XAI_pipeline(region_name, region_geojson_path, 
                      target_time, longest_lead_time_timesteps, shortest_lead_time_timesteps, 
                      target_var=target_var,
                      overwrite=overwrite, 
                      predictions_only=predictions_only,
                      ig_step=ig_step,
                      time_operation=time_operation)