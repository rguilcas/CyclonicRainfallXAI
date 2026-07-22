from ..data.climatology import build_climatology_baseline
from ..models.wrappers import AIFSPrecipRegionWrapperNativeAllPredictions, build_day_groups
from datetime import datetime, timedelta
import tempfile
from earth2studio.data.utils import fetch_data
from collections import OrderedDict
import xarray as xr
from torch.utils.checkpoint import checkpoint
import os
import shutil
from captum.attr import IntegratedGradients, NoiseTunnel
from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import AIFS
import torch
import numpy as np
import pandas as pd
from ..helpers.region_mask import load_region_mask

def get_IG_NT_attribution(model, 
                       init_dt, target_day,
                       data, device, ic,
                       nt_samples=5, nt_stdevs_frac=0.02):

    tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
    os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache

    target_dt = datetime.strptime(target_day, '%Y-%m-%d') + timedelta(hours=18)  # 2016-08-08 18:00 UTC
    if init_dt.date() >= target_dt.date():
        raise ValueError(f"init_dt must be at the latest {(target_dt-timedelta(hours=24)).strftime('%Y-%m-%dT%H')} with target_day {target_day}")
    lead_hours_total = (target_dt - init_dt).total_seconds() / 3600
    if lead_hours_total <= 0 or lead_hours_total % 6 != 0:
        raise ValueError(
            f"target_dt {target_dt} is not reachable from init_dt {init_dt} "
            f"in whole 6h steps (got {lead_hours_total}h)."
        )
    nsteps = int(lead_hours_total // 6)
    target_step = nsteps - 1  # 0-indexed: the final rollout step IS the target
    day_groups = build_day_groups(init_dt, nsteps, boundary_hour=0)  # or 6 for the hydrological day
    lat_day_indices = day_groups[-1]
    print(f"Target day: {target_day}\n    Init: {init_dt} -> target prediction timesteps: [{'-'.join(str(i) for i in lat_day_indices)}]")

    try:
        x, coords = fetch_data(
            source=data,
            time=[init_dt],
            variable=ic["variable"],
            lead_time=ic["lead_time"],
            device=device,
        )
        x_clim, coords_clim = build_climatology_baseline(model, init_dt, device, x, coords)

        x = x.unsqueeze(0)

        with torch.no_grad():
            x_native0 = model._prepare_input(x, coords)
            baseline_native = model._prepare_input(x_clim, coords_clim)  # x_clim already has batch dim (added inside the function)

        coords = OrderedDict([("batch", np.array([0]))] + list(coords.items()))

        model_native_lat = model.latitudes.detach().flatten().cpu().numpy()
        model_native_lon = model.longitudes.detach().flatten().cpu().numpy()

        native_node_mask = load_region_mask(
            "/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/aux/rainfall_regions.geojson", "West Norway", model_native_lat, model_native_lon
        )
        print(f"West Norway native node mask: {len(native_node_mask)} nodes selected out of {len(model_native_lat)} total nodes")
        tp_full_idx = model.VARIABLES.index("tp06")

        for ctx_cls in (torch.no_grad, torch.inference_mode):
            ctx_cls.__enter__ = lambda self: None
            ctx_cls.__exit__ = lambda self, *args: None
        
        wrapper = AIFSPrecipRegionWrapperNativeAllPredictions(model, coords, nsteps, native_node_mask, tp_full_idx, day_groups=day_groups).to(device).eval()
        ig = IntegratedGradients(wrapper)
        smooth_ig = NoiseTunnel(ig)

        x_ig = x_native0.clone().requires_grad_(True)
        target_day_idx = len(day_groups) - 1  # last day group is the target step for attribution

        stdevs = nt_stdevs_frac * x_ig.std().item()  # single global noise scale, same for every variable
        print(f"  Computing SmoothGrad (NoiseTunnel + Integrated Gradients) attribution "
              f"(nt_samples={nt_samples}, stdevs={stdevs:.4g})...")

        attr = smooth_ig.attribute(
            x_ig, baselines=baseline_native, target=target_day_idx,
            n_steps=20, internal_batch_size=1,
            nt_type="smoothgrad", nt_samples=nt_samples, nt_samples_batch_size=1,
            stdevs=stdevs,
        )
        attr = attr.detach().cpu()
        torch.cuda.empty_cache()
        print(f"target_day {target_day_idx}, forecast lead time: {(nsteps-3) * 6}h - sum={attr.sum().item():.4e}")
        
        attr_stack = attr.numpy().squeeze(0)  # (input_lead_time, node, variable)
        lead_time_hours = (model.input_coords()["lead_time"] / np.timedelta64(1, "h")).astype(int)
        node_lat = model.latitudes.detach().flatten().cpu().numpy()
        node_lon = model.longitudes.detach().flatten().cpu().numpy()
        var_names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]

        prediction = wrapper(x_native0).detach().cpu().numpy().squeeze(0)[-1]
        baseline_prediction = wrapper(baseline_native).detach().cpu().numpy().squeeze(0)[-1]

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
                "description": "SmoothGrad (NoiseTunnel + Integrated Gradients) attribution of "
                                "AIFS-predicted west Norway precipitation w.r.t. native-grid initial conditions",
                "target_region": "Vestlandet, Norway",
                "init_time": init_dt.isoformat(),
                "target_day": target_dt.date().isoformat(),
                "forecast_lead_hours": (nsteps-3) * 6,
                "nt_samples": nt_samples,
                "nt_stdevs": stdevs,
            },
        )
        ds['prediction'] = prediction
        ds['baseline_prediction'] = baseline_prediction

        return ds

    finally:
        def _log_rmtree_error(func, path, exc_info):
            print(f"WARNING: failed to remove {path}: {exc_info[1]}")
        shutil.rmtree(tmp_data_cache, onerror=_log_rmtree_error)
        if os.path.exists(tmp_data_cache):
            print(f"WARNING: {tmp_data_cache} still exists after cleanup attempt")
        else:
            print(f"Cleaned up temporary input data cache: {tmp_data_cache}")

def get_IG_attribution(model, 
                       init_dt, target_day,
                       data, device, ic):

    tmp_data_cache = tempfile.mkdtemp(prefix="e2s_data_cache_", dir="/cluster/projects/nn12107k/robin")
    os.environ["EARTH2STUDIO_DATA_CACHE"] = tmp_data_cache

    target_dt = datetime.strptime(target_day, '%Y-%m-%d') + timedelta(hours=18)  # 2016-08-08 18:00 UTC
    if init_dt.date() >= target_dt.date():
        raise ValueError(f"init_dt must be at the latest {(target_dt-timedelta(hours=24)).strftime('%Y-%m-%dT%H')} with target_day {target_day}")
    lead_hours_total = (target_dt - init_dt).total_seconds() / 3600
    if lead_hours_total <= 0 or lead_hours_total % 6 != 0:
        raise ValueError(
            f"target_dt {target_dt} is not reachable from init_dt {init_dt} "
            f"in whole 6h steps (got {lead_hours_total}h)."
        )
    nsteps = int(lead_hours_total // 6)
    target_step = nsteps - 1  # 0-indexed: the final rollout step IS the target
    day_groups = build_day_groups(init_dt, nsteps, boundary_hour=0)  # or 6 for the hydrological day
    lat_day_indices = day_groups[-1]
    print(f"Target day: {target_day}\n    Init: {init_dt} -> target prediction timesteps: [{'-'.join(str(i) for i in lat_day_indices)}]")
    # print(f"Init: {init_dt} -> target valid time {target_dt} ({nsteps} steps, target_step={target_step})")

    try:
        x, coords = fetch_data(
            source=data,
            time=[init_dt],
            variable=ic["variable"],
            lead_time=ic["lead_time"],
            device=device,
        )
        x_clim, coords_clim = build_climatology_baseline(model, init_dt, device, x, coords)

        x = x.unsqueeze(0)

        with torch.no_grad():
            x_native0 = model._prepare_input(x, coords)
            baseline_native = model._prepare_input(x_clim, coords_clim)  # x_clim already has batch dim (added inside the function)

        coords = OrderedDict([("batch", np.array([0]))] + list(coords.items()))

        model_native_lat = model.latitudes.detach().flatten().cpu().numpy()
        model_native_lon = model.longitudes.detach().flatten().cpu().numpy()

        native_node_mask = load_region_mask(
            "/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/aux/rainfall_regions.geojson", "West Norway", model_native_lat, model_native_lon
        )
        print(f"West Norway native node mask: {len(native_node_mask)} nodes selected out of {len(model_native_lat)} total nodes")
        tp_full_idx = model.VARIABLES.index("tp06")

        for ctx_cls in (torch.no_grad, torch.inference_mode):
            ctx_cls.__enter__ = lambda self: None
            ctx_cls.__exit__ = lambda self, *args: None
        
        wrapper = AIFSPrecipRegionWrapperNativeAllPredictions(model, coords, nsteps, native_node_mask, tp_full_idx, day_groups=day_groups).to(device).eval()
        ig = IntegratedGradients(wrapper)


        x_ig = x_native0.clone().requires_grad_(True)
        target_day = len(day_groups) - 1  # last day group is the target step for attribution
        print("  Computing Integrated Gradients attribution...")
        attr = ig.attribute(x_ig, baselines=baseline_native, target=target_day, n_steps=50, internal_batch_size=1)
        attr = attr.detach().cpu()
        torch.cuda.empty_cache()
        print(f"target_day {target_day}, forecast lead time: {(nsteps-3) * 6}h - sum={attr.sum().item():.4e}")
        
        attr_stack = attr.numpy().squeeze(0)  # (input_lead_time, node, variable)
        lead_time_hours = (model.input_coords()["lead_time"] / np.timedelta64(1, "h")).astype(int)
        node_lat = model.latitudes.detach().flatten().cpu().numpy()
        node_lon = model.longitudes.detach().flatten().cpu().numpy()
        var_names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]

        prediction = wrapper(x_native0).detach().cpu().numpy().squeeze(0)[-1]  # (nsteps, 1)
        baseline_prediction = wrapper(baseline_native).detach().cpu().numpy().squeeze(0)[-1]  # (nsteps, 1)

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
                "description": "Integrated Gradients attribution of AIFS-predicted west Norway "
                                "precipitation w.r.t. native-grid initial conditions",
                "target_region": "Vestlandet, Norway",
                "init_time": init_dt.isoformat(),
                "target_day": target_dt.date().isoformat(),
                "forecast_lead_hours": (nsteps-3) * 6,
            },
        )
        ds['prediction'] = prediction
        ds['baseline_prediction'] = baseline_prediction

        return ds

    finally:
        def _log_rmtree_error(func, path, exc_info):
            print(f"WARNING: failed to remove {path}: {exc_info[1]}")
        shutil.rmtree(tmp_data_cache, onerror=_log_rmtree_error)
        if os.path.exists(tmp_data_cache):
            print(f"WARNING: {tmp_data_cache} still exists after cleanup attempt")
        else:
            print(f"Cleaned up temporary input data cache: {tmp_data_cache}")