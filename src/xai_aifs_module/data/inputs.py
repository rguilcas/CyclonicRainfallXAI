import torch
from earth2studio.data.utils import fetch_data

from xai_aifs_module.data.climatology import build_climatology_baseline
from xai_aifs_module.data.grids import region_node_indices


def load_inputs(model, data, init_time, device):
    """Fetch the initial state and its climatological baseline.

    Returns
    -------
    x_native, b_native : model input and baseline on the native grid (detached)
    coords : earth2studio coords of the lat-lon input
    x_ll, b_ll : lat-lon input and baseline, shape (1, 1, L, V, nlat, nlon)
    """
    ic = model.input_coords()
    x_ll, coords = fetch_data(source=data, time=[init_time], variable=ic["variable"],
                              lead_time=ic["lead_time"], device=device)
    b_ll, coords_clim = build_climatology_baseline(model, init_time, device)
    x_ll = x_ll.unsqueeze(0)  # add batch dim

    with torch.no_grad():
        x_native = model._prepare_input(x_ll, coords)
        b_native = model._prepare_input(b_ll, coords_clim)

    # Forcings and invariants are computed internally and must be identical in
    # input and baseline, so that their attribution is exactly zero.
    same = ((x_native - b_native).abs().amax(dim=(0, 1, 2)) == 0).cpu().numpy()
    names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]
    print(f"  Channels identical in input and baseline: {[v for v, s in zip(names, same) if s]}")
    return x_native.detach(), b_native.detach(), coords, x_ll, b_ll


def region_nodes(model, region_name, geojson_path):
    """Native-grid node indices inside a named region of the geojson file."""
    lat = model.latitudes.detach().flatten().cpu().numpy()
    lon = model.longitudes.detach().flatten().cpu().numpy()
    idx = region_node_indices(geojson_path, region_name, lat, lon)
    if len(idx) == 0:
        raise ValueError(f"Region {region_name!r} contains no AIFS grid nodes")
    print(f"{region_name}: {len(idx)} of {len(lat)} native nodes")
    return idx
