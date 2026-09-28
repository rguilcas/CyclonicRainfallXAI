
from earth2studio.data.utils import fetch_data
from xai_aifs.data.climatology import build_climatology_baseline
import torch
from xai_aifs.data.grid import load_region_mask

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_inputs(model, data, init_time):
    ic = model.input_coords()
    x, coords = fetch_data(source=data, time=[init_time], variable=ic["variable"],
                           lead_time=ic["lead_time"], device=device)
    x_clim, coords_clim = build_climatology_baseline(model, init_time, device, x, coords)
    x = x.unsqueeze(0)
    with torch.no_grad():
        x_native = model._prepare_input(x, coords)
        b_native = model._prepare_input(x_clim, coords_clim)

    # Sanity check: forcing and invariant channels should be identical in input and
    # baseline, so their attributions are exactly zero.
    same = ((x_native - b_native).abs().amax(dim=(0, 1, 2)) == 0).cpu().numpy()
    var_names = [model.VARIABLES[i] for i in model.input_full_ids.cpu().numpy()]
    print(f"  Channels identical in input and baseline: {[v for v, s in zip(var_names, same) if s]}")
    return x_native.detach(), b_native.detach(), coords, x, x_clim


def region_nodes(model, region_name, geojson):
    lat = model.latitudes.detach().flatten().cpu().numpy()
    lon = model.longitudes.detach().flatten().cpu().numpy()
    idx = load_region_mask(geojson, region_name, lat, lon)
    if len(idx) == 0:
        raise ValueError(f"Region {region_name!r} contains no AIFS grid nodes")
    print(f"{region_name}: {len(idx)} of {len(lat)} native nodes")
    return idx
