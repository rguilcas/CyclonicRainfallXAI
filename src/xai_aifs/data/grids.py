import json
from shapely.geometry import shape
from shapely import vectorized  # shapely >= 2.0
import numpy as np


def native_to_latlon_attribution(model, integrated_grad, x_ll, b_ll, chunk=32):
    """Exact IG attributions on the 0.25 deg lat-lon input grid.

    The native input is a linear map of the lat-lon input, x_native = M @ x_ll
    (model.interpolation_matrix). By the chain rule the lat-lon gradient is
    M^T @ grad_native, and attr_ll = (x_ll - b_ll) * grad_ll. The sum of attr_ll
    equals the sum of the native attributions, so completeness is preserved.

    Only prognostic input variables live on the lat-lon grid; forcings and
    invariants are generated internally and have zero attribution anyway.

    integrated_grad: (T, 2, N_native, C_in)   x_ll, b_ll: (1, 1, 2, V, nlat, nlon)
    Returns: (T, 2, V, nlat, nlon) float32 tensor.
    """
    M = model.interpolation_matrix
    Mt = (M.to_sparse_coo() if M.is_sparse else M.to_sparse()).t().coalesce()

    input_full = model.input_full_ids.cpu().numpy()
    prognostic = model.input_ids.cpu().numpy()
    chan = [int(np.where(input_full == pid)[0][0]) for pid in prognostic]

    g = integrated_grad[..., chan]  # (T, L, N, V)
    T, L, N, V = g.shape
    nlat, nlon = x_ll.shape[-2:]
    g2 = g.permute(2, 0, 1, 3).reshape(N, T * L * V)

    out = torch.empty((T * L * V, nlat * nlon), dtype=torch.float32, device=g.device)
    for i in range(0, T * L * V, chunk):
        cols = g2[:, i:i + chunk].to(dtype=Mt.dtype)
        out[i:i + chunk] = torch.sparse.mm(Mt, cols).t().float()
    out = out.reshape(T, L, V, nlat, nlon)

    diff = (x_ll - b_ll)[0, 0].float()  # (L, V, nlat, nlon)
    return out * diff.unsqueeze(0)



def get_lat_band_node_mask(nodes_lat, target_lat, lat_half_width):
    """Boolean mask of native-grid nodes falling within a latitudinal band."""
    lat_min = target_lat - lat_half_width
    lat_max = target_lat + lat_half_width
    test = (nodes_lat >= lat_min) & (nodes_lat <= lat_max)
    return np.where(test)[0]

def get_lon_band_node_mask(nodes_lon, target_lon, lon_half_width):
    """Boolean mask of native-grid nodes falling within a longitudinal band."""
    lon_min = target_lon - lon_half_width
    lon_max = target_lon + lon_half_width
    test = (nodes_lon >= lon_min) & (nodes_lon <= lon_max)
    return np.where(test)[0]

def get_lon_lat_box_node_mask(nodes_lon, nodes_lat, target_lon, target_lat, lon_half_width, lat_half_width):
    """Boolean mask of native-grid nodes falling within a lon-lat box."""
    lat_mask = get_lat_band_node_mask(nodes_lat, target_lat, lat_half_width)
    lon_mask = get_lon_band_node_mask(nodes_lon, target_lon, lon_half_width)
    combined_mask = np.intersect1d(lat_mask, lon_mask)
    return combined_mask


def load_region_mask(geojson_path, region_name, node_lat, node_lon):
    """Boolean mask of native-grid nodes falling inside a named region polygon
    from a geojson file (e.g. aux/rainfall_regions.geojson)."""
    with open(geojson_path) as f:
        gj = json.load(f)

    feature = next(
        (feat for feat in gj["features"] if feat["properties"]["index"] == region_name), None
    )
    if feature is None:
        raise ValueError(f"No region named {region_name!r} found in {geojson_path}")
    polygon = shape(feature["geometry"])

    node_lon_180 = ((node_lon + 180) % 360) - 180  # match geojson's -180/180 convention
    inside = vectorized.contains(polygon, node_lon_180, node_lat)
    return np.where(inside)[0]

