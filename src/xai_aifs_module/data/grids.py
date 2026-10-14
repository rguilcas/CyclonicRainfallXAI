import json

import numpy as np
import torch
from shapely import contains_xy
from shapely.geometry import shape


def native_to_latlon_attribution(model, integrated_grad, x_ll, b_ll, chunk=32):
    """Exact IG attributions on the 0.25 deg lat-lon input grid.

    The native input is a linear map of the lat-lon input, x_native = M @ x_ll
    (model.interpolation_matrix). By the chain rule the lat-lon gradient is
    M^T @ grad_native, and attr_ll = (x_ll - b_ll) * grad_ll. The sum of attr_ll
    equals the sum of the native attributions, so completeness is preserved.

    Only prognostic input variables live on the lat-lon grid; forcings and
    invariants are generated internally and have zero attribution anyway.

    integrated_grad: (T, L, N_native, C_in)   x_ll, b_ll: (1, 1, L, V, nlat, nlon)
    Returns: (T, L, V, nlat, nlon) float32 tensor.
    """
    M = model.interpolation_matrix
    Mt = (M.to_sparse_coo() if M.is_sparse else M.to_sparse()).t().coalesce()

    input_full = model.input_full_ids.cpu().numpy()
    prognostic = model.input_ids.cpu().numpy()
    chan = [int(np.flatnonzero(input_full == pid)[0]) for pid in prognostic]

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


def box_node_indices(node_lon, node_lat, lon_range=None, lat_range=None):
    """Indices of nodes inside a lon/lat box. Ranges are (min, max), inclusive; None = unbounded."""
    inside = np.ones_like(node_lat, dtype=bool)
    if lat_range is not None:
        inside &= (node_lat >= lat_range[0]) & (node_lat <= lat_range[1])
    if lon_range is not None:
        inside &= (node_lon >= lon_range[0]) & (node_lon <= lon_range[1])
    return np.flatnonzero(inside)


def region_node_indices(geojson_path, region_name, node_lat, node_lon):
    """Indices of nodes inside a named polygon of a geojson file
    (matched on the feature's ``properties.index``)."""
    with open(geojson_path) as f:
        features = json.load(f)["features"]
    feature = next((ft for ft in features if ft["properties"]["index"] == region_name), None)
    if feature is None:
        names = [ft["properties"]["index"] for ft in features]
        raise ValueError(f"No region {region_name!r} in {geojson_path}. Available: {names}")
    polygon = shape(feature["geometry"])

    node_lon_180 = ((node_lon + 180) % 360) - 180  # geojson uses -180/180
    return np.flatnonzero(contains_xy(polygon, node_lon_180, node_lat))
