import json
from shapely.geometry import shape
from shapely import vectorized  # shapely >= 2.0
import numpy as np


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
