import json
from shapely.geometry import shape
from shapely import vectorized  # shapely >= 2.0


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
