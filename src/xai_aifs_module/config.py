"""Paths and constants shared across the pipeline.

Cluster paths default to the Olivia/NRIS layout but can be overridden with
environment variables, so the package runs elsewhere without code edits.
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(os.environ.get("XAI_AIFS_PROJECT_DIR", "/cluster/projects/nn12107k/robin"))

ERA5_CLIMATOLOGY_DIR = Path(os.environ.get("XAI_AIFS_ERA5_CLIM_DIR", PROJECT_DIR / "era5_climatology"))
OUT_ROOT = Path(os.environ.get("XAI_AIFS_OUT_ROOT", PROJECT_DIR / "xai_aifs_v3"))
DEFAULT_REGION_GEOJSON = REPO_DIR / "aux" / "rainfall_regions.geojson"

WB2_CLIMATOLOGY_STORE = "1990-2019_6h_1440x721.zarr"
STEP_HOURS = 6  # AIFS time step

# Output scaling. 6-hourly accumulations are in metres; attribute them in mm
# (x1000) so gradients stay well inside fp16 range under the model's autocast.
ACCUMULATED_VARS = {"tp", "cp", "sf", "ro"}
RADIATION_VARS = {"ssrd", "strd"}


def default_scale(var):
    """Return (multiplier, units) applied to target variable `var`."""
    if var in ACCUMULATED_VARS:
        return 1000.0, "mm"
    if var in RADIATION_VARS:
        return 1e-6, "MJ m-2"
    return 1.0, "native"


def set_earth2studio_env():
    """Cache settings that must be in place BEFORE earth2studio is imported."""
    os.environ.setdefault("EARTH2STUDIO_CACHE", str(PROJECT_DIR / "earth2studio_cache"))
    os.environ.setdefault("EARTH2STUDIO_DATA_CACHE", str(PROJECT_DIR / "e2s_data_cache"))
    os.environ.setdefault("EARTH2STUDIO_DISABLE_MSC", "1")
