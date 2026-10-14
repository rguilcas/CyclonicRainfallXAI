"""ERA5 'truth' aggregated over a region exactly as the AIFS forecasts are."""

from datetime import timedelta

import torch
from earth2studio.data.utils import fetch_data

try:  # earth2studio >= 0.13
    from earth2studio.lexicon.arco import ARCO_ERA5Lexicon as _ARCOLexicon
except ImportError:  # older releases
    from earth2studio.lexicon.arco import ARCOLexicon as _ARCOLexicon

from xai_aifs_module.config import ACCUMULATED_VARS, RADIATION_VARS
from xai_aifs_module.models.wrappers import _reduce


_SURFACE_NAMES = {"10u": "u10m", "10v": "v10m", "2d": "d2m", "2t": "t2m",
                  "100u": "u100m", "100v": "v100m"}


def _e2s_name(var):
    """earth2studio ID of an AIFS checkpoint name: 2t -> t2m, q_850 -> q850, tp -> tp."""
    if var in _SURFACE_NAMES:
        return _SURFACE_NAMES[var]
    base, _, level = var.partition("_")
    return f"{base}{level}" if level.isdigit() else var


def era5_name(var):
    """ARCO name for an AIFS output variable, and the number of trailing hourly
    values to sum (1 for instantaneous fields).

    AIFS accumulations (tp, cp, sf, ro, ssrd, strd) cover the 6 h ending at the
    valid time. Newer earth2studio provides these as tp06 etc.; with older
    versions the six hourly ERA5 values are summed here instead."""
    name = _e2s_name(var)
    if var not in ACCUMULATED_VARS | RADIATION_VARS:
        return name, 1
    if f"{name}06" in _ARCOLexicon.VOCAB:
        return f"{name}06", 1
    if name in _ARCOLexicon.VOCAB:
        return name, 6
    raise ValueError(f"ERA5 {name!r} is not available from ARCO in this earth2studio version")


def _fetch_native(model, data, times, names, device):
    """ERA5 (times, var) on the AIFS native grid: (n_times, n_vars, nodes)."""
    x, _ = fetch_data(source=data, time=list(times), variable=names, device=device)
    n_t, n_v = x.shape[0], x.shape[2]
    flat = x[:, 0].reshape(n_t * n_v, -1).T.to(torch.float64)          # (lat*lon, time*var)
    native = (model.interpolation_matrix @ flat).to(torch.float32)     # (nodes, time*var)
    return native.T.reshape(n_t, n_v, -1)


@torch.no_grad()
def era5_region_series(model, data, valid_times, node_idx, target_vars, device,
                       space_operation="mean"):
    """ERA5 at ``valid_times`` interpolated to the AIFS native grid with the model's
    own interpolation matrix, then reduced over ``node_idx``.

    Returns a float32 tensor (n_times, n_targets) in the same raw units as the
    model output (apply config.default_scale afterwards, as for forecasts).
    """
    fields = []
    for v in target_vars:
        name, n_hours = era5_name(v)
        if n_hours == 1:
            fields.append(_fetch_native(model, data, valid_times, [name], device)[:, 0])
        else:  # sum the trailing hourly accumulations ending at each valid time
            hours = [t - timedelta(hours=h) for t in valid_times for h in range(n_hours)]
            hourly = _fetch_native(model, data, hours, [name], device)[:, 0]
            fields.append(hourly.reshape(len(valid_times), n_hours, -1).sum(dim=1))
    native = torch.stack(fields, dim=1)                                 # (time, var, nodes)
    node_idx = torch.as_tensor(node_idx, dtype=torch.long, device=native.device)
    vals = native[:, :, node_idx].permute(0, 2, 1)                     # (time, nodes, var)
    return _reduce(vals, space_operation, dim=1)                        # (time, var)
