"""ERA5 'truth' aggregated over a region exactly as the AIFS forecasts are."""

import torch
from earth2studio.data.utils import fetch_data
from earth2studio.lexicon.arco import ARCO_ERA5Lexicon
from earth2studio.models.px import AIFS

from xai_aifs_module.config import ACCUMULATED_VARS, RADIATION_VARS
from xai_aifs_module.models.wrappers import _reduce


def era5_name(var):
    """ARCO/earth2studio name for an AIFS output variable.

    AIFS accumulations (tp, cp, sf, ro, ssrd, strd) cover the 6 h ending at the
    valid time, so they map to the trailing 6 h sums (tp06, ...)."""
    name = AIFS._ckpt_var_to_e2s([var])[0]
    if var in ACCUMULATED_VARS | RADIATION_VARS:
        if f"{name}06" not in ARCO_ERA5Lexicon.VOCAB:
            raise ValueError(f"No 6-hourly ERA5 accumulation for {var!r} in this earth2studio "
                             f"version ({name}06 missing from the ARCO lexicon)")
        name = f"{name}06"
    return name


@torch.no_grad()
def era5_region_series(model, data, valid_times, node_idx, target_vars, device,
                       space_operation="mean"):
    """ERA5 at ``valid_times`` interpolated to the AIFS native grid with the model's
    own interpolation matrix, then reduced over ``node_idx``.

    Returns a float32 tensor (n_times, n_targets) in the same raw units as the
    model output (apply config.default_scale afterwards, as for forecasts).
    """
    names = [era5_name(v) for v in target_vars]
    x, _ = fetch_data(source=data, time=list(valid_times), variable=names, device=device)
    # x: (time, lead=1, var, lat, lon) -> (lat*lon, time*var)
    n_t, n_v = x.shape[0], x.shape[2]
    flat = x[:, 0].reshape(n_t * n_v, -1).T.to(torch.float64)
    native = (model.interpolation_matrix @ flat).to(torch.float32)  # (nodes, time*var)
    native = native.T.reshape(n_t, n_v, -1)                            # (time, var, nodes)
    node_idx = torch.as_tensor(node_idx, dtype=torch.long, device=native.device)
    vals = native[:, :, node_idx].permute(0, 2, 1)                     # (time, nodes, var)
    return _reduce(vals, space_operation, dim=1)                        # (time, var)
