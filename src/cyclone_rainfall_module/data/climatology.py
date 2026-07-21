from earth2studio.data import WB2Climatology
from earth2studio.data.utils import fetch_data
from earth2studio.lexicon.wb2 import WB2ClimatetologyLexicon
import torch
from collections import OrderedDict
import numpy as np

def build_climatology_baseline(model, init_dt, device, x_actual, coords_actual):
    ic = model.input_coords()
    all_vars = list(ic["variable"])
    covered = [v for v in all_vars if v in WB2ClimatetologyLexicon.VOCAB]
    missing = [v for v in all_vars if v not in WB2ClimatetologyLexicon.VOCAB]
    if missing:
        print(f"No climatology for {len(missing)} vars, using sample spatial mean: {missing}")

    clim = WB2Climatology(climatology_zarr_store="1990-2019_6h_1440x721.zarr")
    x_cov, coords_cov = fetch_data(
        source=clim, time=[init_dt], variable=covered,
        lead_time=ic["lead_time"], device=device,
    )

    x_clim = torch.zeros(
        (*x_cov.shape[:2], len(all_vars), *x_cov.shape[3:]),
        device=device, dtype=x_cov.dtype,
    )
    for i, v in enumerate(all_vars):
        if v in covered:
            x_clim[..., i, :, :] = x_cov[..., covered.index(v), :, :]
        else:
            j = list(coords_actual["variable"]).index(v)
            sample_mean = x_actual[..., j, :, :].mean(dim=(-2, -1), keepdim=True)
            x_clim[..., i, :, :] = sample_mean.expand(x_clim[..., i, :, :].shape)

    x_clim = x_clim.unsqueeze(0)
    coords_clim = OrderedDict([("batch", np.array([0]))] + list(coords_cov.items()))
    coords_clim["variable"] = np.array(all_vars)
    return x_clim, coords_clim