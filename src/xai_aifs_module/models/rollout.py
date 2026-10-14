"""Forecast-only AIFS rollouts for several initialisations at once."""

import torch

from xai_aifs_module.models.wrappers import _reduce


def target_indices(model, target_vars):
    """Indices of AIFS output variables (checkpoint names, e.g. 2t, tp)."""
    name_to_index = model.model.data_indices.data.output.name_to_index
    missing = [v for v in target_vars if v not in name_to_index]
    if missing:
        raise ValueError(f"Not AIFS outputs: {missing}. Available: {sorted(name_to_index)}")
    return [name_to_index[v] for v in target_vars]


@torch.no_grad()
def batched_region_rollout(model, x_natives, coords_list, n_steps, node_idx, target_vars,
                           space_operation="mean"):
    """Roll several initial states out together and return regional aggregates.

    earth2studio's AIFS wrapper computes the time-dependent forcings (solar
    zenith angle, local time, julian day) from a single time stamp, so each
    initial state is prepared separately (``x_natives``, ``coords_list``) and
    only the forward pass is batched; forcings are then updated per member.

    Returns a float32 tensor (n_init, n_steps, n_targets): the target variables
    aggregated over ``node_idx`` with ``space_operation`` at every step, where
    step i is valid at init + (i + 1) * 6 h.
    """
    dev = x_natives[0].device
    node_idx = torch.as_tensor(node_idx, dtype=torch.long, device=dev)
    tgt_idx = torch.tensor(target_indices(model, target_vars), device=dev)
    dt = model.output_coords(model.input_coords())["lead_time"]

    x = torch.cat(x_natives, dim=0)
    coords_list = [c.copy() for c in coords_list]
    series = []
    for i in range(n_steps):
        out, _ = model._forward(x, coords_list[0], step=i + 1)  # coords only give output shapes here
        vals = out[:, 1][:, node_idx][:, :, tgt_idx].float()     # (B, nodes, T)
        series.append(_reduce(vals, space_operation, dim=1))      # (B, T)

        if i < n_steps - 1:
            new_x = []
            for b, coords in enumerate(coords_list):
                coords = coords.copy()
                coords["lead_time"] = coords["lead_time"] + dt
                coords_list[b] = coords
                new_x.append(model._update_input(out[b:b + 1], coords))
            x = torch.cat(new_x, dim=0)
            del out
    return torch.stack(series, dim=1)
