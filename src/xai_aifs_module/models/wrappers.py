import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from xai_aifs_module.config import default_scale

REDUCTIONS = ("mean", "sum", "max", "min")


def _reduce(x, op, dim):
    """torch.<op> along `dim`, returning values only (max/min return a tuple)."""
    if op not in REDUCTIONS:
        raise ValueError(f"Unknown reduction {op!r}; choose from {REDUCTIONS}")
    out = getattr(torch, op)(x, dim=dim)
    return out.values if op in ("max", "min") else out


class AIFSRegionTarget(nn.Module):
    """Rolls AIFS out for ``n_steps`` and returns, for each target variable, the
    space- and time-aggregated value over the region nodes and window steps.

    Step i (0-based) of the rollout is valid at init + (i + 1) * 6 h.
    Output shape: (batch, n_targets), in scaled units (see config.default_scale).
    """

    def __init__(self, model, coords, n_steps, window_steps, node_idx, target_vars,
                 time_operation="sum", space_operation="mean", use_checkpoint=True):
        super().__init__()
        for op in (time_operation, space_operation):
            if op not in REDUCTIONS:
                raise ValueError(f"Unknown reduction {op!r}; choose from {REDUCTIONS}")
        if not window_steps or max(window_steps) >= n_steps:
            raise ValueError(f"window_steps {window_steps} must be non-empty and < n_steps={n_steps}")

        self.model = model
        self.coords = coords
        self.n_steps = n_steps
        self.window_steps = set(window_steps)
        self.time_operation = time_operation
        self.space_operation = space_operation
        self.use_checkpoint = use_checkpoint

        name_to_index = model.model.data_indices.data.output.name_to_index
        missing = [v for v in target_vars if v not in name_to_index]
        if missing:
            raise ValueError(f"Not AIFS outputs: {missing}. Available: {sorted(name_to_index)}")
        self.target_vars = list(target_vars)

        dev = next(model.parameters()).device
        self.target_idx = torch.tensor([name_to_index[v] for v in target_vars], device=dev)
        self.node_idx = torch.as_tensor(node_idx, dtype=torch.long, device=dev)
        self.scale = torch.tensor([default_scale(v)[0] for v in target_vars],
                                  dtype=torch.float32, device=dev)
        self.dt = model.output_coords(model.input_coords())["lead_time"]

    def _step(self, x, coords, step):
        out, _ = self.model._forward(x, coords, step=step)
        return out

    def forward(self, x):
        coords = self.coords.copy()
        per_step = []
        for i in range(self.n_steps):
            if self.use_checkpoint and torch.is_grad_enabled():
                out = checkpoint(self._step, x, coords, i + 1, use_reentrant=False)
            else:
                out = self._step(x, coords, i + 1)

            if i in self.window_steps:
                # out[:, 1] is the newest time level of the output state
                vals = out[:, 1][:, self.node_idx][:, :, self.target_idx].float()  # (B, nodes, T)
                per_step.append(_reduce(vals, self.space_operation, dim=1))       # (B, T)

            if i < self.n_steps - 1:
                coords = coords.copy()
                coords["lead_time"] = coords["lead_time"] + self.dt
                x = self.model._update_input(out, coords)

        series = torch.stack(per_step, dim=1)  # (B, n_window, T)
        return _reduce(series, self.time_operation, dim=1) * self.scale
