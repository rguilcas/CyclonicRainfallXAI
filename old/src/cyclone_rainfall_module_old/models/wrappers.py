import torch.nn as nn
import torch
from datetime import timedelta
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

class AIFSWrapper(nn.Module):
    def __init__(self, 
                 model, 
                 n_steps, 
                 coords, 
                 target_var='tp',
                 return_timeseries=False,
                 node_id=10, 
                 space_operation='mean',
                 time_operation='sum',
                 delta_t_hours=6,
                 return_last_n_steps=4,
                 progress_bar=True):
        super().__init__()
        self.model = model
        self.coords = coords
        self.n_steps = n_steps
        self.return_timeseries = return_timeseries
        self.node_id = node_id
        self.space_operation = space_operation
        self.time_operation = time_operation
        self.return_last_n_steps = return_last_n_steps
        self.delta_t_hours = delta_t_hours
        self.progress_bar = progress_bar

        raw_map = model.model.data_indices.data.output.name_to_index
        if target_var not in raw_map:
            raise ValueError("Variable name not in AIFS outputs")
        self.target_var = target_var
        self.target_index = raw_map[target_var]


        
    def forward(self, x):
        m = self.model
        coords = self.coords.copy()
        dt = m.output_coords(m.input_coords())["lead_time"]   # the 6h increment
        region_aggs = []

        for step in tqdm(range(self.n_steps), disable=not self.progress_bar):
            c_in = coords  # bind current coords explicitly (avoid late-binding closures)
            def _step(x_in, c_in=c_in, step=step):
                out, _ = m._forward(x_in, c_in, step=step)
                return out
            out = checkpoint(_step, x, use_reentrant=False)

            coords = coords.copy()
            coords["lead_time"] = coords["lead_time"] + dt      # advance by 6h each step
            x = m._update_input(out, coords)
            x_out = out[:, 1, self.node_id, self.target_index].float()
            region_agg = getattr(torch, self.space_operation)(x_out, dim=-1)
            region_aggs.append(region_agg)
        out =  torch.stack(region_aggs, dim=1)  # (batch, n_days)

        if self.return_timeseries:
            return out
        else:
            return getattr(torch, self.time_operation)(out[:,-self.return_last_n_steps:], dim=1)






# class AIFSWrapper_old(nn.Module):
#     def __init__(self, 
#                  model, 
#                  n_steps, 
#                  coords, 
#                  target_var='tp',
#                  return_timeseries=False,
#                  node_id=10, 
#                  space_operation='mean',
#                  time_operation='sum',
#                  delta_t_hours=6,
#                  return_last_n_steps=4,
#                  progress_bar=True):
#         super().__init__()
#         self.model = model
#         self.coords = coords
#         self.n_steps = n_steps
#         self.return_timeseries = return_timeseries
#         self.node_id = node_id
#         self.space_operation = space_operation
#         self.time_operation = time_operation
#         self.return_last_n_steps = return_last_n_steps
#         self.delta_t_hours = delta_t_hours
#         self.progress_bar = progress_bar

#         raw_map = model.model.data_indices.data.output.name_to_index
#         if target_var not in raw_map:
#             raise ValueError("Variable name not in AIFS outputs")
#         self.target_var = target_var
#         self.target_index = raw_map[target_var]


        
#     def forward(self, x_native):
        
#         region_aggs = []
#         m = self.model
#         step = 0
#         for _ in tqdm(range(self.n_steps), disable=not self.progress_bar):
#             def _step(x_in, coords=self.coords, step=step):
#                 out, coords_out = m._forward(x_in,coords, step=step)
#                 return out
#             out = checkpoint(_step, x_native, use_reentrant=False)
#             # print(out.shape)
#             coords = self.coords.copy()
#             coords["lead_time"] = coords["lead_time"] + m.output_coords(m.input_coords())["lead_time"]
#             x_native = m._update_input(out, coords)
#             target_variable_native = out[:, 1, self.node_id, self.target_index].float()
#             # target_variable_native = out[:, 1, :, self.target_index].float()
#             # region_agg = getattr(torch, self.space_operation)(target_variable_native, dim=-1)
#             region_agg = target_variable_native
#             region_aggs.append(region_agg)
#             step += 1
        
#         out =  torch.stack(region_aggs, dim=1)  # (batch, n_days)

#         if self.return_timeseries:
#             return out
#         else:
#             return getattr(torch, self.time_operation)(out[:,-self.return_last_n_steps:], dim=1)


def build_day_groups(init_dt, nsteps, boundary_hour=0):
    """Group the model's 6-hourly steps (step k valid at init_dt + k*6h, k=1..nsteps)
    into fixed calendar-day buckets. boundary_hour=0 -> standard UTC day (00-24 UTC).
    boundary_hour=6 -> 06-06 UTC "hydrological day". Returns a list of lists of
    0-indexed step positions (index 0 == first step), matching the order forward()
    accumulates them in.
    """
    valid_times = [init_dt + timedelta(hours=6 * step) for step in range(1, nsteps + 1)]
    groups = {}
    for i, vt in enumerate(valid_times):
        day_key = (vt - timedelta(hours=boundary_hour)).date()
        groups.setdefault(day_key, []).append(i)
    return [groups[k] for k in sorted(groups)]


class AIFSPrecipRegionWrapperNativeAllPredictions(nn.Module):
    """
    Wraps AIFS to return mean rainfall over a certain region, defined by the native_node_mask, for a given number of autoregressive steps. 
    This returns the mean rainfall for all intermediate steps.
    """
    def __init__(self, model, coords, nsteps, native_node_mask, tp_full_idx, day_groups=None):
        super().__init__()
        self.model = model
        self.coords = coords
        self.nsteps = nsteps
        self.node_mask = torch.as_tensor(native_node_mask, device=next(model.parameters()).device)
        self.tp_full_idx = tp_full_idx
        # day_groups: optional list of lists of 0-indexed step numbers to sum together
        # (e.g. from build_day_groups). If None, every step is returned individually.
        if day_groups is not None:
            covered = sorted(i for g in day_groups for i in g)
            assert covered == list(range(nsteps)), (
                f"day_groups must partition all {nsteps} steps exactly once, got {covered}"
            )
        self.day_groups = day_groups

    def forward(self, x_native0):
        m = self.model
        coords_t = self.coords.copy()
        x_native = x_native0
        step = 1
        region_avgs = []

        for _ in range(self.nsteps):
            def _step(x_in, coords_t=coords_t, step=step):
                out, coords_out = m._forward(x_in, coords_t, step=step)
                return out

            out = checkpoint(_step, x_native, use_reentrant=False)

            precip_native = out[:, 1, :, self.tp_full_idx].float()
            region_avg = precip_native.index_select(-1, self.node_mask).mean(dim=-1)
            region_avgs.append(region_avg)

            coords_t = coords_t.copy()
            coords_t["lead_time"] = coords_t["lead_time"] + m.output_coords(m.input_coords())["lead_time"]
            x_native = m._update_input(out, coords_t)
            step += 1

        if self.day_groups is None:
            return torch.stack(region_avgs, dim=1)  # (batch, nsteps)

        grouped = [
            torch.stack([region_avgs[i] for i in group], dim=1).sum(dim=1)
            for group in self.day_groups
        ]
        return torch.stack(grouped, dim=1)  # (batch, n_days)


class AIFSPrecipRegionWrapperNative(nn.Module):
    """
    Wraps AIFS to return mean rainfall over a certain region, defined by the native_node_mask, for a given number of autoregressive steps. 
    This only returns the mean rainfall for the last step (target step), but runs all intermediate steps to maintain autoregressive dependency.
    """
    def __init__(self, model, coords, nsteps, native_node_mask, tp_full_idx):
        super().__init__()
        self.model = model
        self.coords = coords
        self.nsteps = nsteps
        self.node_mask = torch.as_tensor(native_node_mask, device=next(model.parameters()).device)
        self.tp_full_idx = tp_full_idx

    def forward(self, x_native0):
        m = self.model
        coords_t = self.coords.copy()
        x_native = x_native0
        step = 1
        region_avg = None

        for _ in range(self.nsteps):
            def _step(x_in, coords_t=coords_t, step=step):
                out, coords_out = m._forward(x_in, coords_t, step=step)
                return out

            out = checkpoint(_step, x_native, use_reentrant=False)

            # Only the last step (== target_step) is kept; intermediate
            # steps are still run (autoregressive dependency) but not stored.
            precip_native = out[:, 1, :, self.tp_full_idx]
            region_avg = precip_native.index_select(-1, self.node_mask).mean(dim=-1)

            coords_t = coords_t.copy()
            coords_t["lead_time"] = coords_t["lead_time"] + m.output_coords(m.input_coords())["lead_time"]
            x_native = m._update_input(out, coords_t)
            step += 1

        return region_avg.unsqueeze(1)  # (batch, 1) -- single target column
