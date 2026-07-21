import torch.nn as nn

class AIFSPrecipRegionWrapperNative(nn.Module):
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
