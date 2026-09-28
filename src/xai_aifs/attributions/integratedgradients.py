import torch
import numpy as np
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Integrated Gradients (Gauss-Legendre, batched path, several targets per forward)
# ----------------------------------------------------------------------------
def integrated_gradients(f, x, baseline, n_points=16, batch_size=1, log=True):
    """Returns (attributions, integrated_gradients), both of shape
    (n_targets, *x.shape[1:]). x and baseline have shape (1, ...).
    attributions = (x - baseline) * integrated_gradients."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    alphas = torch.tensor((nodes + 1) / 2, dtype=x.dtype, device=x.device)
    weights = torch.tensor(weights / 2, dtype=torch.float32, device=x.device)
    bshape = (-1,) + (1,) * (x.dim() - 1)

    diff = (x - baseline).detach()
    grad_sum = None
    with tqdm(total=n_points, desc="    IG path points", unit="pt", disable=not log) as bar:
        for start in range(0, n_points, batch_size):
            a = alphas[start:start + batch_size].view(bshape)
            w = weights[start:start + batch_size].view(bshape)
            xa = (baseline.detach() + a * diff).requires_grad_(True)
            out = f(xa)  # (k, T)
            n_targets = out.shape[1]
            if grad_sum is None:
                grad_sum = torch.zeros((n_targets, *x.shape[1:]), dtype=torch.float32, device=x.device)
            for t in range(n_targets):
                (g,) = torch.autograd.grad(out[:, t].sum(), xa, retain_graph=t < n_targets - 1)
                grad_sum[t] += (g.float() * w).sum(dim=0)
            del out, xa
            bar.update(len(range(start, min(start + batch_size, n_points))))
    return grad_sum * diff[0].float(), grad_sum
