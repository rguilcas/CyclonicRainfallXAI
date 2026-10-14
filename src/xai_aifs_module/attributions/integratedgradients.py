import numpy as np
import torch
from tqdm import tqdm


def integrated_gradients(f, x, baseline, n_points=16, batch_size=1, log=True, grad_scale=1.0):
    """Integrated Gradients with Gauss-Legendre quadrature along the straight path.

    Several path points can be evaluated per forward pass (``batch_size``), and
    all targets returned by ``f`` share the same forward pass.

    Parameters
    ----------
    f : callable
        Maps a batch of inputs (k, ...) to outputs (k, n_targets).
    x, baseline : torch.Tensor
        Input and baseline, both of shape (1, ...).
    n_points : int
        Number of Gauss-Legendre nodes on [0, 1].
    batch_size : int
        Path points evaluated per forward pass.
    grad_scale : float
        The output is multiplied by this factor before backpropagation and the
        gradients divided by it afterwards. Raises small gradients out of the fp16
        underflow range under autocast; results are unchanged in exact arithmetic.

    Returns
    -------
    attributions : torch.Tensor, (n_targets, *x.shape[1:])
        ``(x - baseline) * integrated_gradients``; sums to ~ f(x) - f(baseline).
    integrated_gradients : torch.Tensor, (n_targets, *x.shape[1:])
        Path-averaged gradients (needed to map attributions to another grid).
    """
    if x.shape[0] != 1 or baseline.shape != x.shape:
        raise ValueError(f"x and baseline must both have shape (1, ...); got {tuple(x.shape)}, {tuple(baseline.shape)}")

    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    alphas = torch.tensor((nodes + 1) / 2, dtype=x.dtype, device=x.device)    # nodes mapped to [0, 1]
    weights = torch.tensor(weights / 2, dtype=torch.float32, device=x.device)  # weights sum to 1
    bshape = (-1,) + (1,) * (x.dim() - 1)

    baseline = baseline.detach()
    diff = (x.detach() - baseline)
    grad_sum = None
    with tqdm(total=n_points, desc="    IG path points", unit="pt", disable=not log) as bar:
        for start in range(0, n_points, batch_size):
            stop = min(start + batch_size, n_points)
            a = alphas[start:stop].view(bshape)
            w = weights[start:stop].view(bshape)
            xa = (baseline + a * diff).requires_grad_(True)
            out = f(xa) * grad_scale  # (k, n_targets)
            n_targets = out.shape[1]
            if grad_sum is None:
                grad_sum = torch.zeros((n_targets, *x.shape[1:]), dtype=torch.float32, device=x.device)
            for t in range(n_targets):
                (g,) = torch.autograd.grad(out[:, t].sum(), xa, retain_graph=t < n_targets - 1)
                grad_sum[t] += (g.float() * w).sum(dim=0)
            del out, xa
            bar.update(stop - start)
    if not torch.isfinite(grad_sum).all():
        raise FloatingPointError(f"Non-finite gradients with grad_scale={grad_scale}: lower --grad-scale")
    grad_sum /= grad_scale
    return grad_sum * diff[0].float(), grad_sum
