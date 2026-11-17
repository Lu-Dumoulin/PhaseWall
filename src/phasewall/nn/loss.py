"""The contrastive loss and the checks of a training (adapted from hfml Step 7).

info_nce (SimCLR): a batch holds 2N unit vectors z; rows i and i + N are the two views of
the same run. Row i is a classification among the 2N - 1 other views whose correct answer is
its partner: softmax of z_i . z_k / T, cross-entropy. If every z is the same (collapse), the
loss is ln(2N - 1).

Checks (logged, not used to stop the training): alignment (mean cosine between the two views
of a run, 1 at best), uniformity (log mean exp(-2 |z_i - z_j|^2) over different runs; more
negative = more spread on the sphere), effective rank of h (exp of the entropy of the
normalised singular values: 1 if all h lie on a line, up to the dimension of h).
"""

import torch
import torch.nn.functional as F

T = 0.2   # temperature (a choice, as in hfml)


def info_nce(z: torch.Tensor, T: float = T) -> torch.Tensor:
    """Mean InfoNCE loss of 2N views (z: (2N, d), unit rows, views i and i + N paired)."""
    n2 = z.shape[0]
    sim = z @ z.T / T
    sim = sim.masked_fill(torch.eye(n2, dtype=torch.bool, device=z.device), float("-inf"))
    partner = (torch.arange(n2, device=z.device) + n2 // 2) % n2
    return F.cross_entropy(sim, partner)


def alignment(z: torch.Tensor) -> float:
    """Mean cosine between the two views of each run (1: identical)."""
    n = z.shape[0] // 2
    return float((z[:n] * z[n:]).sum(dim=1).mean())


def uniformity(z: torch.Tensor) -> float:
    """log mean exp(-2 |z_i - z_j|^2) over the first views of different runs."""
    a = z[: z.shape[0] // 2]
    d2 = torch.cdist(a, a).pow(2)
    off = ~torch.eye(a.shape[0], dtype=torch.bool, device=a.device)
    return float(torch.log(torch.exp(-2 * d2[off]).mean()))


def effective_rank(h: torch.Tensor) -> float:
    """exp of the entropy of the normalised singular values of the centred h (rows: samples)."""
    s = torch.linalg.svdvals((h - h.mean(dim=0)).double())
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))
