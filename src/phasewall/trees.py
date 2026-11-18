"""An unsupervised threshold tree: phases as the leaves of a tree of single-descriptor thresholds,
chosen and tested without any label.

A phase boundary often shows as a jump of one descriptor (an order parameter: the skewness between
hexagons and stripes, the amplitude at the uniform state). The tree looks for such a descriptor
and a threshold, splits the parameter points of a region in two, and repeats on each side. The
leaves are the candidate phases, and each wall between them comes with a rule that can be read
("u.mom.skew > 0.31").

1. Values. Every descriptor is standardised over all snapshots; the value of a parameter point is
   the median over its snapshots, computed separately on the two halves of its runs (or of its
   snapshots in time, for a single run), as in walls.halves: v1 places the splits, v2 tests them.

2. Splits (on v1). For a region (a connected set of points) and every descriptor, candidate
   thresholds are N_THRESHOLDS quantiles of the point values. A candidate splits the points into
   S (value above) and its complement, both with at least MIN_PHASE points, and is scored by
   - separation: the share of the variance of the values between the two sides,
     eta^2 = n_S n_S' (m_S - m_S')^2 / (n^2 var), 1 for two tight clouds, small for a smooth drift
     (Otsu's criterion for a threshold);
   - coherence on the parameter grid: the normalised cut of the grid links between the sides,
     Ncut = cut / vol(S) + cut / vol(S') (0 for two sides that share no link, 2 at most), small
     when the sides are two contiguous regions, large for scattered points;
   score = eta^2 (1 - Ncut / 2). The threshold is adaptive: it is chosen anew in every region.

3. Test (on v2, which did not choose the split), three conditions:
   - the separation is real: the v2 values of the points on the two sides differ (two-sided
     Mann-Whitney between the points, p < ALPHA); pure noise fails, since v2 is independent of
     the threshold chosen on v1;
   - it is a jump, not a drift: along each parameter, the median jump of the descriptor across the
     crossing links is compared with its median jump across the inner links along the same
     parameter, and the ratios (weighted by the number of crossing links) must average at least
     RATIO. Comparing along the same parameter matters: a straight cut through a drift crosses
     mostly links along the steep direction, while the inner links mix steep and flat directions,
     which would make any drift look like a jump;
   - the threshold puts at least AGREE of the points on the same side.
   (A Mann-Whitney test on the link jumps themselves, as for the walls, has too little power for
   one noisy descriptor and a wall of a few links.) Accepted: both sides are split again;
   otherwise the region is a leaf.

Physics parallel: choosing the descriptor and the threshold is choosing an order parameter and
the value that separates two phases (the midpoint of a bimodal distribution, as for liquid and gas
densities at coexistence); the normalised cut is the length of the interface between the two
regions of the parameter plane, relative to their sizes, as an interfacial energy.

No label is used anywhere; supervised trees (fitted to labels) are not part of phasewall.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .describe import descriptor_columns
from .graph import grid_edges, parameter_nodes
from .walls import ALPHA, MIN_PHASE, RATIO, Walls, halves, pieces

N_THRESHOLDS = 32       # candidate thresholds per descriptor and region, at quantiles (a choice)
AGREE = 0.9             # share of points on the same side with the other half (a choice)
TINY = 1e-12


@dataclass
class Tree:
    nodes: pd.DataFrame     # one row per parameter point: parameters, leaf, phase
    edges: pd.DataFrame     # grid links: a, b, axis, param, wall (between different leaves)
    splits: pd.DataFrame    # every split tried: region size, column, threshold, scores, test, accepted
    rules: dict             # leaf -> rule (the conditions on the path from the root)
    parameters: list[str]
    columns: list[str]

    @property
    def n_phases(self) -> int:
        return int(self.nodes.loc[self.nodes.phase >= 0, "phase"].nunique())

    def as_walls(self) -> Walls:
        """The tree as a walls.Walls object, for the plots of report.py."""
        nodes = self.nodes.assign(region=self.nodes.leaf, mixed=False, runs=1, snapshots=1, spread=np.nan)
        edges = self.edges.assign(jump=1.0, contrast=1.0)
        return Walls(nodes, edges, self.splits, float("nan"), float("nan"), self.parameters, self.columns,
                     {"tree": True})


def components(X: np.ndarray, variance: float = 0.95) -> np.ndarray:
    """Principal components of X explaining `variance` of its variance (for a learned
    description, whose single columns have no meaning of their own)."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), variance) + 1)
    return Xc @ vt[:k].T


def point_values(table: pd.DataFrame, parameters: list[str], columns: list[str], use_components: bool = False):
    """(node of every row, points, standardised values X, their names, medians per point on the
    first and the second half)."""
    X = table[columns].to_numpy(dtype=float)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    keep = X.std(axis=0) > 0
    X, names = X[:, keep], [c for c, k in zip(columns, keep) if k]
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / std
    if use_components:
        X = components(X)
        names = [f"pc.{j + 1}" for j in range(X.shape[1])]
        mean, std = np.zeros(X.shape[1]), np.ones(X.shape[1])
    node, points = parameter_nodes(table[parameters].to_numpy())
    run, time = table["run"].to_numpy(), table["time"].to_numpy(dtype=float)
    v1, v2 = np.empty((len(points), X.shape[1])), np.empty((len(points), X.shape[1]))
    for n in range(len(points)):
        idx = np.flatnonzero(node == n)
        h = halves(run[idx], time[idx])
        a, b = (idx[h], idx[~h]) if h.any() and (~h).any() else (idx, idx)
        v1[n], v2[n] = np.median(X[a], axis=0), np.median(X[b], axis=0)
    return node, points, names, v1, v2, mean, std


def best_split(members, a, b, v, min_phase=MIN_PHASE, n_thresholds=N_THRESHOLDS):
    """(column, threshold, eta^2, Ncut, score) of the best candidate in a region, or None."""
    pos = np.full(max(a.max(), b.max(), members.max()) + 1, -1)
    pos[members] = np.arange(len(members))
    inside = (pos[a] >= 0) & (pos[b] >= 0)
    ia, ib = pos[a[inside]], pos[b[inside]]
    deg = np.bincount(np.r_[ia, ib], minlength=len(members)).astype(float)
    vol = deg.sum()
    if len(members) < 2 * min_phase or vol == 0:
        return None
    best = None
    vals = v[members]                                       # (points, columns)
    for j in range(vals.shape[1]):
        x = vals[:, j]
        var = x.var()
        if var < TINY:
            continue
        thr = np.unique(np.quantile(x, np.linspace(0, 1, n_thresholds + 2)[1:-1]))
        side = x[None, :] > thr[:, None]                     # (thresholds, points)
        n1 = side.sum(axis=1)
        ok = (n1 >= min_phase) & (len(x) - n1 >= min_phase)
        if not ok.any():
            continue
        side, thr, n1 = side[ok], thr[ok], n1[ok]
        n2 = len(x) - n1
        m1 = (side * x).sum(axis=1) / n1
        m2 = ((~side) * x).sum(axis=1) / n2
        eta2 = n1 * n2 * (m1 - m2) ** 2 / (len(x) ** 2 * var)
        cut = (side[:, ia] != side[:, ib]).sum(axis=1)
        vs = side @ deg
        ncut = cut / np.maximum(vs, TINY) + cut / np.maximum(vol - vs, TINY)
        score = eta2 * (1 - np.minimum(ncut, 2) / 2)
        k = int(np.argmax(score))
        if best is None or score[k] > best[4]:
            best = (j, float(thr[k]), float(eta2[k]), float(ncut[k]), float(score[k]))
    return best


def test_split(members, a, b, v2, j, threshold, side1, axis=None):
    """On v2: agreement of the sides, Mann-Whitney p between the points of the two sides, and
    ratio of the median jumps of the descriptor across crossing and inner links."""
    x = v2[members, j]
    agree = float(np.mean((x > threshold) == side1))
    p = float(mannwhitneyu(x[side1], x[~side1], alternative="two-sided").pvalue)
    pos = np.full(max(a.max(), b.max(), members.max()) + 1, -1)
    pos[members] = np.arange(len(members))
    inside = (pos[a] >= 0) & (pos[b] >= 0)
    ia, ib = pos[a[inside]], pos[b[inside]]
    jump = np.abs(x[ia] - x[ib])
    cross = side1[ia] != side1[ib]
    if not cross.any() or cross.all():
        return agree, p, 1.0
    ax = np.zeros(len(jump), dtype=int) if axis is None else np.asarray(axis)[inside]
    ratios, weights = [], []
    for k in np.unique(ax[cross]):
        c, i = cross & (ax == k), ~cross & (ax == k)
        if i.any():
            ratios.append(np.median(jump[c]) / max(np.median(jump[i]), TINY))
            weights.append(c.sum())
    ratio = float(np.average(ratios, weights=weights)) if ratios else 1.0
    return agree, p, ratio


def fit_tree(table: pd.DataFrame, parameters: list[str], columns: list[str] | None = None,
             use_components: bool = False, swap: bool = False, alpha: float = ALPHA, ratio: float = RATIO,
             min_phase: int = MIN_PHASE, agree_min: float = AGREE) -> Tree:
    """The unsupervised threshold tree of a table of descriptors (see the module docstring).
    swap: place the splits with the second halves and test them with the first (stability)."""
    cols = columns or descriptor_columns(table)
    _, points, names, v1, v2, mean, std = point_values(table, parameters, cols, use_components)
    if swap:
        v1, v2 = v2, v1
    edges = grid_edges(points)
    a, b = edges["a"].to_numpy(), edges["b"].to_numpy()
    leaf = np.zeros(len(points), dtype=int)
    rules, tried = {}, []
    queue, done = [(np.arange(len(points)), [])], []
    while queue:
        members, path = queue.pop(0)
        parts = pieces(members, a, b)
        if len(parts) > 1:
            queue += [(p, path) for p in parts]
            continue
        s = best_split(members, a, b, v1, min_phase)
        if s is None:
            done.append((members, path))
            continue
        j, thr, eta2, ncut, score = s
        side1 = v1[members, j] > thr
        agree, p, r = test_split(members, a, b, v2, j, thr, side1, edges["axis"].to_numpy())
        ok = agree >= agree_min and p < alpha and r >= ratio
        value = thr * std[j] + mean[j]                     # threshold in the units of the descriptor
        tried.append({"size": len(members), "column": names[j], "threshold": value, "above": int(side1.sum()),
                      "below": int((~side1).sum()), "eta2": eta2, "ncut": ncut, "score": score, "agree": agree,
                      "p": p, "ratio": r, "accepted": ok})
        if ok:
            queue += [(members[side1], [*path, f"{names[j]} > {value:.4g}"]),
                      (members[~side1], [*path, f"{names[j]} <= {value:.4g}"])]
        else:
            done.append((members, path))
    done.sort(key=lambda mp: (-len(mp[0]), mp[0].min()))
    for k, (members, path) in enumerate(done):
        leaf[members] = k
        rules[k] = " and ".join(path) if path else "(all points)"
    sizes = np.bincount(leaf)
    phase_of_leaf = np.where(sizes >= min_phase, np.cumsum(sizes >= min_phase) - 1, -1)
    nodes = pd.DataFrame(points, columns=parameters).assign(leaf=leaf, phase=phase_of_leaf[leaf])
    edges["param"] = [parameters[x] for x in edges["axis"]]
    edges["wall"] = leaf[a] != leaf[b]
    return Tree(nodes, edges, pd.DataFrame(tried), rules, list(parameters), names)


def rules_text(tree: Tree) -> str:
    """The phases with their rules, one per line."""
    sizes = tree.nodes.groupby("leaf").size()
    phase = tree.nodes.groupby("leaf").phase.first()
    return "\n".join(f"phase {phase[k]} ({sizes[k]} points): {tree.rules[k]}" for k in sorted(tree.rules))
