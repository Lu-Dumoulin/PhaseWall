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

4. Both halves (symmetric, the default). The best split placed with v1 must pass the test on v2,
   the best split placed with v2 must pass on v1, and the two must put at least AGREE of the
   points on the same sides. One tree comes out, whichever half is called the first (with
   swapped halves only the threshold shown changes), and a split that one half sees by chance is
   not kept.

5. One run per point (halves_by "grid", chosen automatically when most points have one run). The
   two halves in time of a single run are not independent (a frozen pattern gives v1 = v2), so the
   test no longer sees the run-to-run noise and the tree over-splits (11 phases instead of 3 on
   Swift-Hohenberg with one run per point). The points are then coloured as a checkerboard (parity
   of the sum of their grid steps): a split is placed on the points of one colour, with the links
   between them (two steps along a parameter, or a diagonal); each point of the other colour gets
   the side of the majority of its neighbours, and the test is made on these points, whose runs
   are independent of those that placed the split: agreement of the threshold with the predicted
   sides, Mann-Whitney between them, and the jump ratio on their own links. With symmetric, the
   colours are exchanged too. The rule then applies to every point.

6. Separated leaves (merge, the default): leaves that are one phase in two places are joined as
   in walls step 4 (cores compared with their own spread).

Physics parallel: choosing the descriptor and the threshold is choosing an order parameter and
the value that separates two phases (the midpoint of a bimodal distribution, as for liquid and gas
densities at coexistence); the normalised cut is the length of the interface between the two
regions of the parameter plane, relative to their sizes, as an interfacial energy.

No label is used anywhere; supervised trees (fitted to labels) are not part of phasewall.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .describe import descriptor_columns
from .graph import links, parameter_nodes
from .walls import ALPHA, MERGE, MIN_PHASE, RATIO, Walls, halves, merge_table_phases, pieces

N_THRESHOLDS = 32       # candidate thresholds per descriptor and region, at quantiles (a choice)
AGREE = 0.9             # share of points on the same side with the other half (a choice)
SYMMETRIC = True        # a split must be found by both halves, each tested on the other (a choice, see fit_tree)
TINY = 1e-12


@dataclass
class Tree:
    nodes: pd.DataFrame     # one row per parameter point: parameters, leaf, phase
    edges: pd.DataFrame     # grid links: a, b, axis, param, wall (between different leaves)
    splits: pd.DataFrame    # every split tried: region size, column, threshold, scores, test, accepted
    rules: dict             # leaf -> rule (the conditions on the path from the root)
    parameters: list[str]
    columns: list[str]
    merges: pd.DataFrame = field(default_factory=pd.DataFrame)   # separated leaves joined (walls step 4)
    halves_by: str = "runs"                                       # "runs" or "grid" (checkerboard)

    @property
    def n_phases(self) -> int:
        return int(self.nodes.loc[self.nodes.phase >= 0, "phase"].nunique())

    def as_walls(self) -> Walls:
        """The tree as a walls.Walls object, for the plots of report.py."""
        nodes = self.nodes.assign(region=self.nodes.leaf, mixed=False, runs=1, snapshots=1,
                                  spread=np.nan)
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


def point_medians(table: pd.DataFrame, parameters: list[str], columns: list[str], use_components: bool = False):
    """Median of every standardised descriptor over all the snapshots of each point, and the
    number of runs of each point (for the checkerboard test)."""
    X = table[columns].to_numpy(dtype=float)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    if use_components:
        X = components(X)
    node, points = parameter_nodes(table[parameters].to_numpy())
    runs = pd.Series(table["run"].to_numpy()).groupby(node).nunique().to_numpy()
    return np.stack([np.median(X[node == n], axis=0) for n in range(len(points))]), runs


def checkerboard(points: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Colour of every point (0 or 1: parity of the sum of its grid steps, counted by rank) and the
    links between points of the same colour: two steps along one parameter (axis k) or one step
    along two (axis n_parameters + 0, the diagonals)."""
    from .walls import grid_coordinates
    pos = grid_coordinates(points).astype(int)
    colour = pos.sum(axis=1) % 2
    where = {tuple(p): i for i, p in enumerate(pos)}
    rows = []
    d = points.shape[1]
    for i, p in enumerate(pos):
        for k in range(d):
            q = p.copy()
            q[k] += 2
            if tuple(q) in where:
                rows.append((i, where[tuple(q)], k))
            for m in range(k + 1, d):
                for sign in (1, -1):
                    q = p.copy()
                    q[k] += 1
                    q[m] += sign
                    if tuple(q) in where:
                        rows.append((i, where[tuple(q)], d))
    return colour, pd.DataFrame(rows, columns=["a", "b", "axis"])


def test_split_on_colour(members, colour, c, a, b, same_a, same_b, same_axis, v, j, threshold):
    """Checkerboard test of a split placed on the points of colour 1 - c: the points of colour c
    get the side of the majority of their grid neighbours of the other colour (ties left out); then
    agreement of the threshold with these sides, Mann-Whitney between the sides, and the jump ratio
    on the links between points of colour c (as test_split)."""
    inside = np.isin(a, members) & np.isin(b, members)
    placed_side = {}
    for x, y in zip(a[inside], b[inside]):
        for p, q in ((x, y), (y, x)):
            if colour[p] == c and colour[q] != c:
                placed_side.setdefault(p, []).append(v[q, j] > threshold)
    test = np.array(sorted(p for p, sides in placed_side.items() if np.mean(sides) != 0.5))
    if len(test) < 2 * MIN_PHASE:
        return 0.0, 1.0, 1.0
    side = np.array([np.mean(placed_side[p]) > 0.5 for p in test])
    if side.all() or not side.any():
        return 0.0, 1.0, 1.0
    x = v[test, j]
    agree = float(np.mean((x > threshold) == side))
    p = float(mannwhitneyu(x[side], x[~side], alternative="two-sided").pvalue)
    keep = np.isin(same_a, test) & np.isin(same_b, test)
    pos = np.full(len(v), -1)
    pos[test] = np.arange(len(test))
    ia, ib = pos[same_a[keep]], pos[same_b[keep]]
    jump = np.abs(x[ia] - x[ib])
    cross = side[ia] != side[ib]
    if not cross.any() or cross.all():
        return agree, p, 1.0
    ax = same_axis[keep]
    ratios, weights = [], []
    for k in np.unique(ax[cross]):
        cc, ii = cross & (ax == k), ~cross & (ax == k)
        if ii.any():
            ratios.append(np.median(jump[cc]) / max(np.median(jump[ii]), TINY))
            weights.append(cc.sum())
    return agree, p, float(np.average(ratios, weights=weights)) if ratios else 1.0


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


def split_score(members, a, b, v, j, threshold) -> float:
    """eta^2 (1 - Ncut / 2) of one threshold on the values v (as in best_split)."""
    pos = np.full(max(a.max(), b.max(), members.max()) + 1, -1)
    pos[members] = np.arange(len(members))
    inside = (pos[a] >= 0) & (pos[b] >= 0)
    ia, ib = pos[a[inside]], pos[b[inside]]
    x = v[members, j]
    side = x > threshold
    n1, n = side.sum(), len(x)
    if n1 == 0 or n1 == n or x.var() < TINY:
        return 0.0
    eta2 = n1 * (n - n1) * (x[side].mean() - x[~side].mean()) ** 2 / (n**2 * x.var())
    deg = np.bincount(np.r_[ia, ib], minlength=n).astype(float)
    cut = (side[ia] != side[ib]).sum()
    vs = deg[side].sum()
    ncut = cut / max(vs, TINY) + cut / max(deg.sum() - vs, TINY)
    return float(eta2 * (1 - min(ncut, 2) / 2))


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
             min_phase: int = MIN_PHASE, agree_min: float = AGREE, symmetric: bool = SYMMETRIC,
             merge: bool = MERGE, halves_by: str = "auto") -> Tree:
    """The unsupervised threshold tree of a table of descriptors (see the module docstring).
    symmetric: a split is kept only if the best split placed with the first halves passes the test
    on the second, the best split placed with the second passes on the first, and the two put at
    least AGREE of the points on the same sides; the rule shown is the first one (the second with
    swap). Not symmetric: placed with one half, tested with the other (swap exchanges them).
    merge: leaves that are one phase in separate places are joined (walls step 4).
    halves_by: "runs" (the two halves of the runs of each point, or of its snapshots in time when
    it has one run), "grid" (checkerboard of the points: placed on one colour, tested on the other,
    for points with a single run, whose two halves in time are not independent), or "auto": "grid"
    when most points have a single run."""
    cols = columns or descriptor_columns(table)
    _, points, names, v1, v2, mean, std = point_values(table, parameters, cols, use_components)
    v_all, runs_per_point = point_medians(table, parameters, cols, use_components)
    lk = links(points)                  # grid links, or k-NN links for scattered points
    if halves_by == "auto":             # the checkerboard needs a grid; scattered points use the runs
        halves_by = "grid" if np.median(runs_per_point) < 2 and lk.kind == "grid" else "runs"
    if halves_by not in ("runs", "grid"):
        raise ValueError(f"halves_by must be 'auto', 'runs' or 'grid', not {halves_by!r}")
    if halves_by == "grid" and lk.kind != "grid":
        raise ValueError("halves_by='grid' (checkerboard) needs a grid scan; these points are scattered")
    if swap:
        v1, v2 = v2, v1
    edges = lk.edges.drop(columns="length")
    if halves_by == "grid":
        colour, same = checkerboard(points)
        same_a, same_b, same_axis = same["a"].to_numpy(), same["b"].to_numpy(), same["axis"].to_numpy()
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
        if halves_by == "grid":
            ok, entry, side1, j, thr = _grid_split(members, a, b, colour, same_a, same_b, same_axis, v_all,
                                                   min_phase, agree_min, alpha, ratio, symmetric, swap)
            if entry is None:
                done.append((members, path))
                continue
            value = thr * std[j] + mean[j]
            tried.append({"size": len(members), "column": names[j], "threshold": value, **entry, "accepted": ok})
            if ok:
                queue += [(members[side1], [*path, f"{names[j]} > {value:.4g}"]),
                          (members[~side1], [*path, f"{names[j]} <= {value:.4g}"])]
            else:
                done.append((members, path))
            continue
        s = best_split(members, a, b, v1, min_phase)
        if s is None:
            done.append((members, path))
            continue
        j, thr, eta2, ncut, score = s
        side1 = v1[members, j] > thr
        agree, p, r = test_split(members, a, b, v2, j, thr, side1, edges["axis"].to_numpy())
        ok = agree >= agree_min and p < alpha and r >= ratio
        both = np.nan
        if symmetric and ok:                                # the other half must find the same split
            s2 = best_split(members, a, b, v2, min_phase)
            if s2 is None:
                ok = False
            else:
                side2 = v2[members, s2[0]] > s2[1]
                agree2, p2, r2 = test_split(members, a, b, v1, s2[0], s2[1], side2, edges["axis"].to_numpy())
                both = float(max(np.mean(side1 == side2), np.mean(side1 != side2)))
                ok = agree2 >= agree_min and p2 < alpha and r2 >= ratio and both >= agree_min
                # of the two rules, the one that scores better on all the snapshots (so that the
                # tree does not depend on which half is called the first)
                if ok and split_score(members, a, b, v_all, s2[0], s2[1]) > \
                        split_score(members, a, b, v_all, j, thr):
                    j, thr = s2[0], s2[1]
        value = thr * std[j] + mean[j]                     # threshold in the units of the descriptor
        tried.append({"size": len(members), "column": names[j], "threshold": value, "above": int(side1.sum()),
                      "below": int((~side1).sum()), "eta2": eta2, "ncut": ncut, "score": score, "agree": agree,
                      "p": p, "ratio": r, "both_halves": both, "accepted": ok})
        if ok:
            # every point goes to its side by the median over all its snapshots (both halves),
            # so the leaves do not depend on which half is called the first (a choice)
            side = v_all[members, j] > thr if symmetric else side1
            if side.sum() < 1 or (~side).sum() < 1:
                side = side1
            queue += [(members[side], [*path, f"{names[j]} > {value:.4g}"]),
                      (members[~side], [*path, f"{names[j]} <= {value:.4g}"])]
        else:
            done.append((members, path))
    done.sort(key=lambda mp: (-len(mp[0]), mp[0].min()))
    for k, (members, path) in enumerate(done):
        leaf[members] = k
        rules[k] = " and ".join(path) if path else "(all points)"
    sizes = np.bincount(leaf)
    phase_of_leaf = np.where(sizes >= min_phase, np.cumsum(sizes >= min_phase) - 1, -1)
    nodes = pd.DataFrame(points, columns=parameters).assign(leaf=leaf, phase=phase_of_leaf[leaf])
    merges = pd.DataFrame()
    if merge:
        nodes["phase_unmerged"] = nodes["phase"]
        nodes["phase"], merges = merge_table_phases(table, parameters, nodes["phase"].to_numpy(), cols, ratio)
    edges["param"] = [parameters[x] if x >= 0 else "k-NN" for x in edges["axis"]]
    edges["wall"] = leaf[a] != leaf[b]
    return Tree(nodes, edges, pd.DataFrame(tried), rules, list(parameters), names, merges, halves_by)


def _grid_split(members, a, b, colour, same_a, same_b, same_axis, v, min_phase, agree_min, alpha, ratio,
                symmetric, swap):
    """One split of a region with checkerboard halves: placed on the points of one colour (with
    the links between them), tested on the other; with symmetric, also the other way round.
    Returns (accepted, row of the table of splits, side of every member, column, threshold)."""
    first = 1 if swap else 0
    found = []
    for c in ((first, 1 - first) if symmetric else (first,)):
        own = members[colour[members] == c]
        # each colour holds half of the points: sides of at least MIN_PHASE / 2 of them (a choice)
        s = best_split(own, same_a, same_b, v, max(1, min_phase // 2)) if len(own) else None
        if s is None:
            if not found:
                return False, None, None, 0, 0.0
            (j, thr), entry, _ = found[0]
            return False, entry | {"both_halves": np.nan}, v[members, j] > thr, j, thr
        j, thr = s[0], s[1]
        agree, p, r = test_split_on_colour(members, colour, 1 - c, a, b, same_a, same_b, same_axis, v, j, thr)
        found.append(((j, thr), {"above": int(np.sum(v[members, j] > thr)), "below": int(np.sum(v[members, j] <= thr)),
                                 "eta2": s[2], "ncut": s[3], "score": s[4], "agree": agree, "p": p, "ratio": r},
                      agree >= agree_min and p < alpha and r >= ratio))
    (j, thr), entry, ok = found[0]
    side = v[members, j] > thr
    entry["both_halves"] = np.nan
    if symmetric:
        (j2, thr2), _, ok2 = found[1]
        both = float(max(np.mean(side == (v[members, j2] > thr2)), np.mean(side != (v[members, j2] > thr2))))
        entry["both_halves"] = both
        ok = ok and ok2 and both >= agree_min
        if ok and split_score(members, a, b, v, j2, thr2) > split_score(members, a, b, v, j, thr):
            j, thr = j2, thr2                       # the better rule on all the points (as for runs)
            side = v[members, j] > thr
    ok = ok and side.sum() >= min_phase and (~side).sum() >= min_phase
    return ok, entry, side, j, thr


def rules_text(tree: Tree) -> str:
    """The phases with their rules, one per line."""
    sizes = tree.nodes.groupby("leaf").size()
    phase = tree.nodes.groupby("leaf").phase.first()
    return "\n".join(f"phase {phase[k]} ({sizes[k]} points): {tree.rules[k]}" for k in sorted(tree.rules))
