"""Walls between phases: lines of links of the parameter graph across which the description
of the snapshots jumps more than it does on either side.

1. Jumps. For each link a-b of the scan graph, J(a, b) is the energy distance between the
   snapshots of the two points, averaged over the pairs of halves (a1, b1), (a1, b2), ...;
   each point also gets a spread s(a) = D(a1, a2) between its own two halves. Jump and
   spread compare sets of the same size, so J / s is about 1 when b looks like a again.
   Two more jumps are kept apart: J1 = D(a1, b1) between the first halves, used to place
   the cuts (step 2), and J2 = D(a2, b2) between the second halves, used to test them
   (step 3). They come from different snapshots, so they are independent estimates.

2. Cuts. Each link gets a weight w = exp(-(J1 / median J1)^2): large for small jumps. A random
   walker hopping along the links with these weights stays trapped in regions bounded by
   large jumps. The slowest mode of this diffusion (the second eigenvector of the normalised
   graph Laplacian, the Fiedler vector) orders the points from one side of the strongest
   barrier to the other; the cut along this order with the smallest normalised cut
   (Shi and Malik 2000) splits the region in two.

        o---o---o | o---o
        |   |   | | |   |        crossing links: the four links cut by the wall
        o---o---o | o---o        inner links:    all the others
        |   |   | | |   |
        o---o---o | o---o

3. Test. The cut is kept as a wall only if the crossing links jump more than the inner ones,
   measured with J2: one-sided Mann-Whitney test with p < ALPHA, and median crossing J2 at
   least RATIO times the median inner J2. A single phase whose description changes smoothly
   has crossing links like the others, so no wall is kept. The cut must be tested on other
   data than those that placed it: step 2 puts it where the noisy J1 happen to be largest,
   so J1 on the crossing links is large by construction, and testing with J1 found two or
   three phases in pure noise (synthetic tests, 2026-11-17). This is the selection bias of
   fitting a peak and judging its height on the same noisy spectrum; J2 does not know where
   the cut was put. Each side is then split again, until no cut
   passes: the regions left are the candidate phases, regions with fewer than MIN_PHASE
   points are boundary regions.

The links are not independent (neighbouring jumps are correlated), so p is used as a score
with a strict threshold, not as an exact probability.

A flag per point (it does not change the walls):
- `mixed`: next to a first-order transition the runs of one point can fall in different states
  (bistability, hysteresis); the spread between the halves of runs of such a point is an
  outlier among the points with several runs. Only points with several runs are judged, and only
  against each other: two runs at one point can differ far more than the two time halves of one
  run, so in a scan where most points have one run, a baseline set by all points flags almost
  every point with several runs. With fewer than MIN_MIXED_GROUP such points, nothing is flagged.
  A flag comparing early and late snapshots was tried and dropped (2026-11-23): over a short window
  it cannot tell a run still converging from a steady state with slow dynamics.

0. Runs that ended early. A run whose last snapshot comes before END_FRACTION of the median end
   time of all runs stopped long before the others (a crashed or unstable simulation); its
   snapshots are early transients, not a state to compare with steady ones. Such runs are left
   out (Walls.ended lists them); a point whose runs all ended early is an unstable point, outside
   every phase (Walls.unstable); the other points count their ended runs (nodes.ended_runs).

4. Separated parts of one phase (merge=True). A phase can lie in two places with another phase
   between them (in Gray-Scott, the uniform state on both sides of the pattern band); the cuts
   give two regions, since no link joins them. For every pair of phases that share no link,
   their cores (points at least CORE_DEPTH grid steps
   from any other phase, grid steps counted by rank of the parameter values) are compared: the
   gap is the energy distance between the snapshots of the two cores, the spread of a core the
   energy distance between its two halves in space (either side of the median along its longest
   axis), which includes the drift of the description across the phase. The two are one phase
   when gap / larger spread < RATIO: they differ no more than the two halves of one of them.
   A phase of fewer than 2 MIN_PHASE points cannot be split in two halves of at least
   MIN_PHASE points; it has no spread of its own and is compared with the spread of the other
   phase, or with the median spread of a point when neither has one (before 2026-11-23 such
   phases were not tested, so small pieces of one state could stay apart).
   Cores, not the points facing each other across the gap, because the points at the edge of a
   phase are the least typical of it (faint leftover patterns, transients); two earlier versions
   that compared facing points joined different states or none (Gray-Scott, 2026-11-18). Merges
   are joined transitively (A with B and B with C makes one phase), they only rename phases (the
   walls stay), and the cores are chosen by the geometry alone.

5. Points near the walls (reassign=True). The walls are cuts of the parameter grid: a phase is
   a connected region of at least MIN_PHASE points, so a single point in a state that differs
   from its region cannot be a phase of its own. Where the states alternate from one point to the
   next along a boundary (one run per point in a range where both states are possible), such
   points stay with their region. Every point within NEAR grid steps of another phase is
   therefore compared with the core of its own phase (the core without the point itself) and
   with the cores of the phases within NEAR steps of it; it joins another phase when its
   energy distance to that core is REASSIGN_RATIO times smaller than to its own. Moves are decided from
   the phases before any move (no chain), the walls stay, and the points moved are marked
   (nodes.reassigned, Walls.reassigned): a cluster of them along a boundary is a sign that both
   states are possible there.

Physics parallel: domain walls are found where the order parameter changes abruptly in
space. Here "space" is the parameter grid and the "order parameter" is the distribution of
the descriptors at each point; the diffusion is the walker exploring the grid, slowed down
at the walls, as a particle in a landscape with barriers between basins.

Energy distance (Szekely and Rizzo): E = 2 <|x - y|> - <|x - x'|> - <|y - y'|> for samples X
and Y, zero only when the two distributions are equal; its square root is used.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu

from .describe import descriptor_columns
from .graph import grid_edges, parameter_nodes

ALPHA = 1e-3           # a cut is a wall if the Mann-Whitney p is below this ... (a choice)
RATIO = 1.5            # ... and its crossing links jump RATIO times more than the inner ones (a choice)
MIN_PHASE = 3          # smaller regions are boundary regions, not phases; also the smallest side of a cut (a choice)
FLOOR_QUANTILE = 0.1   # jumps below this quantile of all positive spreads count as no jump (a choice)
MAX_PER_NODE = 200     # at most this many snapshots per point, drawn at random (a choice)
MIXED_K = 6.0          # a point is mixed when its energy statistic > median + MIXED_K robust std (a choice:
                       # the statistic is right-skewed; on pure noise, ordinary points reach 5-6)
END_FRACTION = 0.1     # a run ended early if its last snapshot comes before 0.1 x the median end time of all
                       # runs (a choice: crashed runs stop at a few % of the others; shorter scans at 1/3 or more)
MIN_MIXED_GROUP = 10   # mixed: at least this many points with several runs set the baseline (a choice)
MAD_TO_STD = 1.4826    # median absolute deviation of a Gaussian times this = its standard deviation
MIN_WEIGHT = 1e-8      # links never weigh less than this, so that a strong wall leaves the graph connected (a choice)
MERGE = True           # join separated parts of one phase (step 4) by default (a choice: on the examples it
                       # joins the three pieces of the uniform state of Gray-Scott and changes nothing else)
CORE_DEPTH = 2         # merge test: core of a phase = its points at least 2 grid steps from other phases (a choice)
REASSIGN = True        # step 5 by default (a choice: on the five examples every score is equal or better)
REASSIGN_RATIO = 1.2   # step 5: a point moves when another core is 1.2 times closer than its own (a choice, made on
                       # the examples: 1.2 is the smallest value that moves no correct point on Swift-Hohenberg)
NEAR = 2               # step 5: points within 2 grid steps of another phase are compared with the cores (a choice)
MAX_CORE = 400         # merge test: at most this many snapshots per core, drawn at random (cost; a choice)
SEED = 0


# ---------------------------------------------------------------- features
def feature_matrix(table: pd.DataFrame, columns: list[str] | None = None,
                   weight_blocks: bool = True) -> tuple[np.ndarray, list[str]]:
    """Standardised descriptors (mean 0, standard deviation 1 over all snapshots) and their
    names. Constant columns are dropped; missing values (for example the dynamics of a
    snapshot without partner) are set to the column median (a choice). With weight_blocks,
    each block of columns (field.block, for example rho.shape) is divided by the square root
    of its size, so that every block counts the same in distances, whatever its number of
    columns (a choice)."""
    cols = columns or descriptor_columns(table)
    X = table[cols].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    std = X.std(axis=0)
    keep = std > 0
    X = (X[:, keep] - X[:, keep].mean(axis=0)) / std[keep]
    names = [c for c, k in zip(cols, keep) if k]
    if weight_blocks:
        block = np.array([".".join(c.split(".")[:2]) for c in names])
        for b in np.unique(block):
            X[:, block == b] /= np.sqrt(np.sum(block == b))
    return X, names


# ---------------------------------------------------------------- jumps and spreads
def energy_statistic(A: np.ndarray, B: np.ndarray) -> float:
    """The energy distance 2 <|x - y|> - <|x - x'|> - <|y - y'|> between two samples (rows);
    the terms within a sample leave out the pairs of a point with itself, so the estimate is
    unbiased and comes out negative about half of the time when the two distributions are
    equal."""
    ab = cdist(A, B).mean()
    aa = cdist(A, A).sum() / max(len(A) * (len(A) - 1), 1)
    bb = cdist(B, B).sum() / max(len(B) * (len(B) - 1), 1)
    return float(2 * ab - aa - bb)


def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Square root of the energy statistic; 0 when it comes out negative."""
    return float(np.sqrt(max(energy_statistic(A, B), 0.0)))


def halves(run: np.ndarray, time: np.ndarray) -> np.ndarray:
    """True for the rows in the first half of a point: half of its runs if it has several
    (alternately, by run number), otherwise its earlier half of snapshots in time."""
    ids = np.unique(run)
    if len(ids) > 1:
        return np.isin(run, ids[::2])
    return time < np.median(time) if len(np.unique(time)) > 1 else np.arange(len(time)) % 2 == 0


def jump(X: np.ndarray, sa, sb, ia: np.ndarray, ib: np.ndarray) -> float:
    """Mean energy distance between the halves of two points (the whole points when one of
    them cannot be split)."""
    if sa is None or sb is None:
        return energy_distance(X[ia], X[ib])
    return float(np.mean([energy_distance(X[p], X[q]) for p in sa for q in sb]))


def half_jumps(X: np.ndarray, sa, sb, ia: np.ndarray, ib: np.ndarray) -> tuple[float, float]:
    """(J1, J2): energy distance between the first halves of two points and between their
    second halves. When one of them cannot be split, both are the distance between the
    whole points (then they are not independent)."""
    if sa is None or sb is None:
        d = energy_distance(X[ia], X[ib])
        return d, d
    return energy_distance(X[sa[0]], X[sb[0]]), energy_distance(X[sa[1]], X[sb[1]])


def robust_outliers(values: np.ndarray, k: float, min_scale: float = 0.0) -> np.ndarray:
    """values > median + k robust standard deviations (from the median absolute deviation,
    never below min_scale)."""
    v = values[np.isfinite(values)]
    med = float(np.median(v))
    scale = max(MAD_TO_STD * float(np.median(np.abs(v - med))), min_scale)
    return values > med + k * scale


# ---------------------------------------------------------------- cuts
def _positions(members: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pos = np.full(max(a.max(), b.max(), members.max()) + 1, -1)
    pos[members] = np.arange(len(members))
    return pos


def pieces(members: np.ndarray, a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """The members split into the connected pieces of their subgraph (links between two
    members only), each piece sorted."""
    pos = _positions(members, a, b)
    inside = (pos[a] >= 0) & (pos[b] >= 0)
    m = len(members)
    adj = csr_matrix((np.ones(int(inside.sum())), (pos[a[inside]], pos[b[inside]])), shape=(m, m))
    n, lab = connected_components(adj, directed=False)
    return [np.sort(members[lab == k]) for k in range(n)]


def best_cut(members: np.ndarray, a: np.ndarray, b: np.ndarray, weight: np.ndarray,
             min_side: int = MIN_PHASE) -> np.ndarray | None:
    """Boolean side of each member for the smallest normalised cut along the Fiedler vector
    of the members' subgraph, which must be connected (see pieces); None if it is too small.
    Weights must be positive: a very strong wall then still leaves a small but non-zero link
    between its sides, and the Fiedler vector separates them."""
    m = len(members)
    if m < 2 * min_side:
        return None
    pos = _positions(members, a, b)
    inside = (pos[a] >= 0) & (pos[b] >= 0)
    A = np.zeros((m, m))
    A[pos[a[inside]], pos[b[inside]]] = weight[inside]
    A += A.T
    d = A.sum(axis=1)
    if np.any(d == 0):
        return None
    dm = 1 / np.sqrt(d)
    _, vec = np.linalg.eigh(np.eye(m) - dm[:, None] * A * dm[None, :])
    order = np.argsort(vec[:, 1] * dm)
    best, side = np.inf, None
    vol = d.sum()
    S = np.zeros(m, dtype=bool)
    for k in range(m - min_side):
        S[order[k]] = True
        if k + 1 < min_side:
            continue
        cut = A[S][:, ~S].sum()
        vs = d[S].sum()
        ncut = cut / vs + cut / (vol - vs)
        if ncut < best:
            best, side = ncut, S.copy()
    return side


def test_cut(side: np.ndarray, members: np.ndarray, a: np.ndarray, b: np.ndarray,
             log_jump: np.ndarray) -> dict:
    """Crossing against inner links of a cut: the crossing links, Mann-Whitney p, ratio of
    the median jumps."""
    pos = _positions(members, a, b)
    inside = np.flatnonzero((pos[a] >= 0) & (pos[b] >= 0))
    crossing = side[pos[a[inside]]] != side[pos[b[inside]]]
    cross, inner = inside[crossing], inside[~crossing]
    if len(cross) == 0 or len(inner) == 0:
        return {"crossing": cross, "p": 1.0, "ratio": 1.0}
    p = float(mannwhitneyu(log_jump[cross], log_jump[inner], alternative="greater").pvalue)
    ratio = float(np.exp(np.median(log_jump[cross]) - np.median(log_jump[inner])))
    return {"crossing": cross, "p": p, "ratio": ratio}


def split_regions(n_nodes: int, a: np.ndarray, b: np.ndarray, log_jump: np.ndarray,
                  alpha: float = ALPHA, ratio: float = RATIO, min_phase: int = MIN_PHASE,
                  log_test: np.ndarray | None = None) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Repeated cuts of the graph (see the module docstring): log_jump (log J1) places the
    cuts, log_test (log J2; log_jump itself if None) tests them. Returns the region of every
    node (0 = largest), the list of the cuts tried (accepted or not) and a boolean per link:
    True for the walls."""
    if log_test is None:
        log_test = log_jump
    weight = np.maximum(np.exp(-(np.exp(log_jump - np.median(log_jump))) ** 2), MIN_WEIGHT)
    wall = np.zeros(len(a), dtype=bool)
    queue, done, tried = [np.arange(n_nodes)], [], []
    while queue:
        members = queue.pop(0)
        parts = pieces(members, a, b)
        if len(parts) > 1:          # a side of a cut may be in pieces that share no link
            queue += parts
            continue
        side = best_cut(members, a, b, weight, min_phase)
        if side is None:
            done.append(members)
            continue
        t = test_cut(side, members, a, b, log_test)
        accepted = t["p"] < alpha and t["ratio"] >= ratio
        tried.append({"size": len(members), "side_1": int(side.sum()), "side_2": int((~side).sum()),
                      "crossing_links": len(t["crossing"]), "p": t["p"], "ratio": t["ratio"],
                      "accepted": accepted})
        if accepted:
            wall[t["crossing"]] = True
            queue += [members[side], members[~side]]
        else:
            done.append(members)
    done.sort(key=lambda m: (-len(m), m.min()))
    region = np.empty(n_nodes, dtype=int)
    for r, m in enumerate(done):
        region[m] = r
    return region, tried, wall


# ---------------------------------------------------------------- the whole method
@dataclass
class Walls:
    nodes: pd.DataFrame     # one row per parameter point
    edges: pd.DataFrame     # one row per link: jump J, jump_select J1, jump_test J2, contrast J / noise, wall
    cuts: pd.DataFrame      # every cut tried, accepted or not
    floor: float            # jumps below this count as no jump
    noise: float            # median spread of one point
    parameters: list[str]
    features: list[str]
    settings: dict = field(default_factory=dict)    # alpha, ratio, min_phase used
    merges: pd.DataFrame = field(default_factory=pd.DataFrame)   # merge tests of separated phases (merge=True)
    reassigned: pd.DataFrame = field(default_factory=pd.DataFrame)  # points moved to a nearby phase (step 5)
    ended: pd.DataFrame = field(default_factory=pd.DataFrame)    # runs left out: ended early (run, parameters, end)
    unstable: pd.DataFrame = field(default_factory=pd.DataFrame)  # points whose runs all ended early

    @property
    def n_phases(self) -> int:
        return int(self.nodes.loc[self.nodes["phase"] >= 0, "phase"].nunique())


def grid_coordinates(points: np.ndarray) -> np.ndarray:
    """Position of every point in grid steps: the rank of its value among the values of each
    parameter (so a logarithmic grid has equal steps; a choice)."""
    return np.stack([np.searchsorted(np.unique(points[:, k]), points[:, k]) for k in range(points.shape[1])],
                    axis=1).astype(float)


def core(members: np.ndarray, phase: np.ndarray, pos: np.ndarray, depth: int) -> np.ndarray:
    """The members at least `depth` grid steps (largest of the steps along each parameter) from
    every node outside their phase; all members if fewer than MIN_PHASE are that deep."""
    others = np.flatnonzero(phase != phase[members[0]])
    if not len(others):
        return members
    deep = members[cdist(pos[members], pos[others], "chebyshev").min(axis=1) >= depth]
    return deep if len(deep) >= MIN_PHASE else members


def split_in_space(members: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two halves of a set of nodes on either side of the median along its longest axis."""
    centred = pos[members] - pos[members].mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    t = centred @ axis
    lower = t <= np.median(t)
    return members[lower], members[~lower]


def merge_phases(phase: np.ndarray, points: np.ndarray, a: np.ndarray, b: np.ndarray, X: np.ndarray,
                 rows_of: list[np.ndarray], floor: float, ratio: float = RATIO, rng=None,
                 noise: float | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Step 4 of the module docstring. phase: phase of every node (-1: boundary region); a, b:
    the links; X, rows_of: the feature rows of every node; noise: median spread of a point, the
    scale when neither phase of a pair can be split (default: floor). Returns the new phase of
    every node (numbered from the largest) and one row per pair of phases tested."""
    rng = rng or np.random.default_rng(SEED)
    pos = grid_coordinates(points)
    labels = sorted(set(phase.tolist()) - {-1})
    touching = {(min(p, q), max(p, q)) for p, q in zip(phase[a], phase[b]) if p != q}

    def rows(nodes):
        r = np.concatenate([rows_of[n] for n in nodes])
        return X[rng.choice(r, MAX_CORE, replace=False)] if len(r) > MAX_CORE else X[r]

    cores, spread = {}, {}
    for p in labels:
        members = np.flatnonzero(phase == p)
        cores[p] = core(members, phase, pos, CORE_DEPTH)
        spread[p] = np.nan                       # too small to split in two halves of MIN_PHASE points
        if len(members) >= 2 * MIN_PHASE:
            h1, h2 = split_in_space(cores[p], pos)
            spread[p] = energy_distance(rows(h1), rows(h2)) if len(h1) and len(h2) else np.nan
    parent = {p: p for p in labels}

    def root(p):
        while parent[p] != p:
            p = parent[p]
        return p

    out = []
    for i, pa in enumerate(labels):
        for pb in labels[i + 1:]:
            if (pa, pb) in touching or pa not in cores or pb not in cores:
                continue
            gap = energy_distance(rows(cores[pa]), rows(cores[pb]))
            known = [x for x in (spread[pa], spread[pb]) if np.isfinite(x)]
            scale = max(max(known) if known else (noise if noise is not None else floor), floor)
            merged = gap / scale < ratio
            out.append({"phase_a": pa, "phase_b": pb, "core_a": len(cores[pa]), "core_b": len(cores[pb]),
                        "gap": gap, "spread_a": spread[pa], "spread_b": spread[pb], "ratio": gap / scale,
                        "merged": merged})
            if merged:
                parent[root(pb)] = root(pa)
    group = np.array([root(p) if p >= 0 else -1 for p in phase])
    sizes = pd.Series(group[group >= 0]).value_counts()
    order = sorted(sizes.index, key=lambda g: (-sizes[g], g))
    new = {g: k for k, g in enumerate(order)}
    return np.array([new.get(g, -1) for g in group]), pd.DataFrame(out)


def reassign_near_walls(phase: np.ndarray, points: np.ndarray, X: np.ndarray, rows_of: list[np.ndarray],
                        ratio: float = REASSIGN_RATIO, near: int = NEAR, rng=None) -> tuple[np.ndarray, pd.DataFrame]:
    """Step 5 of the module docstring: the new phase of every node and one row per point moved
    (node, from, to, distance to its own core, distance to the new core, their ratio)."""
    rng = rng or np.random.default_rng(SEED)
    pos = grid_coordinates(points)
    labels = sorted(set(phase.tolist()) - {-1})
    if len(labels) < 2:
        return phase.copy(), pd.DataFrame()

    def rows(nodes):
        r = np.concatenate([rows_of[n] for n in nodes]) if len(nodes) else np.array([], dtype=int)
        return X[rng.choice(r, MAX_CORE, replace=False)] if len(r) > MAX_CORE else X[r]

    cores = {p: core(np.flatnonzero(phase == p), phase, pos, CORE_DEPTH) for p in labels}
    steps = cdist(pos, pos, "chebyshev")
    new, out = phase.copy(), []
    for a in np.flatnonzero(phase >= 0):
        close = (steps[a] <= near) & (phase >= 0) & (phase != phase[a])
        others = sorted(set(phase[close].tolist()))
        if not others:
            continue
        own_core = cores[phase[a]][cores[phase[a]] != a]
        if not len(own_core):
            continue
        xa = X[rows_of[a]]
        d_own = energy_distance(xa, rows(own_core))
        d = {q: energy_distance(xa, rows(cores[q])) for q in others}
        best = min(d, key=d.get)
        if d[best] * ratio < d_own:
            new[a] = best
            out.append({"node": int(a), "from": int(phase[a]), "to": int(best), "d_own": d_own, "d_to": d[best],
                        "ratio": d_own / max(d[best], 1e-12)})
    return new, pd.DataFrame(out)


def merge_table_phases(table: pd.DataFrame, parameters: list[str], phase: np.ndarray,
                       columns: list[str] | None = None, ratio: float = RATIO, seed: int = SEED
                       ) -> tuple[np.ndarray, pd.DataFrame]:
    """Step 4 for phases found by another method (the threshold tree): `phase` of every
    parameter point, numbered as graph.parameter_nodes numbers them."""
    X, _ = feature_matrix(table, columns)
    node, points = parameter_nodes(table[parameters].to_numpy())
    rng = np.random.default_rng(seed)
    rows_of = []
    for n in range(len(points)):
        idx = np.flatnonzero(node == n)
        rows_of.append(np.sort(rng.choice(idx, MAX_PER_NODE, replace=False)) if len(idx) > MAX_PER_NODE else idx)
    run, time = table["run"].to_numpy(), table["time"].to_numpy(dtype=float)
    spreads = []
    for i in rows_of:
        h = halves(run[i], time[i])
        if h.any() and (~h).any():
            spreads.append(energy_distance(X[i[h]], X[i[~h]]))
    positive = np.array([x for x in spreads if x > 0])
    floor = float(np.quantile(positive, FLOOR_QUANTILE)) if len(positive) else 1.0
    noise = max(float(np.median(positive)), floor) if len(positive) else floor
    edges = grid_edges(points)
    return merge_phases(np.asarray(phase), points, edges["a"].to_numpy(), edges["b"].to_numpy(), X, rows_of,
                        floor, ratio, rng, noise=noise)


def ended_early(table: pd.DataFrame, parameters: list[str], fraction: float | None = END_FRACTION
                ) -> pd.DataFrame:
    """The runs whose last snapshot comes before `fraction` times the median end time of all
    runs (step 0 of the module docstring): run, the parameters, end. None: no run."""
    last = table.groupby("run").agg({**{p: "first" for p in parameters}, "time": "max"})
    last = last.rename(columns={"time": "end"}).reset_index()
    if fraction is None or not len(last):
        return last.iloc[:0]
    ref = float(last["end"].median())
    return last[last["end"] < fraction * ref].reset_index(drop=True) if ref > 0 else last.iloc[:0]


def _point_keys(frame: pd.DataFrame, parameters: list[str]) -> list[tuple]:
    return [tuple(r) for r in frame[parameters].to_numpy(dtype=float).round(9)]


def find_walls(table: pd.DataFrame, parameters: list[str], columns: list[str] | None = None,
               alpha: float = ALPHA, ratio: float = RATIO, min_phase: int = MIN_PHASE,
               seed: int = SEED, merge: bool = MERGE, end_fraction: float | None = END_FRACTION,
               reassign: bool = REASSIGN) -> Walls:
    """Walls and candidate phases of a table of descriptors (one row per snapshot, columns
    run, time, the parameters and the descriptors). merge: join phases in separate places
    that are one phase (step 4 of the module docstring); end_fraction: runs that ended early
    are left out (step 0; None keeps every run); reassign: points near the walls join the
    phase whose core they are much closer to (step 5)."""
    ended = ended_early(table, parameters, end_fraction)
    if len(ended):
        table = table[~table["run"].isin(ended["run"])]
    X, names = feature_matrix(table, columns)
    node, points = parameter_nodes(table[parameters].to_numpy())
    rng = np.random.default_rng(seed)
    rows_of = []
    for n in range(len(points)):
        idx = np.flatnonzero(node == n)
        if len(idx) > MAX_PER_NODE:
            idx = np.sort(rng.choice(idx, MAX_PER_NODE, replace=False))
        rows_of.append(idx)
    run, time = table["run"].to_numpy(), table["time"].to_numpy(dtype=float)
    split = []                                   # (first half, second half) of every point
    for i in rows_of:
        h = halves(run[i], time[i])
        split.append((i[h], i[~h]) if h.any() and (~h).any() else None)
    # energy statistic between the two halves of each point (can be negative), and its root
    within = np.array([energy_statistic(X[p[0]], X[p[1]]) if p else np.nan for p in split])
    spread = np.sqrt(np.maximum(within, 0.0))
    by_runs = np.array([len(np.unique(run[i])) > 1 for i in rows_of])

    edges = grid_edges(points)
    edges["param"] = [parameters[a] for a in edges["axis"]]
    edges["jump"] = [jump(X, split[a], split[b], rows_of[a], rows_of[b]) for a, b in zip(edges.a, edges.b)]
    positive = spread[spread > 0]
    # no point with any spread: floor and noise come from the jumps themselves (a choice)
    pool = positive if len(positive) else edges["jump"][edges["jump"] > 0].to_numpy()
    floor = float(np.quantile(pool, FLOOR_QUANTILE)) if len(pool) else 1.0
    noise = max(float(np.median(pool)), floor) if len(pool) else 1.0
    log_jump = np.log(np.maximum(edges["jump"].to_numpy(), floor))
    edges["contrast"] = np.exp(log_jump) / noise
    j12 = np.array([half_jumps(X, split[a], split[b], rows_of[a], rows_of[b])
                    for a, b in zip(edges.a, edges.b)]).reshape(-1, 2)
    edges["jump_select"], edges["jump_test"] = j12[:, 0], j12[:, 1]
    log_select, log_test = np.log(np.maximum(j12, floor)).T

    region, tried, wall = split_regions(len(points), edges["a"].to_numpy(), edges["b"].to_numpy(),
                                        log_select, alpha, ratio, min_phase, log_test=log_test)
    edges["wall"] = wall
    sizes = np.bincount(region)
    phase_of_region = np.where(sizes >= min_phase, np.cumsum(sizes >= min_phase) - 1, -1)

    nodes = pd.DataFrame(points, columns=parameters)
    nodes["runs"] = [len(np.unique(run[node == n])) for n in range(len(points))]
    nodes["snapshots"] = np.bincount(node, minlength=len(points))
    nodes["halves"] = np.where(by_runs, "runs", "time")
    nodes["spread"] = spread
    # the unclipped statistic scatters around 0 for an ordinary point, so its median and MAD
    # are meaningful; deviations below floor^2 (the floor in the same units) never count.
    # mixed: points with several runs only, judged against each other (module docstring)
    mixed = np.zeros(len(points), dtype=bool)
    judged = by_runs & ~np.isnan(within)
    if judged.sum() >= MIN_MIXED_GROUP:
        mixed[judged] = robust_outliers(within[judged], MIXED_K, min_scale=floor**2)
    nodes["mixed"] = mixed
    nodes["region"] = region
    nodes["phase"] = phase_of_region[region]
    merges = pd.DataFrame()
    if merge:
        nodes["phase_unmerged"] = nodes["phase"]
        nodes["phase"], merges = merge_phases(
            nodes["phase"].to_numpy(), points, edges["a"].to_numpy(), edges["b"].to_numpy(), X, rows_of,
            floor, ratio, rng, noise=noise)
    moved = pd.DataFrame()
    nodes["reassigned"] = False
    if reassign:
        nodes["phase_walls"] = nodes["phase"]
        new, moved = reassign_near_walls(nodes["phase"].to_numpy(), points, X, rows_of, REASSIGN_RATIO, NEAR, rng)
        nodes["phase"] = new
        nodes["reassigned"] = nodes["phase"] != nodes["phase_walls"]
    kept = set(_point_keys(nodes, parameters))
    ended_at = pd.Series(_point_keys(ended, parameters), dtype=object).value_counts()
    nodes["ended_runs"] = [int(ended_at.get(k, 0)) for k in _point_keys(nodes, parameters)]
    unstable = ended.assign(_key=_point_keys(ended, parameters))
    unstable = unstable[~unstable["_key"].isin(kept)].groupby(parameters, as_index=False).agg(
        runs=("run", "size"), end=("end", "max"))
    settings = {"alpha": alpha, "ratio": ratio, "min_phase": min_phase, "merge": merge, "end_fraction": end_fraction,
                "reassign": reassign}
    return Walls(nodes, edges, pd.DataFrame(tried), floor, noise, list(parameters), names, settings, merges,
                 moved, ended, unstable)


def phases_against_ratio(table: pd.DataFrame, parameters: list[str],
                         ratios=(1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0), **kw) -> pd.DataFrame:
    """Number of phases for several values of RATIO: a count that stays the same over a range
    of ratios is a robust answer; one that changes with every value is not."""
    out = []
    for r in ratios:
        w = find_walls(table, parameters, ratio=r, **kw)
        out.append({"ratio": r, "phases": w.n_phases, "walls": int(w.edges["wall"].sum()),
                    "boundary_points": int((w.nodes["phase"] < 0).sum())})
    return pd.DataFrame(out)
