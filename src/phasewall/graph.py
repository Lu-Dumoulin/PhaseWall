"""The graph of the parameter scan: one node per parameter point, links between neighbours.

Runs at the same parameter values share a node (replicates: other seeds or initial
conditions). Two nodes are neighbours when they have the same values of all parameters but
one and are consecutive along that one, among all the nodes of the scan. A refined line of
the scan (smaller steps near a boundary) therefore links to the coarser points around it
when they share the other values.

Scattered points (a random or space-filling sampling, not a grid) share no values, so they get
no grid link. They are linked to their nearest neighbours instead (`knn_edges`):
1. each parameter is divided by the median gap between its sorted distinct values (on a grid the
   step, for scattered points about range / points: the same factor for every parameter); a
   positive parameter whose values are spread more evenly in log10 than in linear scale (a log
   scan) is taken in log10 first (`sampled_in_log`);
2. every point is linked to its k = max(K_MIN, K_PER_DIMENSION d) nearest points (d: number of
   parameters; 6 up to 3 parameters, 8 for 4, 10 for 5), and a link is kept if either point
   chose the other (about 7 links per point for k = 6);
3. links longer than MAX_LENGTH times the median link are dropped: they cross a hole in the
   sampling, where the state can change without a wall.
`links` chooses: grid links when the scan is a grid (`is_grid`), else k-nearest-neighbour links.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

DECIMALS = 9      # parameter values are rounded to 9 decimals to identify a point (a choice)
K_MIN = 6         # k-NN links: at least 6 neighbours chosen by every point (a choice: about the number of
                  # natural neighbours in two dimensions; in 2D, k = 4, 6, 8, 10 gave 0, 2, 1, 5 false walls in
                  # 100 synthetic noise tables of scattered points, because links that share a point share its
                  # samples and the link test takes them as independent; k = 4 finds fewer real walls)
K_PER_DIMENSION = 2  # ... and 2 per parameter beyond (a choice: k = max(6, 2 d); with 16 parameters k = 6
                  # gave 1 false wall in 10 noise tables and missed walls, k = 32 none, 2026-11-26)
MAX_LENGTH = 3.0  # k-NN links longer than 3 x the median link are dropped (a choice)
LOG_SPAN = 4.0    # sampled_in_log: only positive parameters spanning at least 4 x (max / min) are candidates
                  # (a choice: on a narrower range log and linear spacing look alike)
GRID_SHARE = 0.9  # a scan is a grid when at least 90 % of its points have a grid link (a choice: a grid
                  # with a few extra points keeps its grid links; scattered points have almost none)


def parameter_nodes(values: np.ndarray, decimals: int = DECIMALS) -> tuple[np.ndarray, np.ndarray]:
    """(node of every row, parameter values of every node) for rows of parameter values."""
    points, node = np.unique(np.round(np.asarray(values, dtype=float), decimals), axis=0,
                             return_inverse=True)
    return node.ravel(), points


def grid_edges(points: np.ndarray) -> pd.DataFrame:
    """Links between neighbouring nodes: columns a, b (node numbers, a before b along the
    axis) and axis (the parameter that changes)."""
    rows = []
    for axis in range(points.shape[1]):
        others = [j for j in range(points.shape[1]) if j != axis]
        if others:
            lines = pd.DataFrame(points[:, others]).groupby(list(range(len(others)))).indices.values()
        else:
            lines = [np.arange(len(points))]
        for idx in lines:
            order = np.asarray(idx)[np.argsort(points[idx, axis])]
            rows += [(int(a), int(b), axis) for a, b in itertools.pairwise(order)]
    return pd.DataFrame(rows, columns=["a", "b", "axis"]).astype(int)


def regions(n_nodes: int, edges: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
    """Connected regions of the graph that keeps only the links where `keep` is True;
    numbered 0, 1, ... from the largest region to the smallest (ties: lowest node first)."""
    e = edges[np.asarray(keep, dtype=bool)]
    adj = csr_matrix((np.ones(len(e)), (e["a"].to_numpy(), e["b"].to_numpy())), shape=(n_nodes, n_nodes))
    _, lab = connected_components(adj, directed=False)
    sizes = np.bincount(lab)
    first = np.array([np.flatnonzero(lab == r)[0] for r in range(len(sizes))])
    order = np.lexsort((first, -sizes))          # largest first, then by first node
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[lab]


def is_grid(points: np.ndarray) -> bool:
    """True when at least GRID_SHARE of the points have a grid link and the grid links join
    every point into one piece (a full grid, or a grid with refined lines)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return True
    e = grid_edges(points)
    linked = np.zeros(len(points), dtype=bool)
    linked[e["a"].to_numpy()] = True
    linked[e["b"].to_numpy()] = True
    if linked.mean() < GRID_SHARE:
        return False
    return regions(len(points), e, np.ones(len(e), dtype=bool)).max() == 0


def sampled_in_log(values: np.ndarray) -> bool:
    """True for a positive parameter spanning at least LOG_SPAN whose distinct values are spread
    more evenly in log10 than in linear scale: the largest gap between their cumulative
    distribution and a uniform one (Kolmogorov-Smirnov distance) is smaller in log10. A grid with
    equal ratios or points drawn uniformly in log10 give True; a linear grid or points drawn
    uniformly in linear scale give False. On FitzHugh-Nagumo sampled in log, linear distances
    found 1 phase and log distances 2."""
    u = np.unique(np.asarray(values, dtype=float))
    if len(u) < 3 or u[0] <= 0 or u[-1] / u[0] < LOG_SPAN:
        return False

    def ks(x):
        f = (x - x[0]) / (x[-1] - x[0])
        c = np.arange(len(x)) / (len(x) - 1)
        return float(np.max(np.abs(f - c)))

    return ks(np.log10(u)) < ks(u)


def scaled(points: np.ndarray) -> np.ndarray:
    """Every parameter divided by the median gap between its sorted distinct values, in log10 when
    it was sampled in log (point coordinates in steps; see the module docstring)."""
    p = np.asarray(points, dtype=float).copy()
    for k in range(p.shape[1]):
        if sampled_in_log(p[:, k]):
            p[:, k] = np.log10(p[:, k])
        u = np.unique(p[:, k])
        gap = float(np.median(np.diff(u))) if len(u) > 1 else 1.0
        p[:, k] = (p[:, k] - u[0]) / gap
    return p


def neighbours(dimensions: int) -> int:
    """Number of neighbours each point chooses: max(K_MIN, K_PER_DIMENSION x parameters)."""
    return max(K_MIN, K_PER_DIMENSION * dimensions)


def knn_edges(points: np.ndarray, k: int | None = None, max_length: float = MAX_LENGTH) -> pd.DataFrame:
    """k-nearest-neighbour links of the points (coordinates from `scaled`): columns a < b, axis
    (-1: not along one parameter) and length (in steps). k: neighbours chosen by every point
    (default `neighbours` of the number of parameters). See the module docstring."""
    pos = scaled(points)
    k = neighbours(pos.shape[1]) if k is None else k
    n = len(pos)
    if n < 2:
        return pd.DataFrame({"a": [], "b": [], "axis": [], "length": []}).astype(
            {"a": int, "b": int, "axis": int, "length": float})
    _, near = cKDTree(pos).query(pos, k=min(k + 1, n))
    pairs = sorted({(min(i, int(j)), max(i, int(j))) for i in range(n) for j in near[i, 1:]})
    e = pd.DataFrame(pairs, columns=["a", "b"])
    e["axis"] = -1
    e["length"] = np.linalg.norm(pos[e["a"]] - pos[e["b"]], axis=1)
    return e[e["length"] <= max_length * e["length"].median()].reset_index(drop=True)


@dataclass
class Links:
    """The links of a scan and where its points are: edges (a, b, axis, length), kind ("grid" or
    "knn"), steps (point coordinates in steps: grid ranks, or `scaled` divided by the median
    link length), used for the depths of the walls' steps 4 and 5."""
    edges: pd.DataFrame
    kind: str
    steps: np.ndarray


def links(points: np.ndarray, graph: str = "auto") -> Links:
    """Grid links ("grid"), k-nearest-neighbour links ("knn"), or the grid links if the scan is a
    grid and k-NN links otherwise ("auto")."""
    points = np.asarray(points, dtype=float)
    if graph not in ("auto", "grid", "knn"):
        raise ValueError(f"graph must be 'auto', 'grid' or 'knn', not {graph!r}")
    if graph == "auto":
        graph = "grid" if is_grid(points) else "knn"
    if graph == "grid":
        e = grid_edges(points)
        e["length"] = 1.0
        ranks = np.stack([np.searchsorted(np.unique(points[:, k]), points[:, k]) for k in range(points.shape[1])],
                         axis=1).astype(float)
        return Links(e, "grid", ranks)
    e = knn_edges(points)
    unit = float(e["length"].median()) if len(e) else 1.0
    return Links(e, "knn", scaled(points) / unit)
