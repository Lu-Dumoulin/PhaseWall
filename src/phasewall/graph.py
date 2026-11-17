"""The graph of the parameter scan: one node per parameter point, links between neighbours.

Runs at the same parameter values share a node (replicates: other seeds or initial
conditions). Two nodes are neighbours when they have the same values of all parameters but
one and are consecutive along that one, among all the nodes of the scan. A refined line of
the scan (smaller steps near a boundary) therefore links to the coarser points around it
when they share the other values.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DECIMALS = 9      # parameter values are rounded to 9 decimals to identify a point (a choice)


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
