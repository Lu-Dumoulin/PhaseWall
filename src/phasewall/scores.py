"""Scores of a grouping of the runs (walls, or a blind method) against labels, at the level of
the parameter points; needs no scikit-learn, so the walls can be scored without the extra
[baselines].

The label of a point is the most common label of its runs, and its group the most common
group (-1: noise, for HDBSCAN). agree = points whose label is the most common label of their
group, as for the walls; ari = adjusted Rand index between the two partitions of the points
(1: identical, about 0: as by chance), on the points that are placed; ari_all = the same with
noise counted as one more group, so that leaving points out is not rewarded; pieces = connected
pieces of the groups on the parameter graph.
"""

import numpy as np
import pandas as pd
from scipy.special import comb

from .graph import grid_edges, parameter_nodes, regions


def n_groups(labels: np.ndarray) -> int:
    return len(set(np.asarray(labels).tolist()) - {-1})


def adjusted_rand(a, b) -> float:
    """Adjusted Rand index of two labellings of the same items (Hubert and Arabie 1985):
    (RI - E[RI]) / (max RI - E[RI]) from the contingency table n_ij with row sums a_i and
    column sums b_j: sum C(n_ij, 2) against sum C(a_i, 2) sum C(b_j, 2) / C(n, 2)."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return float("nan")
    table = pd.crosstab(a, b).to_numpy()
    s_ij = comb(table, 2).sum()
    s_a, s_b = comb(table.sum(axis=1), 2).sum(), comb(table.sum(axis=0), 2).sum()
    expected = s_a * s_b / comb(n, 2)
    top = (s_a + s_b) / 2
    if top == expected:                       # both labellings trivial (one group, or all apart)
        return 1.0
    return float((s_ij - expected) / (top - expected))


def point_scores(groups: pd.DataFrame, column: str, parameters: list[str], labels: pd.Series | None = None) -> dict:
    """Compare a grouping of the runs with labels of the runs (Series indexed by run): at the
    level of the points (majority group and majority label of each point). Without labels only
    groups, pieces and noise are given."""
    g = groups.set_index("run")[column]
    node, points = parameter_nodes(groups[parameters].to_numpy())
    runs = groups["run"].to_numpy()

    def majority(values):
        return pd.Series(values).groupby(node).agg(lambda s: s.value_counts().idxmax()).to_numpy()

    pg = majority(g.loc[runs].to_numpy())
    kept = pg != -1
    edges = grid_edges(points)
    same = pg[edges["a"].to_numpy()] == pg[edges["b"].to_numpy()]
    pieces = len(np.unique(regions(len(points), edges, same)[kept]))
    if labels is None:
        return {"groups": n_groups(pg), "noise_points": int((~kept).sum()), "points": len(points),
                "pieces": pieces, "noise_runs": float(np.mean(g.to_numpy() == -1))}
    pl = majority(labels.loc[runs].to_numpy())
    table = pd.crosstab(pl[kept], pg[kept])
    return {"groups": n_groups(pg), "noise_points": int((~kept).sum()), "agree": int(table.max(axis=0).sum()),
            "points": len(points),
            "ari": adjusted_rand(pl[kept], pg[kept]) if kept.any() else float("nan"),
            "ari_all": adjusted_rand(pl, pg),
            "pieces": pieces, "noise_runs": float(np.mean(g.to_numpy() == -1))}


