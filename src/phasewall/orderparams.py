"""Candidate order parameters of the phases found: which descriptors, with one threshold each,
separate two phases, or one phase from all the others, and how well.

After the walls (or the tree), for two groups of parameter points A and B (two phases, sharing a
wall or not; or one phase and all the other phases) and every descriptor c:

- Values. The median of c over the snapshots of each half of the runs of a point (walls.halves:
  half of its runs, or of its snapshots in time when it has a single run): v1 places, v2 tests,
  as in the walls and the tree (trees.point_values).
- Rule. On v1, the threshold t and the side that maximise the balanced accuracy, the mean of the
  share of A below t and the share of B above it (or the reverse): "c > t: B".
- Score. The balanced accuracy of that rule on v2, which did not choose it: the share of points it
  puts on the right side. With about a hundred descriptors, one of them would separate two groups
  of a few points by chance on the values that chose it; on the other half it would not.
- Ranking. By the held-out score; the TOP best are reported with the median value of c in each
  group (in its own units), so that the rule can be read as a candidate order parameter.

A combination is tried when no single descriptor is enough: the Fisher discriminant (the direction
that best separates the two groups relative to their spread) of up to COMBO_MAX descriptors,
chosen one by one on v1 among the COMBO_POOL best single ones (each kept only if it raises the v1
score by COMBO_STEP), its threshold placed on v1 and scored on v2. It is reported only when its
held-out score beats the best single descriptor's by COMBO_GAIN; it is written on standardised
descriptors, z(c) = (c - mean) / std over all the snapshots.

With one run per point the two halves are the two halves in time of one run, which are not
independent (the same limitation as for the tree): the scores are then optimistic.

Physics parallel: an order parameter is a quantity whose value tells the phases apart (the
magnetisation, the density difference at liquid-gas coexistence, a structure factor peak); the
threshold is where it crosses between the two phases. Here candidates are proposed among the
descriptors and kept only if they also work on runs that did not propose them.
"""

import numpy as np
import pandas as pd

from .describe import descriptor_columns
from .trees import point_medians, point_values

TOP = 5            # descriptors reported per comparison (a choice)
COMBO_MAX = 3      # combination: at most 3 descriptors (a choice: short formulas)
COMBO_POOL = 10    # combination: chosen among the 10 best single descriptors on v1 (a choice)
COMBO_STEP = 0.01  # combination: a descriptor is added only if it raises the v1 score by 1 point (a choice)
COMBO_GAIN = 0.03  # combination: reported only if its held-out score beats the best single by 3 points (a choice)
RIDGE = 1e-6       # Fisher discriminant: added to the diagonal of the spread (numerical safety)


def best_threshold(xa: np.ndarray, xb: np.ndarray) -> tuple[float, bool, float]:
    """(threshold, above, balanced accuracy) of the best rule "x > t: B" (above True) or
    "x <= t: B" (above False) on two samples; thresholds at midpoints between sorted values."""
    vals = np.unique(np.concatenate([xa, xb]))
    if len(vals) < 2:
        return float(vals[0]) if len(vals) else 0.0, True, 0.5
    thr = (vals[:-1] + vals[1:]) / 2
    a_below = (xa[None, :] <= thr[:, None]).mean(axis=1)
    b_above = (xb[None, :] > thr[:, None]).mean(axis=1)
    up = (a_below + b_above) / 2                     # "x > t: B"
    k = int(np.argmax(np.maximum(up, 1 - up)))
    return float(thr[k]), bool(up[k] >= 1 - up[k]), float(max(up[k], 1 - up[k]))


def score(xa: np.ndarray, xb: np.ndarray, threshold: float, above: bool) -> float:
    """Balanced accuracy of a rule on two samples."""
    if above:
        return float(((xa <= threshold).mean() + (xb > threshold).mean()) / 2)
    return float(((xa > threshold).mean() + (xb <= threshold).mean()) / 2)


def fisher(Za: np.ndarray, Zb: np.ndarray) -> np.ndarray:
    """Unit direction w maximising the separation of the means relative to the spread within the
    groups (Fisher's linear discriminant)."""
    S = np.cov(Za, rowvar=False) * (len(Za) - 1) + np.cov(Zb, rowvar=False) * (len(Zb) - 1)
    S = np.atleast_2d(S) + RIDGE * np.eye(Za.shape[1])
    wv = np.linalg.solve(S, Zb.mean(axis=0) - Za.mean(axis=0))
    return wv / max(np.linalg.norm(wv), 1e-12)


def compare(ia: np.ndarray, ib: np.ndarray, v1: np.ndarray, v2: np.ndarray, v_all: np.ndarray,
            names: list[str], mean: np.ndarray, std: np.ndarray, top: int = TOP) -> list[dict]:
    """The rules for groups of points ia (A) and ib (B): the top single descriptors, then the
    combination if it earns its place (module docstring). v1, v2, v_all: standardised point
    values (rows: points); mean, std: to write thresholds and values in the units of each descriptor."""
    rows = []
    for j in range(v1.shape[1]):
        t, above, s1 = best_threshold(v1[ia, j], v1[ib, j])
        s2 = score(v2[ia, j], v2[ib, j], t, above)
        rows.append((s2, s1, j, t, above))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    out = []
    for rank, (s2, s1, j, t, above) in enumerate(rows[:top], start=1):
        out.append({"rank": rank, "rule": f"{names[j]} {'>' if above else '<='} {t * std[j] + mean[j]:.4g}",
                    "column": names[j], "held_out": s2, "placing": s1,
                    "value_a": float(np.median(v_all[ia, j]) * std[j] + mean[j]),
                    "value_b": float(np.median(v_all[ib, j]) * std[j] + mean[j])})
    # combination, chosen on v1 only
    pool = [r[2] for r in sorted(rows, key=lambda r: -r[1])[:COMBO_POOL]]
    chosen, best1 = [pool[0]], max(r[1] for r in rows if r[2] == pool[0])
    for j in pool[1:]:
        if len(chosen) == COMBO_MAX:
            break
        trial = chosen + [j]
        wv = fisher(v1[np.ix_(ia, trial)], v1[np.ix_(ib, trial)])
        _, _, s = best_threshold(v1[ia][:, trial] @ wv, v1[ib][:, trial] @ wv)
        if s >= best1 + COMBO_STEP:
            chosen, best1 = trial, s
    if len(chosen) > 1:
        wv = fisher(v1[np.ix_(ia, chosen)], v1[np.ix_(ib, chosen)])
        t, above, s1 = best_threshold(v1[ia][:, chosen] @ wv, v1[ib][:, chosen] @ wv)
        s2 = score(v2[ia][:, chosen] @ wv, v2[ib][:, chosen] @ wv, t, above)
        if s2 >= rows[0][0] + COMBO_GAIN:
            formula = " ".join(f"{'+' if c >= 0 and k else ''}{'-' if c < 0 else ''}{abs(c):.2f} z({names[j]})"
                               for k, (c, j) in enumerate(zip(wv, chosen)))
            out.append({"rank": 0, "rule": f"{formula} {'>' if above else '<='} {t:.3g}", "column": "combination",
                        "held_out": s2, "placing": s1, "value_a": np.nan, "value_b": np.nan})
    return out


def hand_columns(table: pd.DataFrame) -> list[str]:
    """The descriptors a rule is written on: the hand-made ones, which have a meaning; the learned
    ones (cnn.h.dNNN) only when the table has nothing else."""
    cols = descriptor_columns(table)
    return [c for c in cols if not c.startswith("cnn.")] or cols


def order_parameters(w, table: pd.DataFrame, top: int = TOP, columns: list[str] | None = None) -> pd.DataFrame:
    """Candidate order parameters of the phases of w (walls.Walls, or trees.Tree.as_walls()) on a
    table of descriptors of the same runs (by default its hand-made descriptors, even when the
    phases were found on a learned description: a rule on cnn.h.d017 says nothing to a physicist).
    For every pair of phases ("phase 0 | phase 1"; shared_neighbour True when they touch on the grid) and, with
    three phases or more, every phase against all the others ("others | phase 0"), the top
    descriptors with their rule, held-out and placing scores (balanced accuracy, 0.5: chance, 1:
    every point on its side) and median value in each group (a, b); a combination row (rank 0)
    when it earns its place. Empty with fewer than two phases."""
    cols = columns or hand_columns(table)
    _, points, names, v1, v2, mean, std = point_values(table, w.parameters, cols)
    v_all, _ = point_medians(table, w.parameters, cols)
    key = pd.MultiIndex.from_frame(pd.DataFrame(points, columns=w.parameters).round(9))
    node_phase = w.nodes["phase"].to_numpy()
    phase_of = pd.Series(node_phase, index=pd.MultiIndex.from_frame(w.nodes[w.parameters].round(9)))
    phase = phase_of.reindex(key).fillna(-1).to_numpy().astype(int)
    labels = sorted(int(p) for p in set(phase) if p >= 0)
    if len(labels) < 2:
        return pd.DataFrame()
    pa, pb = node_phase[w.edges.a.to_numpy()], node_phase[w.edges.b.to_numpy()]
    touching = {(min(x, y), max(x, y)) for x, y in zip(pa, pb) if x != y and x >= 0 and y >= 0}
    out = []
    for i, p in enumerate(labels):
        for q in labels[i + 1:]:
            for r in compare(np.flatnonzero(phase == p), np.flatnonzero(phase == q), v1, v2, v_all, names, mean, std,
                             top):
                out.append({"comparison": f"phase {p} | phase {q}", "a": p, "b": q, "shared_neighbour": (p, q) in touching} | r)
    if len(labels) > 2:
        for p in labels:
            for r in compare(np.flatnonzero((phase >= 0) & (phase != p)), np.flatnonzero(phase == p), v1, v2, v_all,
                             names, mean, std, top):
                out.append({"comparison": f"others | phase {p}", "a": -1, "b": p, "shared_neighbour": False} | r)
    return pd.DataFrame(out)
