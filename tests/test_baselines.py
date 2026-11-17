"""Tests of baselines.py (skipped without scikit-learn) on synthetic tables with known groups:
three well separated clouds are found by the three methods, a grouping that splits one region
into two places counts two pieces, and the probe is near 1 for separable labels and near 1/2
for labels that the description does not contain."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from phasewall import baselines as b

DIM = 6


def three_states(seed: int = 0, shift: float = 6.0, look=(0, 1, 2)) -> tuple[pd.DataFrame, pd.Series]:
    """A 12 x 6 grid; state 0 for a < 4, 1 for 4 <= a < 8, 2 for a >= 8; 2 runs per point,
    3 snapshots per run; descriptors Gaussian around shift x (one-hot of look[state])."""
    rng = np.random.default_rng(seed)
    rows, labels, run = [], {}, 0
    for a in range(12):
        for g in range(6):
            state = a // 4
            for _ in range(2):
                labels[run] = ["s0", "s1", "s2"][state]
                for t in range(3):
                    x = rng.standard_normal(DIM)
                    x[look[state]] += shift
                    rows.append({"run": run, "time": float(t), "a": float(a), "g": float(g)}
                                | {f"f.blk.x{j}": x[j] for j in range(DIM)})
                run += 1
    return pd.DataFrame(rows), pd.Series(labels)


def test_run_matrix_averages_snapshots():
    t, _ = three_states()
    M, runs = b.run_matrix(t, ["a", "g"])
    assert M.shape == (144, DIM) and list(runs.columns) == ["run", "a", "g"]


def test_three_clouds_are_found_by_every_method():
    t, labels = three_states()
    groups, info = b.blind_groups(t, ["a", "g"], n_init=3)
    assert info["mixture"]["k"] == 3 and info["kmeans"]["k"] == 3
    for col in ("mixture", "kmeans", "hdbscan_10"):
        s = b.point_scores(groups, col, ["a", "g"], labels)
        assert (s["groups"], s["agree"], s["points"], s["pieces"]) == (3, 72, 72, 3), (col, s)
        assert s["ari"] == pytest.approx(1) and s["ari_all"] == pytest.approx(1)


def test_pieces_count_separate_places_of_one_group():
    """States 0 and 2 described alike: one group in two separate places of the grid."""
    t, labels = three_states(look=(0, 1, 0))
    groups, _ = b.blind_groups(t, ["a", "g"], n_init=3)
    s = b.point_scores(groups, "kmeans", ["a", "g"], labels)
    assert s["groups"] == 2 and s["pieces"] == 3


def test_probe():
    t, labels = three_states(shift=3.0)
    assert b.probe(t, ["a", "g"], labels) > 0.95
    rng = np.random.default_rng(1)
    random = pd.Series(rng.choice(["x", "y"], size=len(labels)), index=labels.index)
    assert abs(b.probe(t, ["a", "g"], random) - 0.5) < 0.1


def test_no_point_placed_gives_no_ari():
    t, labels = three_states()
    groups, _ = b.blind_groups(t, ["a", "g"], n_init=3)
    groups["hdbscan_10"] = -1                                    # everything is noise
    s = b.point_scores(groups, "hdbscan_10", ["a", "g"], labels)
    assert s["groups"] == 0 and np.isnan(s["ari"]) and s["ari_all"] == pytest.approx(0)
