"""Tests of trees.py on the synthetic tables of test_walls (no label used): no split in noise or
under a smooth drift, the jumping descriptor and its threshold for two phases, three phases with
two rules, the same tree with the halves swapped, principal components; splits found by both
halves; one run per point (checkerboard halves); one phase in two places joined."""

import numpy as np
import pytest
from test_walls import DIM, bands, table

from phasewall import trees


def jump_at(a, b, col=0, size=3.0):
    m = np.zeros(DIM)
    m[col] = size * (a > 5)
    m[1] += 0.3 * a                       # a smooth drift in another descriptor
    return m


def test_no_split_in_noise_or_drift():
    """Pure noise, a drift along both parameters, and a strong drift along one parameter only
    (a straight cut through it crosses only steep links)."""
    for f in (lambda a, b: np.zeros(DIM), lambda a, b: np.full(DIM, 0.3 * a + 0.2 * b),
              lambda a, b: np.full(DIM, 0.8 * a)):
        t = trees.fit_tree(table(f), ["a", "b"])
        assert t.n_phases == 1 and not t.edges.wall.any()
        assert not t.splits.accepted.any()


def test_two_phases_found_by_the_jumping_descriptor():
    t = trees.fit_tree(table(jump_at), ["a", "b"])
    assert t.n_phases == 2
    first = t.splits.iloc[0]
    assert first.column == "f.blk.x0" and first.accepted
    assert 0.5 < first.threshold < 2.5                   # between the two levels 0 and 3
    left = t.nodes.a <= 5
    assert t.nodes.loc[left, "leaf"].nunique() == 1 and t.nodes.loc[~left, "leaf"].nunique() == 1
    assert "f.blk.x0 >" in trees.rules_text(t) and "f.blk.x0 <=" in trees.rules_text(t)


def test_three_phases_and_their_rules():
    def mean(a, b):
        m = np.zeros(DIM)
        m[0] = 3.0 * (a > 5)
        m[3] = 3.0 * ((a > 5) & (b > 3))
        return m

    t = trees.fit_tree(table(mean), ["a", "b"])
    assert t.n_phases == 3
    assert set(t.splits[t.splits.accepted].column) == {"f.blk.x0", "f.blk.x3"}
    assert any(r.count(" and ") == 1 for r in t.rules.values())     # nested rule of depth 2


def test_swapped_halves_give_the_same_tree():
    tab = table(jump_at, seed=3)
    t1, t2 = trees.fit_tree(tab, ["a", "b"]), trees.fit_tree(tab, ["a", "b"], swap=True)
    assert np.array_equal(t1.nodes.leaf.to_numpy(), t2.nodes.leaf.to_numpy())


def test_components_and_walls_view():
    t = trees.fit_tree(table(jump_at), ["a", "b"], use_components=True)
    assert t.n_phases == 2 and t.columns[0] == "pc.1"
    w = t.as_walls()
    assert w.n_phases == 2 and w.edges.wall.sum() == t.edges.wall.sum()


def test_components_explain_the_variance():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)) @ np.diag([5.0, 1.0, 0.1])
    assert trees.components(X, 0.95).shape[1] == 1          # variances 25, 1, 0.01: the first explains 96%
    assert trees.components(X, 0.99).shape[1] == 2
    assert np.var(trees.components(X, 0.999), axis=0).sum() == pytest.approx(np.var(X, axis=0).sum(), rel=1e-3)


def one_run(t):
    """The first of the two runs of every point: one run per point."""
    return t[t.run % 2 == 0]


def test_accepted_splits_are_found_by_both_halves():
    t = trees.fit_tree(table(jump_at), ["a", "b"])
    acc = t.splits[t.splits.accepted]
    assert len(acc) and (acc.both_halves >= trees.AGREE).all()
    assert t.halves_by == "runs"


def test_one_run_per_point_uses_the_checkerboard():
    for f in (lambda a, b: np.zeros(DIM), lambda a, b: np.full(DIM, 0.3 * a + 0.2 * b)):
        t = trees.fit_tree(one_run(table(f, seed=5)), ["a", "b"])
        assert t.halves_by == "grid" and t.n_phases == 1
    t = trees.fit_tree(one_run(table(jump_at, seed=5)), ["a", "b"])
    assert t.n_phases == 2
    n = t.nodes[t.nodes.phase >= 0]                 # a single noisy point may stay a boundary region
    assert len(n) >= len(t.nodes) - 2
    left = n.a <= 5
    assert n.loc[left, "phase"].nunique() == 1 and n.loc[~left, "phase"].nunique() == 1


def test_checkerboard_colours_and_links():
    pts = np.array([[x, y] for x in range(3) for y in range(3)], dtype=float)
    colour, same = trees.checkerboard(pts)
    assert colour.reshape(3, 3).tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert (colour[same.a] == colour[same.b]).all()
    assert len(same) == 14                      # 3 two-step links along each axis, 2 diagonals in each of 4 squares


def test_one_phase_in_two_places_is_one_phase_of_the_tree():
    t = trees.fit_tree(table(bands((0.0, 8.0, 0.0)), noise=0.5, seed=3), ["a", "b"])
    assert t.nodes.leaf.nunique() == 3 and t.n_phases == 2
    assert t.nodes[t.nodes.a == 0].phase.iloc[0] == t.nodes[t.nodes.a == 9].phase.iloc[0]
