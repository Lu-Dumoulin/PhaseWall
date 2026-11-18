"""Tests of trees.py on the synthetic tables of test_walls (no label used): no split in noise or
under a smooth drift, the jumping descriptor and its threshold for two phases, three phases with
two rules, the same tree with the halves swapped, principal components."""

import numpy as np
import pytest
from test_walls import DIM, table

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
