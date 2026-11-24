"""Tests of orderparams.py with known answers: the threshold rule on two samples, the descriptor
that tells two phases apart ranked first with the right threshold and values, a phase that no
single descriptor tells from the others but a combination of two does, the noise descriptors
scored near chance on the half that did not choose their threshold, learned columns set aside
when hand-made ones exist, and nothing with a single phase."""

import numpy as np
import pytest
from test_walls import DIM, table

from phasewall import orderparams as op
from phasewall import walls as w


def three_phases():
    """Along a: x0 jumps by 6 at a = 3, x1 by 6 at a = 7; the other columns are noise."""
    def mean_of(a, b):
        m = np.zeros(DIM)
        m[0] = 6.0 if a >= 3 else 0.0
        m[1] = 6.0 if a >= 7 else 0.0
        return m
    t = table(mean_of)
    res = w.find_walls(t, ["a", "b"])
    assert res.n_phases == 3
    n = res.nodes
    low, mid, high = (int(n.loc[n.a == a, "phase"].iloc[0]) for a in (0, 5, 9))
    return t, res, low, mid, high


def pair(out, p, q):
    key = f"phase {min(p, q)} | phase {max(p, q)}"
    return out[out.comparison == key]


def test_best_threshold_and_score():
    t, above, s = op.best_threshold(np.array([0.0, 1, 2]), np.array([5.0, 6]))
    assert (t, above, s) == (3.5, True, 1.0)
    t, above, s = op.best_threshold(np.array([5.0, 6]), np.array([0.0, 1, 2]))
    assert (t, above, s) == (3.5, False, 1.0)
    assert op.score(np.array([0.0, 4]), np.array([5.0, 6]), 3.5, True) == 0.75
    assert op.best_threshold(np.array([1.0]), np.array([1.0]))[2] == 0.5


def test_the_column_that_jumps_is_the_order_parameter():
    t, res, low, mid, high = three_phases()
    out = op.order_parameters(res, t, top=DIM)
    first = pair(out, low, mid).query("rank == 1").iloc[0]
    assert first.column == "f.blk.x0" and first.held_out == 1.0 and bool(first.shared_neighbour)
    thr = float(first.rule.split()[-1])
    assert 0.5 < thr < 5.5                                          # between the two phases, in units
    a_low = first.a == low
    assert (first.value_b - first.value_a) * (1 if a_low else -1) == pytest.approx(6.0, abs=1.0)
    assert pair(out, mid, high).query("rank == 1").iloc[0].column == "f.blk.x1"
    far = pair(out, low, high)
    assert not far.shared_neighbour.any()                                       # low and high do not touch
    assert far.query("rank == 1").iloc[0].column in ("f.blk.x0", "f.blk.x1")


def test_noise_descriptors_score_near_chance_on_the_other_half():
    t, res, low, mid, _ = three_phases()
    out = pair(op.order_parameters(res, t, top=DIM), low, mid)
    noise = out[~out.column.isin(["f.blk.x0", "f.blk.x1", "combination"])]
    assert len(noise) == DIM - 2
    assert (noise.held_out < 0.8).all()


def test_a_combination_tells_the_middle_phase_from_the_others():
    t, res, _, mid, _ = three_phases()
    out = op.order_parameters(res, t)
    rows = out[out.comparison == f"others | phase {mid}"]
    best = rows.query("rank == 1").iloc[0]
    assert best.held_out == pytest.approx(0.75, abs=0.05)           # x0 alone leaves the high phase out
    combo = rows[rows.column == "combination"]
    assert len(combo) == 1 and combo.held_out.iloc[0] > 0.95
    formula = combo.rule.iloc[0]
    assert "z(f.blk.x0)" in formula and "z(f.blk.x1)" in formula   # x0 - x1: 6 in the middle only


def test_learned_columns_set_aside_and_single_phase_empty():
    t, res, *_ = three_phases()
    t = t.assign(**{"cnn.h.d000": t["f.blk.x0"]})
    assert "cnn.h.d000" not in set(op.order_parameters(res, t).column)
    assert op.hand_columns(t[["run", "time", "cnn.h.d000"]]) == ["cnn.h.d000"]
    one = table(lambda a, b: np.zeros(DIM))
    assert op.order_parameters(w.find_walls(one, ["a", "b"]), one).empty
