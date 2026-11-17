"""Tests of toy.py: tiny Swift-Hohenberg runs end in the expected state with the amplitude of
the weakly nonlinear theory, and make_ensemble writes an ensemble that io.py can read."""

import numpy as np
import pytest

from phasewall import toy
from phasewall.describe import describe_ensemble
from phasewall.io import load_ensemble, run_snapshots

TINY = {"n": 32, "waves": 4}      # 0.1 s per run


@pytest.mark.parametrize(("eps", "g", "state"), [
    (-0.2, 0.0, "uniform"), (-0.075, 1.0, "uniform"),
    (0.3, 0.0, "stripes"), (0.6, 0.0, "stripes"),
    (0.075, 1.0, "hexagons"), (0.125, 1.2, "hexagons")])
def test_states(eps, g, state):
    snaps = toy.simulate(eps, g, seed=0, **TINY)
    assert [t for t, _ in snaps] == [toy.T_RELAX + j * toy.EVERY for j in range(toy.N_SAVE)]
    assert all(toy.reference_label(u) == state for _, u in snaps)


@pytest.mark.parametrize("eps", [0.1, 0.2])
def test_stripe_amplitude(eps):
    """Near onset, stripes are u = A cos(x) with A^2 = 4 eps / 3 (for g = 0), so the standard
    deviation of u is sqrt(2 eps / 3)."""
    u = toy.simulate(eps, 0.0, seed=1, **TINY)[-1][1]
    assert u.std() == pytest.approx(np.sqrt(2 * eps / 3), rel=0.02)


def test_seeds_give_different_runs():
    a = toy.simulate(0.3, 0.0, seed=0, **TINY)[-1][1]
    b = toy.simulate(0.3, 0.0, seed=1, **TINY)[-1][1]
    assert not np.allclose(a, b)


def test_make_ensemble_is_readable(tmp_path):
    runs = toy.make_ensemble(tmp_path, eps=[-0.1, 0.3], g=[0.0], seeds=(0, 1), n_save=3, **TINY)
    assert runs.reference.tolist() == ["uniform", "uniform", "stripes", "stripes"]
    ens = load_ensemble(tmp_path / "phasewall.toml")
    assert ens.parameters == ["eps", "g"]
    assert ens.dx == pytest.approx(toy.box(**TINY)[0])
    assert ens.last == 3 and ens.pair_dt == toy.EVERY
    assert len(run_snapshots(ens, ens.runs.path[0])) == 3
    table, errors = describe_ensemble(ens)
    assert errors == [] and len(table) == 4 * 3
    # the first snapshot has no partner pair_dt earlier, the two others have one
    assert table["u.dyn.still"].isna().sum() == 4
