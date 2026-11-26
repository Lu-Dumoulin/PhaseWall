"""Tests of the FitzHugh-Nagumo, Barkley and localised Swift-Hohenberg examples: reference labels
of synthetic fields with a known state, the spiral-tip counter, short runs with a known end
state, and the ensembles they write."""

import numpy as np
import pytest

from phasewall import barkley as bk
from phasewall import fitzhugh_nagumo as fhn
from phasewall import sh_localized as sl
from phasewall.examples import EXAMPLES
from phasewall.io import load_ensemble

N = 64
X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")


def test_registry_holds_every_example():
    assert set(EXAMPLES) == {"swift-hohenberg", "gray-scott", "fitzhugh-nagumo", "sh-localized", "barkley",
                             "ginzburg-landau-1d"}
    for m in EXAMPLES.values():
        assert {"make_ensemble", "simulate", "reference_label", "GRID", "SETTINGS", "SEEDS", "QUICK", "N_FIELDS",
                "seconds_per_run"} <= set(dir(m))
        assert m.seconds_per_run() > 0


def test_fitzhugh_nagumo_labels():
    flat = {"u": np.full((N, N), -0.7)}
    assert fhn.reference_label([flat, flat]) == "uniform"
    two = {"u": np.where((X > 16) & (X < 48), 0.7, -0.7)}          # two straight fronts
    assert fhn.reference_label([two, two]) == "fronts"
    d = np.hypot((X % 16) - 8, (Y % 16) - 8)
    spots = {"u": np.where(d < 3, 0.7, -0.7)}
    assert fhn.reference_label([spots, spots]) == "patterns"
    holes = {"u": -spots["u"]}                     # the image of the spots under u -> -u
    assert fhn.reference_label([holes, holes]) == "patterns"
    rng = np.random.default_rng(0)
    assert fhn.reference_label([{"u": rng.standard_normal((N, N))}, {"u": rng.standard_normal((N, N))}]) == "dynamic"


def test_fitzhugh_nagumo_run_in_the_uniform_band():
    """A short run in the band between moving domains and Turing patterns ends uniform."""
    snaps = fhn.simulate(0.05, 0.632, 0, n=64, t_relax=500, n_save=2)
    assert fhn.reference_label([s for _, s in snaps]) == "uniform"


def test_barkley_tips_of_a_spiral_pair_and_of_plane_waves():
    """On a periodic box tips come in pairs of opposite charge: the phase of (u, v) winds once
    around each core of a vortex pair; plane waves have no tip."""
    phase = np.arctan2(Y - 31.7, X - 24.3) - np.arctan2(Y - 31.7, X - 40.3)
    assert bk.tips(0.5 + 0.5 * np.cos(phase), 0.3 + 0.3 * np.sin(phase)) == 2
    plane = 2 * np.pi * X / 16
    assert bk.tips(0.5 + 0.5 * np.cos(plane), 0.3 + 0.3 * np.sin(plane)) == 0


def test_barkley_runs():
    rest = bk.simulate(0.3, 0.1, 0, n=64, t_relax=100, n_save=2)
    assert bk.reference_label([s for _, s in rest]) == "rest"
    spirals = bk.simulate(0.8, 0.02, 0, n=64, t_relax=100, n_save=2)
    u = spirals[-1][1]["u"]
    assert 0 <= u.min() and u.max() <= 1           # Barkley's step keeps u in [0, 1]
    assert bk.reference_label([s for _, s in spirals]) == "spirals"


def test_localized_labels_and_initial_patch():
    assert sl.reference_label(np.zeros((128, 128))) == "uniform"
    u0 = sl.initial(128, 16, seed=0)
    assert 0.05 < sl.patterned_share(u0) < 0.6 and abs(u0).max() == pytest.approx(sl.AMPLITUDE, rel=0.05)
    assert sl.reference_label(u0) == "localized"
    full = 1.4 * np.cos(2 * np.pi * 16 * X / 128.0)
    assert sl.reference_label(np.pad(full, ((0, 64), (0, 64)), mode="wrap")) == "extended"


@pytest.mark.parametrize("name", ["fitzhugh-nagumo", "barkley", "sh-localized"])
def test_quick_ensembles_are_readable(tmp_path, name):
    m = EXAMPLES[name]
    quick = dict(m.QUICK)
    grid = {p: list(quick.pop(p))[:2] for p in m.GRID}
    quick["t_relax"] = 20.0
    runs = m.make_ensemble(tmp_path, seeds=(0,), **grid, **quick, n_save=3)
    ens = load_ensemble(tmp_path / "phasewall.toml")
    assert len(runs) == 4 and ens.parameters == list(m.GRID) and len(ens.fields) == m.N_FIELDS
    from phasewall.examples import fields_of
    assert list(ens.fields) == fields_of(name)
    from phasewall.io import run_snapshots
    assert len(run_snapshots(ens, ens.runs.path[0])) == 3
