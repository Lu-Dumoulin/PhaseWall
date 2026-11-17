"""Tests of gray_scott.py and of the examples registry: the discrete Laplacian of a wave, the
reference labels of synthetic fields with a known state, a run that decays to the uniform
state, the ensemble files, and the demo command on the quick grid."""

import numpy as np
import pandas as pd
import pytest

from phasewall import cli
from phasewall import gray_scott as gs
from phasewall.examples import EXAMPLES, example
from phasewall.io import load_ensemble

N = 64
x = np.arange(N)
X, Y = np.meshgrid(x, x, indexing="ij")


def test_laplacian_of_a_wave():
    """Five-point Laplacian of cos(q x): -(4 / dx^2) sin^2(q dx / 2) cos(q x)."""
    dx, q = 0.1, 2 * np.pi * 3 / (N * 0.1)
    u = np.cos(q * X * dx)
    assert np.allclose(gs.laplacian(u, dx), -(4 / dx**2) * np.sin(q * dx / 2) ** 2 * u)


def lattice(period=8, radius=2.0):
    """v = 0.3 inside discs on a square lattice, 0 outside."""
    d = np.hypot((X % period) - period / 2, (Y % period) - period / 2)
    return np.where(d < radius, 0.3, 0.0)


def labels_of(v_before, v_after=None):
    v_after = v_before if v_after is None else v_after
    return gs.reference_label([{"v": v_before}, {"v": v_after}])


def test_reference_labels_of_known_fields():
    assert labels_of(np.zeros((N, N))) == "uniform"
    assert labels_of(np.full((N, N), 0.2)) == "uniform (high v)"
    assert labels_of(lattice()) == "spots"
    assert labels_of(0.3 - lattice()) == "holes"
    assert labels_of(np.where(np.cos(2 * np.pi * X / 8) > 0, 0.3, 0.0)) == "stripes"
    lone = np.where(np.hypot((X + 2) % N - 2, (Y + 2) % N - 2) < 3, 0.3, 0.0)   # one spot on the corner
    assert labels_of(lone) == "isolated spots"
    rng = np.random.default_rng(0)
    assert labels_of(rng.random((N, N)), rng.random((N, N))) == "dynamic"


def test_compact_share():
    assert gs.compact_share(lattice() > 0)[1] == pytest.approx(1)
    n, share = gs.compact_share(np.cos(2 * np.pi * X / 8) > 0)
    assert share == 0 and n == 9      # 8 stripes, the one through the edge counted twice (not periodic)


def test_run_in_the_uniform_region_decays():
    snaps = gs.simulate(0.01, 0.07, seed=0, n=32, t_relax=3000, n_save=2)
    assert [t for t, _ in snaps] == [3000.0, 3000.0 + gs.EVERY]
    assert gs.reference_label([s for _, s in snaps]) == "uniform"
    assert np.allclose(snaps[-1][1]["u"], 1, atol=1e-3)


def test_registry_and_ensemble(tmp_path):
    assert example("gray-scott") is gs and {"swift-hohenberg", "gray-scott"} <= set(EXAMPLES)   # full set: test_more_examples
    with pytest.raises(ValueError):
        example("brusselator")
    runs = gs.make_ensemble(tmp_path, F=[0.01], k=[0.07], seeds=(0, 1), n=32, t_relax=1000, n_save=3)
    assert runs.reference.tolist() == ["uniform", "uniform"]
    ens = load_ensemble(tmp_path / "phasewall.toml")
    assert ens.parameters == ["F", "k"] and [f.name for f in ens.fields] == ["u", "v"]
    assert ens.dx == pytest.approx(gs.DX) and ens.pair_dt == gs.EVERY


def test_demo_command_on_the_quick_grid(tmp_path):
    out = cli.run(["demo", str(tmp_path / "gs"), "--example", "gray-scott", "--quick", "--workers", "4"])
    text = (out / "summary.md").read_text()
    assert "gray-scott: reference_label" in text
    runs = pd.read_csv(tmp_path / "gs" / "runs.csv")
    assert len(runs) == len(gs.QUICK["F"]) * len(gs.QUICK["k"]) * 2
