"""Tests of the blocks of observables (describe.py) chosen per field: the defaults of each kind,
a field's own choice, the checks of that choice, and the role-specific blocks against fields
whose answer is known: Gaussian vortices, a cellular flow and a shear flow (Okubo-Weiss), asters, vortices and
saddles of a polarity (+1, +1, -1), and +-1/2 defects of a nematic."""

import numpy as np
import pandas as pd
import pytest

from phasewall import describe as d
from phasewall.fields import BLOCKS, DEFAULT_BLOCKS, Field
from phasewall.io import load_ensemble

N, DX = 64, 0.5
x = DX * (np.arange(N) + 0.5)
X, Y = np.meshgrid(x, x, indexing="ij")
L = N * DX


def blocks_in(row: dict, name: str) -> set[str]:
    return {c.split(".")[1] for c in row if c.startswith(name + ".")}


def test_default_blocks_of_every_kind():
    a = np.random.default_rng(0).normal(size=(N, N, 2))
    for kind, blocks in DEFAULT_BLOCKS.items():
        f = Field("a", ("a",), kind)
        value = a[..., 0] if kind == "scalar" else a
        row = d.describe_snapshot({"a": value}, [f], DX, before={"a": value})
        assert blocks_in(row, "a") == set(blocks), kind
        assert all(kind in BLOCKS[b] for b in blocks)


def test_a_field_computes_only_its_blocks():
    a = np.random.default_rng(1).normal(size=(N, N, 2))
    f = Field("p", ("p",), "polarity", blocks=("amp", "defects"))
    row = d.describe_snapshot({"p": a}, [f], DX, before={"p": a})
    assert set(row) == {"p.amp.log10var", "p.defects.plus", "p.defects.minus"}       # no dyn either


def test_blocks_must_fit_the_kind():
    with pytest.raises(ValueError, match="do not apply"):
        Field("rho", ("rho",), "scalar", blocks=("flow",))
    with pytest.raises(ValueError, match="do not apply"):
        Field("v", ("v",), "velocity", blocks=("defects",))
    with pytest.raises(ValueError, match="do not apply"):
        Field("Q", ("Q",), "nematic", blocks=("amp", "vortx"))
    Field("w", ("w",), "vector", blocks=("vortex", "defects"))          # an unknown role may ask for both


def test_blocks_from_the_configuration(tmp_path):
    pd.DataFrame({"path": ["r0"], "a": [0.1]}).to_csv(tmp_path / "runs.csv", index=False)
    (tmp_path / "ens.toml").write_text('[ensemble]\nruns = "runs.csv"\nparameters = ["a"]\ndx = 1.0\n\n'
                                       '[[fields]]\nname = "p"\nkind = "polarity"\nblocks = ["amp", "defects"]\n')
    assert load_ensemble(tmp_path / "ens.toml").fields[0].selected == ("amp", "defects")


def from_stream(psi: np.ndarray) -> np.ndarray:
    """Incompressible velocity v = (d_y psi, -d_x psi) of a periodic stream function (spectral)."""
    k = 2 * np.pi * np.fft.fftfreq(N, DX)
    f = np.fft.fft2(psi)
    return np.stack([np.fft.ifft2(1j * k[None, :] * f).real, -np.fft.ifft2(1j * k[:, None] * f).real], axis=-1)


def gaussian_vortex(cx: float, cy: float, sign: float = 1.0, r0: float = 3.0) -> np.ndarray:
    return from_stream(sign * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / r0**2))


def test_vortex_cores_count_vortices_and_ignore_shear():
    one = d.vortex_cores(gaussian_vortex(L / 2, L / 2), DX)
    two = d.vortex_cores(gaussian_vortex(L / 4, L / 4) + gaussian_vortex(3 * L / 4, 3 * L / 4, -1), DX)
    assert one["cores"] == pytest.approx(1 / L**2) and two["cores"] == pytest.approx(2 / L**2)
    assert 0 < one["share"] < 0.2
    cells = d.vortex_cores(from_stream(np.sin(2 * np.pi * X / L) * np.sin(2 * np.pi * Y / L)), DX)
    assert cells["cores"] == pytest.approx(4 / L**2)                  # four counter-rotating cells
    shear = np.stack([np.sin(2 * np.pi * Y / L), np.zeros_like(X)], axis=-1)   # W = 0 everywhere: no core
    assert d.vortex_cores(shear, DX)["cores"] == 0
    assert d.vortex_cores(np.zeros((N, N, 2)), DX) == {"share": 0.0, "cores": 0.0}


def angle_field(defects: list[tuple[float, float, float]]) -> np.ndarray:
    """Angle of a field with point defects (x, y, charge), summed over the defects."""
    theta = np.zeros_like(X)
    for cx, cy, q in defects:
        theta += q * np.arctan2(Y - cy, X - cx)
    return theta


def test_winding_defects_of_a_polarity():
    # a box with a +1 and a -1 defect inside, read as a non-periodic frame
    th = angle_field([(L / 3, L / 2, 1), (2 * L / 3, L / 2, -1)])
    p = np.stack([np.cos(th), np.sin(th)], axis=-1)
    q = d.winding_charges(p, "vector", periodic=False)
    assert np.sort(q[np.abs(q) > 0.5]).round(6).tolist() == [-1.0, 1.0]
    counts = d.defect_counts(p, "vector", DX, periodic=False)
    assert counts["plus"] * q.size * DX**2 == pytest.approx(1) and counts["minus"] * q.size * DX**2 == pytest.approx(1)
    # a vortex is +1 too (its angle is that of an aster turned by 90 degrees)
    th = angle_field([(L / 2, L / 2, 1)]) + np.pi / 2
    q = d.winding_charges(np.stack([np.cos(th), np.sin(th)], axis=-1), "vector", periodic=False)
    assert q[np.abs(q) > 0.5].round(6).tolist() == [1.0]


def test_winding_defects_of_a_nematic():
    th = angle_field([(L / 3, L / 2, 0.5), (2 * L / 3, L / 2, -0.5)])      # director angle
    Q = 0.5 * np.stack([np.cos(2 * th), np.sin(2 * th)], axis=-1)
    q = d.winding_charges(Q, "nematic", periodic=False)
    assert np.sort(q[np.abs(q) > 0.25]).round(6).tolist() == [-0.5, 0.5]
    # read as a polarity, the same angles have no integer defect at all
    th_p = angle_field([(L / 3, L / 2, 0.5), (2 * L / 3, L / 2, -0.5)])
    assert np.abs(d.winding_charges(np.stack([np.cos(th_p), np.sin(th_p)], -1), "vector", False)).max() < 0.75


def test_a_uniform_field_has_no_defect():
    p = np.zeros((N, N, 2))
    p[..., 0] = 1.0
    assert d.defect_counts(p, "vector", DX) == {"plus": 0.0, "minus": 0.0}
    assert d.defect_counts(np.zeros((N, N, 2)), "nematic", DX) == {"plus": 0.0, "minus": 0.0}
