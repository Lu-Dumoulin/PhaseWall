"""Tests of the blocks of observables (describe.py) chosen per field: the defaults of each kind,
a field's own choice, the checks of that choice, and the role-specific blocks against fields
whose answer is known: Gaussian vortices, a cellular flow and a shear flow (Okubo-Weiss), asters, vortices and
saddles of a polarity (+1, +1, -1), and +-1/2 defects of a nematic; the parts block (two states side by
side against one state everywhere, its block sums, its invariance under the symmetries of the box); the
frank block (Frank energy of known orientations, aster and vortex, its change over the lag), and the
numerical noise of a uniform vector or nematic field not described as a pattern, nor that of a size that
stays uniform while the direction turns."""

import numpy as np
import pandas as pd
import pytest

from phasewall import describe as d
from phasewall.fields import BLOCKS, DEFAULT_BLOCKS, Field, transform
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


# ---------------------------------------------------------------- parts
def test_box_means_are_the_block_averages():
    """Every block position, against a direct sum: n^2 positions wrapped in a periodic frame,
    (n - side + 1)^2 inside a non-periodic one."""
    a = np.random.default_rng(0).standard_normal((7, 7))
    side = 3
    per = d.box_means(a, side, periodic=True)
    wrapped = np.tile(a, (2, 2))
    assert per.shape == (7, 7)
    assert per[5, 6] == pytest.approx(wrapped[5:8, 6:9].mean())
    inner = d.box_means(a, side, periodic=False)
    assert inner.shape == (5, 5)
    assert inner[2, 4] == pytest.approx(a[2:5, 4:7].mean())


def test_parts_sees_two_states_side_by_side():
    """Stripes next to hexagons, and spots on a flat background, differ far more from block to
    block than stripes or hexagons alone; a flat field gives the floor."""
    k = 2 * np.pi * 8 / L                          # 8 wavelengths per box
    stripes = np.cos(k * X)
    hexagons = sum(np.cos(k * (np.cos(t) * X + np.sin(t) * Y)) for t in (0, 2 * np.pi / 3, 4 * np.pi / 3))
    half = np.where(X < L / 2, stripes, hexagons / 2)
    spots = np.zeros((N, N))
    spots[10:14, 10:14] = spots[40:44, 30:34] = 1.0
    u = Field("u", ("u",), blocks=("parts",))

    def h(a):
        return d.describe_snapshot({"u": a}, [u], DX)["u.parts.log10H"]

    one_state = max(h(stripes), h(hexagons))
    assert h(half) > one_state + 2
    assert h(spots) > h(half)
    assert h(np.zeros((N, N))) == pytest.approx(np.log10(d.PARTS_H_FLOOR))


def test_parts_is_unchanged_by_the_symmetries_of_the_box():
    """Rotations, mirrors and (whole periodic box) shifts, for every kind of field; D4 also in a
    non-periodic frame, where the blocks lie inside the frame."""
    rng = np.random.default_rng(3)

    def smooth(shape):
        f = np.fft.fft2(rng.standard_normal(shape), axes=(0, 1))
        kk = np.hypot(*np.meshgrid(np.fft.fftfreq(N, 1 / N), np.fft.fftfreq(N, 1 / N), indexing="ij"))
        f[kk > 6] = 0
        return np.fft.ifft2(f, axes=(0, 1)).real

    fields = [Field("u", ("u",), blocks=("parts",)), Field("p", ("p",), "vector", blocks=("parts",)),
              Field("Q", ("Q",), "nematic", blocks=("parts",)), Field("v", ("v",), "velocity", blocks=("parts",)),
              Field("q", ("q",), "polarity", blocks=("parts",))]
    snap = {"u": smooth((N, N)), "p": smooth((N, N, 2)), "Q": smooth((N, N, 2)), "v": smooth((N, N, 2)),
            "q": smooth((N, N, 2))}
    for periodic, shifts in ((True, [(0, 0), (7, 19)]), (False, [(0, 0)])):
        ref = d.describe_snapshot(snap, fields, DX, periodic=periodic)
        assert len(ref) == 5
        for turns in range(4):
            for mirror in (False, True):
                for shift in shifts:
                    t = d.describe_snapshot(transform(snap, fields, turns, mirror, shift), fields, DX,
                                            periodic=periodic)
                    for c in ref:
                        assert t[c] == pytest.approx(ref[c], rel=1e-9, abs=1e-9), (c, periodic, turns, mirror, shift)


# ---------------------------------------------------------------- frank block
K = 2 * np.pi * 3 / L                                  # three turns of the orientation across the box


def test_frank_energy_of_known_fields():
    """An orientation turning at rate k: F = k^2 (polarity, and nematic with S = 1), half splay and
    half bend; an aster is splay and a vortex bend; a change of the size alone has no Frank energy."""
    p = np.stack([np.cos(K * X), np.sin(K * X)], -1)
    s, b = d.frank_density(p, "vector", DX)
    assert np.mean(s + b) == pytest.approx(K**2, rel=1e-9) and np.mean(s) == pytest.approx(K**2 / 2, rel=1e-9)
    q = 0.5 * np.stack([np.cos(2 * K * X), np.sin(2 * K * X)], -1)           # S = 2 |Q| = 1
    s, b = d.frank_density(q, "nematic", DX)
    assert np.mean(s + b) == pytest.approx(K**2, rel=1e-9)
    c = L / 2
    r = np.hypot(X - c, Y - c)
    aster = np.stack([(X - c) / r, (Y - c) / r], -1)
    vortex = np.stack([-(Y - c) / r, (X - c) / r], -1)
    for a, splay in ((aster, 1.0), (vortex, 0.0)):
        s, b = d.frank_density(a, "vector", DX, periodic=False)
        assert np.mean(s) / np.mean(s + b) == pytest.approx(splay, abs=0.01)
    size_only = np.stack([1 + 0.5 * np.sin(K * X), np.zeros_like(X)], -1)
    s, b = d.frank_density(size_only, "vector", DX)
    assert np.max(s + b) < 1e-20


def test_frank_change_over_the_lag():
    """Three times fewer turns at t - lag than at t: change = ln F(t) - ln F(t - lag) = ln 9;
    the same field twice: 0."""
    p = Field("p", ("p",), "polarity")
    now = np.stack([np.cos(K * X), np.sin(K * X)], -1)
    before = np.stack([np.cos(K * X / 3), np.sin(K * X / 3)], -1)
    row = d.describe_snapshot({"p": now}, [p], DX, before={"p": before})
    assert row["p.frank.change"] == pytest.approx(np.log(9), rel=1e-9)
    assert row["p.frank.splay_share"] == pytest.approx(0.5)
    assert d.describe_snapshot({"p": now}, [p], DX, before={"p": now})["p.frank.change"] == 0


@pytest.mark.parametrize("kind", ["polarity", "nematic"])
def test_numerical_noise_on_a_uniform_field_is_not_a_pattern(kind):
    """A uniform field and the same field with fluctuations of 2.5e-5 of its size (numerical
    noise) give the same descriptors; fluctuations of 1% do not."""
    f = Field("p", ("p",), kind)
    u = np.zeros((N, N, 2))
    u[..., 0] = 1.0
    rng = np.random.default_rng(0)
    rows = [d.describe_snapshot({"p": a}, [f], DX, before={"p": a})
            for a in (u, u + 2.5e-5 * rng.standard_normal(u.shape), u + 1e-2 * rng.standard_normal(u.shape))]
    assert max(abs(rows[0][c] - rows[1][c]) for c in rows[0]) < 1e-6
    assert abs(rows[0]["p.amp.log10var"] - rows[2]["p.amp.log10var"]) > 5


def test_a_turning_polarity_of_uniform_size_has_uniform_size_descriptors():
    """The direction of p turns slowly (by up to 3 degrees) while |p| = 1 with noise of 1e-4: the
    moments and domains of |p| are those of a uniform size (0), the turn itself is kept (variance,
    Frank energy above the floor); a size that varies by 5% keeps its moments and domains."""
    f = Field("p", ("p",), "polarity")
    theta = 0.05 * np.sin(2 * np.pi * X / L)
    rng = np.random.default_rng(1)
    size = 1 + 1e-4 * rng.standard_normal(X.shape)
    p = np.stack([size * np.cos(theta), size * np.sin(theta)], -1)
    row = d.describe_snapshot({"p": p}, [f], DX, before={"p": p})
    assert row["p.mom.std"] < 1e-12 and row["p.mom.skew"] == 0 and row["p.mom.kurt"] == 0
    assert all(row[c] == 0 for c in row if c.startswith("p.topo."))
    assert row["p.amp.log10var"] > -5 and row["p.frank.log10F"] > -5
    varying = np.stack([(1 + 0.05 * np.sin(2 * np.pi * Y / L)) * np.cos(theta), np.sin(theta)], -1)
    row = d.describe_snapshot({"p": varying}, [f], DX)
    assert row["p.mom.std"] > 0.01 and any(row[c] > 0 for c in row if c.startswith("p.topo."))
