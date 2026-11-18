"""Tests of frames.py and of the non-periodic rules of describe.py, with known answers:
crop sizes and errors, block averaging, S(k) of crops against the exact S(k), finite
differences against spectral derivatives, the windowed moving correlation, rotation and
mirror invariance, and the same descriptors at dx and dx / 2 after coarsening."""

import numpy as np
import pandas as pd
import pytest

from phasewall import describe as d
from phasewall.fields import Field, transform
from phasewall.frames import MAX_SHIFT_FRACTION, coarsen, cut, frame_for
from phasewall.io import Ensemble, write_snapshot

U = Field("u", ("u",))
P = Field("p", ("p",), "vector")
Q = Field("Q", ("Q",), "nematic")


def band_limited(n: int, seed: int, kmin: float, kmax: float, comps: int = 0) -> np.ndarray:
    """Periodic random field made of the modes with kmin <= |k| <= kmax (units 2 pi / box)."""
    rng = np.random.default_rng(seed)
    shape = (n, n, comps) if comps else (n, n)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n, 1 / n), np.fft.fftfreq(n, 1 / n), indexing="ij"))
    if comps:
        k = k[..., None]
    f = np.fft.fft2(rng.standard_normal(shape), axes=(0, 1))
    return np.fft.ifft2(f * ((k >= kmin) & (k <= kmax)), axes=(0, 1)).real


# ---------------------------------------------------------------- frames
def test_whole_periodic_box():
    f = frame_for((64, 64), 0.5, periodic=True)
    assert (f.periodic, f.crop, f.factor, f.dx) == (True, None, 1, 0.5)


def test_default_crop_of_a_non_periodic_box():
    """max_shift defaults to a fifth of the side; the crop keeps that margin from the walls."""
    f = frame_for((100, 100), 0.1, periodic=False)
    assert f.shift == int(MAX_SHIFT_FRACTION * 100)
    assert (f.periodic, f.crop) == (False, 100 - 2 * 20)


def test_crop_coarsening_and_their_errors():
    f = frame_for((128, 128), 0.25, periodic=True, crop=16.0, dx_target=0.5, max_shift=2.0)
    assert (f.periodic, f.crop, f.factor, f.shift, f.dx, f.box_periodic) == (False, 64, 2, 4, 0.5, True)
    with pytest.raises(ValueError, match="dx_target"):
        frame_for((128, 128), 0.25, periodic=True, dx_target=0.6)
    with pytest.raises(ValueError, match="crop"):
        frame_for((128, 128), 0.25, periodic=True, crop=16.2, dx_target=0.5)
    with pytest.raises(ValueError, match="does not fit"):
        frame_for((128, 128), 0.25, periodic=False, crop=24.0, max_shift=5.0)
    frame_for((128, 128), 0.25, periodic=True, crop=24.0, max_shift=5.0)   # the margin wraps around


def test_coarsen_and_cut():
    a = np.kron(np.arange(16.0).reshape(4, 4), np.ones((2, 2)))      # 8 x 8, constant 2 x 2 blocks
    assert np.array_equal(coarsen(a, 2), np.arange(16.0).reshape(4, 4))
    b = np.arange(100.0).reshape(10, 10)
    f = frame_for((10, 10), 1.0, periodic=False, crop=4.0, max_shift=2.0)
    assert np.array_equal(cut(b, f), b[3:7, 3:7])
    assert np.array_equal(cut(b, f, margin=True), b[1:9, 1:9])
    g = frame_for((10, 10), 1.0, periodic=True, crop=8.0, max_shift=2.0)
    assert np.array_equal(cut(b, g, margin=True), b[np.ix_(np.arange(-1, 11) % 10, np.arange(-1, 11) % 10)])


# ---------------------------------------------------------------- non-periodic descriptors
def test_structure_factor_of_crops():
    """Crops of periodic broadband fields: S(k) close to the exact one for wavelengths from
    side / 3 down to side / 11 (mean over 480 crops of 24 fields; measured within 5%, tested
    within 6% for the sampling noise). Longer wavelengths are off by 15-30%, which is why
    they are dropped."""
    n, nc = 256, 48
    dk = 2 * np.pi / nc
    edges = dk * (np.arange(nc // 2) + 0.5)                     # shells centred on m 2 pi / nc
    m = np.arange(1, nc // 2)
    exact = np.exp(-(((m - 6) / 4) ** 2))
    exact /= exact.sum()
    rng = np.random.default_rng(0)
    k = np.hypot(*np.meshgrid(*(2 * np.pi * np.fft.fftfreq(n),) * 2, indexing="ij"))
    amp = np.sqrt(np.exp(-(((k - 6 * dk) / (4 * dk)) ** 2)))
    est = []
    for _ in range(24):
        u = np.fft.ifft2(amp * np.fft.fft2(rng.standard_normal((n, n)))).real
        for _ in range(20):
            i, j = rng.integers(0, n - nc, 2)
            s = d.structure_factor(u[i:i + nc, j:j + nc], 1.0, edges, periodic=False)
            est.append(s / s.sum())
    ratio = np.mean(est, axis=0) / exact
    assert np.all(np.abs(ratio[2:11] - 1) < 0.06), ratio.round(3)     # wavelengths nc/3 ... nc/11
    assert ratio[0] > 1.15                                            # wavelength nc: biased


def test_first_shell_of_a_non_periodic_frame():
    edges = d.k_shells(60, 0.5, min_wavelengths=3)
    dk = 2 * np.pi / 30
    assert edges[0] == pytest.approx(2.5 * dk)
    assert edges[0] < 3 * dk < edges[1]


def test_finite_differences_agree_with_spectral_derivatives():
    p = band_limited(128, 1, 1, 6, comps=2)
    dx = 0.3
    ds, ws = d.div_curl(p, dx)
    df, wf = d.div_curl(p, dx, periodic=False)
    assert np.allclose(df, ds[1:-1, 1:-1], atol=0.03 * np.abs(ds).max())
    assert np.allclose(wf, ws[1:-1, 1:-1], atol=0.03 * np.abs(ws).max())


def test_windowed_moving_correlation():
    """The later frame is the earlier one moved by (3, -2) points: still decorrelated, moving
    at the floor; a move larger than max_shift is not found."""
    u = band_limited(96, 2, 2, 8)
    s, c, x0 = 5, 40, 28
    before = u[x0 - s:x0 + c + s, x0 - s:x0 + c + s]
    after = u[x0 - 3:x0 - 3 + c, x0 + 2:x0 + 2 + c]
    out = d.pair_descriptors(before, after, U, shift=s)
    floor = np.log10(d.DECOR_FLOOR)
    assert out["u.dyn.moving"] == pytest.approx(floor)
    assert out["u.dyn.still"] > floor + 1
    far = u[x0 - 9:x0 - 9 + c, x0:x0 + c]
    assert d.pair_descriptors(before, far, U, shift=s)["u.dyn.moving"] > floor + 1
    frozen = d.pair_descriptors(before, before[s:-s, s:-s], U, shift=s)
    assert frozen["u.dyn.still"] == pytest.approx(floor)


def test_non_periodic_descriptors_are_unchanged_by_rotations_and_mirrors():
    fields = [Field("rho", ("rho",), relative=True), P, Q, Field("v", ("v",), "velocity"),
              Field("q", ("q",), "polarity")]
    n = 72
    snap = {"rho": 1 + 0.2 * band_limited(n, 3, 1, 9), "p": band_limited(n, 4, 1, 9, 2),
            "Q": band_limited(n, 5, 1, 9, 2), "v": band_limited(n, 9, 1, 9, 2), "q": band_limited(n, 10, 1, 9, 2)}
    old = {"rho": 1 + 0.2 * band_limited(n, 6, 1, 9), "p": band_limited(n, 7, 1, 9, 2),
           "Q": band_limited(n, 8, 1, 9, 2), "v": band_limited(n, 11, 1, 9, 2), "q": band_limited(n, 12, 1, 9, 2)}
    frame = frame_for((n, n), 0.5, periodic=False, crop=20.0, max_shift=4.0)

    def describe(s, o):
        from phasewall.frames import prepare
        return d.describe_snapshot(prepare(s, fields, frame), fields, frame.dx,
                                   prepare(o, fields, frame, margin=True), False, frame.shift, 3)

    ref = describe(snap, old)
    for turns in range(4):
        for mirror in (False, True):
            t = describe(transform(snap, fields, turns, mirror), transform(old, fields, turns, mirror))
            for col in ref:
                assert t[col] == pytest.approx(ref[col], rel=1e-9, abs=1e-9), (col, turns, mirror)


def test_same_descriptors_at_dx_and_dx_over_2_after_coarsening():
    """A smooth pattern sampled at dx and at dx / 2, the fine one coarsened to dx: amplitude,
    S(k), moments, order and flow agree within 2%."""
    L, kmax = 20.0, 4

    def sample(n, seed=9):
        x = L * np.arange(n) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        rng = np.random.default_rng(seed)
        u, px, py = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for _ in range(12):
            kx, ky = rng.integers(-kmax, kmax + 1, 2) * 2 * np.pi / L
            ph, a = rng.uniform(0, 2 * np.pi), rng.normal()
            # cell-averaged values, so that a 2 x 2 block average of the fine grid is exact
            u += a * np.cos(kx * X + ky * Y + ph)
            px += a * ky * np.sin(kx * X + ky * Y + ph)
            py -= a * kx * np.sin(kx * X + ky * Y + ph)
        return {"u": u, "p": np.stack([px, py], -1)}

    coarse_frame = frame_for((64, 64), L / 64, periodic=True)
    fine_frame = frame_for((128, 128), L / 128, periodic=True, dx_target=L / 64)
    from phasewall.frames import prepare
    a = d.describe_snapshot(prepare(sample(64), [U, P], coarse_frame), [U, P], coarse_frame.dx)
    b = d.describe_snapshot(prepare(sample(128), [U, P], fine_frame), [U, P], fine_frame.dx)
    assert a.keys() == b.keys()
    for col in a:
        if ".topo." in col:
            continue
        assert b[col] == pytest.approx(a[col], rel=0.02, abs=0.01), col


# ---------------------------------------------------------------- ensembles of mixed grids
def mixed_runs(folder, crop=None, dx_target=None) -> Ensemble:
    """Two runs of the same box, one at 64 points and spacing 0.5, one at 128 and 0.25."""
    rows = []
    for r, n in enumerate((64, 128)):
        for t in (0, 5):
            write_snapshot(folder / f"r{r}" / "Data" / f"{t:04d}.h5", {"u": band_limited(n, t, 2, 6)})
        rows.append({"run": r, "path": str(folder / f"r{r}"), "a": 0.1 * r, "dx": 32.0 / n})
    return Ensemble(runs=pd.DataFrame(rows), parameters=["a"], fields=[U], snapshots="Data/*.h5",
                    time_regex=r"(\d+)\.h5$", last=2, pair_dt=5.0, crop=crop, dx_target=dx_target)


def test_mixed_grids_need_a_common_frame(tmp_path):
    with pytest.raises(ValueError, match="different S\\(k\\) shells"):
        d.describe_ensemble(mixed_runs(tmp_path))
    table, errors = d.describe_ensemble(mixed_runs(tmp_path, crop=24.0, dx_target=0.5))
    assert errors == [] and len(table) == 4
    assert not table.filter(like=".shape.").isna().any().any()
