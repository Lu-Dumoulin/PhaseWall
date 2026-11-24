"""1D systems, hand-made observables of a window: symmetries and known answers."""

import numpy as np
import pytest

from phasewall.oned.describe1d import (
    AMP_FLOOR,
    blocks_of,
    describe_window,
    field_descriptors,
)
from phasewall.oned.kymo import Field1D

NX, NT, DX, DT = 64, 128, 0.5, 0.25
L, T = NX * DX, NT * DT
X = np.arange(NX)[:, None] * DX
TIMES = 100.0 + DT * np.arange(NT)
TT = TIMES[None, :]


def smooth_random(seed, comps=1, scale=3):
    """A random field smooth in x and t, a[x, t, comps]."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((NX, NT, comps))
    kx = np.fft.fftfreq(NX)[:, None, None]
    kt = np.fft.fftfreq(NT)[None, :, None]
    filt = np.exp(-(kx**2 + kt**2) * (2 * np.pi * scale) ** 2 / 2)
    return np.fft.ifft2(np.fft.fft2(a, axes=(0, 1)) * filt, axes=(0, 1)).real * 10


def mirror(a, f):
    """x -> -x on the periodic grid (index i -> -i mod NX), odd components flipped."""
    m = np.roll(a[::-1], 1, axis=0)
    return m * np.array([-1.0 if o else 1.0 for o in f.odd])


FIELDS = [Field1D("u"), Field1D("v", kind="velocity"), Field1D("p", kind="polarity"),
          Field1D("q", kind="polarity", components=("qx", "qy")),
          Field1D("A", kind="complex", components=("re", "im"))]


def describe(a, f, periodic=True):
    return field_descriptors(a, f, DX, DT, TIMES, periodic)


def assert_same(d1, d2):
    assert d1.keys() == d2.keys()
    for k in d1:
        assert d1[k] == pytest.approx(d2[k], rel=1e-7, abs=1e-6), k    # |P+ - P-| loses digits when it nearly cancels


@pytest.mark.parametrize("f", FIELDS, ids=lambda f: f.kind + str(f.n_components))
def test_shift_and_mirror_change_nothing(f):
    a = smooth_random(1, f.n_components)
    if f.kind == "polarity" and f.n_components == 2:
        a[..., 1] += 0.5                       # p_y of one sign: a polarity across the line
    d = describe(a, f)
    assert_same(d, describe(np.roll(a, 17, axis=0), f))
    assert_same(d, describe(mirror(a, f), f))


def test_global_phase_changes_nothing():
    f = FIELDS[-1]
    a = smooth_random(2, 2)
    z = (a[..., 0] + 1j * a[..., 1]) * np.exp(1j * 0.9)
    assert_same(describe(a, f), describe(np.stack([z.real, z.imag], axis=-1), f))


def test_blocks_must_fit_the_kind():
    assert "flow" not in blocks_of(Field1D("u"))
    assert "defects" in blocks_of(Field1D("p", kind="polarity"))
    with pytest.raises(ValueError):
        blocks_of(Field1D("u", blocks=("flow",)))
    with pytest.raises(ValueError):
        blocks_of(Field1D("A", kind="complex", components=("re", "im"), blocks=("kinks",)))
    d = describe(smooth_random(3), Field1D("u", blocks=("amp", "skw")))
    assert {k.split(".")[1] for k in d} == {"amp", "skw"}


def test_travelling_and_standing_waves():
    k, w = 2 * np.pi * 4 / L, 2 * np.pi * 3 / T           # whole periods in x and t
    travelling = describe(np.cos(k * X - w * TT), Field1D("u"))
    standing = describe(np.cos(k * X) * np.cos(w * TT), Field1D("u"))
    assert travelling["u.skw.travelling"] > 0.99
    assert standing["u.skw.travelling"] < 0.01
    assert travelling["u.skw.static"] < 0.1               # 3 periods: just past the STATIC_BINS of a frozen pattern
    # the drift speed is the phase speed w / k = 0.75: at the lag of the moving correlation (16 rows,
    # 4 time units) the wave moves by 3 length units = 6 grid points, less than half a wavelength (8)
    assert 10 ** travelling["u.corr.log10drift"] == pytest.approx(w / k, rel=1e-6)
    assert travelling["u.corr.moving"] == pytest.approx(np.log10(1e-3), abs=1e-6)  # C = 1 after the shift
    assert travelling["u.corr.still"] > -1                                        # not frozen in place


def test_frozen_pattern():
    d = describe(np.cos(2 * np.pi * 3 * X / L) + 0 * TT, Field1D("u"))
    assert d["u.skw.static"] > 0.95
    assert d["u.corr.still"] == pytest.approx(np.log10(1e-3))
    assert d["u.corr.log10tau"] == pytest.approx(np.log10(64 * DT))    # longer than half the window
    assert d["u.kinks.log10births"] == d["u.kinks.log10deaths"] == pytest.approx(np.log10(1 / (L * T)))
    assert d["u.trend.amp_slope"] == 0.0
    assert 10 ** d["u.kinks.log10zeros"] == pytest.approx(6 / L)       # 3 wavelengths, 6 zeros
    assert d["u.kinks.lifetime"] == 1.0


def test_flat_field_floors():
    d = describe(np.ones((NX, NT)), Field1D("u"))
    assert d["u.amp.log10var"] == AMP_FLOOR
    assert d["u.skw.static"] == 1.0 and d["u.skw.travelling"] == 0.0
    assert d["u.mom.std"] == 0.0 and d["u.topo.above+0"] == 0.0


def test_decorrelation_time():
    tau0 = 2.0
    rho = np.exp(-DT / tau0)
    rng = np.random.default_rng(4)
    a = np.zeros((NX, NT * 8))                       # a long AR(1) process in time at every point
    a[:, 0] = rng.standard_normal(NX)
    for j in range(1, a.shape[1]):
        a[:, j] = rho * a[:, j - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(NX)
    d = field_descriptors(a, Field1D("u"), DX, DT, 100 + DT * np.arange(a.shape[1]), True)
    assert 10 ** d["u.corr.log10tau"] == pytest.approx(tau0, rel=0.2)


def test_coarsening_trend():
    a = np.cos(2 * np.pi * 4 * X / L) / np.sqrt(TT)      # variance falls like 1 / t
    assert describe(a, Field1D("u"))["u.trend.amp_slope"] == pytest.approx(-1.0, abs=1e-6)


def test_two_zeros_annihilate():
    half = (NT // 2) * DT
    width = np.clip(4.0 * (1 - (TT - TIMES[0]) / half), 0, None)    # shrinks to 0 at mid-window
    s = np.abs(X - L / 2) - width + 0.1                              # one negative domain, then none
    d = describe(s, Field1D("v", kind="velocity"))
    assert 10 ** d["v.kinks.log10deaths"] == pytest.approx(2 / (L * T))
    assert 10 ** d["v.kinks.log10births"] == pytest.approx(1 / (L * T))     # floor: no birth


def test_polarity_zeros_by_sign():
    p = np.sin(2 * np.pi * 3 * X / L) + 0 * TT
    d = describe(p, Field1D("p", kind="polarity"))
    assert d["p.defects.plus"] == pytest.approx(3 / L) and d["p.defects.minus"] == pytest.approx(3 / L)


def test_complex_plane_wave_and_a_slip():
    f = FIELDS[-1]
    q, w = 2 * np.pi * 5 / L, 0.7
    z = np.exp(1j * (q * X - w * TT))
    d = describe(np.stack([z.real, z.imag], axis=-1), f)
    assert d["A.defects.slips"] == 0.0
    assert d["A.defects.twist"] == pytest.approx(q, rel=1e-6)
    assert d["A.mom.std"] == 0.0 and d["A.order.global"] == pytest.approx(0.0, abs=1e-9)
    # A = (x - x0) + i (t - t0): the phase winds once around (x0, t0); a non-periodic window
    z = (X - L / 2 - DX / 2) + 1j * (TT - TIMES[NT // 2] - DT / 2)
    d = describe(np.stack([z.real, z.imag], axis=-1), f, periodic=False)
    assert d["A.defects.slips"] == pytest.approx(1 / ((NX - 1) * DX * (NT - 1) * DT))


def test_describe_window_prefixes_every_field():
    row = describe_window({"u": smooth_random(5)[..., 0], "v": smooth_random(6)[..., 0]},
                          [Field1D("u"), Field1D("v", kind="velocity")], DX, DT, TIMES)
    assert all(k.startswith(("u.", "v.")) for k in row)
    assert "v.flow.log10div2" in row and "u.flow.log10div2" not in row
