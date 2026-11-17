"""Tests of describe.py on small synthetic fields whose descriptors are known: a single
wave, spots against stripes, gradient and rotational flows, frozen, travelling and changing
patterns, and the invariance of everything under the symmetries of the box."""

import numpy as np
import pandas as pd
import pytest

from phasewall import describe as d
from phasewall.fields import Field, transform
from phasewall.io import Ensemble, write_snapshot

N = 32
DX = 0.5
L = N * DX
x = DX * np.arange(N)
X, Y = np.meshgrid(x, x, indexing="ij")
U = Field("u", ("u",))
P = Field("p", ("p",), "vector")
Q = Field("Q", ("Q",), "nematic")


def smooth_noise(seed: int, shape=(N, N), kmax: int = 4) -> np.ndarray:
    """Random periodic field made of the Fourier modes with |k| <= kmax (in units 2 pi / L)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal(shape), axes=(0, 1))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(N, 1 / N), np.fft.fftfreq(N, 1 / N), indexing="ij"))
    if len(shape) == 3:
        k = k[..., None]
    return np.fft.ifft2(f * (k <= kmax), axes=(0, 1)).real


def test_k_shells_cover_the_grid_without_empty_shells():
    edges = d.k_shells(N, DX)
    dk = 2 * np.pi / L
    assert edges[0] == pytest.approx(dk / 2)
    assert edges[-1] == pytest.approx(np.pi / DX)
    assert np.all(np.diff(edges) >= dk - 1e-12)
    k = np.hypot(*np.meshgrid(2 * np.pi * np.fft.fftfreq(N, DX), 2 * np.pi * np.fft.fftfreq(N, DX)))
    counts = np.histogram(k.ravel(), edges)[0]
    assert np.all(counts > 0)


@pytest.mark.parametrize("m", [2, 5, 9])
def test_structure_factor_peaks_at_the_wavenumber_of_a_wave(m):
    k0 = 2 * np.pi * m / L
    edges = d.k_shells(N, DX)
    S = d.structure_factor(np.cos(k0 * X), DX, edges)
    peak = int(np.argmax(S))
    assert edges[peak] <= k0 < edges[peak + 1]
    # all the power is in that shell (a single Fourier mode and its mirror image)
    assert S.sum() == pytest.approx(S[peak])


def test_structure_factor_sums_components_and_ignores_a_global_rotation():
    edges = d.k_shells(N, DX)
    p = smooth_noise(0, (N, N, 2))
    c, s = np.cos(0.7), np.sin(0.7)
    turned = np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], axis=-1)
    assert np.allclose(d.structure_factor(turned, DX, edges), d.structure_factor(p, DX, edges))


def test_moments_of_a_two_valued_field():
    """Values 1 with probability 1/4, else 0: skewness 2/sqrt(3), excess kurtosis -2/3."""
    mean, std, skew, kurt = d.moments(np.array([0, 0, 0, 1] * 8, dtype=float))
    assert (mean, std) == pytest.approx((0.25, np.sqrt(3) / 4))
    assert skew == pytest.approx(2 / np.sqrt(3))
    assert kurt == pytest.approx(-2 / 3)
    assert d.moments(np.full(10, 3.0)) == (3.0, 0.0, 0.0, 0.0)


def test_count_domains_joins_across_the_periodic_edges():
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, :] = mask[-1, :] = True       # one stripe cut by the edge of the box
    assert d.count_domains(mask, periodic=False) == 2
    assert d.count_domains(mask) == 1
    corners = np.zeros((8, 8), dtype=bool)
    corners[[0, 0, -1, -1], [0, -1, 0, -1]] = True     # one spot cut in four by the corners
    assert d.count_domains(corners, periodic=False) == 4
    assert d.count_domains(corners) == 1
    assert d.count_domains(np.zeros((8, 8), dtype=bool)) == 0
    assert d.count_domains(np.ones((8, 8), dtype=bool)) == 1


def test_domains_of_spots_and_stripes():
    """m x m spots (cos kx + cos ky) have m^2 peaks above one standard deviation; m stripes
    (cos kx) have m domains, each closed on itself through the periodic edge."""
    m = 4
    k = 2 * np.pi * m / L
    area = L * L
    spots = d.domain_counts(np.cos(k * X) + np.cos(k * Y), area)
    stripes = d.domain_counts(np.cos(k * X), area)
    assert spots["above+1"] * area == pytest.approx(m * m)
    assert spots["below-1"] * area == pytest.approx(m * m)
    assert stripes["above+1"] * area == pytest.approx(m)
    assert stripes["below-1"] * area == pytest.approx(m)


def test_flat_field_gives_floor_values():
    out = d.field_descriptors(np.full((N, N), 2.0), U, DX, d.k_shells(N, DX))
    assert out["u.amp.log10var"] == pytest.approx(np.log10(d.FLAT**2))
    assert all(v == 0 for c, v in out.items() if ".shape." in c or ".topo." in c)
    assert (out["u.mom.mean"], out["u.mom.std"], out["u.mom.skew"]) == (2.0, 0.0, 0.0)


def test_relative_scalar_has_no_mean_and_ignores_the_mean_value():
    f = 1 + 0.1 * smooth_noise(1)
    rel = Field("u", ("u",), relative=True)
    edges = d.k_shells(N, DX)
    a = d.field_descriptors(f, rel, DX, edges)
    b = d.field_descriptors(3 * f, rel, DX, edges)
    assert "u.mom.mean" not in a
    assert a == pytest.approx(b)


def test_shape_is_normalised():
    out = d.field_descriptors(smooth_noise(2), U, DX, d.k_shells(N, DX))
    assert sum(v for c, v in out.items() if ".shape." in c) == pytest.approx(1)


def test_flow_of_gradient_and_rotational_fields():
    """The gradient of a potential has no curl, its rotation by 90 degrees no divergence."""
    phi = smooth_noise(3)
    phik = np.fft.fft2(phi)
    kx = 2j * np.pi * np.fft.fftfreq(N, DX)[:, None]
    ky = 2j * np.pi * np.fft.fftfreq(N, DX)[None, :]
    grad = np.stack([np.fft.ifft2(kx * phik).real, np.fft.ifft2(ky * phik).real], axis=-1)
    swirl = np.stack([-grad[..., 1], grad[..., 0]], axis=-1)
    edges = d.k_shells(N, DX)
    g = d.field_descriptors(grad, P, DX, edges)
    s = d.field_descriptors(swirl, P, DX, edges)
    assert g["p.flow.curl_share"] == pytest.approx(0, abs=1e-12)
    assert s["p.flow.curl_share"] == pytest.approx(1)
    assert g["p.flow.log10div2"] == pytest.approx(s["p.flow.log10curl2"])


def test_global_order():
    edges = d.k_shells(N, DX)
    aligned = np.stack([np.ones((N, N)), np.zeros((N, N))], axis=-1)
    assert d.field_descriptors(aligned, P, DX, edges)["p.order.global"] == pytest.approx(1)
    # (cos ky, 0) points both ways equally often: no global order
    opposed = np.stack([np.cos(2 * np.pi * Y / L), np.zeros((N, N))], axis=-1)
    assert d.field_descriptors(opposed, P, DX, edges)["p.order.global"] == pytest.approx(0, abs=1e-12)


def test_dynamics_of_frozen_travelling_and_changing_patterns():
    f = smooth_noise(4)
    floor = np.log10(d.DECOR_FLOOR)
    frozen = d.pair_descriptors(f, f, U)
    assert frozen["u.dyn.still"] == pytest.approx(floor)
    assert frozen["u.dyn.moving"] == pytest.approx(floor)
    moved = d.pair_descriptors(f, np.roll(f, (5, 3), axis=(0, 1)), U)
    assert moved["u.dyn.still"] > floor + 1
    assert moved["u.dyn.moving"] == pytest.approx(floor)
    changed = d.pair_descriptors(f, smooth_noise(5), U)
    assert changed["u.dyn.still"] > floor + 1
    assert changed["u.dyn.moving"] > floor + 1
    flat = d.pair_descriptors(np.ones((N, N)), f, U)
    assert flat["u.dyn.still"] == pytest.approx(floor)


def test_every_descriptor_is_unchanged_by_the_symmetries_of_the_box():
    fields = [Field("rho", ("rho",), relative=True), P, Q]
    snap = {"rho": 1 + 0.2 * smooth_noise(6), "p": smooth_noise(7, (N, N, 2)), "Q": smooth_noise(8, (N, N, 2))}
    before = {"rho": 1 + 0.2 * smooth_noise(9), "p": smooth_noise(10, (N, N, 2)), "Q": smooth_noise(11, (N, N, 2))}
    ref = d.describe_snapshot(snap, fields, DX, before)
    assert len(ref) > 50
    for turns in range(4):
        for mirror in (False, True):
            for shift in [(0, 0), (7, 19)]:
                t = d.describe_snapshot(transform(snap, fields, turns, mirror, shift), fields, DX,
                                        transform(before, fields, turns, mirror, shift))
                assert t.keys() == ref.keys()
                for c in ref:
                    assert t[c] == pytest.approx(ref[c], rel=1e-9, abs=1e-9), (c, turns, mirror, shift)


def make_runs(folder, n_runs: int = 2, times=(0, 5, 10, 15)) -> Ensemble:
    rows = []
    for r in range(n_runs):
        for t in times:
            write_snapshot(folder / f"r{r}" / "Data" / f"{t:04d}.h5", {"u": smooth_noise(10 * r + t)})
        rows.append({"run": r, "path": str(folder / f"r{r}"), "a": 0.1 * r})
    return Ensemble(runs=pd.DataFrame(rows), parameters=["a"], fields=[U], dx=DX, snapshots="Data/*.h5",
                    time_regex=r"(\d+)\.h5$", last=3, pair_dt=5.0)


def test_describe_run_takes_the_last_snapshots_and_their_partners(tmp_path):
    ens = make_runs(tmp_path, times=(0, 5, 10, 20))
    rows = d.describe_run(ens, 0, ens.runs.path[0])
    assert [r["time"] for r in rows] == [5, 10, 20]
    assert "u.dyn.still" in rows[0] and "u.dyn.still" in rows[1]
    assert "u.dyn.still" not in rows[2]                # no snapshot at t = 15


def test_describe_ensemble_table_and_errors(tmp_path):
    ens = make_runs(tmp_path, n_runs=3)
    write_snapshot(tmp_path / "r2" / "Data" / "0015.h5", {"v": np.zeros((N, N))})   # broken run
    table, errors = d.describe_ensemble(ens)
    assert len(errors) == 1 and errors[0].startswith("run 2: KeyError")
    assert len(table) == 2 * 3
    assert list(table.columns[:3]) == ["run", "time", "a"]
    assert table.groupby("run").a.first().tolist() == [0.0, 0.1]
    assert d.descriptor_columns(table) == [c for c in table.columns if c.startswith("u.")]
    parallel, _ = d.describe_ensemble(ens, workers=2)
    pd.testing.assert_frame_equal(parallel, table)
