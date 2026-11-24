"""1D learned description: channels, views, symmetry of h, and a short training (extra [nn])."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phasewall.oned.kymo import Field1D
from phasewall.oned.nn1d import (
    Settings1D,
    WindowSet,
    channels_of,
    check_network,
    embed1d,
    input_channels,
    mirror,
    new_encoder1d,
    normalise,
    train1d,
)

NX, NT, DX = 64, 48, 0.5
X = np.arange(NX)[:, None] * DX


def test_channels_of_each_kind():
    fields = [Field1D("u"), Field1D("v", kind="velocity"), Field1D("p", kind="polarity", components=("px", "py")),
              Field1D("A", kind="complex", components=("re", "im"))]
    ch = channels_of(fields)
    assert [c.name for c in ch] == ["u", "v.x", "p.x", "p.y", "|A|", "J(A)"]
    assert [c.odd for c in ch] == [False, True, True, False, False, True]
    assert [c.norm for c in ch] == ["centre", "scale", "scale", "scale", "centre", "centre"]


def test_complex_channels_ignore_the_global_phase():
    rng = np.random.default_rng(0)
    z = np.exp(1j * (2 * np.pi * 3 * X / (NX * DX) + 0.3 * rng.standard_normal((NX, NT)))) * (1 + 0.1 * X / X.max())
    f = Field1D("A", kind="complex", components=("re", "im"))
    a = input_channels({"A": np.stack([z.real, z.imag], -1)}, [f], DX, True)
    zr = z * np.exp(1.1j)
    b = input_channels({"A": np.stack([zr.real, zr.imag], -1)}, [f], DX, True)
    assert np.allclose(a, b, atol=1e-5)
    # a plane wave of wavenumber q: J = |A|^2 q
    q = 2 * np.pi * 3 / (NX * DX)
    w = np.exp(1j * q * X) * np.ones((1, NT))
    c = input_channels({"A": np.stack([w.real, w.imag], -1)}, [f], DX, True)
    assert np.allclose(c[1], q, atol=1e-5) and np.allclose(c[0], 1.0, atol=1e-6)


def test_views_are_normalised():
    rng = np.random.default_rng(1)
    ch = channels_of([Field1D("u"), Field1D("v", kind="velocity")])
    x = torch.from_numpy(rng.standard_normal((5, 2, NX, NT)).astype(np.float32) * 3 + 2)
    v = normalise(x, ch)
    assert torch.allclose(v[:, 0].mean(dim=(1, 2)), torch.zeros(5), atol=1e-5)
    assert torch.allclose(v[:, 0].std(dim=(1, 2), unbiased=False), torch.ones(5), atol=1e-4)
    assert torch.allclose(torch.sqrt((v[:, 1] ** 2).mean(dim=(1, 2))), torch.ones(5), atol=1e-4)
    assert (v[:, 1].mean(dim=(1, 2)) > 0.3).all()                  # the net flow is kept


def _frames(n_runs=24, windows=4):
    """Travelling waves (odd runs) and standing waves (even runs), random wavenumber and phase."""
    rng = np.random.default_rng(2)
    t = np.arange(NT * windows)[None, :] * 0.25
    xs, runs, times = [], [], []
    for r in range(n_runs):
        k = 2 * np.pi * rng.integers(3, 6) / (NX * DX)
        w, ph = rng.uniform(0.8, 1.2), rng.uniform(0, 2 * np.pi)
        u = np.cos(k * X - w * t + ph) if r % 2 else np.cos(k * X + ph) * np.cos(w * t)
        for j in range(windows):
            xs.append(u[None, :, j * NT:(j + 1) * NT].astype(np.float32))
            runs.append(r)
            times.append(j)
    return WindowSet(np.stack(xs), np.array(runs), np.array(times, dtype=float), channels_of([Field1D("u")]))


def test_h_is_unchanged_by_the_mirror():
    frames = _frames(4, 2)
    enc = new_encoder1d(frames, Settings1D(channels=(8, 16)))
    flipped = WindowSet(mirror(torch.from_numpy(frames.x), torch.tensor([False])).numpy(), frames.run, frames.time,
                        frames.channels)
    assert np.allclose(embed1d(enc, frames), embed1d(enc, flipped), atol=1e-5)


def test_network_needs_long_enough_windows():
    with pytest.raises(ValueError, match="blocks need"):
        check_network(256, 20)                     # crop of 15 rows, 4 blocks need 32
    check_network(256, 48)


def test_short_training_separates_travelling_from_standing():
    frames = _frames()
    enc, log = train1d(frames, Settings1D(epochs=15, batch=24, channels=(16, 32, 64), seed=0), dev="cpu",
                       verbose=False)
    assert log.loss.iloc[-1] < log.loss.iloc[0]
    h = embed1d(enc, frames)
    h = (h - h.mean(0)) / (h.std(0) + 1e-9)
    kind = frames.run % 2
    d = np.linalg.norm(h[:, None] - h[None], axis=2)
    same, other = d[kind[:, None] == kind[None]].mean(), d[kind[:, None] != kind[None]].mean()
    assert other > 1.2 * same
