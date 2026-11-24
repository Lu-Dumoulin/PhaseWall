"""1D predictive learned description (JEPA): masks, EMA, symmetry of h, a short training."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phasewall.nn.jepa import MapEncoder, coarse_shape, ema_rate, ema_update, input_mask, random_masks
from phasewall.oned.jepa1d import embed_jepa, settings_1d, train_jepa
from phasewall.oned.kymo import Field1D
from phasewall.oned.nn1d import WindowSet, channels_of, mirror

NX, NT, DX = 64, 48, 0.5
X = np.arange(NX)[:, None] * DX


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


def test_coarse_shape_matches_the_encoder():
    enc = MapEncoder(2, (8, 16, 32))
    for nx, nt in ((256, 50), (64, 48), (63, 17)):
        assert enc(torch.zeros(1, 2, nx, nt)).shape[2:] == coarse_shape(nx, nt, enc.stride)


def test_masks_hide_rectangles_and_reach_the_input():
    hide = random_masks(5, 8, 4, 1, 3, 2, torch.Generator().manual_seed(0))
    assert (hide.sum(dim=(1, 2)) == 6).all()                       # one 3 x 2 rectangle each
    vis = input_mask(hide, 64, 30, 8)
    assert vis.shape == (5, 1, 64, 30)
    i, j, k = map(int, torch.nonzero(hide)[0])
    assert vis[i, 0, j * 8:(j + 1) * 8, k * 8:(k + 1) * 8].max() == 0   # a hidden cell hides its 8 x 8 points


def test_ema_moves_the_target_slowly():
    a, b = MapEncoder(1, (4,)), MapEncoder(1, (4,))
    wa = [p.clone() for p in a.parameters()]
    ema_update(a, b, 0.9)
    for p0, pa, pb in zip(wa, a.parameters(), b.parameters()):
        assert torch.allclose(pa, 0.9 * p0 + 0.1 * pb)
    assert ema_rate(0, 100, 0.996) == pytest.approx(0.996) and ema_rate(100, 100, 0.996) == pytest.approx(1.0)


def test_h_is_unchanged_by_the_mirror():
    frames = _frames(4, 2)
    enc = MapEncoder(2, (8, 16))
    flipped = WindowSet(mirror(torch.from_numpy(frames.x), torch.tensor([False])).numpy(), frames.run, frames.time,
                        frames.channels)
    assert np.allclose(embed_jepa(enc, frames), embed_jepa(enc, flipped), atol=1e-5)


def test_short_training_lowers_the_loss_without_collapse():
    frames = _frames()
    _, log = train_jepa(frames, settings_1d(epochs=8, batch=32, channels=(16, 32), mask_share=(0.125, 0.17)), dev="cpu",
                        verbose=False)
    assert log.loss.iloc[-1] < log.loss.iloc[0]
    assert log.h_std.iloc[-1] > 1e-3
