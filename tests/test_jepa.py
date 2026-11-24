"""2D predictive learned description (JEPA): symmetries of the square, invariance of h, a training."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phasewall.fields import Field, mirror_x, rotate90
from phasewall.nn.data import FrameSet
from phasewall.nn.jepa import (
    MapEncoder,
    SettingsJEPA,
    d4_turns,
    embed_jepa,
    train_jepa,
)

FIELDS = [Field("u", ("u",)), Field("v", (), kind="velocity", components=("vx", "vy")),
          Field("Q", (), kind="nematic", components=("qxx", "qxy"))]


def _pair(rng, n=16):
    """One pair (2 frames x (1 + 2 + 2) channels) of random fields."""
    return rng.standard_normal((10, n, n)).astype(np.float32)


def _numpy_turn(x, turns, mirror):
    """The rules of fields.py applied channel by channel to a pair (2C, nx, ny)."""
    out = []
    for start in (0, 5):
        u = x[start]
        v = np.moveaxis(x[start + 1:start + 3], 0, -1)
        q = np.moveaxis(x[start + 3:start + 5], 0, -1)
        for _ in range(turns):
            u, v, q = rotate90(u, "scalar"), rotate90(v, "velocity"), rotate90(q, "nematic")
        if mirror:
            u, v, q = mirror_x(u, "scalar"), mirror_x(v, "velocity"), mirror_x(q, "nematic")
        out += [u[None], np.moveaxis(v, -1, 0), np.moveaxis(q, -1, 0)]
    return np.concatenate(out)


def test_d4_turns_follow_the_field_rules():
    x = _pair(np.random.default_rng(0))
    images = d4_turns(torch.from_numpy(x[None]), FIELDS)
    assert len(images) == 8
    for k in range(4):
        for m in (False, True):
            assert np.allclose(images[2 * k + m][0].numpy(), _numpy_turn(x, k, m), atol=1e-6)


def _frames(xs, periodic=True):
    xs = np.stack(xs)
    return FrameSet(xs, np.arange(len(xs)), np.zeros(len(xs)), FIELDS, periodic, np.ones(3))


def test_h_is_unchanged_by_the_symmetries_of_the_square():
    rng = np.random.default_rng(1)
    x = [_pair(rng) for _ in range(3)]
    enc = MapEncoder(11, (8, 16), periodic=True)
    h = embed_jepa(enc, _frames(x))
    for turns, mirror in ((1, False), (2, True), (3, False)):
        h2 = embed_jepa(enc, _frames([_numpy_turn(a, turns, mirror) for a in x]))
        assert np.allclose(h, h2, atol=1e-5)


def test_short_training_lowers_the_loss_without_collapse():
    rng = np.random.default_rng(2)
    n = 32
    xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xs = []
    for r in range(32):
        k = 2 * np.pi * rng.integers(3, 5) / n
        u = np.cos(k * xx + rng.uniform(0, 6)) if r % 2 else np.cos(k * xx) + np.cos(k * yy)
        pair = np.zeros((2, n, n), dtype=np.float32)
        pair[0] = pair[1] = u
        xs.append(pair)
    frames = FrameSet(np.stack(xs), np.arange(32), np.zeros(32), [Field("u", ("u",))], True, np.ones(1))
    _, log = train_jepa(frames, SettingsJEPA(epochs=8, batch=16, channels=(16, 32)), dev="cpu", verbose=False)
    assert log.loss.iloc[-1] < log.loss.iloc[0]
    assert log.h_std.iloc[-1] > 1e-3
