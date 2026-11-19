"""Tests of the learned description (phasewall.nn), skipped without PyTorch: channel layout
and scaling, augmentations against the rules of fields.py, the shift invariance of a periodic
encoder, the loss at known points, a short training that separates two kinds of patterns, and
saving and loading."""

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from phasewall.fields import Field, transform
from phasewall.io import Ensemble, write_snapshot
from phasewall.nn.data import FrameSet, PairViews, augment, load_frames, to_channels
from phasewall.nn.loss import effective_rank, info_nce
from phasewall.nn.model import Encoder
from phasewall.nn.train import Settings, embed, h_table, load, save, train

U = Field("u", ("u",), relative=True)
P = Field("p", ("p",), "vector")
SMALL = (8, 16, 16, 32)      # small channels for fast tests


def field(n, seed, comps=0):
    rng = np.random.default_rng(seed)
    shape = (n, n, comps) if comps else (n, n)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n, 1 / n), np.fft.fftfreq(n, 1 / n), indexing="ij"))
    if comps:
        k = k[..., None]
    return np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape), axes=(0, 1)) * (k <= 5), axes=(0, 1)).real


def test_channels_and_augmentation_follow_the_field_rules():
    """A pair turned by augment() equals the snapshots turned by fields.transform()."""
    fields = [U, P]
    old = {"u": 2 + field(16, 0), "p": field(16, 1, 2)}
    now = {"u": 2 + field(16, 2), "p": field(16, 3, 2)}
    x = np.concatenate([to_channels(old, fields), to_channels(now, fields)])
    assert x.shape == (6, 16, 16)
    assert np.allclose(x[0], (old["u"] - old["u"].mean()) / old["u"].mean())      # relative scalar
    rng = np.random.default_rng(5)
    for _ in range(8):
        state = rng.bit_generator.state
        y = augment(x, fields, rng, periodic=True)
        r = np.random.default_rng()
        r.bit_generator.state = state
        turns, mirror = int(r.integers(4)), bool(r.integers(2))
        shift = (int(r.integers(16)), int(r.integers(16)))
        expect = np.concatenate([to_channels(transform(s, fields, turns, mirror, shift), fields) for s in (old, now)])
        assert np.allclose(y, expect)


def test_periodic_encoder_ignores_shifts_by_its_stride():
    """Four stride-2 blocks: a circular shift by 16 points moves the feature maps by exactly
    one cell, so the average over the frame does not change."""
    enc = Encoder(2, SMALL, periodic=True).eval()
    x = torch.from_numpy(np.stack([field(64, 4), field(64, 5)]).astype(np.float32))[None]
    h = enc(x)
    assert torch.allclose(enc(torch.roll(x, (16, 32), dims=(2, 3))), h, atol=1e-5)
    assert not torch.allclose(enc(torch.roll(x, (5, 0), dims=(2, 3))), h, atol=1e-5)


def test_info_nce_at_known_points():
    n = 8
    z = torch.nn.functional.normalize(torch.ones(2 * n, 4), dim=1)          # collapse
    assert float(info_nce(z)) == pytest.approx(math.log(2 * n - 1), rel=1e-5)
    e = torch.eye(2 * n)[:n]                                                  # partners equal, others orthogonal
    good = torch.cat([e, e])
    assert float(info_nce(good, T=0.05)) < 1e-6
    assert effective_rank(torch.randn(500, 3) * torch.tensor([1.0, 1.0, 1.0])) == pytest.approx(3, rel=0.1)


def two_kinds(tmp_path, n_runs=24, n=32):
    """Runs of stripes (a = 0) or spots (a = 1), random orientation and phase, 4 snapshots."""
    rng = np.random.default_rng(0)
    x = 2 * np.pi * np.arange(n) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rows = []
    for r in range(n_runs):
        a = r % 2
        th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi, 2)
        for t in range(4):
            if a == 0:
                u = np.cos(4 * (np.cos(th) * X + np.sin(th) * Y) + ph[0] + 0.1 * t)
            else:
                u = np.cos(4 * X + ph[0] + 0.1 * t) + np.cos(4 * Y + ph[1])
            write_snapshot(tmp_path / f"r{r}" / "Data" / f"{t:04d}.h5", {"u": u + 0.05 * rng.standard_normal((n, n))})
        rows.append({"run": r, "path": str(tmp_path / f"r{r}"), "a": float(a)})
    return Ensemble(runs=pd.DataFrame(rows), parameters=["a"], fields=[Field("u", ("u",))], dx=0.2,
                    snapshots="Data/*.h5", time_regex=r"(\d+)\.h5$", last=4, pair_dt=1.0)


def test_load_frames(tmp_path):
    ens = two_kinds(tmp_path, n_runs=4)
    fr = load_frames(ens)
    assert isinstance(fr, FrameSet) and fr.x.shape == (4 * 3, 2, 32, 32) and fr.periodic
    assert np.sqrt(np.mean(fr.x**2)) == pytest.approx(1, rel=1e-5)          # scaled to order 1
    assert sorted(set(fr.time)) == [1.0, 2.0, 3.0]                          # snapshots with a partner
    items = PairViews(fr, draws=2)
    assert len(items) == 8 and items[0][0].shape == (2, 32, 32)


def separation(ens, fr, h):
    kind = ens.runs.set_index("run").a.loc[fr.run].to_numpy()
    hs = (h - h.mean(0)) / (h.std(0) + 1e-9)
    between = np.linalg.norm(hs[kind == 0].mean(0) - hs[kind == 1].mean(0))
    within = np.mean([np.linalg.norm(hs[kind == k] - hs[kind == k].mean(0), axis=1).mean() for k in (0, 1)])
    return between, within


def test_training_with_the_old_views_separates_two_kinds(tmp_path):
    """The views used before 2026-11-19 (D4, not normalised): after 6 epochs stripes and spots are
    apart; there the two kinds also differ in amplitude (1 against sqrt 2), an easy cue."""
    ens = two_kinds(tmp_path)
    fr = load_frames(ens)
    enc, log = train(fr, Settings(epochs=6, batch=12, channels=SMALL, warmup=1, views="d4", normalise=False),
                     dev=torch.device("cpu"), verbose=False)
    assert log.loss.iloc[-1] < log.collapse_loss.iloc[-1]
    between, within = separation(ens, fr, embed(enc, fr))
    assert between > within


def test_training_separates_two_kinds_of_pattern(tmp_path):
    """Default views (any angle, zoom, crop of 24 of 32 points, normalised: the amplitude is gone,
    only the shape is left, which needs more steps on 24 runs: 30 epochs, 3 blocks for the crop):
    h of stripes and of spots are farther apart than within each kind, and saving then loading
    gives the same h."""
    ens = two_kinds(tmp_path)
    fr = load_frames(ens)
    enc, log = train(fr, Settings(epochs=30, batch=12, channels=(8, 16, 32), warmup=1), dev=torch.device("cpu"),
                     verbose=False)
    assert log.loss.iloc[-1] < log.collapse_loss.iloc[-1]
    h = embed(enc, fr)
    between, within = separation(ens, fr, h)
    assert between > within
    table = h_table(fr, h, ens)
    assert list(table.columns[:3]) == ["run", "time", "a"] and table.columns[3] == "cnn.h.d000"
    save(tmp_path / "enc", enc, fr, Settings(channels=(8, 16, 32)), log)
    enc2, meta = load(tmp_path / "enc")
    assert np.allclose(embed(enc2, fr), h, atol=1e-5)
    assert meta["periodic"] and meta["fields"][0].name == "u"


def test_learn_and_describe_commands(tmp_path):
    from phasewall import cli, toy
    toy.make_ensemble(tmp_path, eps=[-0.1, 0.3], g=[0.0, 1.0], seeds=(0, 1), n=32, waves=4)
    out = cli.run(["learn", str(tmp_path / "phasewall.toml"), "--epochs", "1", "--workers", "1", "--blocks", "3"])
    t1 = pd.read_csv(out / "descriptors_cnn.csv")
    assert len(t1) == 8 * (toy.N_SAVE - 1) and "cnn.h.d255" in t1
    csv = cli.run(["describe", str(tmp_path / "phasewall.toml"), "--encoder", str(out), "--workers", "1"])
    t2 = pd.read_csv(csv)
    assert csv.name == "descriptors_cnn.csv"
    assert np.allclose(t1.filter(like="cnn.").to_numpy(), t2.filter(like="cnn.").to_numpy(), atol=1e-4)
