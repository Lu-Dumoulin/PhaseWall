"""Tests of nn/views.py, the views of the learned encoder: what the network sees (input side), the
normalisation of every field of every view (vectors and nematics keep their mean), the resampling (a turn by 90 degrees is exact, the
wavelength is kept at zoom 1), components that turn with the grid (a vector field that is the
gradient of a scalar stays the gradient of the turned scalar; a nematic built from that gradient
stays the nematic of the turned gradient), the input of the description after training, and the
views kept on saving and loading an encoder (old encoders read with the old views)."""

import json
import math

import numpy as np
import pytest

from phasewall.fields import Field
from phasewall.nn.architecture import Network, input_side

torch = pytest.importorskip("torch")
from phasewall.nn import views as v

U = Field("u", ("u",))
P = Field("p", ("p",), "vector")
Q = Field("Q", ("Q",), "nematic")
N = 64


def test_input_side():
    """d4: the frame; rotzoom: the crop, at least what the blocks need in a periodic frame, and
    smaller in a non-periodic frame so that a turned and zoomed crop stays inside."""
    assert input_side(Network(views="d4"), N) == N
    assert input_side(Network(), N, periodic=True) == 48
    assert input_side(Network(), N, periodic=False) == math.floor(N / (1.25 * math.sqrt(2))) == 36
    assert input_side(Network(zoom=1.0, crop=0.5), N, periodic=False) == 32
    # periodic: at least what the blocks need (2 x 2^blocks), at most the frame
    assert input_side(Network(blocks=5), N) == 64 and input_side(Network(blocks=6), N) == 64
    assert input_side(Network(blocks=4), 32) == 32 and input_side(Network(blocks=2), 32) == 24


def test_normalise_every_field_of_every_view():
    """Old normalisation (keep_vector_mean False): every field centred per component and divided
    by its rms; a flat view is only centred."""
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(3, 6, 8, 8)) * [[[[1.0]], [[5.0]], [[5.0]], [[2.0]], [[7.0]], [[7.0]]]] + 3,
                     dtype=torch.float32)            # a pair of (u, p_x, p_y)
    x[2] = 4.0                                       # a flat view
    y = v.normalise(x, [U, P], keep_vector_mean=False)
    for b in (0, 1):
        u, p = y[b, [0, 3]], y[b, [1, 2, 4, 5]]
        assert abs(float(u.mean())) < 1e-5 and float(u.pow(2).mean().sqrt()) == pytest.approx(1, rel=1e-5)
        assert float(p.pow(2).mean().sqrt()) == pytest.approx(1, rel=1e-5)
        for c in (1, 2, 4, 5):
            assert abs(float(y[b, c].mean())) < 1e-5
    assert torch.all(y[2] == 0)                      # only centred


def test_vectors_keep_their_mean():
    """Default (keep_vector_mean): scalars as before; a vector field is only divided by its rms, so
    its mean direction and the share |<p>| / rms (polarisation) are kept; a uniform vector field
    keeps its direction whatever its amplitude; a nematic keeps its mean too; a rotation of the
    components before or after the normalisation gives the same view."""
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(size=(2, 6, 8, 8)), dtype=torch.float32)
    x[:, [1, 4]] += 3.0                              # p_x has a mean: polarised along x
    x[:, [2, 5]] *= 0.5
    y = v.normalise(x, [U, P])
    assert torch.allclose(y[:, [0, 3]], v.normalise(x, [U, P], keep_vector_mean=False)[:, [0, 3]])
    for b in (0, 1):
        p, q = x[b, [1, 2, 4, 5]], y[b, [1, 2, 4, 5]]
        rms = float(p.pow(2).mean().sqrt())
        assert torch.allclose(q, p / rms, atol=1e-6)
        assert float(q.pow(2).mean().sqrt()) == pytest.approx(1, rel=1e-5)
        assert float(q[[0, 2]].mean()) == pytest.approx(float(p[[0, 2]].mean()) / rms, rel=1e-5)
    uniform = torch.zeros(1, 6, 4, 4)
    uniform[:, [1, 4]], uniform[:, [2, 5]] = 0.6, 0.8   # |p| = 1 along (0.6, 0.8)
    yu = v.normalise(2 * uniform, [U, P])
    assert torch.allclose(yu, v.normalise(uniform, [U, P]), atol=1e-6)        # amplitude gone
    assert torch.allclose(yu, uniform * math.sqrt(2), atol=1e-6)   # direction kept; rms per component 1
    xq = x.clone()
    assert abs(float(v.normalise(xq, [U, Q])[:, [1, 4]].mean())) > 0.5   # nematic keeps <Q_xx>
    theta, mirror = torch.tensor([0.7, 2.1]), torch.tensor([False, True])
    a = v.normalise(v.turn_components(x, [U, P], theta, mirror), [U, P])
    b = v.turn_components(v.normalise(x, [U, P]), [U, P], theta, mirror)
    assert torch.allclose(a, b, atol=1e-5)


def stripes(n=N, k=8, angle=0.0):
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.cos(2 * np.pi * k / n * (np.cos(angle) * i + np.sin(angle) * j))


def peak_wavenumber(a):
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    P_ = np.abs(np.fft.fft2((a - a.mean()) * w)) ** 2
    P_[0, 0] = 0
    k = np.fft.fftfreq(a.shape[0]) * a.shape[0]
    i, j = np.unravel_index(P_.argmax(), P_.shape)
    return float(np.hypot(k[i], k[j]))


def one(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)[None, None]


def test_a_turn_by_90_degrees_is_exact_and_the_wavelength_is_kept():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(N, N))
    side = 48
    centre = torch.tensor([[31.5, 31.5]])
    out = v.resample(one(a), [U], torch.tensor([math.pi / 2]), torch.tensor([1.0]), torch.tensor([False]),
                     centre, side, periodic=True)[0, 0].numpy()
    crops = [np.rot90(a, k)[8:56, 8:56] for k in range(4)]
    assert min(np.abs(out - c).max() for c in crops) < 1e-5
    s = stripes()
    for theta in (0.3, 1.1, 2.6):
        t = v.resample(one(s), [U], torch.tensor([theta]), torch.tensor([1.0]), torch.tensor([False]),
                       torch.tensor([[10.0, 20.0]]), side, periodic=True)[0, 0].numpy()
        assert peak_wavenumber(t) == pytest.approx(8 * side / N, abs=0.6)
    z = v.resample(one(s), [U], torch.tensor([0.0]), torch.tensor([1.25]), torch.tensor([False]),
                   torch.tensor([[10.0, 20.0]]), side, periodic=True)[0, 0].numpy()
    assert peak_wavenumber(z) == pytest.approx(8 * side / N / 1.25, abs=0.6)   # zoom 1.25 magnifies: longer waves


def smooth_scalar(seed, n=N, kmax=3):
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.standard_normal((n, n)))
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    f[np.hypot(kx, ky) > kmax] = 0
    phi = np.fft.ifft2(f).real
    gx = np.fft.ifft2(2j * np.pi * kx / n * f).real
    gy = np.fft.ifft2(2j * np.pi * ky / n * f).real
    return phi / phi.std(), gx / phi.std(), gy / phi.std()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_components_turn_with_the_grid(seed):
    """Turned vector = gradient of the turned scalar; turned nematic = nematic of that gradient
    (central differences inside the view; the views are not zoomed, so lengths are kept)."""
    phi, gx, gy = smooth_scalar(seed)
    rng = np.random.default_rng(seed + 10)
    theta = torch.tensor([rng.uniform(0, 2 * math.pi)])
    mirror = torch.tensor([bool(seed % 2)])
    centre = torch.tensor(rng.uniform(0, N, (1, 2)), dtype=torch.float32)
    side = 40
    x = torch.tensor(np.stack([phi, gx, gy, (gx**2 - gy**2) / 2, gx * gy] * 2), dtype=torch.float32)[None]
    out = v.resample(x, [U, P, Q], theta, torch.tensor([1.0]), mirror, centre, side, periodic=True)[0].numpy()
    p_ = out[0]
    dx = (p_[2:, 1:-1] - p_[:-2, 1:-1]) / 2
    dy = (p_[1:-1, 2:] - p_[1:-1, :-2]) / 2
    ax, ay = out[1][1:-1, 1:-1], out[2][1:-1, 1:-1]
    scale = np.sqrt(np.mean(ax**2 + ay**2))
    assert np.sqrt(np.mean((ax - dx) ** 2 + (ay - dy) ** 2)) < 0.05 * scale
    qxx, qxy = out[3][1:-1, 1:-1], out[4][1:-1, 1:-1]
    qscale = np.sqrt(np.mean(qxx**2 + qxy**2))
    assert np.sqrt(np.mean((qxx - (dx**2 - dy**2) / 2) ** 2 + (qxy - dx * dy) ** 2)) < 0.1 * qscale


def test_training_views_and_embedding_input():
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(size=(5, 2, N, N)) * 4 + 1, dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    t = v.training_views(x, [U], v.ViewSettings(), gen, periodic=True)
    assert t.shape == (5, 2, 48, 48)
    assert torch.allclose(t.pow(2).mean(dim=(1, 2, 3)), torch.ones(5), atol=1e-4)
    t2 = v.training_views(x, [U], v.ViewSettings(), gen, periodic=False)
    assert t2.shape == (5, 2, 36, 36)
    e = v.embedding_input(x, [U], v.ViewSettings(), periodic=True)
    assert e.shape == (5, 2, 48, 48)
    centred = x[:, :, 8:56, 8:56] - x[:, :, 8:56, 8:56].mean(dim=(2, 3), keepdim=True)
    assert torch.allclose(e * centred.pow(2).mean(dim=(1, 2, 3)).sqrt().view(-1, 1, 1, 1), centred, atol=1e-4)
    assert torch.equal(v.embedding_input(x, [U], v.OLD, periodic=True), x)       # old encoders: as they were
    assert torch.equal(v.training_views(x, [U], v.OLD, gen, periodic=True), x)   # d4: turned on the CPU before


def test_views_are_kept_with_the_encoder(tmp_path):
    from phasewall.nn.data import FrameSet
    from phasewall.nn.train import Settings, embed, load, new_encoder, save
    x = np.random.default_rng(4).normal(size=(6, 2, 32, 32)).astype(np.float32)
    fr = FrameSet(x, np.arange(6), np.ones(6), [U], True, np.ones(1))
    s = Settings(channels=(8, 16), zoom=1.5, crop=0.5)
    enc = new_encoder(fr, s)
    assert enc.views == v.ViewSettings("rotzoom", 1.5, 0.5, True, 2, True) and not enc.circular
    import pandas as pd
    save(tmp_path / "e", enc, fr, s, pd.DataFrame())
    back, meta = load(tmp_path / "e")
    assert back.views == enc.views and meta["circular"] is False
    # saved before keep_vector_mean existed (2026-11-20): vectors and nematics centred as then
    before = json.loads((tmp_path / "e" / "settings.json").read_text())
    del before["settings"]["keep_vector_mean"]
    (tmp_path / "e" / "settings.json").write_text(json.dumps(before))
    assert load(tmp_path / "e")[0].views.keep_vector_mean is False
    assert np.allclose(embed(back, fr), embed(enc, fr), atol=1e-6)
    # an encoder saved before the views existed: no views in its settings, padding as the frame
    old = json.loads((tmp_path / "e" / "settings.json").read_text())
    for k in ("views", "zoom", "crop", "normalise"):
        del old["settings"][k]
    del old["circular"]
    d4 = new_encoder(fr, Settings(channels=(8, 16), views="d4", normalise=False))
    save(tmp_path / "old", d4, fr, Settings(channels=(8, 16), views="d4", normalise=False), pd.DataFrame())
    (tmp_path / "old" / "settings.json").write_text(json.dumps(old))
    back_old, _ = load(tmp_path / "old")
    assert back_old.views == v.OLD and back_old.circular
    assert np.allclose(embed(back_old, fr), embed(d4, fr), atol=1e-6)
