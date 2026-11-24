"""The learned description of 1D runs: a convolutional network on the (x, t) window (extra [nn]).

A window of a kymograph is an image with x across and t up, one channel per field component. A
3 x 3 convolution over (x, t) is a learned stencil that holds d_x, d_t, d_x^2, d_x d_t and d_t^2 at
once (an advection d_t + c d_x, a diffusion, a wave operator), so the network of the 2D package
(nn/model.py: four blocks, 32 to 256 channels, mean over the frame) is used as it is, with zero
padding on both axes (the views below are crops, which are not periodic).

Channels (input_channels): a scalar gives itself; a velocity or a one-component polarity its
value (odd under the mirror); a two-component polarity (p_x odd, p_y even); a complex amplitude A
gives two channels that do not depend on its global phase (Ludo's choice, 2026-11-23): its size
|A| and the current J = Im(A* d_x A) = |A|^2 d_x(phase), which is smooth through the zeros of A
and odd under the mirror.

Views (training): two different windows of the same run (positives), each cut to a crop of
CROP_X of the line and CROP_T of the window at a random place, mirrored (x -> -x, odd channels
flipped) with probability 1/2, then normalised channel by channel: scalars, |A| and J minus their
own mean and divided by their own standard deviation; the components of a velocity or polarity
only divided by their root mean square, so that a net flow or polarisation is kept (as in 2D).
No rotation and no zoom: they would mix x with t, and drift and wave speeds are differences of
state (Ludo, 2026-11-23).

Description (embed): the centred crop, normalised as in training, and h = (f(view) +
f(mirrored view)) / 2, so that h is exactly unchanged by the mirror. Loss and training as in 2D
(InfoNCE, AdamW, warm-up then cosine); h_table1d gives the rows run, time, the parameters and
cnn.h.d000, ... like the hand-made table.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..nn.data import PairViews
from ..nn.model import CHANNELS, Encoder
from ..nn.train import TrainingSettings, device, fit_contrastive, h_table
from ..parallel import run_jobs
from .describe1d import d_dx, prepared
from .kymo import Ensemble1D, Field1D, read_kymograph, window_slices

CROP_X = 0.75      # views: share of the line kept (a choice, as the 2D crop)
CROP_T = 0.75      # views: share of the window kept (a choice)
EPS = 1e-12


@dataclass(frozen=True)
class Channel:
    """One input channel: its field, name, normalisation ("centre": own mean and std; "scale":
    divided by the rms of its field's scaled channels together) and whether the mirror flips it."""

    field: str
    name: str
    norm: str
    odd: bool


def channels_of(fields: list[Field1D]) -> list[Channel]:
    out = []
    for f in fields:
        if f.kind == "scalar":
            out.append(Channel(f.name, f.name, "centre", False))
        elif f.kind == "complex":
            out += [Channel(f.name, f"|{f.name}|", "centre", False), Channel(f.name, f"J({f.name})", "centre", True)]
        else:
            out += [Channel(f.name, f"{f.name}.{c}", "scale", odd) for c, odd in zip("xy", f.odd)]
    return out


def input_channels(arrays: dict[str, np.ndarray], fields: list[Field1D], dx: float, periodic: bool) -> np.ndarray:
    """The channels of one window, float32 [channel, x, t] (module docstring)."""
    out = []
    for f in fields:
        a = prepared(arrays[f.name], f)
        if f.kind == "complex":
            z = a[..., 0] + 1j * a[..., 1]
            dz = d_dx(a[..., 0], dx, periodic) + 1j * d_dx(a[..., 1], dx, periodic)
            out += [np.abs(z), np.imag(np.conj(z) * dz)]
        else:
            out += [a[..., c] for c in range(a.shape[-1])]
    return np.stack(out).astype(np.float32)


@dataclass
class WindowSet:
    """Every window of an ensemble as network input: x [n, channel, nx, nt], with its run and
    time (end of the window)."""

    x: np.ndarray
    run: np.ndarray
    time: np.ndarray
    channels: list[Channel]


def _windows_one(args):
    ens, run, path = args
    data = read_kymograph(Path(path) / ens.file, ens.fields, ens.julia_order, ens.order)
    n_t = next(iter(data.values())).shape[1]
    t_all = ens.t0 + ens.dt * np.arange(n_t)
    xs, times = [], []
    for sl in window_slices(n_t, ens.share, ens.windows):
        xs.append(input_channels({k: v[:, sl] for k, v in data.items()}, ens.fields, ens.dx_of(run), ens.periodic))
        times.append(float(t_all[sl][-1]))
    return run, np.stack(xs), times


def windows_of(ens: Ensemble1D, workers: int = 1) -> WindowSet:
    """Read every run and cut it into its windows as network input."""
    jobs = [(ens, int(r.run), r.path) for r in ens.runs.itertuples()]
    res = run_jobs(_windows_one, jobs, workers, None, "frames")
    x = np.concatenate([r[1] for r in res])
    run = np.concatenate([[r[0]] * len(r[2]) for r in res])
    time = np.concatenate([r[2] for r in res])
    return WindowSet(x, run, time, channels_of(ens.fields))


# ---------------------------------------------------------------- views
def crop_sides(nx: int, nt: int) -> tuple[int, int]:
    """Sides of a view; nx - cx is even, so that the centred crop of a mirrored window is the
    mirrored centred crop (h exactly unchanged by the mirror)."""
    cx = max(1, round(CROP_X * nx))
    cx -= (nx - cx) % 2
    return max(1, cx), max(1, round(CROP_T * nt))


def check_network(nx: int, nt: int, channels: tuple = CHANNELS) -> None:
    """The crops must keep at least 2 points per side after the last block (as in 2D)."""
    cx, ct = crop_sides(nx, nt)
    need = 2 * 2 ** len(channels)
    if min(cx, ct) < need:
        raise ValueError(f"the crop of a window is {cx} x {ct} points; {len(channels)} blocks need at least "
                         f"{need} on each side: use longer windows (fewer windows or a larger share) or fewer blocks")


def mirror(x: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """x -> -x on a batch [b, channel, x, t], odd channels flipped."""
    return torch.flip(x, dims=[2]) * torch.where(odd, -1.0, 1.0).to(x.dtype)[None, :, None, None]


def normalise(x: torch.Tensor, channels: list[Channel]) -> torch.Tensor:
    """Each view on its own (module docstring)."""
    out = x.clone()
    for i, c in enumerate(channels):
        if c.norm == "centre":
            v = x[:, i]
            m = v.mean(dim=(1, 2), keepdim=True)
            s = v.std(dim=(1, 2), keepdim=True, unbiased=False)
            out[:, i] = (v - m) / (s + EPS)
    for name in {c.field for c in channels if c.norm == "scale"}:
        idx = [i for i, c in enumerate(channels) if c.field == name and c.norm == "scale"]
        rms = torch.sqrt(torch.mean(x[:, idx] ** 2, dim=(1, 2, 3), keepdim=True))
        out[:, idx] = x[:, idx] / (rms + EPS)
    return out


def random_views(x: torch.Tensor, channels: list[Channel], gen: torch.Generator) -> torch.Tensor:
    """A random crop of every window of the batch, mirrored with probability 1/2, normalised."""
    b, _, nx, nt = x.shape
    cx, ct = crop_sides(nx, nt)
    ox = torch.randint(0, nx - cx + 1, (b,), generator=gen)
    ot = torch.randint(0, nt - ct + 1, (b,), generator=gen)
    v = torch.stack([x[i, :, ox[i]:ox[i] + cx, ot[i]:ot[i] + ct] for i in range(b)])
    odd = torch.tensor([c.odd for c in channels], device=x.device)
    flip = torch.rand(b, generator=gen) < 0.5
    v[flip.to(x.device)] = mirror(v[flip.to(x.device)], odd)
    return normalise(v, channels)


def centred_view(x: torch.Tensor, channels: list[Channel]) -> torch.Tensor:
    nx, nt = x.shape[2:]
    cx, ct = crop_sides(nx, nt)
    ox, ot = (nx - cx) // 2, (nt - ct) // 2
    return normalise(x[:, :, ox:ox + cx, ot:ot + ct], channels)


# ---------------------------------------------------------------- training and description
Settings1D = TrainingSettings   # the settings of nn/train.py (no 2D views)


def new_encoder1d(frames: WindowSet, settings: Settings1D) -> Encoder:
    torch.manual_seed(settings.seed)
    check_network(frames.x.shape[2], frames.x.shape[3], tuple(settings.channels))
    return Encoder(frames.x.shape[1], tuple(settings.channels), periodic=False)


def train1d(frames: WindowSet, settings: Settings1D | None = None, dev: torch.device | str | None = None,
            verbose: bool = True, progress=None) -> tuple[Encoder, pd.DataFrame]:
    """Train f and g on the windows of an ensemble with the training loop of nn/train.py
    (fit_contrastive); positives are two windows of a run (nn/data.PairViews); returns f (on the
    CPU) and the log."""
    dev = torch.device(dev) if isinstance(dev, str) else (dev or device())
    s = settings or Settings1D()
    encoder = new_encoder1d(frames, s).to(dev)
    data = PairViews(frames, draws=s.draws, seed=s.seed, augment=False)

    def views(a, b, gen):
        return torch.cat([random_views(a, frames.channels, gen), random_views(b, frames.channels, gen)]).to(dev)

    log = fit_contrastive(encoder, data, views, lambda: embed1d(encoder, frames), s, dev, verbose, progress)
    return encoder.cpu(), log


@torch.no_grad()
def embed1d(encoder: Encoder, frames: WindowSet, batch: int = 256) -> np.ndarray:
    """h of every window: (f(centred view) + f(its mirror)) / 2, encoder in evaluation mode."""
    dev = next(encoder.parameters()).device
    was = encoder.training
    encoder.eval()
    odd = torch.tensor([c.odd for c in frames.channels], device=dev)
    out = []
    for i in range(0, len(frames.x), batch):
        v = centred_view(torch.from_numpy(frames.x[i:i + batch]).to(dev), frames.channels)
        out.append(((encoder(v) + encoder(mirror(v, odd))) / 2).cpu())
    encoder.train(was)
    return torch.cat(out).numpy()


def h_table1d(encoder: Encoder, frames: WindowSet, ens: Ensemble1D) -> pd.DataFrame:
    """One row per window: run, time, the parameters, then cnn.h.d000, ... (nn/train.h_table)."""
    return h_table(frames, embed1d(encoder, frames), ens)
