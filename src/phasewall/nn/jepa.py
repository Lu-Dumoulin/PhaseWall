"""A predictive learned description (JEPA style; extra [nn]): predict the description of hidden parts
of a frame from the visible rest (Assran et al. 2023, I-JEPA; Bardes et al. 2024, V-JEPA).

The contrastive training (train.py) is rewarded for telling runs apart, so it can win by
recognising a run (a wavevector of the box, its noise) instead of its state. Here nothing rewards
that: the network must predict the description of hidden parts of a sample from the rest, which it
can only do by knowing the local structure and dynamics of the pattern.

A sample is a pair of frames (t - pair_dt and t, as for the contrastive training: nn/data.py), so
the dynamics over one lag is seen. Training:
- Views: a random element of the symmetries of the square with the rules of fields.py (and a random
  shift in a periodic frame), then each field normalised on its own (views.normalise). No crop: the
  masks replace it.
- Encoder: the convolutional blocks of model.py without the final mean, kept as a map F[c, X, Y]
  on the grid coarsened by 2^blocks (BLOCKS: 3, so 8 times coarser); circular padding for a
  periodic frame.
- Masks: N_MASKS squares of MASK_SIDE of the coarse side at random places (on the coarse grid,
  spread to the points of the frame); the context encoder sees the sample with them set to 0 and a
  channel of ones where it is visible.
- Target: the whole sample through the target encoder, a copy whose weights follow the context
  encoder's as an exponential moving average (EMA_START raised to 1 by a cosine), layer-normalised
  at each cell, no gradient.
- Predictor: three 3 x 3 convolutions from the context map and the mask to the target map; loss:
  mean squared error on the hidden cells.
Description: h = mean over the cells of the target encoder's map of the whole sample, averaged over
the eight symmetries of the square, so that h is exactly unchanged by them (the components of
vectors and nematics turned by the rules of fields.py). Columns cnn.h.d000, ... as the other
learned descriptions.

Collapse (every h the same) would also lower the loss; the EMA target and the predictor prevent it
in practice (no proof). The log keeps the loss and the spread of h at each epoch.

The pieces that do not depend on the dimension (MapEncoder, Predictor, masks, EMA) are also used by
the 1D version (oned/jepa1d.py).
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..fields import Field
from .data import FrameSet, augment, channels_of
from .model import Encoder
from .train import device, h_table, schedule
from .views import normalise

CHANNELS = (32, 64, 128)   # three blocks: a coarse grid 8 times coarser than the frame (a choice)
N_MASKS = 4                # hidden squares per sample (a choice, as I-JEPA)
MASK_SIDE = 0.3            # side of a hidden square, share of the coarse side (a choice)
EMA_START = 0.996          # EMA of the target encoder, raised to 1 by a cosine (I-JEPA's value)


# ---------------------------------------------------------------- pieces common to 1D and 2D
class MapEncoder(nn.Module):
    """The blocks of model.Encoder without the final mean: x -> map [b, c, X, Y]."""

    def __init__(self, in_channels: int, channels: tuple, periodic: bool = False):
        super().__init__()
        self.blocks = Encoder(in_channels, channels, periodic=periodic).blocks
        self.dim = channels[-1]
        self.stride = 2 ** len(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Predictor(nn.Module):
    """Three 3 x 3 convolutions: (context map, visible mask) -> predicted target map."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim + 1, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, dim, 3, padding=1))

    def forward(self, ctx: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([ctx, visible], dim=1))


def coarse_shape(nx: int, ny: int, stride: int) -> tuple[int, int]:
    """Size of the map of a frame of nx x ny points (each stride-2 convolution of size 3 and
    padding 1 gives ceil(n / 2))."""
    for _ in range(int(np.log2(stride))):
        nx, ny = (nx + 1) // 2, (ny + 1) // 2
    return nx, ny


def random_masks(b: int, cx: int, cy: int, n_masks: int, mx: int, my: int, gen: torch.Generator) -> torch.Tensor:
    """Hidden cells [b, cx, cy] (True: hidden): n_masks rectangles of mx x my cells per sample."""
    hide = torch.zeros(b, cx, cy, dtype=torch.bool)
    mx, my = min(mx, cx), min(my, cy)
    for i in range(b):
        for _ in range(n_masks):
            x0 = int(torch.randint(0, cx - mx + 1, (1,), generator=gen))
            y0 = int(torch.randint(0, cy - my + 1, (1,), generator=gen))
            hide[i, x0:x0 + mx, y0:y0 + my] = True
    return hide


def input_mask(hide: torch.Tensor, nx: int, ny: int, stride: int) -> torch.Tensor:
    """The coarse hidden cells spread to the points of the frame: [b, 1, nx, ny], 1 = visible."""
    up = hide.repeat_interleave(stride, dim=1).repeat_interleave(stride, dim=2)[:, :nx, :ny]
    return (~up).float()[:, None]


def ema_rate(step: int, total: int, start: float) -> float:
    return 1.0 - (1.0 - start) * 0.5 * (1 + np.cos(np.pi * min(step / max(total, 1), 1.0)))


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, rate: float) -> None:
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(rate).add_(po.detach(), alpha=1 - rate)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)


def jepa_loss(online: MapEncoder, target: MapEncoder, pred: Predictor, x: torch.Tensor, hide: torch.Tensor
              ) -> torch.Tensor:
    """Mean squared error between the predicted and the target maps on the hidden cells."""
    vis = input_mask(hide, x.shape[2], x.shape[3], online.stride).to(x.device)
    with torch.no_grad():
        tgt = target(torch.cat([x, torch.ones_like(vis)], dim=1))
        tgt = F.layer_norm(tgt.permute(0, 2, 3, 1), (tgt.shape[1],)).permute(0, 3, 1, 2)
    ctx = online(torch.cat([x * vis, vis], dim=1))
    out = pred(ctx, (~hide).float()[:, None].to(x.device))
    m = hide[:, None].float().to(x.device)
    return (((out - tgt) ** 2) * m).sum() / (m.sum() * out.shape[1]).clamp(min=1)


# ---------------------------------------------------------------- training (any dimension)
@dataclass
class SettingsJEPA:
    epochs: int = 40
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 3
    channels: tuple = CHANNELS
    seed: int = 0
    n_masks: int = N_MASKS
    mask_share: tuple = (MASK_SIDE, MASK_SIDE)   # side of a hidden rectangle, share of the coarse grid per axis
    ema_start: float = EMA_START


class Samples(Dataset):
    """Item k: sample k of x, through `turn` (a random symmetry, or nothing)."""

    def __init__(self, x: np.ndarray, turn=None):
        self.x, self.turn = x, turn

    def __len__(self):
        return len(self.x)

    def __getitem__(self, k):
        return torch.from_numpy(self.turn(self.x[k]) if self.turn else self.x[k])


def fit_jepa(data: Dataset, x_all: np.ndarray, periodic: bool, prepare, images, normalise_fn,
             settings: SettingsJEPA | None = None, dev: torch.device | str | None = None,
             verbose: bool = True) -> tuple[MapEncoder, pd.DataFrame]:
    """The training of the module docstring, for samples of any dimension laid out as images
    [channel, X, Y] (2D: x, y; 1D: x, t). data: the training samples (Dataset of tensors);
    prepare(batch, gen): the views of a batch (CPU); images(v): the symmetric images averaged in
    h; normalise_fn(batch): the normalisation of embed_maps. Returns the target encoder (CPU) and
    the log (loss and spread of h per epoch)."""
    dev = torch.device(dev) if isinstance(dev, str) else (dev or device())
    s = settings or SettingsJEPA()
    torch.manual_seed(s.seed)
    n, c, nx, ny = x_all.shape
    online = MapEncoder(c + 1, tuple(s.channels), periodic).to(dev)      # + the visible channel
    target = copy.deepcopy(online).to(dev)
    for p in target.parameters():
        p.requires_grad_(False)
    pred = Predictor(online.dim).to(dev)
    cx, cy = coarse_shape(nx, ny, online.stride)
    mx, my = max(1, round(s.mask_share[0] * cx)), max(1, round(s.mask_share[1] * cy))
    loader = DataLoader(data, batch_size=min(s.batch, n), shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(s.seed))
    gen = torch.Generator().manual_seed(s.seed)
    opt = torch.optim.AdamW([*online.parameters(), *pred.parameters()], lr=s.lr, weight_decay=s.weight_decay)
    steps = len(loader)
    total = s.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: schedule(k, s.warmup * steps, total))
    rows, step = [], 0
    for epoch in range(s.epochs):
        losses = []
        for x in loader:
            x = prepare(x, gen)
            hide = random_masks(len(x), cx, cy, s.n_masks, mx, my, gen)
            loss = jepa_loss(online, target, pred, x.to(dev), hide)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ema_update(target, online, ema_rate(step, total, s.ema_start))
            step += 1
            losses.append(float(loss.detach()))
        h = embed_maps(target, x_all[:512], normalise_fn, images)
        rows.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "h_std": float(h.std(axis=0).mean())})
        if verbose:
            print(f"epoch {epoch + 1}/{s.epochs}: loss {rows[-1]['loss']:.4f}, h std {rows[-1]['h_std']:.4f}", flush=True)
    return target.cpu(), pd.DataFrame(rows)


@torch.no_grad()
def embed_maps(encoder: MapEncoder, x: np.ndarray, normalise_fn, images, batch: int = 128) -> np.ndarray:
    """h of every sample: mean over the cells of the encoder's map, averaged over images(v) (the
    symmetric images of the normalised sample), with the visible channel at 1."""
    dev = next(encoder.parameters()).device
    was = encoder.training
    encoder.eval()
    out = []
    for i in range(0, len(x), batch):
        v = normalise_fn(torch.from_numpy(x[i:i + batch])).to(dev)
        ones = torch.ones_like(v[:, :1])
        out.append(torch.stack([encoder(torch.cat([w, ones], 1)).mean(dim=(2, 3)) for w in images(v)]).mean(0).cpu())
    encoder.train(was)
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- 2D
def train_jepa(frames: FrameSet, settings: SettingsJEPA | None = None, dev: torch.device | str | None = None,
               verbose: bool = True) -> tuple[MapEncoder, pd.DataFrame]:
    """2D: views = a random symmetry of the square (and shift if periodic), then the
    normalisation of views.py; h averaged over the eight symmetries."""
    s = settings or SettingsJEPA()
    rng = np.random.default_rng(s.seed)
    data = Samples(frames.x, lambda a: augment(a, frames.fields, rng, frames.periodic))
    return fit_jepa(data, frames.x, frames.periodic, lambda x, gen: normalise(x, frames.fields),
                    lambda v: d4_turns(v, frames.fields), lambda x: normalise(x, frames.fields), s, dev, verbose)


def d4_turns(x: torch.Tensor, fields: list[Field]) -> list[torch.Tensor]:
    """The eight images of a batch of pairs [b, 2C, nx, ny] under the symmetries of the square,
    the components of vectors and nematics turned with the grid (fields.rotate90, mirror_x)."""
    c = x.shape[1] // 2
    kinds = []
    for f in fields:
        kinds += [f.symmetry] * channels_of(f)
    kinds = kinds * 2

    def rot(a):                                      # rotate90 of fields.py: a[x, y] -> a[-y, x]
        out = torch.rot90(a, 1, dims=(2, 3)).clone()
        i = 0
        while i < 2 * c:
            if kinds[i] == "vector":
                out[:, i], out[:, i + 1] = -out[:, i + 1].clone(), out[:, i].clone()
                i += 2
            elif kinds[i] == "nematic":
                out[:, i], out[:, i + 1] = -out[:, i].clone(), -out[:, i + 1].clone()
                i += 2
            else:
                i += 1
        return out

    def mir(a):                                      # mirror_x: x -> -x
        out = torch.flip(a, dims=(2,)).clone()
        i = 0
        while i < 2 * c:
            if kinds[i] == "vector":
                out[:, i] = -out[:, i]
                i += 2
            elif kinds[i] == "nematic":
                out[:, i + 1] = -out[:, i + 1]
                i += 2
            else:
                i += 1
        return out

    out, a = [], x
    for _ in range(4):
        out += [a, mir(a)]
        a = rot(a)
    return out


def embed_jepa(encoder: MapEncoder, frames: FrameSet, limit: int | None = None) -> np.ndarray:
    """h of every pair (or of the first `limit`), averaged over the eight symmetries of the square."""
    x = frames.x if limit is None else frames.x[:limit]
    return embed_maps(encoder, x, lambda b: normalise(b, frames.fields), lambda v: d4_turns(v, frames.fields))


def h_table_jepa(encoder: MapEncoder, frames, ens) -> pd.DataFrame:
    """One row per sample: run, time, the parameters, then cnn.h.d000, ... (train.h_table)."""
    return h_table(frames, embed_jepa(encoder, frames), ens)
