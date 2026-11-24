"""A predictive learned description of 1D windows (JEPA style; extra [nn]; a test, Ludo 2026-11-23).

The contrastive training (nn1d.py) is rewarded for telling runs apart, so it can win by
recognising a run (its wavenumber, its noise) instead of its regime. Here nothing rewards that:
the network must predict the description of hidden parts of a window from the visible rest, which
it can only do by knowing the local dynamics (Assran et al. 2023, I-JEPA; Bardes et al. 2024,
V-JEPA).

- Encoder: the convolutional blocks of nn/model.py (zero padding), kept as a map F[c, X, T] on
  the grid coarsened by 2^blocks (BLOCKS = 3: 8 times; a window of 256 x 50 points gives 32 x 7).
- Masks: on that coarse grid, N_MASKS rectangles of MASK_X x MASK_T cells at random places (the
  same in the input, cell by cell); about MASK_SHARE of the cells are hidden.
- Context: the window, normalised and mirrored at random as in nn1d.py, with the hidden cells set
  to 0 and a channel of ones where the input is visible, through the context encoder f.
- Target: the whole window through the target encoder, a copy of f whose weights follow f slowly
  (exponential moving average, EMA from EMA_START to 1 over the training); no gradient through it.
  Its map is normalised over the channels at each cell (layer normalisation).
- Predictor: three 3 x 3 convolutions on the context map and the mask, giving a map like the
  target's. Loss: mean squared error between prediction and target on the hidden cells only.
- Description: h = mean over the cells of the target encoder's map of the whole window (no crop:
  the masks replace the crops of nn1d.py), averaged with its mirror, so exactly unchanged by the mirror; columns
  cnn.h.d000, ... as the other learned descriptions.

Collapse (every h the same) would lower the loss too; the EMA target and the predictor are what
prevent it in practice (no proof). The log keeps the loss and the spread of h (standard deviation
over the windows, averaged over the dimensions) at each epoch to see it.

The training loop, the encoder map, the predictor, the masks and the moving average are those of
the 2D version (nn/jepa.py, fit_jepa); this module only gives the 1D views and symmetry.
"""

import numpy as np
import pandas as pd
import torch

from ..nn.jepa import MapEncoder, Samples, SettingsJEPA, embed_maps, fit_jepa
from ..nn.train import h_table
from .kymo import Ensemble1D
from .nn1d import WindowSet, mirror, normalise

MASK_SHARE = (6 / 32, 3 / 7)   # hidden rectangles: 6 x 3 cells of the coarse grid of a 256 x 50 window (a choice)


def settings_1d(**kw) -> SettingsJEPA:
    """The settings of nn/jepa.py with the 1D masks."""
    return SettingsJEPA(**({"mask_share": MASK_SHARE} | kw))


def train_jepa(frames: WindowSet, settings: SettingsJEPA | None = None, dev: torch.device | str | None = None,
               verbose: bool = True) -> tuple[MapEncoder, pd.DataFrame]:
    """1D: views = the mirror at random (odd channels flipped) and the normalisation of nn1d.py;
    h averaged over a window and its mirror."""
    odd = torch.tensor([c.odd for c in frames.channels])

    def prepare(x, gen):
        flip = torch.rand(len(x), generator=gen) < 0.5
        x = x.clone()
        x[flip] = mirror(x[flip], odd)
        return normalise(x, frames.channels)

    return fit_jepa(Samples(frames.x), frames.x, False, prepare, _images(frames), lambda b: normalise(b, frames.channels),
                    settings or settings_1d(), dev, verbose)


def _images(frames: WindowSet):
    odd = [c.odd for c in frames.channels]
    return lambda v: [v, mirror(v, torch.tensor(odd, device=v.device))]


def embed_jepa(encoder: MapEncoder, frames: WindowSet, limit: int | None = None) -> np.ndarray:
    """h of every window (or of the first `limit`): mean of the map of the whole window (no crop)
    and of its mirror."""
    x = frames.x if limit is None else frames.x[:limit]
    return embed_maps(encoder, x, lambda b: normalise(b, frames.channels), _images(frames))


def h_table_jepa(encoder: MapEncoder, frames: WindowSet, ens: Ensemble1D) -> pd.DataFrame:
    return h_table(frames, embed_jepa(encoder, frames), ens)
