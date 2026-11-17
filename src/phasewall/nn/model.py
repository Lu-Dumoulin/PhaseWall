"""The encoder f and the projection head g (adapted from hfml Step 7).

f: four blocks, each a 3 x 3 convolution with stride 2 (halves the grid) then a 3 x 3
convolution, batch normalisation and ReLU after each; channels in -> 32 -> 64 -> 128 -> 256;
then the average over the frame: h, 256 numbers. Padding is circular for a periodic frame (a
shift of the pattern by a multiple of 16 points leaves h unchanged) and zeros otherwise.
g: Linear(256, 256), ReLU, Linear(256, 64), then z = g / |g| on the unit sphere. Only f is
kept after training; h is the description.
"""

import itertools

import torch
import torch.nn.functional as F
from torch import nn

CHANNELS = (32, 64, 128, 256)   # channels of the four blocks (a choice, as in hfml)
HIDDEN, OUT = 256, 64           # sizes of the head g (a choice, as in hfml)


def block(c_in: int, c_out: int, periodic: bool) -> nn.Sequential:
    """3 x 3 convolution with stride 2, then 3 x 3 with stride 1, each followed by batch
    normalisation and ReLU."""
    mode = "circular" if periodic else "zeros"

    def conv(a, b, stride):
        return nn.Conv2d(a, b, 3, stride=stride, padding=1, padding_mode=mode)
    return nn.Sequential(conv(c_in, c_out, 2), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                         conv(c_out, c_out, 1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    """f: (batch, in_channels, nx, ny) -> h, (batch, channels[-1])."""

    def __init__(self, in_channels: int, channels=CHANNELS, periodic: bool = True):
        super().__init__()
        sizes = (in_channels, *channels)
        self.blocks = nn.Sequential(*[block(a, b, periodic) for a, b in itertools.pairwise(sizes)])
        self.dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).mean(dim=(2, 3))          # average over the frame


class ProjectionHead(nn.Module):
    """g followed by the normalisation: h -> z on the unit sphere."""

    def __init__(self, dim: int, hidden: int = HIDDEN, out: int = OUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=1)


def count_weights(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
