"""Views of a run for the learned encoder, made on the device for a whole batch.

A training item is two pairs of snapshots of one run (data.PairViews); each becomes a view:

- "rotzoom" (default since 2026-11-19): rotation by an angle drawn in [0, 2 pi), a mirror with
  probability 1/2, a zoom by a factor drawn log-uniformly in [1/zoom, zoom] (a factor above 1
  magnifies: the view covers a smaller region and the waves look longer), and a square crop of
  architecture.input_side() points at a random place (crop x the frame side, but at least what
  the network's blocks need in a periodic frame). In a periodic frame the field is tiled 3 x 3 first, so the
  turned crop never reaches an edge; in a non-periodic frame the crop is placed so that it stays
  inside. The field is read at the turned points by bilinear interpolation (grid_sample).
  Components turn with the grid: an output point o reads the input at M o + c, M = R(theta) D / s
  (D the mirror of the second axis, s the zoom), so the map from the input to the view has the
  linear part D R(-theta) (times s, which does not change directions): a vector becomes
  a' = D R(-theta) a and a nematic Q' = A Q A^T with A = D R(-theta), that is (Q_xx, Q_xy) turned
  by -2 theta, then Q_xy -> -Q_xy under the mirror. The two frames of a pair get the same view.
- "d4": the views of data.augment (rotation by k x 90 degrees, mirror, and a periodic shift in a
  periodic frame), made on the CPU when the item is drawn.

Then, if normalise: every scalar field of every view minus its own mean, divided by its own
standard deviation (both frames of the pair together, so that the change between the frames is
kept); a view flatter than STD_FLOOR is only centred. h then describes the shape of the pattern,
not its amplitude: on Swift-Hohenberg the amplitude grows like sqrt(eps) inside a phase and hid the
hexagon-stripe boundary (README, plan step 18).
Vector and nematic fields (keep_vector_mean, default since 2026-11-20) are only divided by their
root mean square (all components and both frames together; a rotation of the components changes
nothing): their mean is an order parameter (global polarisation, global alignment), invariant under
the turns of the views up to the turn itself, and removing it made a uniformly polarised state look
like a disordered one. With keep_vector_mean False (encoders saved before 2026-11-20) they are
centred per component like scalars.

For the description h after training (embedding_input): "rotzoom" takes the centred crop of
input_side() points without turning or zooming it; "d4" the whole frame; both normalised as in
training.

Physics parallel: the views are the symmetry operations the description should not see (any
rotation and mirror of an isotropic system, a change of the pattern's wavelength by +-25%, the
amplitude), so the loss can only be lowered by what survives them: the shape of the pattern.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..fields import Field
from .architecture import Network, view_side
from .data import channels_of

STD_FLOOR = 1e-8     # views whose fields vary less than this are only centred (a choice)


@dataclass(frozen=True)
class ViewSettings:
    kind: str = "rotzoom"
    zoom: float = 1.25
    crop: float = 0.75
    normalise: bool = True
    blocks: int = 4          # blocks of the network (the crop is at least what they need)
    keep_vector_mean: bool = True   # vectors and nematics: divided by their rms, not centred

    @classmethod
    def of(cls, s) -> "ViewSettings":
        """From a Network or a training Settings (views, zoom, crop, normalise, keep_vector_mean;
        blocks or channels)."""
        blocks = s.blocks if isinstance(s, Network) else len(s.channels)
        return cls(s.views, s.zoom, s.crop, s.normalise, blocks, s.keep_vector_mean)

    def side(self, n: int, periodic: bool) -> int:
        """Points per side of the network's input for a frame of n points (architecture.view_side)."""
        return view_side(self.kind, self.zoom, self.crop, self.blocks, n, periodic)


OLD = ViewSettings("d4", normalise=False, keep_vector_mean=False)    # encoders saved before 2026-11-19


def field_channels(fields: list[Field], c: int) -> list[tuple[str, list[int], list[int]]]:
    """For every field: its symmetry, its channels at t - dt and at t (a pair stacks the frames)."""
    out, k = [], 0
    for f in fields:
        n = channels_of(f)
        out.append((f.symmetry, list(range(k, k + n)), list(range(c + k, c + k + n))))
        k += n
    return out


def normalise(x: torch.Tensor, fields: list[Field], keep_vector_mean: bool = True) -> torch.Tensor:
    """x (B, C, h, w): every field of every view divided by its own root mean square; scalars (and
    vectors and nematics if not keep_vector_mean) first minus their mean, per component."""
    out = x.clone()
    for sym, before, now in field_channels(fields, x.shape[1] // 2):
        ch = before + now
        a = x[:, ch]
        if sym == "scalar" or not keep_vector_mean:
            a = a - a.mean(dim=(2, 3), keepdim=True)
        s = a.pow(2).mean(dim=(1, 2, 3)).sqrt().view(-1, 1, 1, 1)
        out[:, ch] = torch.where(s > STD_FLOOR, a / s.clamp_min(STD_FLOOR), a)
    return out


def turn_components(x: torch.Tensor, fields: list[Field], theta: torch.Tensor, mirror: torch.Tensor) -> torch.Tensor:
    """Turn the components of vectors and nematics of already resampled views (module docstring)."""
    out = x.clone()
    c, s = torch.cos(theta).view(-1, 1, 1), torch.sin(theta).view(-1, 1, 1)
    m = torch.where(mirror, -1.0, 1.0).view(-1, 1, 1).to(x.dtype)
    for sym, before, now in field_channels(fields, x.shape[1] // 2):
        if sym == "scalar":
            continue
        for i, j in (before, now):
            a, b = x[:, i], x[:, j]
            if sym == "vector":
                out[:, i], out[:, j] = c * a + s * b, m * (-s * a + c * b)
            else:                                                     # nematic: twice the angle
                c2, s2 = c * c - s * s, 2 * s * c
                out[:, i], out[:, j] = c2 * a + s2 * b, m * (-s2 * a + c2 * b)
    return out


def resample(x: torch.Tensor, fields: list[Field], theta: torch.Tensor, zoom: torch.Tensor, mirror: torch.Tensor,
             centre: torch.Tensor, side: int, periodic: bool) -> torch.Tensor:
    """Views of side points of x (B, C, n, n): view b reads the field at M o + centre[b], M =
    R(theta) D / zoom (module docstring), bilinear; the components are turned with the grid.
    centre in points of x (index along each axis); a periodic x is tiled 3 x 3 (centre + n)."""
    n = x.shape[-1]
    dev = x.device
    source, size, shift = (x.repeat(1, 1, 3, 3), 3 * n, n) if periodic else (x, n, 0)
    o = torch.arange(side, device=dev, dtype=x.dtype) - (side - 1) / 2
    oi, oj = torch.meshgrid(o, o, indexing="ij")
    oj = torch.where(mirror.view(-1, 1, 1), -oj, oj)
    cos, sin = (torch.cos(theta) / zoom).view(-1, 1, 1), (torch.sin(theta) / zoom).view(-1, 1, 1)
    pi = cos * oi - sin * oj + (centre[:, 0] + shift).view(-1, 1, 1)     # index along the first axis
    pj = sin * oi + cos * oj + (centre[:, 1] + shift).view(-1, 1, 1)     # index along the second axis
    grid = torch.stack([(pj + 0.5) / size * 2 - 1, (pi + 0.5) / size * 2 - 1], dim=-1).to(x.dtype)
    v = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return turn_components(v, fields, theta, mirror)


def rotzoom(x: torch.Tensor, fields: list[Field], vs: ViewSettings, gen: torch.Generator,
            periodic: bool) -> torch.Tensor:
    """x (B, C, n, n) on the device -> views (B, C, side, side), one draw per view (module docstring)."""
    b, _, n, _ = x.shape
    dev = x.device
    side = vs.side(n, periodic)
    theta = (torch.rand(b, generator=gen) * 2 * math.pi).to(dev)
    zoom = torch.exp((torch.rand(b, generator=gen) * 2 - 1) * math.log(vs.zoom)).to(dev)
    mirror = (torch.rand(b, generator=gen) < 0.5).to(dev)
    if periodic:
        centre = torch.rand(b, 2, generator=gen) * n
    else:
        r = side * vs.zoom * math.sqrt(2) / 2                   # half-diagonal of the largest crop
        lo, hi = r - 0.5, n - 0.5 - r
        u = torch.rand(b, 2, generator=gen)
        centre = lo + u * (hi - lo) if hi > lo else torch.full((b, 2), (n - 1) / 2)
    return resample(x, fields, theta, zoom, mirror, centre.to(dev), side, periodic)


def training_views(x: torch.Tensor, fields: list[Field], vs: ViewSettings, gen: torch.Generator,
                   periodic: bool) -> torch.Tensor:
    """The views of a batch of pairs (already turned by data.augment for "d4")."""
    if vs.kind == "rotzoom":
        x = rotzoom(x, fields, vs, gen, periodic)
    return normalise(x, fields, vs.keep_vector_mean) if vs.normalise else x


def embedding_input(x: torch.Tensor, fields: list[Field], vs: ViewSettings, periodic: bool) -> torch.Tensor:
    """What the trained encoder reads to give h: the centred crop (rotzoom) or the whole frame."""
    if vs.kind == "rotzoom":
        n = x.shape[-1]
        side = vs.side(n, periodic)
        a = (n - side) // 2
        x = x[:, :, a:a + side, a:a + side]
    return normalise(x, fields, vs.keep_vector_mean) if vs.normalise else x
