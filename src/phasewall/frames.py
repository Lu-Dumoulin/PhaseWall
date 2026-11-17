"""The frame that is described: a square crop of fixed physical size, coarsened to a common
grid spacing.

Why. Descriptors of different runs must mean the same thing column by column. The shells
of S(k) start from the side of the frame, so frames of different sizes give columns at
different wavenumbers; and domain counts depend a little on the grid spacing. Describing,
in every run, a crop of the same side `crop`, coarsened to the same spacing `dx_target`,
makes all columns comparable even when boxes and grids differ between runs.

Boundaries. A whole periodic box is described as it is (periodic frame: FFTs, periodic
domains, circular shifts). A crop, or any frame of a non-periodic box, is not periodic and
is described with the non-periodic rules of describe.py. For the moving correlation, the
frame at t is compared with the frame at t - pair_dt shifted by up to `max_shift` in each
direction, so the crop must stay `max_shift` away from the walls of a non-periodic box:

    +-----------------------------+   box of side L (non-periodic)
    |   max_shift                 |
    |   +---------------------+   |
    |   |  crop of side Lc    |   |   Lc + 2 max_shift <= L
    |   |  (centred)          |   |
    |   +---------------------+   |
    +-----------------------------+

In a periodic box the margin wraps around, so only Lc <= L is needed.

Settings (TOML [ensemble], all optional): crop (side, physical units; default the whole
periodic box, or L - 2 max_shift for a non-periodic one), dx_target (coarsening by block
averaging; dx_target / dx must be an integer), max_shift (physical units; default
MAX_SHIFT_FRACTION L).
"""

import math
from dataclasses import dataclass

import numpy as np

from .fields import Field

MAX_SHIFT_FRACTION = 0.2   # default max_shift: a fifth of the box side (a choice, Ludo 2026-11-17)
TOLERANCE = 1e-6           # relative tolerance for "is an integer multiple" (a choice)


@dataclass(frozen=True)
class Frame:
    """How one run is cut and coarsened.

    periodic: the described frame is a whole periodic box.
    crop: side of the crop in points of the run grid (None: the whole box).
    factor: coarsening factor (block of factor x factor points).
    shift: largest shift of the moving correlation, in points of the described grid
        (unused for a periodic frame, where every circular shift is tried).
    dx: grid spacing of the described frame.
    box_periodic: the box of the run is periodic (the margin around a crop may wrap).
    """

    periodic: bool
    crop: int | None
    factor: int
    shift: int
    dx: float
    box_periodic: bool


def _multiple(x: float, unit: float, what: str) -> int:
    """x / unit as an integer; ValueError if it is not one."""
    q = x / unit
    if abs(q - round(q)) > TOLERANCE * max(1.0, abs(q)) or round(q) < 1:
        raise ValueError(f"{what} = {x:g} must be a positive integer multiple of {unit:g}")
    return round(q)


def frame_for(shape: tuple[int, ...], dx: float, periodic: bool, crop: float | None = None,
              dx_target: float | None = None, max_shift: float | None = None) -> Frame:
    """The frame of a run whose snapshots have grid `shape` (nx, ny, ...) and spacing dx."""
    nx, ny = shape[:2]
    side = min(nx, ny)
    factor = 1 if dx_target is None else _multiple(dx_target, dx, "dx_target (in units of the run's dx)")
    dx_frame = dx * factor
    s = MAX_SHIFT_FRACTION * side * dx if max_shift is None else max_shift
    if s < 0:
        raise ValueError("max_shift must be >= 0")
    shift = math.floor(s / dx_frame + TOLERANCE)
    margin = shift * factor                       # in points of the run grid
    if crop is None and periodic:
        if nx % factor or ny % factor:
            raise ValueError(f"a box of {nx} x {ny} points cannot be coarsened by {factor}")
        return Frame(True, None, factor, shift, dx_frame, True)
    if crop is None:                              # non-periodic box: the largest centred crop
        n = (side - 2 * margin) // factor
        if n < 1:
            raise ValueError(f"max_shift = {s:g} leaves no room for a crop in a box of side {side * dx:g}")
        return Frame(False, n * factor, factor, shift, dx_frame, False)
    n = _multiple(crop, dx_frame, "crop (in units of the grid spacing of the frame)")
    need = n * factor + (0 if periodic else 2 * margin)
    if need > side:
        raise ValueError(f"crop = {crop:g}" + ("" if periodic else f" plus 2 max_shift = {2 * s:g}")
                         + f" does not fit in a box of side {side * dx:g}")
    return Frame(False, n * factor, factor, shift, dx_frame, periodic)


def coarsen(a: np.ndarray, factor: int) -> np.ndarray:
    """Block average over factor x factor points (the first two axes)."""
    if factor == 1:
        return a
    nx, ny = a.shape[:2]
    b = a.reshape(nx // factor, factor, ny // factor, factor, *a.shape[2:])
    return b.mean(axis=(1, 3))


def cut(a: np.ndarray, frame: Frame, margin: bool = False) -> np.ndarray:
    """The frame of one field, coarsened; with `margin`, extended by frame.shift points of the
    described grid on every side (wrapping around in a periodic box)."""
    if frame.crop is None:
        return coarsen(a, frame.factor)
    nx, ny = a.shape[:2]
    m = frame.shift * frame.factor if margin else 0
    rows = np.arange((nx - frame.crop) // 2 - m, (nx - frame.crop) // 2 + frame.crop + m)
    cols = np.arange((ny - frame.crop) // 2 - m, (ny - frame.crop) // 2 + frame.crop + m)
    if frame.box_periodic:
        rows, cols = rows % nx, cols % ny
    return coarsen(a[np.ix_(rows, cols)], frame.factor)


def prepare(snapshot: dict[str, np.ndarray], fields: list[Field], frame: Frame,
            margin: bool = False) -> dict[str, np.ndarray]:
    """cut() applied to every listed field of a snapshot."""
    return {f.name: cut(snapshot[f.name], frame, margin) for f in fields}
