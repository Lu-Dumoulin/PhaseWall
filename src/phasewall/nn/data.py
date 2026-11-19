"""Network inputs: the frames of an ensemble as channels, and the two augmented views of a run.

A sample is a pair of frames of one run, at t - pair_dt and at t (the same frames as the
hand-made description: crop and coarsening from frames.py). Each field gives channels: one
for a scalar (relative scalars as (f - <f>) / <f>), two for a vector (a_x, a_y) or a nematic
(Q_xx, Q_xy); the pair stacks the channels at t - pair_dt, then at t. Each field is divided by
one number, its standard deviation over the whole ensemble, so that all inputs are of order 1;
both components of a vector or a nematic share that number, so the rotation rules still hold.

A training item is two views of one run: two different pairs of the run (the same one if it
has a single pair). With the views "d4" each is turned here by a random element of D4 with the
rules of fields.py (vectors and nematics turn with the grid) and, in a periodic frame only,
shifted at random; with the default views "rotzoom" the pairs are returned as they are and the
views are made on the device for the whole batch (views.py).
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..describe import prepared, run_frame
from ..fields import Field, mirror_x, rotate90
from ..frames import prepare
from ..io import Ensemble, read_snapshot, run_snapshots
from ..parallel import Progress, run_jobs

DRAWS = 4       # every run is drawn this many times per epoch (a choice)


def channels_of(f: Field) -> int:
    return 1 if f.symmetry == "scalar" else 2


def to_channels(frame: dict[str, np.ndarray], fields: list[Field]) -> np.ndarray:
    """One frame as an array (channels, nx, ny), float32."""
    out = []
    for f in fields:
        a = prepared(frame[f.name], f)
        out.append(a[None] if f.symmetry == "scalar" else np.moveaxis(a, -1, 0))
    return np.concatenate(out).astype(np.float32)


@dataclass
class FrameSet:
    """The pairs of an ensemble, ready for the network.

    x: (rows, 2C, nx, ny) float32, scaled; rows are the described snapshots that have a
    partner pair_dt earlier. run, time: of each row. scale: per field."""

    x: np.ndarray
    run: np.ndarray
    time: np.ndarray
    fields: list[Field]
    periodic: bool
    scale: np.ndarray

    @property
    def in_channels(self) -> int:
        return self.x.shape[1]


def _run_pairs(args):
    ens, run, path, dx = args
    snaps = run_snapshots(ens, path)
    times = np.array([t for t, _ in snaps])
    frame, rows = None, []
    for t, p in snaps[-ens.last:]:
        j = np.flatnonzero(np.isclose(times, t - ens.pair_dt, rtol=0, atol=1e-6 * ens.pair_dt))
        if not len(j):
            continue
        now = read_snapshot(p, ens.fields, ens.julia_order)
        frame = frame or run_frame(ens, now[ens.fields[0].name].shape, dx)
        old = read_snapshot(snaps[j[0]][1], ens.fields, ens.julia_order)
        pair = np.concatenate([to_channels(prepare(old, ens.fields, frame), ens.fields),
                               to_channels(prepare(now, ens.fields, frame), ens.fields)])
        rows.append((t, pair))
    return run, rows, frame.periodic if frame else None


def load_frames(ens: Ensemble, workers: int = 1, scale: np.ndarray | None = None,
                progress: Progress | None = None) -> FrameSet:
    """All pairs of an ensemble. scale: per field, from a training ensemble (default: the
    standard deviation of each field over this ensemble)."""
    if ens.pair_dt is None:
        raise ValueError("the learned description needs pairs of snapshots: set pair_dt")
    dxs = ens.runs["dx"] if "dx" in ens.runs else [ens.dx] * len(ens.runs)
    jobs = [(ens, int(r.run), r.path, float(d)) for r, d in zip(ens.runs.itertuples(), dxs)]
    results = run_jobs(_run_pairs, jobs, workers, progress, "frames")
    xs, runs, times, periodic = [], [], [], set()
    for run, rows, per in results:
        for t, pair in rows:
            xs.append(pair)
            runs.append(run)
            times.append(t)
        if per is not None:
            periodic.add(per)
    if not xs:
        raise ValueError("no pair of snapshots pair_dt apart was found")
    shapes = {a.shape for a in xs}
    if len(shapes) > 1:
        raise ValueError(f"frames of different sizes {sorted(shapes)}: set crop and dx_target")
    x = np.stack(xs)
    c = x.shape[1] // 2
    if scale is None:
        scale, k = [], 0
        for f in ens.fields:
            n = channels_of(f)
            chans = list(range(k, k + n)) + list(range(c + k, c + k + n))
            scale.append(float(np.sqrt(np.mean(x[:, chans] ** 2))) or 1.0)
            k += n
        scale = np.array(scale)
    k = 0
    for f, s in zip(ens.fields, scale):
        n = channels_of(f)
        x[:, k:k + n] /= s
        x[:, c + k:c + k + n] /= s
        k += n
    return FrameSet(x, np.array(runs), np.array(times), list(ens.fields), periodic.pop(), np.asarray(scale))


# ---------------------------------------------------------------- augmentation
def augment(x: np.ndarray, fields: list[Field], rng: np.random.Generator, periodic: bool) -> np.ndarray:
    """A pair (2C, nx, ny) turned by a random element of D4 (and shifted at random if
    periodic); both frames of the pair get the same transformation."""
    turns, mirror = int(rng.integers(4)), bool(rng.integers(2))
    shift = (int(rng.integers(x.shape[1])), int(rng.integers(x.shape[2]))) if periodic else (0, 0)
    c = x.shape[0] // 2
    out = []
    for start in (0, c):
        k = start
        for f in fields:
            n = channels_of(f)
            a = x[k] if f.symmetry == "scalar" else np.moveaxis(x[k:k + n], 0, -1)
            for _ in range(turns):
                a = rotate90(a, f.kind)
            if mirror:
                a = mirror_x(a, f.kind)
            a = np.roll(a, shift, axis=(0, 1))
            out.append(a[None] if f.symmetry == "scalar" else np.moveaxis(a, -1, 0))
            k += n
    return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)


class PairViews(Dataset):
    """Training items: (view 1, view 2) of one run. Item i is run i % runs; its random draws
    depend only on (seed, epoch, i), so a training can be repeated. Set `epoch` before each
    epoch."""

    def __init__(self, frames: FrameSet, draws: int = DRAWS, seed: int = 0, augment: bool = True):
        self.frames, self.draws, self.seed, self.epoch = frames, draws, seed, 0
        self.augment = augment        # False: the pairs as they are (views made later, nn/views.py)
        self.rows = [np.flatnonzero(frames.run == r) for r in np.unique(frames.run)]

    def __len__(self) -> int:
        return len(self.rows) * self.draws

    def __getitem__(self, i: int):
        rows = self.rows[i % len(self.rows)]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        a = int(rng.choice(rows))
        others = rows[rows != a]
        b = int(rng.choice(others)) if len(others) else a
        f = self.frames
        if not self.augment:
            return torch.from_numpy(f.x[a]), torch.from_numpy(f.x[b])
        v1 = augment(f.x[a], f.fields, rng, f.periodic)
        v2 = augment(f.x[b], f.fields, rng, f.periodic)
        return torch.from_numpy(v1), torch.from_numpy(v2)
