"""Field types and how they change under the symmetries of a square periodic grid.

A snapshot is a dict {name: array}. Arrays are indexed [x, y] for a scalar field and
[x, y, component] for the other kinds:

- "scalar": density, concentration of nucleators, ...: (nx, ny).
- "vector": velocity, polarity: (nx, ny, 2), components (a_x, a_y).
- "nematic": traceless symmetric tensor Q stored by its two independent components
  (Q_xx, Q_xy), with Q_yy = -Q_xx: (nx, ny, 2). For a director at angle theta,
  (Q_xx, Q_xy) = S (cos 2 theta, sin 2 theta) / 2.

The square grid has eight symmetries (the group D4): rotations by multiples of 90 degrees,
each with or without a mirror. Turning the grid is not enough: the vectors and tensors on it
must turn too. Rotation by +90 degrees (x towards y) sends a vector (a_x, a_y) to
(-a_y, a_x) and turns a director by 90 degrees, so 2 theta by 180 degrees: Q -> -Q.
The mirror x -> -x sends a_x to -a_x, and theta to pi - theta: Q_xx unchanged, Q_xy -> -Q_xy.
"""

from dataclasses import dataclass

import numpy as np

KINDS = ("scalar", "vector", "nematic")


@dataclass(frozen=True)
class Field:
    """One field of a snapshot.

    name: the name used inside phasewall (for example "rho", "p").
    keys: names the field may have in the files, tried in order (for example ["C", "rho"]).
    kind: "scalar", "vector" or "nematic".
    relative: scalars only; describe (f - <f>) / <f> instead of f, so that the mean value,
    which often changes across the parameter grid (the mean density), does not by itself
    separate the runs.
    """

    name: str
    keys: tuple[str, ...]
    kind: str = "scalar"
    relative: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"field {self.name}: kind must be one of {KINDS}, not {self.kind!r}")
        if self.relative and self.kind != "scalar":
            raise ValueError(f"field {self.name}: 'relative' is only for scalar fields")

    def check(self, a: np.ndarray) -> None:
        """Raise ValueError if the array does not have the shape of this kind of field."""
        if self.kind == "scalar" and a.ndim != 2:
            raise ValueError(f"field {self.name}: a scalar field needs shape (nx, ny), got {a.shape}")
        if self.kind != "scalar" and (a.ndim != 3 or a.shape[2] != 2):
            raise ValueError(f"field {self.name}: a {self.kind} field needs shape (nx, ny, 2), "
                             f"got {a.shape}")


def rotate90(a: np.ndarray, kind: str) -> np.ndarray:
    """The field turned by +90 degrees (x towards y), grid and components together."""
    b = np.rot90(a, 1, axes=(0, 1)).copy()
    if kind == "vector":
        b = np.stack([-b[..., 1], b[..., 0]], axis=-1)
    elif kind == "nematic":
        b = -b
    return b


def mirror_x(a: np.ndarray, kind: str) -> np.ndarray:
    """The field mirrored by x -> -x, grid and components together."""
    b = a[::-1].copy()
    if kind == "vector":
        b[..., 0] = -b[..., 0]
    elif kind == "nematic":
        b[..., 1] = -b[..., 1]
    return b


def transform(snapshot: dict[str, np.ndarray], fields: list[Field], turns: int = 0,
              mirror: bool = False, shift: tuple[int, int] = (0, 0)) -> dict[str, np.ndarray]:
    """Apply one symmetry of the periodic square box to every field of a snapshot: `turns`
    rotations by +90 degrees, then the mirror x -> -x if `mirror`, then a periodic shift of
    the grid by `shift` points. Fields keep their names; fields not listed are dropped."""
    out = {}
    for f in fields:
        a = snapshot[f.name]
        for _ in range(turns % 4):
            a = rotate90(a, f.kind)
        if mirror:
            a = mirror_x(a, f.kind)
        out[f.name] = np.roll(a, shift, axis=(0, 1))
    return out


def random_transform(snapshot: dict[str, np.ndarray], fields: list[Field],
                     rng: np.random.Generator, periodic: bool = True) -> dict[str, np.ndarray]:
    """A random element of D4 and, for a periodic box, a random periodic shift."""
    nx, ny = snapshot[fields[0].name].shape[:2]
    if nx != ny:
        raise ValueError("rotations by 90 degrees need a square grid")
    shift = (int(rng.integers(nx)), int(rng.integers(ny))) if periodic else (0, 0)
    return transform(snapshot, fields, int(rng.integers(4)), bool(rng.integers(2)), shift)
