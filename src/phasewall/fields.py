"""Field types and how they change under the symmetries of a square periodic grid.

A snapshot is a dict {name: array}. Arrays are indexed [x, y] for a scalar field and
[x, y, component] for the other kinds:

- "scalar": density, concentration of nucleators, ...: (nx, ny).
- "velocity", "polarity", or "vector" when the role is not given: (nx, ny, 2), components
  (a_x, a_y). The three turn alike under rotations and mirrors (their `symmetry` is "vector");
  the role only says which observables are worth reading (describe.py).
- "nematic": traceless symmetric tensor Q stored by its two independent components
  (Q_xx, Q_xy), with Q_yy = -Q_xx: (nx, ny, 2). For a director at angle theta,
  (Q_xx, Q_xy) = S (cos 2 theta, sin 2 theta) / 2.

A vector or nematic field is read either from one dataset holding both components, or from
two scalar datasets (`components`, for example ("vx", "vy")), stacked in that order. A
component must never be described as a scalar of its own: a rotation by 90 degrees sends
v_x to -v_y, so v_x alone is not invariant. `component_of` recognises such names (vx, p_y,
Qxx, ...), and io.load_ensemble refuses a scalar field with such a name unless the field says
`single_component`.

The square grid has eight symmetries (the group D4): rotations by multiples of 90 degrees,
each with or without a mirror. Turning the grid is not enough: the vectors and tensors on it
must turn too. Rotation by +90 degrees (x towards y) sends a vector (a_x, a_y) to
(-a_y, a_x) and turns a director by 90 degrees, so 2 theta by 180 degrees: Q -> -Q.
The mirror x -> -x sends a_x to -a_x, and theta to pi - theta: Q_xx unchanged, Q_xy -> -Q_xy.
"""

import re
from dataclasses import dataclass

import numpy as np

KINDS = ("scalar", "vector", "velocity", "polarity", "nematic")
SYMMETRY = {"scalar": "scalar", "vector": "vector", "velocity": "vector", "polarity": "vector",
            "nematic": "nematic"}
# a name made of a short stem and a component suffix: vx, Vy, p_x, Qxx, Q_xy (a choice: stems of
# 1 to 3 letters, so that names such as "rho" or "phi" are not taken for components)
COMPONENT_NAME = re.compile(r"^(?P<stem>[A-Za-z]{1,3}?)_?(?P<comp>xx|xy|yy|x|y)$")


# Blocks of observables of the hand-made encoder (describe.py) and the kinds they apply to.
# DEFAULT_BLOCKS: what is computed when a field does not choose (a choice: the role-specific
# blocks are on for their role; a "vector" of unknown role gets only the common ones).
VECTORS = ("vector", "velocity", "polarity")
BLOCKS = {
    "amp": KINDS, "shape": KINDS, "mom": KINDS, "topo": KINDS, "dyn": KINDS,
    "order": (*VECTORS, "nematic"),
    "flow": VECTORS,                          # divergence and curl
    "vortex": ("velocity", "vector"),         # Okubo-Weiss vortex cores
    "defects": ("polarity", "vector", "nematic"),   # winding-number defects (+-1, or +-1/2 for nematics)
}
COMMON = ("amp", "shape", "mom", "topo")
DEFAULT_BLOCKS = {
    "scalar": (*COMMON, "dyn"),
    "vector": (*COMMON, "order", "flow", "dyn"),
    "velocity": (*COMMON, "order", "flow", "vortex", "dyn"),
    "polarity": (*COMMON, "order", "flow", "defects", "dyn"),
    "nematic": (*COMMON, "order", "defects", "dyn"),
}


def symmetry_of(kind: str) -> str:
    """How a field of this kind turns: "scalar", "vector" or "nematic"."""
    if kind not in SYMMETRY:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    return SYMMETRY[kind]


def component_of(name: str) -> tuple[str, str] | None:
    """(stem, component) if the name looks like one component of a vector or tensor
    ("vx" -> ("v", "x"), "Q_xy" -> ("Q", "xy")), else None."""
    m = COMPONENT_NAME.match(name)
    return (m["stem"], m["comp"]) if m else None


@dataclass(frozen=True)
class Field:
    """One field of a snapshot.

    name: the name used inside phasewall (for example "rho", "p").
    keys: names the field may have in the files, tried in order (for example ["C", "rho"]),
    each a dataset with all components of the field.
    kind: "scalar", "vector", "velocity", "polarity" or "nematic".
    relative: scalars only; describe (f - <f>) / <f> instead of f, so that the mean value,
    which often changes across the parameter grid (the mean density), does not by itself
    separate the runs.
    components: vector and nematic only; the two scalar datasets that hold the components, in
    the order (x, y) or (xx, xy); used instead of `keys`.
    single_component: scalars only; the user confirms that a name such as "vx" is meant as a
    scalar of its own (see the module docstring).
    blocks: the blocks of observables computed for this field by the hand-made encoder
    (names of BLOCKS that apply to its kind); empty: DEFAULT_BLOCKS of its kind.
    """

    name: str
    keys: tuple[str, ...]
    kind: str = "scalar"
    relative: bool = False
    components: tuple[str, ...] = ()
    single_component: bool = False
    blocks: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"field {self.name}: kind must be one of {KINDS}, not {self.kind!r}")
        if self.relative and self.kind != "scalar":
            raise ValueError(f"field {self.name}: 'relative' is only for scalar fields")
        if self.components and (self.kind == "scalar" or len(self.components) != 2):
            raise ValueError(f"field {self.name}: 'components' takes two dataset names, for a vector or "
                             f"nematic field")
        if self.single_component and self.kind != "scalar":
            raise ValueError(f"field {self.name}: 'single_component' is only for scalar fields")
        wrong = [b for b in self.blocks if self.kind not in BLOCKS.get(b, ())]
        if wrong:
            allowed = [b for b, kinds in BLOCKS.items() if self.kind in kinds]
            raise ValueError(f"field {self.name}: blocks {wrong} do not apply to a {self.kind} field; "
                             f"blocks for this kind: {allowed}")

    @property
    def selected(self) -> tuple[str, ...]:
        """The blocks computed for this field: its own choice, or the defaults of its kind."""
        return self.blocks or DEFAULT_BLOCKS[self.kind]

    @property
    def symmetry(self) -> str:
        """How the field turns: "scalar", "vector" or "nematic"."""
        return SYMMETRY[self.kind]

    def check(self, a: np.ndarray) -> None:
        """Raise ValueError if the array does not have the shape of this kind of field."""
        if self.symmetry == "scalar" and a.ndim != 2:
            raise ValueError(f"field {self.name}: a scalar field needs shape (nx, ny), got {a.shape}")
        if self.symmetry != "scalar" and (a.ndim != 3 or a.shape[2] != 2):
            raise ValueError(f"field {self.name}: a {self.kind} field needs shape (nx, ny, 2), "
                             f"got {a.shape}")


def rotate90(a: np.ndarray, kind: str) -> np.ndarray:
    """The field turned by +90 degrees (x towards y), grid and components together."""
    kind = symmetry_of(kind)
    b = np.rot90(a, 1, axes=(0, 1)).copy()
    if kind == "vector":
        b = np.stack([-b[..., 1], b[..., 0]], axis=-1)
    elif kind == "nematic":
        b = -b
    return b


def mirror_x(a: np.ndarray, kind: str) -> np.ndarray:
    """The field mirrored by x -> -x, grid and components together."""
    kind = symmetry_of(kind)
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
