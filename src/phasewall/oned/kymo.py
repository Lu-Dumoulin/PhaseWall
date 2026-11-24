"""One-dimensional systems: fields, the ensemble of runs, kymograph files and time windows.

A run of a 1D system is stored as one kymograph per field: an array a[x, t] (one component) or
a[x, t, component]. The ensemble is described by a TOML file, for example:

    [kymograph]
    runs = "runs.csv"            # one row per run: a column `path`, one column per parameter
    parameters = ["b", "c"]
    file = "kymograph.h5"        # the kymograph file inside each run folder (HDF5, so JLD2 too)
    julia_order = true           # arrays written by Julia (column-major): transposed on reading
    order = "xt"                 # axes of an array once read: "xt" (x first) or "tx" (t first)
    dx = 0.78                    # grid spacing (or a column `dx` in the table of runs)
    dt = 0.5                     # time between two rows of the kymograph
    t0 = 0.0                     # time of the first row
    periodic = true              # periodic in x
    share = 0.5                  # the last half of every run is described ...
    windows = 8                  # ... cut into 8 windows of equal length, one row of the table each

    [[fields]]
    name = "A"
    kind = "complex"             # scalar, velocity, polarity or complex
    components = ["Are", "Aim"]  # two datasets stacked as components; or keys = ["A"] for one

Kinds and their symmetry under the mirror x -> -x (the only symmetry of a line besides shifts):
- scalar: one component, unchanged (density, concentration); `relative = true` describes
  (a - mean) / mean.
- velocity: one component along the line, changes sign.
- polarity: one component p_x (changes sign), or two (p_x, p_y): p_x along the line changes
  sign, p_y across it does not.
- complex: an amplitude A = (Re A, Im A) whose global phase does not matter (the complex
  Ginzburg-Landau equation): unchanged by the mirror, and every descriptor is unchanged by
  A -> A e^{i phi}.

Unknown keys are errors, as for the 2D ensembles (a misspelt key must not be ignored silently).
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..io import read_dataset, write_snapshot

KINDS = ("scalar", "velocity", "polarity", "complex")
N_COMPONENTS = {"scalar": (1,), "velocity": (1,), "polarity": (1, 2), "complex": (2,)}
ORDERS = ("xt", "tx")
SHARE = 0.5           # default: the last half of each run is described (Ludo, 2026-11-23; a choice)
WINDOWS = 8           # default: windows per run (a choice: enough rows for halves by time)
MIN_WINDOW_ROWS = 16  # a window holds at least 16 rows of the kymograph (a choice: spectra in time)
RUN_KEYS = {"runs", "parameters", "file", "julia_order", "order", "dx", "dt", "t0", "periodic", "share",
            "windows"}
FIELD_KEYS = {"name", "keys", "kind", "relative", "components", "blocks"}


@dataclass(frozen=True)
class Field1D:
    """A field of a 1D system. keys: names the dataset may have in the files (one component);
    components: names of the datasets of each component, stacked in this order (p_x then p_y;
    Re A then Im A); blocks: the observables to compute (None: the defaults of the kind)."""

    name: str
    keys: tuple[str, ...] = ()
    kind: str = "scalar"
    relative: bool = False
    components: tuple[str, ...] = ()
    blocks: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"field {self.name}: kind must be one of {KINDS}, not {self.kind!r}")
        n = len(self.components) or 1
        if n not in N_COMPONENTS[self.kind]:
            raise ValueError(f"field {self.name}: a {self.kind} has {N_COMPONENTS[self.kind]} components, not {n}")
        if self.relative and self.kind != "scalar":
            raise ValueError(f"field {self.name}: only a scalar can be relative")
        if not self.keys and not self.components:
            object.__setattr__(self, "keys", (self.name,))

    @property
    def n_components(self) -> int:
        return len(self.components) or 1

    @property
    def odd(self) -> tuple[bool, ...]:
        """For each component: does it change sign under the mirror x -> -x?"""
        if self.kind in ("velocity", "polarity"):
            return (True,) + (False,) * (self.n_components - 1)
        return (False,) * self.n_components

    @property
    def selected(self) -> tuple[str, ...]:
        """The blocks of observables computed for this field (describe1d.blocks_of), as Field.selected in 2D."""
        from .describe1d import blocks_of
        return blocks_of(self)


@dataclass
class Ensemble1D:
    """Everything PhaseWall needs to know about a set of 1D runs."""

    runs: pd.DataFrame                  # columns: run, path (absolute), one per parameter[, dx]
    parameters: list[str]
    fields: list[Field1D]
    dt: float
    dx: float | None = None             # unless the table of runs has a column dx
    file: str = "kymograph.h5"
    julia_order: bool = True
    order: str = "xt"
    t0: float = 0.0
    periodic: bool = True
    share: float = SHARE
    windows: int = WINDOWS

    def __post_init__(self):
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, not {self.order!r}")
        if not 0 < self.share <= 1:
            raise ValueError(f"share must be in (0, 1], not {self.share}")
        if self.windows < 1:
            raise ValueError("windows must be at least 1")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.dx is None and "dx" not in self.runs.columns:
            raise ValueError("dx is needed, in [kymograph] or as a column of the table of runs")

    def dx_of(self, run: int) -> float:
        if "dx" in self.runs.columns:
            return float(self.runs.loc[self.runs.run == run, "dx"].iloc[0])
        return float(self.dx)


def load_ensemble1d(path: str | Path) -> Ensemble1D:
    """Read a 1D ensemble from its TOML file (module docstring); paths are relative to it."""
    path = Path(path)
    cfg = tomllib.loads(path.read_text())
    if "kymograph" not in cfg:
        raise ValueError(f"{path}: no [kymograph] section (a 2D ensemble uses [ensemble])")
    k = cfg["kymograph"]
    unknown = set(k) - RUN_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys in [kymograph]: {sorted(unknown)}")
    runs = pd.read_csv(path.parent / k["runs"])
    if "run" not in runs.columns:
        runs.insert(0, "run", range(len(runs)))
    runs["path"] = [str((path.parent / p).resolve()) for p in runs["path"]]
    fields = []
    for f in cfg.get("fields", []):
        unknown = set(f) - FIELD_KEYS
        if unknown:
            raise ValueError(f"{path}: unknown keys in [[fields]] {f.get('name')}: {sorted(unknown)}")
        fields.append(Field1D(f["name"], tuple(f.get("keys", ())), f.get("kind", "scalar"),
                              bool(f.get("relative", False)), tuple(f.get("components", ())),
                              tuple(f["blocks"]) if "blocks" in f else None))
    if not fields:
        raise ValueError(f"{path}: no [[fields]]")
    if not isinstance(k.get("periodic", True), bool):
        raise TypeError(f"{path}: periodic must be true or false")
    return Ensemble1D(runs, list(k["parameters"]), fields, float(k["dt"]), k.get("dx"), k.get("file", "kymograph.h5"),
                      bool(k.get("julia_order", True)), k.get("order", "xt"), float(k.get("t0", 0.0)),
                      k.get("periodic", True), float(k.get("share", SHARE)), int(k.get("windows", WINDOWS)))


def _read(f: h5py.File, key: str, julia_order: bool, order: str) -> np.ndarray:
    a = read_dataset(f, key, julia_order)
    if a.ndim != 2:
        raise ValueError(f"dataset {key}: a kymograph is a 2D array, not {a.ndim}D")
    return a if order == "xt" else a.T


def read_kymograph(path: str | Path, fields: list[Field1D], julia_order: bool = True, order: str = "xt"
                   ) -> dict[str, np.ndarray]:
    """The listed fields of one kymograph file as float64 arrays indexed [x, t] (one
    component) or [x, t, component]. KeyError if a field is missing."""
    out = {}
    with h5py.File(path, "r") as f:
        for fl in fields:
            if fl.components:
                missing = [c for c in fl.components if c not in f]
                if missing:
                    raise KeyError(f"{path}: components {missing} not found (field {fl.name})")
                out[fl.name] = np.stack([_read(f, c, julia_order, order) for c in fl.components], axis=-1)
            else:
                key = next((k for k in fl.keys if k in f), None)
                if key is None:
                    raise KeyError(f"{path}: none of {list(fl.keys)} found (field {fl.name})")
                out[fl.name] = _read(f, key, julia_order, order)
    shapes = {a.shape[:2] for a in out.values()}
    if len(shapes) > 1:
        raise ValueError(f"{path}: fields of different shapes {sorted(shapes)}")
    return out


def write_kymograph(path: str | Path, arrays: dict[str, np.ndarray], julia_order: bool = True,
                    order: str = "xt") -> None:
    """Write arrays indexed [x, t] to an HDF5 file, as read_kymograph reads them back (the
    inverse; used by the example)."""
    write_snapshot(path, {name: a if order == "xt" else a.T for name, a in arrays.items()}, julia_order)


def window_slices(n_t: int, share: float = SHARE, windows: int = WINDOWS) -> list[slice]:
    """The rows of each window: the last `share` of the n_t rows cut into `windows` contiguous
    windows of equal length (rows left over go to the earliest part, not described).
    ValueError if a window would hold fewer than MIN_WINDOW_ROWS rows."""
    used = int(np.floor(share * n_t))
    length = used // windows
    if length < MIN_WINDOW_ROWS:
        raise ValueError(f"{windows} windows in the last {share:g} of {n_t} rows hold {length} rows each, "
                         f"fewer than {MIN_WINDOW_ROWS}: use fewer windows or a larger share")
    start = n_t - windows * length
    return [slice(start + i * length, start + (i + 1) * length) for i in range(windows)]
