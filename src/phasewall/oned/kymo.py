"""One-dimensional systems: fields, the ensemble of runs, kymograph files and time windows.

A run of a 1D system is stored as one kymograph per field: an array a[x, t] (one component) or
a[x, t, component]. The ensemble is described by a TOML file, for example:

    [kymograph]
    runs = "runs.csv"            # one row per run: a column `path`, one column per parameter
    parameters = ["b", "c"]
    file = "kymograph.h5"        # the kymograph file inside each run folder (HDF5, so JLD2 too);
                                 # file = "": the column `path` gives the file of each run itself
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

    def file_of(self, path: str | Path) -> Path:
        """The kymograph file of a run: `path` itself when `file` is empty (one file per run, the
        table of runs gives the files), else the file `file` inside the run folder `path`."""
        return Path(path) / self.file if self.file else Path(path)

    def dx_of(self, run: int) -> float:
        if "dx" in self.runs.columns:
            return float(self.runs.loc[self.runs.run == run, "dx"].iloc[0])
        return float(self.dx)


def load_ensemble1d(path: str | Path) -> Ensemble1D:
    """Read a 1D ensemble from its TOML file (module docstring). The table of runs is relative to
    the TOML file, the paths in the table relative to the table (as for the 2D ensembles)."""
    path = Path(path)
    cfg = tomllib.loads(path.read_text())
    if "kymograph" not in cfg:
        raise ValueError(f"{path}: no [kymograph] section (a 2D ensemble uses [ensemble])")
    k = cfg["kymograph"]
    unknown = set(k) - RUN_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys in [kymograph]: {sorted(unknown)}")
    table_path = (path.parent / k["runs"]).resolve()
    runs = pd.read_csv(table_path)
    if "run" not in runs.columns:
        runs.insert(0, "run", range(len(runs)))
    runs["path"] = [str((table_path.parent / p).resolve()) for p in runs["path"]]   # relative to the table, as in 2D
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


# names that suggest a role on a line (a choice; suggestions are only shown, never applied)
VELOCITY_NAMES_1D = {"v", "vel", "velocity", "vx", "v_x", "flow"}
POLARITY_NAMES_1D = {"p", "pol", "polarity", "px", "p_x"}
DENSITY_NAMES_1D = {"rho", "c", "n", "density", "conc", "rho1", "rho2", "c1", "c2"}
COMPLEX_PARTS = (("re", "im"), ("_re", "_im"), ("real", "imag"), ("_real", "_imag"))


def suggest_order(shape: tuple[int, ...]) -> tuple[str, str]:
    """A first guess of the order of the axes of a kymograph array as read (shape (n1, n2)): "xt" if
    the second axis is the longer (runs usually save more times than grid points), else "tx", with
    the reason. Only a suggestion: the interface shows the kymograph so that you can check it."""
    n1, n2 = shape[:2]
    if n2 >= n1:
        return "xt", f"{n2} >= {n1}: the longer axis is taken as time"
    return "tx", f"{n1} > {n2}: the longer axis is taken as time"


def suggest_fields1d(datasets: pd.DataFrame) -> list[dict]:
    """Suggested 1D fields for the datasets of one kymograph file (io.inspect_snapshot): pairs of
    datasets grouped into one field (Are + Aim as a complex amplitude, px + py as a two-component
    polarity), velocity- and polarity-like names, scalars (relative for density-like names). Each has
    a `reason`; datasets that are not 2D numeric arrays get kind None. The user confirms every kind."""
    from ..fields import component_of
    numeric = datasets[datasets.dtype.str.startswith(("float", "int", "uint"))]
    two_d = {r.name: tuple(r.shape) for r in numeric.itertuples() if len(r.shape) == 2 and min(r.shape) > 1}
    out, used = [], set()
    for name, shape in two_d.items():                           # first the pairs, whatever the order of the names
        if name in used:
            continue
        low, pair = name.lower(), None
        for re_, im_ in COMPLEX_PARTS:                          # Are / Aim, A_re / A_im, ...
            if low.endswith(re_):
                stem = name[: len(name) - len(re_)]
                other = next((d for d in two_d if d.lower() == (stem + im_).lower()), None)
                if stem and other and two_d[other] == shape:
                    pair = ("complex", stem, [name, other], "real and imaginary parts grouped as a complex amplitude")
                    break
        c = component_of(name)
        if pair is None and c and c[1] == "x":
            other = next((d for d, sh in two_d.items() if component_of(d) == (c[0], "y") and sh == shape), None)
            if other:
                pair = ("polarity", c[0], [name, other],
                        "components along (x) and across (y) the line grouped as a polarity")
        if pair:
            kind, stem, comps, why = pair
            out.append({"name": stem.rstrip("_") or name, "kind": kind, "components": comps, "reason": why})
            used.update(comps)
    for r in datasets.itertuples():
        name = r.name
        if name in used:
            continue
        used.add(name)
        if name not in two_d:
            why = (f"not numeric ({r.dtype})" if name not in set(numeric.name)
                   else f"shape {tuple(r.shape)} is not a kymograph (two axes of several points)")
            out.append({"name": name, "kind": None, "reason": why})
        elif name.lower() in VELOCITY_NAMES_1D:
            out.append({"name": name, "kind": "velocity", "keys": [name], "reason": f"name {name!r} suggests a velocity"})
        elif name.lower() in POLARITY_NAMES_1D:
            out.append({"name": name, "kind": "polarity", "keys": [name], "reason": f"name {name!r} suggests a polarity"})
        else:
            relative = name.lower() in DENSITY_NAMES_1D
            out.append({"name": name, "kind": "scalar", "keys": [name], "relative": relative,
                        "reason": "scalar" + ("; density-like name, described relative to its mean" if relative else "")})
    return out


def write_config1d(path: str | Path, runs: str | Path, parameters: list[str], fields: list[Field1D], **kymograph) -> Path:
    """Write the [kymograph] TOML of a 1D ensemble (the inverse of load_ensemble1d): `runs` is the
    table of runs (written as an absolute path), `kymograph` the other keys of [kymograph] (None
    values are left out; file = "" when the table gives the file of each run). Read back with
    load_ensemble1d, so a mistake is reported here; an unchanged file is not rewritten."""
    from ..io import _toml_value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[kymograph]", f"runs = {_toml_value(str(Path(runs).resolve()))}",
             f"parameters = {_toml_value(list(parameters))}"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in kymograph.items() if v is not None]
    for f in fields:
        lines += ["", "[[fields]]", f"name = {_toml_value(f.name)}", f"kind = {_toml_value(f.kind)}"]
        if f.components:
            lines.append(f"components = {_toml_value(list(f.components))}")
        else:
            lines.append(f"keys = {_toml_value(list(f.keys))}")
        if f.relative:
            lines.append("relative = true")
        if f.blocks:
            lines.append(f"blocks = {_toml_value(list(f.blocks))}")
    text = "\n".join(lines) + "\n"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    load_ensemble1d(path)
    return path
