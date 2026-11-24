"""The ensemble of runs: its configuration file, its snapshot files and their reading.

An ensemble is described by a TOML file (TOML: a plain text format of `key = value` lines,
read by Python's standard `tomllib`), for example:

    [ensemble]
    runs = "runs.csv"              # one row per run: a column `path`, one column per parameter
    parameters = ["eps", "g"]
    snapshots = "Data/*.h5"        # snapshot files inside each run folder (glob pattern)
    time_regex = '(\\d+)\\.h5$'     # the number in the file name ...
    time_unit = 1.0                # ... times time_unit is the time of the snapshot
    julia_order = true             # arrays written by Julia (column-major)
    dx = 0.785                     # grid spacing (or a column `dx` in the table of runs)
    periodic = true                # periodic boundary conditions
    last = 10                      # the last 10 snapshots of every run are described
    pair_dt = 25.0                 # time between the two snapshots of a pair (dynamics)
    # optional, see frames.py: the frame that is described in every run
    crop = 40.0                    # side of a centred square crop (physical units)
    dx_target = 1.57               # coarsen to this grid spacing (a multiple of dx)
    max_shift = 10.0               # moving correlation: shifts up to this in x and y
    min_wavelengths = 3            # non-periodic frames: S(k) from wavelength side / 3

    [[fields]]                     # one block per field
    name = "rho"
    keys = ["C", "rho"]            # names it may have in the files
    kind = "scalar"                # scalar, velocity, polarity, vector (role not given) or nematic
    relative = true

    [[fields]]
    name = "p"
    kind = "polarity"
    components = ["px", "py"]      # two scalar datasets, stacked as (x, y); or keys = ["P"]
    blocks = ["amp", "shape", "defects"]   # optional: observables to compute (default: by kind)

`phasewall inspect FILE` lists the datasets of a snapshot file and suggests these blocks
(inspect_snapshot, suggest_fields, fields_toml below). A scalar field whose name looks like a
component (vx, p_y, Qxx) is refused unless it says `single_component = true`.

Snapshot files are HDF5 files, which includes JLD2 files written by Julia. Julia stores
arrays in column-major order and HDF5 readers show their dimensions reversed, so with
`julia_order = true` every array is transposed on reading: [x, y] for a scalar,
[x, y, component] otherwise, as in Julia.

Unknown keys are errors (a misspelt `periodc = false` must not be ignored silently).
"""

import re
import tomllib
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .fields import KINDS, Field, component_of

MIN_WAVELENGTHS = 3.0      # non-periodic frames: S(k) only for wavelengths up to side / 3 (a choice)
FIELD_KEYS = {"name", "keys", "kind", "relative", "components", "single_component", "blocks"}

@dataclass
class Ensemble:
    """Everything phasewall needs to know about a set of runs."""

    runs: pd.DataFrame                      # columns: run, path (absolute), one per parameter[, dx]
    parameters: list[str]
    fields: list[Field]
    dx: float | None = None                 # grid spacing, unless the table of runs has a column dx
    snapshots: str = "Data/*.jld2"
    time_regex: str = r"(\d+)\.jld2?$"
    time_unit: float = 1.0
    julia_order: bool = True
    periodic: bool = True
    last: int = 10
    pair_dt: float | None = None
    crop: float | None = None               # frames.py
    dx_target: float | None = None
    max_shift: float | None = None
    min_wavelengths: float = MIN_WAVELENGTHS


def load_ensemble(path: str | Path) -> Ensemble:
    """Read the TOML configuration and the table of runs it points to. Relative paths (the
    table, and the `path` column of the table) are taken from the folder of the file that
    contains them."""
    path = Path(path)
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    e = dict(cfg["ensemble"])
    table_path = (path.parent / e.pop("runs")).resolve()
    runs = pd.read_csv(table_path)
    parameters = list(e.pop("parameters"))
    missing = [c for c in ["path", *parameters] if c not in runs.columns]
    if missing:
        raise ValueError(f"{table_path}: missing columns {missing}")
    runs["path"] = [str((table_path.parent / p).resolve()) for p in runs["path"]]
    if "run" not in runs.columns:
        runs.insert(0, "run", np.arange(len(runs)))
    if runs["run"].duplicated().any():
        raise ValueError(f"{table_path}: the column `run` must hold one value per run")
    known = {f.name for f in dataclass_fields(Ensemble)} - {"runs", "parameters", "fields"}
    unknown = sorted(set(e) - known)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown} in [ensemble]; known keys: "
                         f"runs, parameters, {', '.join(sorted(known))}")
    for f in cfg.get("fields", []):
        bad = sorted(set(f) - FIELD_KEYS)
        if bad:
            raise ValueError(f"{path}: unknown keys {bad} in [[fields]] {f.get('name', '?')!r}; "
                             f"known keys: {', '.join(sorted(FIELD_KEYS))}; kinds: {', '.join(KINDS)}")
    if not cfg.get("fields"):
        raise ValueError(f"{path}: no [[fields]] block")
    if "dx" not in e and "dx" not in runs.columns:
        raise ValueError(f"{path}: give the grid spacing, as `dx` in [ensemble] or as a column dx of {table_path}")
    if "periodic" in e and not isinstance(e["periodic"], bool):
        raise ValueError(f"{path}: `periodic` must be true or false")
    fields = [Field(f["name"], tuple(f.get("keys", [f["name"]])), f.get("kind", "scalar"),
                    bool(f.get("relative", False)), tuple(f.get("components", ())),
                    bool(f.get("single_component", False)), tuple(f.get("blocks", ()))) for f in cfg["fields"]]
    for f in fields:
        check_not_component(f, path)
    return Ensemble(runs=runs, parameters=parameters, fields=fields, **e)


def list_snapshots(run_dir: str | Path, pattern: str, time_regex: str,
                   time_unit: float = 1.0) -> list[tuple[float, Path]]:
    """Snapshot files of one run as (time, path), sorted by time. Files that match the glob
    pattern but not the regular expression are ignored."""
    rx = re.compile(time_regex)
    found = []
    for p in Path(run_dir).glob(pattern):
        m = rx.search(p.name)
        if m:
            found.append((int(m.group(1)) * time_unit, p))
    return sorted(found)


def load_any(path: str | Path):
    """The ensemble of a TOML file: io.Ensemble for snapshots ([ensemble]), oned.kymo.Ensemble1D
    for kymographs ([kymograph])."""
    text = Path(path).read_text()
    if "kymograph" in tomllib.loads(text):
        from .oned.kymo import load_ensemble1d
        return load_ensemble1d(path)
    return load_ensemble(path)


def dims(ens) -> int:
    """1 for a 1D ensemble (kymographs), 2 for snapshots."""
    return 1 if hasattr(ens, "windows") else 2


def read_dataset(f: h5py.File, key: str, julia_order: bool = True) -> np.ndarray:
    """One dataset of an open HDF5 file as float64, transposed if written by Julia (column-major).
    Shared by the snapshots (2D) and the kymographs (1D, oned/kymo.py)."""
    a = np.asarray(f[key][...], dtype=float)
    return a.T if julia_order else a


def read_snapshot(path: str | Path, fields: list[Field], julia_order: bool = True) -> dict[str, np.ndarray]:
    """The listed fields of one snapshot file, as float64 arrays indexed [x, y] (scalar) or
    [x, y, component]. For each field the first of its `keys` found in the file is used;
    KeyError if none is."""
    out = {}
    with h5py.File(path, "r") as f:
        for fl in fields:
            if fl.components:
                missing = [k for k in fl.components if not (k in f and isinstance(f[k], h5py.Dataset))]
                if missing:
                    raise KeyError(f"{path}: components {missing} not found (field {fl.name})")
                out[fl.name] = np.stack([read_dataset(f, k, julia_order) for k in fl.components], axis=-1)
            else:
                key = next((k for k in fl.keys if k in f and isinstance(f[k], h5py.Dataset)), None)
                if key is None:
                    raise KeyError(f"{path}: none of {list(fl.keys)} found (field {fl.name})")
                out[fl.name] = read_dataset(f, key, julia_order)
            fl.check(out[fl.name])
    return out


def write_snapshot(path: str | Path, snapshot: dict[str, np.ndarray], julia_order: bool = True) -> None:
    """Write arrays indexed [x, y] / [x, y, component] to an HDF5 file, laid out as Julia
    would write them if julia_order (the inverse of read_snapshot). Used by the demo."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, a in snapshot.items():
            f.create_dataset(name, data=a.T if julia_order else a)


def run_snapshots(ens: Ensemble, run_path: str | Path) -> list[tuple[float, Path]]:
    """Snapshots of one run of the ensemble."""
    return list_snapshots(run_path, ens.snapshots, ens.time_regex, ens.time_unit)


def load_table(path: str | Path, parameters: list[str], label: str | None = None,
               prefix: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    """A table of descriptors made elsewhere (for example by another program), as input to the walls and
    the other methods. Columns: `run` (one value per run), the parameters, optionally `time`
    (one row per snapshot; without it, one row per run), optionally a label column, and the
    descriptors: the columns named field.block.name, or else every numeric column (starting
    with `prefix` if given), renamed table.<prefix or x>.<name> as one block.
    Returns the table (run, time, parameters, descriptors) and the labels per run (Series
    indexed by run, the most common label of its rows) or None."""
    t = pd.read_csv(path)
    missing = [c for c in ["run", *parameters] if c not in t.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if "time" not in t.columns:
        t["time"] = 0.0
    named = [c for c in t.columns if c.count(".") >= 2]
    if not named:
        skip = {"run", "time", *parameters, *([label] if label else [])}
        cols = [c for c in t.columns if c not in skip and pd.api.types.is_numeric_dtype(t[c])
                and (prefix is None or c.startswith(prefix))]
        if not cols:
            raise ValueError(f"{path}: no descriptor column found")
        block = (prefix or "x").strip("_.") or "x"
        t = t.rename(columns={c: f"table.{block}.{c[len(prefix):] if prefix else c}" for c in cols})
        named = [f"table.{block}.{c[len(prefix):] if prefix else c}" for c in cols]
    labels = None
    if label:
        labels = t.groupby("run")[label].agg(lambda s: s.value_counts().idxmax())
    return t[["run", "time", *parameters, *named]], labels


# ---------------------------------------------------------------- fields: checks and suggestions
def check_not_component(f: Field, where: str | Path = "") -> None:
    """ValueError for a scalar field whose name or keys look like one component of a vector or
    tensor (vx, p_y, Qxx), unless it says single_component (see fields.py)."""
    if f.kind != "scalar" or f.single_component:
        return
    for name in (f.name, *f.keys):
        c = component_of(name)
        if c:
            stem, comp = c
            pair = ("xx", "xy") if comp in ("xx", "xy", "yy") else ("x", "y")
            names = [name[: len(name) - len(comp)] + q for q in pair]
            raise ValueError(
                f"{where}: field {f.name!r} looks like one component of {stem!r} ({name!r}). A component "
                f"alone is not invariant under rotations; declare the whole field instead, for example "
                f"components = {names} with kind = \"velocity\", \"polarity\", \"vector\" or \"nematic\". "
                f"If {name!r} is really a scalar of its own, add single_component = true.")


# names that suggest a role (a choice; suggestions are only shown, never applied)
VELOCITY_NAMES = {"v", "vel", "velocity", "u_vel"}
POLARITY_NAMES = {"p", "pol", "polarity"}
NEMATIC_NAMES = {"q", "nematic"}
DENSITY_NAMES = {"rho", "c", "density", "conc"}


def inspect_snapshot(path: str | Path, julia_order: bool = True) -> pd.DataFrame:
    """Every dataset of one snapshot file: name, shape as read by phasewall ([x, y] or
    [x, y, component], transposed if julia_order), shape in the file, dtype."""
    rows = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = tuple(obj.shape)
                rows.append({"name": name, "shape": shape[::-1] if julia_order else shape,
                             "file_shape": shape, "dtype": str(obj.dtype)})
        f.visititems(visit)
    return pd.DataFrame(rows, columns=["name", "shape", "file_shape", "dtype"])


def _role(stem: str, two_components: str) -> tuple[str, str]:
    s = stem.lower()
    if two_components == "tensor" or s in NEMATIC_NAMES:
        return "nematic", f"name {stem!r} suggests a nematic tensor"
    if s in VELOCITY_NAMES:
        return "velocity", f"name {stem!r} suggests a velocity"
    if s in POLARITY_NAMES:
        return "polarity", f"name {stem!r} suggests a polarity"
    return "vector", "two components; role unknown (velocity or polarity?)"


def suggest_fields(datasets: pd.DataFrame) -> list[dict]:
    """Suggested [[fields]] blocks for the datasets listed by inspect_snapshot: whole vectors
    (one dataset with a last dimension of 2), pairs of component datasets grouped into one
    field (vx + vy, Qxx + Qxy), and scalars (relative for density-like names). Each suggestion
    has a `reason`; datasets of other shapes are listed with kind None. Suggestions only: the
    user confirms every role."""
    numeric = datasets[datasets.dtype.str.startswith(("float", "int", "uint"))]
    two_d = {r.name: r.shape for r in numeric.itertuples() if len(r.shape) == 2}
    out, used = [], set()
    for r in datasets.itertuples():
        if r.name in used:
            continue
        shape = tuple(r.shape)
        if r.name not in set(numeric.name):
            out.append({"name": r.name, "kind": None, "reason": f"not numeric ({r.dtype})"})
        elif len(shape) == 3 and shape[2] == 2:
            kind, why = _role(r.name, "vector")
            out.append({"name": r.name, "kind": kind, "keys": [r.name], "reason": f"one dataset with 2 components; {why}"})
        elif len(shape) == 2:
            c = component_of(r.name)
            if c:
                stem, comp = c
                pair = ("xx", "xy") if comp in ("xx", "xy", "yy") else ("x", "y")
                names = [r.name[: len(r.name) - len(comp)] + q for q in pair]
                if comp == "yy" and any(n in used for n in names):
                    out.append({"name": r.name, "kind": None, "reason": "Q_yy = -Q_xx, not needed"})
                    continue
                if all(n in two_d and two_d[n] == shape for n in names):
                    kind, why = _role(stem.rstrip("_"), "tensor" if pair[0] == "xx" else "vector")
                    out.append({"name": stem.rstrip("_"), "kind": kind, "components": names,
                                "reason": f"components {names[0]}, {names[1]} grouped; {why}"})
                    used.update(names)
                    continue
                out.append({"name": r.name, "kind": None,
                            "reason": "looks like a component, but its partner is missing: declare it by hand"})
                continue
            relative = r.name.lower() in DENSITY_NAMES
            out.append({"name": r.name, "kind": "scalar", "keys": [r.name], "relative": relative,
                        "reason": "scalar" + ("; density-like name, described relative to its mean" if relative else "")})
        else:
            out.append({"name": r.name, "kind": None, "reason": f"shape {shape} is not a field"})
        used.add(r.name)
    return out


def fields_toml(suggestions: list[dict]) -> str:
    """The suggested fields as [[fields]] blocks of a TOML configuration, with the reason as a
    comment; datasets that are not fields are listed as comments."""
    lines = []
    for s in suggestions:
        if s["kind"] is None:
            lines.append(f"# {s['name']}: not used ({s['reason']})")
            continue
        lines += ["", "[[fields]]", f"# {s['reason']}", f'name = "{s["name"]}"']
        if "components" in s:
            lines.append("components = [" + ", ".join(f'"{c}"' for c in s["components"]) + "]")
        else:
            lines.append("keys = [" + ", ".join(f'"{k}"' for k in s["keys"]) + "]")
        lines.append(f'kind = "{s["kind"]}"')
        if s.get("relative"):
            lines.append("relative = true")
    return "\n".join(lines).strip() + "\n"


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        return "'" + v + "'" if "\\" in v and "'" not in v else '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {v!r} in TOML")


def write_config(path: str | Path, runs: str | Path, parameters: list[str], fields: list[Field], **ensemble) -> Path:
    """Write the TOML configuration of an ensemble (the inverse of load_ensemble): `runs` is the
    table of runs (written as an absolute path), `ensemble` the other keys of [ensemble] (None
    values are left out). The file is read back with load_ensemble, so a mistake is reported here;
    a file that already holds the same text is not rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[ensemble]", f"runs = {_toml_value(str(Path(runs).resolve()))}",
             f"parameters = {_toml_value(list(parameters))}"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in ensemble.items() if v is not None]
    for f in fields:
        lines += ["", "[[fields]]", f"name = {_toml_value(f.name)}", f"kind = {_toml_value(f.kind)}"]
        if f.components:
            lines.append(f"components = {_toml_value(list(f.components))}")
        else:
            lines.append(f"keys = {_toml_value(list(f.keys))}")
        if f.relative:
            lines.append("relative = true")
        if f.single_component:
            lines.append("single_component = true")
        if f.blocks:
            lines.append(f"blocks = {_toml_value(list(f.blocks))}")
    text = "\n".join(lines) + "\n"
    if not path.exists() or path.read_text() != text:   # an unchanged file is not touched: caches keyed
        path.write_text(text)                            # on its modification time stay valid
    load_ensemble(path)
    return path
