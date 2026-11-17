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
    kind = "scalar"
    relative = true

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

from .fields import KINDS, Field

MIN_WAVELENGTHS = 3.0      # non-periodic frames: S(k) only for wavelengths up to side / 3 (a choice)
FIELD_KEYS = {"name", "keys", "kind", "relative"}

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
                    bool(f.get("relative", False))) for f in cfg["fields"]]
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


def read_snapshot(path: str | Path, fields: list[Field], julia_order: bool = True) -> dict[str, np.ndarray]:
    """The listed fields of one snapshot file, as float64 arrays indexed [x, y] (scalar) or
    [x, y, component]. For each field the first of its `keys` found in the file is used;
    KeyError if none is."""
    out = {}
    with h5py.File(path, "r") as f:
        for fl in fields:
            key = next((k for k in fl.keys if k in f and isinstance(f[k], h5py.Dataset)), None)
            if key is None:
                raise KeyError(f"{path}: none of {list(fl.keys)} found (field {fl.name})")
            a = np.asarray(f[key][...], dtype=float)
            out[fl.name] = a.T if julia_order else a
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
