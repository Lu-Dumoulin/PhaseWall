"""Tests of io.py: snapshot files written and read back, the Julia layout, the snapshot list
and the TOML configuration of an ensemble."""

import h5py
import numpy as np
import pandas as pd
import pytest

from phasewall.fields import Field
from phasewall.io import list_snapshots, load_ensemble, read_snapshot, write_snapshot

RHO = Field("rho", ("C", "rho"))
P = Field("p", ("P",), "vector")


def snapshot(nx: int = 6, ny: int = 4) -> dict[str, np.ndarray]:
    """Arrays of different sizes along x and y, so that a transposition shows."""
    rng = np.random.default_rng(0)
    return {"rho": rng.random((nx, ny)), "P": rng.random((nx, ny, 2))}


def test_round_trip_in_julia_layout(tmp_path):
    snap = snapshot()
    write_snapshot(tmp_path / "s.h5", snap)
    with h5py.File(tmp_path / "s.h5") as f:          # what an HDF5 reader shows for Julia arrays
        assert f["rho"].shape == (4, 6)
        assert f["P"].shape == (2, 4, 6)
    back = read_snapshot(tmp_path / "s.h5", [RHO, P])
    assert np.array_equal(back["rho"], snap["rho"])
    assert np.array_equal(back["p"], snap["P"])


def test_round_trip_without_julia_layout(tmp_path):
    snap = snapshot()
    write_snapshot(tmp_path / "s.h5", snap, julia_order=False)
    with h5py.File(tmp_path / "s.h5") as f:
        assert f["P"].shape == (6, 4, 2)
    back = read_snapshot(tmp_path / "s.h5", [RHO, P], julia_order=False)
    assert np.array_equal(back["p"], snap["P"])


def test_first_key_found_is_used_and_missing_fields_raise(tmp_path):
    snap = snapshot()
    write_snapshot(tmp_path / "s.h5", {"C": snap["rho"], "rho": 2 * snap["rho"]})
    assert np.array_equal(read_snapshot(tmp_path / "s.h5", [RHO])["rho"], snap["rho"])
    with pytest.raises(KeyError):
        read_snapshot(tmp_path / "s.h5", [P])


def test_wrong_shape_raises(tmp_path):
    write_snapshot(tmp_path / "s.h5", {"P": np.zeros((4, 4, 3))})
    with pytest.raises(ValueError):
        read_snapshot(tmp_path / "s.h5", [P])


def test_list_snapshots_sorts_by_time_and_ignores_other_files(tmp_path):
    for name in ["0000000200.h5", "0000000100.h5", "0000001000.h5", "notes.h5", "0000000300.txt"]:
        (tmp_path / name).touch()
    found = list_snapshots(tmp_path, "*.h5", r"(\d+)\.h5$", time_unit=0.5)
    assert [t for t, _ in found] == [50.0, 100.0, 500.0]
    assert found[0][1].name == "0000000100.h5"


def write_config(folder, runs: pd.DataFrame, extra: str = "") -> None:
    runs.to_csv(folder / "runs.csv", index=False)
    (folder / "ens.toml").write_text(f"""
[ensemble]
runs = "runs.csv"
parameters = ["a", "b"]
snapshots = "Data/*.h5"
time_regex = '(\\d+)\\.h5$'
dx = 0.5
last = 3
pair_dt = 2.0
{extra}

[[fields]]
name = "rho"
keys = ["C", "rho"]
relative = true

[[fields]]
name = "p"
keys = ["P"]
kind = "vector"
""")


def test_load_ensemble(tmp_path):
    (tmp_path / "sub").mkdir()
    runs = pd.DataFrame({"path": ["run0", "run1"], "a": [0.1, 0.2], "b": [1, 1]})
    runs.to_csv(tmp_path / "sub" / "runs.csv", index=False)
    write_config(tmp_path, runs, extra="note = 'any other key is kept'")
    # the table is read from the folder of the TOML file, the run paths from that of the table
    cfg = (tmp_path / "ens.toml").read_text().replace('runs = "runs.csv"', 'runs = "sub/runs.csv"')
    (tmp_path / "ens.toml").write_text(cfg)
    ens = load_ensemble(tmp_path / "ens.toml")
    assert ens.parameters == ["a", "b"]
    assert list(ens.runs["run"]) == [0, 1]
    assert ens.runs["path"].iloc[0] == str((tmp_path / "sub" / "run0").resolve())
    assert [(f.name, f.keys, f.kind, f.relative) for f in ens.fields] == [
        ("rho", ("C", "rho"), "scalar", True), ("p", ("P",), "vector", False)]
    assert (ens.dx, ens.last, ens.pair_dt, ens.julia_order) == (0.5, 3, 2.0, True)
    assert ens.extra == {"note": "any other key is kept"}


def test_load_ensemble_rejects_bad_tables(tmp_path):
    write_config(tmp_path, pd.DataFrame({"path": ["r0"], "a": [0.1]}))
    with pytest.raises(ValueError, match="missing columns"):
        load_ensemble(tmp_path / "ens.toml")
    write_config(tmp_path, pd.DataFrame({"run": [3, 3], "path": ["r0", "r1"], "a": [0, 1], "b": [0, 0]}))
    with pytest.raises(ValueError, match="one value per run"):
        load_ensemble(tmp_path / "ens.toml")
