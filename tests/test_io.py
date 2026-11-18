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
    write_config(tmp_path, runs, extra="crop = 2.0\nmax_shift = 0.5")
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
    assert (ens.crop, ens.max_shift, ens.dx_target, ens.min_wavelengths) == (2.0, 0.5, None, 3.0)


def test_load_ensemble_rejects_bad_tables(tmp_path):
    write_config(tmp_path, pd.DataFrame({"path": ["r0"], "a": [0.1]}))
    with pytest.raises(ValueError, match="missing columns"):
        load_ensemble(tmp_path / "ens.toml")
    write_config(tmp_path, pd.DataFrame({"run": [3, 3], "path": ["r0", "r1"], "a": [0, 1], "b": [0, 0]}))
    with pytest.raises(ValueError, match="one value per run"):
        load_ensemble(tmp_path / "ens.toml")


def test_unknown_keys_and_missing_settings_raise(tmp_path):
    runs = pd.DataFrame({"path": ["r0"], "a": [0.1], "b": [1]})
    write_config(tmp_path, runs, extra="periodc = false")
    with pytest.raises(ValueError, match="unknown keys \\['periodc'\\]"):
        load_ensemble(tmp_path / "ens.toml")
    write_config(tmp_path, runs, extra="periodic = 0")
    with pytest.raises(ValueError, match="true or false"):
        load_ensemble(tmp_path / "ens.toml")
    write_config(tmp_path, runs)
    cfg = (tmp_path / "ens.toml").read_text()
    (tmp_path / "ens.toml").write_text(cfg.replace('kind = "vector"', 'kind = "vector"\nrelatve = true'))
    with pytest.raises(ValueError, match="relatve"):
        load_ensemble(tmp_path / "ens.toml")
    (tmp_path / "ens.toml").write_text(cfg.replace("dx = 0.5\n", ""))
    with pytest.raises(ValueError, match="grid spacing"):
        load_ensemble(tmp_path / "ens.toml")


def test_grid_spacing_per_run(tmp_path):
    runs = pd.DataFrame({"path": ["r0", "r1"], "a": [0.1, 0.2], "b": [1, 1], "dx": [0.5, 0.25]})
    write_config(tmp_path, runs)
    (tmp_path / "ens.toml").write_text((tmp_path / "ens.toml").read_text().replace("dx = 0.5\n", ""))
    ens = load_ensemble(tmp_path / "ens.toml")
    assert ens.dx is None and ens.runs.dx.tolist() == [0.5, 0.25]


def test_write_config_round_trip(tmp_path):
    from phasewall.io import write_config
    pd.DataFrame({"path": ["r0"], "a": [0.1], "b": [1]}).to_csv(tmp_path / "runs.csv", index=False)
    fields = [Field("rho", ("C", "rho"), relative=True), Field("v", (), "velocity", components=("vx", "vy")),
              Field("Q", ("Q",), "nematic", blocks=("amp", "defects"))]
    path = write_config(tmp_path / "cfg" / "ens.toml", tmp_path / "runs.csv", ["a", "b"], fields,
                        snapshots="Data/*.h5", time_regex=r"(\d+)\.h5$", dx=0.5, periodic=False, last=3,
                        pair_dt=None, julia_order=True)
    ens = load_ensemble(path)
    assert ens.parameters == ["a", "b"] and ens.time_regex == r"(\d+)\.h5$" and ens.periodic is False
    assert ens.pair_dt is None and ens.runs.path.iloc[0] == str((tmp_path / "r0").resolve())
    assert [(f.name, f.kind, f.keys, f.components, f.relative, f.blocks) for f in ens.fields] == [
        ("rho", "scalar", ("C", "rho"), (), True, ()), ("v", "velocity", ("v",), ("vx", "vy"), False, ()),
        ("Q", "nematic", ("Q",), (), False, ("amp", "defects"))]
    mtime = path.stat().st_mtime_ns
    write_config(path, tmp_path / "runs.csv", ["a", "b"], fields, snapshots="Data/*.h5", time_regex=r"(\d+)\.h5$",
                 dx=0.5, periodic=False, last=3, pair_dt=None, julia_order=True)
    assert path.stat().st_mtime_ns == mtime                     # the same text: the file is not touched
    with pytest.raises(ValueError, match="single_component"):
        write_config(tmp_path / "bad.toml", tmp_path / "runs.csv", ["a", "b"], [Field("vx", ("vx",))], dx=0.5)
