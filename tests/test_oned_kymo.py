"""1D systems, input side: fields, the ensemble file, kymograph files and time windows."""

import numpy as np
import pandas as pd
import pytest

from phasewall.oned.kymo import (
    MIN_WINDOW_ROWS,
    Field1D,
    load_ensemble1d,
    read_kymograph,
    window_slices,
    write_kymograph,
)


@pytest.mark.parametrize("julia_order", [True, False])
@pytest.mark.parametrize("order", ["xt", "tx"])
def test_kymograph_round_trip(tmp_path, julia_order, order):
    rng = np.random.default_rng(0)
    u, px, py = (rng.standard_normal((12, 30)) for _ in range(3))
    write_kymograph(tmp_path / "k.h5", {"u": u, "px": px, "py": py}, julia_order, order)
    fields = [Field1D("u"), Field1D("p", kind="polarity", components=("px", "py"))]
    out = read_kymograph(tmp_path / "k.h5", fields, julia_order, order)
    assert np.array_equal(out["u"], u)                           # indexed [x, t]
    assert out["p"].shape == (12, 30, 2)
    assert np.array_equal(out["p"][..., 0], px) and np.array_equal(out["p"][..., 1], py)


def test_missing_dataset_is_an_error(tmp_path):
    write_kymograph(tmp_path / "k.h5", {"u": np.zeros((4, 20))})
    with pytest.raises(KeyError):
        read_kymograph(tmp_path / "k.h5", [Field1D("v")])


def test_fields_check_their_components():
    assert Field1D("v", kind="velocity").odd == (True,)
    assert Field1D("p", kind="polarity", components=("px", "py")).odd == (True, False)
    assert Field1D("A", kind="complex", components=("re", "im")).odd == (False, False)
    with pytest.raises(ValueError):
        Field1D("v", kind="velocity", components=("vx", "vy"))     # a velocity on a line has one component
    with pytest.raises(ValueError):
        Field1D("A", kind="complex")                                 # needs two components
    with pytest.raises(ValueError):
        Field1D("n", kind="nematic")
    with pytest.raises(ValueError):
        Field1D("v", kind="velocity", relative=True)


def _write_config(tmp_path, extra=""):
    pd.DataFrame({"path": ["run0", "run1"], "b": [0.0, 1.0]}).to_csv(tmp_path / "runs.csv", index=False)
    (tmp_path / "e.toml").write_text(f"""[kymograph]
runs = "runs.csv"
parameters = ["b"]
dx = 0.5
dt = 0.1
t0 = 10.0
order = "tx"
share = 0.25
windows = 4
{extra}
[[fields]]
name = "A"
kind = "complex"
components = ["Are", "Aim"]
""")
    return tmp_path / "e.toml"


def test_ensemble_file(tmp_path):
    ens = load_ensemble1d(_write_config(tmp_path))
    assert ens.parameters == ["b"] and ens.dt == 0.1 and ens.t0 == 10.0 and ens.order == "tx"
    assert ens.share == 0.25 and ens.windows == 4 and ens.periodic
    assert list(ens.runs.run) == [0, 1] and ens.runs.path.iloc[0].endswith("run0")
    assert ens.fields[0].components == ("Are", "Aim")
    with pytest.raises(ValueError, match="unknown keys"):
        load_ensemble1d(_write_config(tmp_path, 'windws = 3'))
    (tmp_path / "two_d.toml").write_text('[ensemble]\nruns = "runs.csv"\n')
    with pytest.raises(ValueError, match=r"\[kymograph\]"):
        load_ensemble1d(tmp_path / "two_d.toml")


def test_windows_cover_the_last_share():
    sl = window_slices(100, share=0.5, windows=2)
    assert [(s.start, s.stop) for s in sl] == [(50, 75), (75, 100)]
    sl = window_slices(101, share=0.5, windows=3)                  # 50 rows used: 3 windows of 16
    assert [(s.start, s.stop) for s in sl] == [(53, 69), (69, 85), (85, 101)]
    with pytest.raises(ValueError):
        window_slices(100, share=0.5, windows=50 // MIN_WINDOW_ROWS + 1)   # 12 rows each


def one_file_per_run(folder, n_runs=4, n_x=32, n_t=96, julia_order=True, order="xt"):
    """Runs saved as one HDF5 file each, all in one folder, and a table of runs pointing at the files."""
    from phasewall.oned.kymo import write_kymograph
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_runs):
        x = np.linspace(0, 2 * np.pi, n_x, endpoint=False)[:, None]
        t = np.arange(n_t)[None, :] * 0.1
        rho = 1 + 0.1 * np.cos(x - i * t) + 0.01 * rng.standard_normal((n_x, n_t))
        arrays = {"rho": rho, "v": 0.2 * np.sin(x - i * t), "Are": np.cos(x + t), "Aim": np.sin(x + t),
                  "time": np.arange(n_t, dtype=float)[None, :]}
        write_kymograph(folder / f"run{i}.h5", arrays, julia_order, order)
        rows.append({"path": f"run{i}.h5", "speed": float(i), "seed": 0})
    pd.DataFrame(rows).to_csv(folder / "runs.csv", index=False)
    return folder / "runs.csv"


def test_suggestions_and_order_from_one_file(tmp_path):
    from phasewall.io import inspect_snapshot
    from phasewall.oned.kymo import suggest_fields1d, suggest_order
    runs = one_file_per_run(tmp_path / "runs")
    datasets = inspect_snapshot(tmp_path / "runs" / "run0.h5", julia_order=True)
    s = {d["name"]: d for d in suggest_fields1d(datasets)}
    assert s["rho"]["kind"] == "scalar" and s["rho"]["relative"]               # density-like name
    assert s["v"]["kind"] == "velocity"
    assert s["A"]["kind"] == "complex" and s["A"]["components"] == ["Are", "Aim"]
    assert s["time"]["kind"] is None                                             # (1, n_t): not a kymograph
    assert suggest_order((32, 96))[0] == "xt" and suggest_order((96, 32))[0] == "tx"
    assert runs.exists()


@pytest.mark.parametrize("order", ["xt", "tx"])
def test_one_file_per_run_config_written_and_read(tmp_path, order):
    """file = "": the column path gives each run's file; the config written is read back, the arrays
    come back as [x, t] whatever the order in the files."""
    from phasewall.oned.kymo import Field1D, read_kymograph, write_config1d
    runs = one_file_per_run(tmp_path / "runs", order=order)
    fields = [Field1D("rho", ("rho",), "scalar", relative=True), Field1D("v", ("v",), "velocity"),
              Field1D("A", kind="complex", components=("Are", "Aim"))]
    cfg = write_config1d(tmp_path / "ens.toml", runs, ["speed"], fields, file="", julia_order=True, order=order,
                         dx=2 * np.pi / 32, dt=0.1, periodic=True, share=1.0, windows=2)
    ens = load_ensemble1d(cfg)
    assert ens.file == "" and ens.parameters == ["speed"] and [f.kind for f in ens.fields] == ["scalar", "velocity", "complex"]
    f0 = ens.file_of(ens.runs.path.iloc[0])
    assert f0.name == "run0.h5" and f0.is_file()
    a = read_kymograph(f0, ens.fields, ens.julia_order, ens.order)
    assert a["rho"].shape == (32, 96) and a["A"].shape == (32, 96, 2)
    mtime = cfg.stat().st_mtime_ns
    write_config1d(tmp_path / "ens.toml", runs, ["speed"], fields, file="", julia_order=True, order=order,
                   dx=2 * np.pi / 32, dt=0.1, periodic=True, share=1.0, windows=2)
    assert cfg.stat().st_mtime_ns == mtime                                       # unchanged: not rewritten
