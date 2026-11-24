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
