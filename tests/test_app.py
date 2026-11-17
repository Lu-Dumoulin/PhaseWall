"""Tests of the notebook interface apps/explore.py (skipped without the extra [app], marimo).

The helpers that read the controls are tested on known inputs; the notebook itself is run once
from Python on the quick Swift-Hohenberg demo, with every tab drawn, and must show no error.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("marimo")

APP = Path(__file__).resolve().parent.parent / "apps" / "explore.py"


@pytest.fixture(scope="module")
def nb():
    """The notebook as a module (registered in sys.modules, where marimo looks up its functions)."""
    spec = importlib.util.spec_from_file_location("explore", APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["explore"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("explore", None)


def test_parse_values(nb):
    assert nb.parse_values("0.1, 0.2,0.5") == [0.1, 0.2, 0.5]
    assert nb.parse_values("-0.075:0.05:0.075") == [-0.075, -0.025, 0.025, 0.075]   # stop included
    assert nb.parse_values("0:0.1:1.4") == [round(0.1 * i, 10) for i in range(15)]  # no rounding loss
    assert nb.parse_values("3") == [3.0]
    for bad in ["", "a, b", "1:0:2", "2:1:1", "1:2"]:
        with pytest.raises(ValueError):
            nb.parse_values(bad)


def test_format_values_is_the_inverse(nb):
    import numpy as np

    from phasewall import gray_scott, toy
    for values in [toy.EPS, toy.G, gray_scott.F, gray_scott.K, [0.1, 0.3], [1.0, 2.0, 4.0]]:
        assert np.allclose(nb.parse_values(nb.format_values(values)), values)
    assert nb.format_values(toy.EPS) == "-0.075:0.05:0.675"
    assert nb.format_values([1.0, 2.0, 4.0]) == "1, 2, 4"


def test_parse_ints_and_optional_float(nb):
    assert nb.parse_ints("1, 2,4") == (1, 2, 4)
    assert nb.parse_ints(" ") == ()
    with pytest.raises(ValueError):
        nb.parse_ints("1.5")
    assert nb.optional_float("", "crop") is None
    assert nb.optional_float("2.5", "crop") == 2.5
    with pytest.raises(ValueError, match="crop"):
        nb.optional_float("big", "crop")


def test_notebook_runs_the_quick_demo(nb, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["explore.py", "--run", "--quick", "--cache", str(tmp_path)])
    outputs, defs = nb.app.run()
    res = defs["res"]
    assert list(res.tables) == ["hand"] and "hand" in res.walls
    page = " ".join(getattr(o, "text", "") for o in outputs if o is not None)
    assert "The run stopped" not in page and "Traceback" not in page
    assert page.count("data:image/png") >= 2          # phase map and snapshots, drawn without a browser


def test_setting_input_follows_the_type(nb):
    assert nb.setting_input("n", 64).value == 64
    assert nb.setting_input("dx", 0.01).value == 0.01
    assert nb.setting_input("localised", True).value is True
    assert nb.setting_input("init", "noise").value == "noise"
    menu = nb.setting_input("init", "patch", choices=["noise", "patch"])
    assert menu.value == "patch" and "noise" in menu.options
