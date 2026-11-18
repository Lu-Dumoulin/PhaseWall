"""Tests of the notebook interface apps/explore.py (skipped without the extra [app], marimo).

The helpers that read the controls are tested on known inputs; the notebook itself is run from
Python, with every tab drawn, on the quick Swift-Hohenberg demo (with the tree), and on runs given
as a table of runs and snapshot files (fields from the inspector); it must show no error.
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
    assert nb.format_values([1.0, 2.0, 5.0]) == "1, 2, 5"


def test_geometric_ranges(nb):
    import numpy as np

    from phasewall import fitzhugh_nagumo
    assert nb.parse_values("1:*2:16") == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert nb.format_values([1.0, 2.0, 4.0, 8.0]) == "1:*2:8"
    # a factor written with 5 digits still ends exactly on stop
    values = nb.parse_values("0.005:*1.5849:0.5")
    assert len(values) == 11 and values[-1] == 0.5
    assert np.allclose(values, np.geomspace(0.005, 0.5, 11), rtol=1e-4)
    for bad in ["0:*2:8", "1:*1:8", "-1:*2:8", "8:*2:1"]:
        with pytest.raises(ValueError):
            nb.parse_values(bad)
    # the demo's grids, rounded to 4 decimals, are shown as geometric ranges; read back they are
    # within the rounding, and an untouched field gives the demo's values exactly
    for default in [fitzhugh_nagumo.EPS, fitzhugh_nagumo.DELTA]:
        text = nb.format_values(default)
        assert ":*" in text
        assert np.allclose(nb.parse_values(text), default, rtol=2e-2)
        assert nb.grid_values(text, default) == [float(x) for x in default]
    assert nb.grid_values("0.01, 0.1", fitzhugh_nagumo.EPS) == [0.01, 0.1]


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


def test_controls_from_settings(nb):
    from phasewall.builtins import TreeSettings, WallsSettings
    from phasewall.nn.architecture import Network
    c = nb.spec_controls(Network)
    assert set(c.value) == {"blocks", "width", "dim_h", "hidden", "projection", "temperature"}
    assert nb.settings_from(Network, c.value) == Network()                 # defaults round trip
    values = dict(c.value) | {"width": "16", "blocks": 3.0}
    assert nb.settings_from(Network, values) == Network(blocks=3, width=16)
    t = nb.spec_controls(TreeSettings)
    assert nb.settings_from(TreeSettings, t.value) == TreeSettings()
    assert nb.settings_from(WallsSettings, nb.spec_controls(WallsSettings).value) == WallsSettings()


def test_field_lines(nb):
    from phasewall.fields import Field
    fields = nb.parse_field_lines("rho scalar C relative\n# a comment\nv velocity vx,vy\nux scalar ux single")
    assert fields == [Field("rho", ("C",), "scalar", True), Field("v", ("v",), "velocity", components=("vx", "vy")),
                      Field("ux", ("ux",), single_component=True)]
    for bad in ["rho scalar", "rho scalar C fast", "rho flow C"]:
        with pytest.raises(ValueError):
            nb.parse_field_lines(bad)


def test_notebook_runs_with_the_tree(nb, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["explore.py", "--run", "--quick", "--tree", "--cache", str(tmp_path)])
    outputs, defs = nb.app.run()
    res = defs["res"]
    assert ("hand", "tree") in res.extras and "tree" in set(res.scores.method)
    page = " ".join(getattr(o, "text", "") for o in outputs if o is not None)
    assert "The run stopped" not in page and "Traceback" not in page and "phase 0" in page


def test_notebook_runs_your_own_runs(nb, tmp_path, monkeypatch):
    import numpy as np

    from phasewall import toy
    toy.make_ensemble(tmp_path / "runs", eps=[-0.1, 0.3], g=[0.0, 1.0], seeds=(0, 1), n=32, waves=4)
    dx = 2 * np.pi * 4 / 32
    monkeypatch.setattr(sys, "argv", ["explore.py", "--run", "--runs", str(tmp_path / "runs" / "runs.csv"),
                                      "--dx", repr(dx), "--pair-dt", "25", "--cache", str(tmp_path / "cache")])
    outputs, defs = nb.app.run()
    assert [f.name for f in defs["known_fields"]] == ["u"] and defs["settings_problems"] == []
    res = defs["res"]
    assert list(res.parameters) == ["eps", "g"] and "hand" in res.walls     # seed is not a parameter by default
    assert any(c.startswith("u.dyn.") for c in res.tables["hand"].columns)
    page = " ".join(getattr(o, "text", "") for o in outputs if o is not None)
    assert "The run stopped" not in page and "Traceback" not in page


def test_your_runs_with_a_velocity_saved_as_two_datasets(nb, tmp_path, monkeypatch):
    import h5py
    import numpy as np

    from phasewall import toy
    toy.make_ensemble(tmp_path / "runs", eps=[-0.1, 0.3], g=[0.0, 1.0], seeds=(0, 1), n=32, waves=4)
    rng = np.random.default_rng(0)
    for f in sorted((tmp_path / "runs").glob("*/Data/*.h5")):
        with h5py.File(f, "a") as h:                    # a velocity whose components are saved apart
            h["vx"], h["vy"] = rng.normal(size=(32, 32)), rng.normal(size=(32, 32))
    dx = 2 * np.pi * 4 / 32
    monkeypatch.setattr(sys, "argv", ["explore.py", "--run", "--runs", str(tmp_path / "runs" / "runs.csv"),
                                      "--dx", repr(dx), "--cache", str(tmp_path / "cache")])
    _, defs = nb.app.run()
    fields = {f.name: f for f in defs["known_fields"]}
    assert fields["v"].kind == "velocity" and fields["v"].components == ("vx", "vy")
    cols = defs["res"].tables["hand"].columns
    assert any(c.startswith("v.vortex.") for c in cols) and any(c.startswith("v.flow.") for c in cols)
