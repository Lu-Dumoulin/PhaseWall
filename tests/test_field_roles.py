"""Tests of the field roles (velocity, polarity) and of fields assembled from component
datasets: roles turn as vectors; components read from two datasets give the same array and the
same descriptors as one dataset; a component declared as a scalar is refused; the inspector
groups components and suggests roles; its TOML loads back."""

import numpy as np
import pandas as pd
import pytest

from phasewall import cli
from phasewall.describe import describe_snapshot
from phasewall.fields import Field, component_of, mirror_x, rotate90
from phasewall.io import (
    check_not_component,
    fields_toml,
    inspect_snapshot,
    load_ensemble,
    read_snapshot,
    suggest_fields,
    write_snapshot,
)

N = 16


def field(seed: int = 0, comps: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, N, comps)) if comps > 1 else rng.normal(size=(N, N))


@pytest.mark.parametrize("kind", ["velocity", "polarity"])
def test_roles_turn_as_vectors(kind):
    a = field()
    assert Field("a", ("a",), kind).symmetry == "vector"
    assert np.array_equal(rotate90(a, kind), rotate90(a, "vector"))
    assert np.array_equal(mirror_x(a, kind), mirror_x(a, "vector"))


def test_component_names():
    assert component_of("vx") == ("v", "x")
    assert component_of("p_y") == ("p", "y")
    assert component_of("Qxx") == ("Q", "xx")
    assert component_of("Q_xy") == ("Q", "xy")
    for name in ("rho", "u", "v", "phi", "C", "P", "conc"):
        assert component_of(name) is None, name


def test_field_settings_are_checked():
    with pytest.raises(ValueError, match="two dataset names"):
        Field("v", (), "scalar", components=("vx", "vy"))
    with pytest.raises(ValueError, match="two dataset names"):
        Field("v", (), "velocity", components=("vx",))
    with pytest.raises(ValueError, match="only for scalar"):
        Field("v", ("V",), "velocity", single_component=True)
    with pytest.raises(ValueError, match="kind must be"):
        Field("v", ("V",), "flow")


@pytest.mark.parametrize("julia_order", [True, False])
def test_components_read_as_one_field(tmp_path, julia_order):
    a = field()[:, :12]                      # different sizes along x and y, so that a transposition shows
    write_snapshot(tmp_path / "s.h5", {"V": a, "vx": a[..., 0], "vy": a[..., 1]}, julia_order=julia_order)
    whole = read_snapshot(tmp_path / "s.h5", [Field("v", ("V",), "velocity")], julia_order)["v"]
    parts = read_snapshot(tmp_path / "s.h5", [Field("v", (), "velocity", components=("vx", "vy"))], julia_order)["v"]
    assert np.array_equal(whole, a) and np.array_equal(parts, a)
    with pytest.raises(KeyError, match="components"):
        read_snapshot(tmp_path / "s.h5", [Field("v", (), "velocity", components=("vx", "vz"))], julia_order)


def test_descriptors_depend_on_the_symmetry_not_on_the_role():
    a = {"p": field(1)}
    base = describe_snapshot(a, [Field("p", ("P",), "vector")], 1.0, before=a)
    for kind in ("velocity", "polarity"):
        assert describe_snapshot(a, [Field("p", ("P",), kind)], 1.0, before=a) == base


def test_a_component_alone_is_refused_unless_confirmed():
    with pytest.raises(ValueError, match=r"components = \['vx', 'vy'\]"):
        check_not_component(Field("vx", ("vx",)))
    with pytest.raises(ValueError, match="Qxx"):
        check_not_component(Field("q", ("Qxy",)))       # the keys are checked too
    check_not_component(Field("vx", ("vx",), single_component=True))
    check_not_component(Field("rho", ("C", "rho")))


def test_inspector_groups_components_and_suggests_roles(tmp_path):
    a = field()
    snap = {"rho": a[..., 0], "vx": a[..., 0], "vy": a[..., 1], "P": a, "Qxx": a[..., 0], "Qxy": a[..., 1],
            "Qyy": -a[..., 0], "px": a[..., 0], "time": np.array([3.0])}
    write_snapshot(tmp_path / "s.h5", snap)
    ds = inspect_snapshot(tmp_path / "s.h5")
    assert ds.set_index("name").loc["P", "shape"] == (N, N, 2)
    assert ds.set_index("name").loc["P", "file_shape"] == (2, N, N)
    s = {x["name"]: x for x in suggest_fields(ds)}
    assert s["v"]["kind"] == "velocity" and s["v"]["components"] == ["vx", "vy"]
    assert s["P"]["kind"] == "polarity" and s["P"]["keys"] == ["P"]
    assert s["Q"]["kind"] == "nematic" and s["Q"]["components"] == ["Qxx", "Qxy"]
    assert s["Qyy"]["kind"] is None                      # redundant with Qxx
    assert s["px"]["kind"] is None                       # py missing: not guessed
    assert s["rho"]["kind"] == "scalar" and s["rho"]["relative"]
    assert s["time"]["kind"] is None
    assert not {"vx", "vy", "Qxx", "Qxy"} & set(s)


def test_suggested_fields_load_and_read(tmp_path):
    a = field()
    (tmp_path / "r0" / "Data").mkdir(parents=True)
    snap = {"rho": 1 + 0.1 * a[..., 0], "vx": a[..., 0], "vy": a[..., 1]}
    write_snapshot(tmp_path / "r0" / "Data" / "0000000010.h5", snap)
    text = fields_toml(suggest_fields(inspect_snapshot(tmp_path / "r0" / "Data" / "0000000010.h5")))
    pd.DataFrame({"path": ["r0"], "a": [0.1]}).to_csv(tmp_path / "runs.csv", index=False)
    (tmp_path / "ens.toml").write_text('[ensemble]\nruns = "runs.csv"\nparameters = ["a"]\nsnapshots = "Data/*.h5"\n'
                                       "time_regex = '(\\d+)\\.h5$'\ndx = 1.0\n\n" + text)
    ens = load_ensemble(tmp_path / "ens.toml")
    assert [(f.name, f.kind, f.components) for f in ens.fields] == [("rho", "scalar", ()),
                                                                     ("v", "velocity", ("vx", "vy"))]
    out = read_snapshot(tmp_path / "r0" / "Data" / "0000000010.h5", ens.fields)
    assert np.array_equal(out["v"], a)


def test_load_ensemble_refuses_a_component_declared_as_a_scalar(tmp_path):
    pd.DataFrame({"path": ["r0"], "a": [0.1]}).to_csv(tmp_path / "runs.csv", index=False)
    head = '[ensemble]\nruns = "runs.csv"\nparameters = ["a"]\ndx = 1.0\n\n[[fields]]\nname = "vx"\n'
    (tmp_path / "ens.toml").write_text(head)
    with pytest.raises(ValueError, match="single_component"):
        load_ensemble(tmp_path / "ens.toml")
    (tmp_path / "ens.toml").write_text(head + "single_component = true\n")
    assert load_ensemble(tmp_path / "ens.toml").fields[0].single_component


def test_inspect_command(tmp_path, capsys):
    a = field()
    write_snapshot(tmp_path / "s.h5", {"px": a[..., 0], "py": a[..., 1]})
    text = cli.run(["inspect", str(tmp_path / "s.h5")])
    assert 'components = ["px", "py"]' in text and 'kind = "polarity"' in text
    assert "px" in capsys.readouterr().out
