"""Tests of the command-line tool on a tiny Swift-Hohenberg ensemble: each command writes
what it says, and the demo runs end to end with --quick."""

import pandas as pd
import pytest

from phasewall import cli, toy

TINY = {"n": 32, "waves": 4}


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ens")
    toy.make_ensemble(folder, eps=[-0.1, 0.05, 0.3, 0.6], g=[0.0, 1.0], seeds=(0, 1), n_save=3, t_relax=500.0, **TINY)
    cli.run(["describe", str(folder / "phasewall.toml"), "--workers", "1"])
    return folder


def test_describe_and_walls(ensemble):
    csv = ensemble / "descriptors.csv"
    table = pd.read_csv(csv)
    assert len(table) == 4 * 2 * 2 * 3 and {"run", "time", "eps", "g"} <= set(table.columns)
    out = cli.run(["walls", str(csv), "--parameters", "eps", "g", "--out", str(ensemble / "w1")])
    assert {"summary.md", "nodes.csv", "edges.csv", "cuts.csv", "phase_map.png"} <= {p.name for p in out.iterdir()}
    assert (out / "summary.md").read_text().startswith("# PhaseWall result")
    # with --config the parameters come from the ensemble, and example snapshots are drawn
    out = cli.run(["walls", str(csv), "--config", str(ensemble / "phasewall.toml"), "--out", str(ensemble / "w2")])
    assert (out / "snapshots.png").exists()


def test_walls_needs_parameters(ensemble):
    with pytest.raises(SystemExit):
        cli.run(["walls", str(ensemble / "descriptors.csv")])


def test_node_labels_and_agreement(ensemble):
    table = pd.read_csv(ensemble / "descriptors.csv")
    runs = pd.read_csv(ensemble / "runs.csv")
    w = cli.find_walls(table, ["eps", "g"])
    runs.loc[0, "reference"] = "other"      # a tie at point (-0.1, 0): either label is a majority
    labels = cli.node_labels(w, runs)
    for node, row in w.nodes.iterrows():
        here = runs[(runs.eps == row.eps) & (runs.g == row.g)].reference
        assert labels[node] in set(here[here.map(here.value_counts()) == here.value_counts().max()])
    counts, agree = cli.agreement(w, labels)
    assert counts.to_numpy().sum() == len(w.nodes) and agree <= len(w.nodes)


def test_demo_quick(tmp_path):
    out = cli.run(["demo", str(tmp_path / "demo"), "--quick", "--workers", "2"])
    text = (out / "summary.md").read_text()
    assert "## Check against the reference labels (demo only)" in text
    assert {"phase_map.png", "snapshots.png", "robustness.png"} <= {p.name for p in out.iterdir()}


def test_main_returns_none_so_the_command_exits_with_0(ensemble):
    """The installed command runs sys.exit(main()): anything but None would be an error."""
    assert cli.main(["walls", str(ensemble / "descriptors.csv"), "--parameters", "eps", "g",
                     "--out", str(ensemble / "w3")]) is None
