"""1D through the main command line: toy, describe, walls and demo on the quick Ginzburg-Landau
ensemble ([kymograph] configuration read by io.load_any)."""

import pandas as pd

from phasewall.cli import run


def test_toy_describe_walls(tmp_path):
    runs = run(["toy", str(tmp_path), "--example", "ginzburg-landau-1d", "--quick", "--workers", "1"])
    assert len(runs) == 50 and "[kymograph]" in (tmp_path / "phasewall.toml").read_text()
    out = run(["describe", str(tmp_path / "phasewall.toml"), "--workers", "1"])
    table = pd.read_csv(out)
    assert len(table) == 50 * 8 and "A.skw.travelling" in table.columns
    walls = run(["walls", str(out), "--config", str(tmp_path / "phasewall.toml")])
    for name in ("summary.md", "nodes.csv", "edges.csv", "phase_map.png", "kymographs.png"):
        assert (walls / name).exists(), name


def test_demo(tmp_path):
    out = run(["demo", str(tmp_path), "--example", "ginzburg-landau-1d", "--quick", "--workers", "1"])
    assert (out / "kymographs.png").exists() and "reference labels" in (out / "summary.md").read_text()


def test_learn_both_objectives(tmp_path):
    import pytest
    pytest.importorskip("torch")
    run(["toy", str(tmp_path), "--example", "ginzburg-landau-1d", "--quick", "--workers", "1"])
    for objective in ("contrastive", "jepa"):
        out = run(["learn", str(tmp_path / "phasewall.toml"), "--epochs", "1", "--objective", objective,
                   "--out", str(tmp_path / objective), "--width", "8", "--dim-h", "32", "--hidden", "32",
                   "--projection", "16"])
        table = pd.read_csv(out / "descriptors_cnn.csv")
        assert len(table) == 50 * 8 and "cnn.h.d000" in table.columns
