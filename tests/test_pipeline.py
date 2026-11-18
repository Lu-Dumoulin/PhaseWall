"""Tests of pipeline.py and scores.py on tiny Swift-Hohenberg ensembles: the steps run and are
cached (a second run neither simulates nor describes), blind grouping and the probe, a table
or a TOML ensemble as source, a cropped frame, the learned and combined descriptions, the cost
estimate, and the adjusted Rand index against scikit-learn."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from phasewall import pipeline as pl
from phasewall.scores import adjusted_rand

TINY = pl.Source(example="swift-hohenberg", grid={"eps": [-0.1, 0.2, 0.5], "g": [0.0, 0.7, 1.4]}, seeds=2,
                 settings={"n": 32, "waves": 4, "n_save": 3, "t_relax": 500.0})


def settings(tmp_path, **parts) -> pl.Settings:
    return pl.Settings(source=parts.pop("source", TINY), execution=pl.Execution(workers=1, cache=str(tmp_path)),
                       **parts)


def test_adjusted_rand():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand(a, a) == pytest.approx(1)
    assert adjusted_rand(a, [5, 5, 3, 3, 9, 9]) == pytest.approx(1)          # names do not matter
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, y = rng.integers(0, 4, 40), rng.integers(0, 3, 40)
        assert adjusted_rand(x, y) == pytest.approx(sklearn.adjusted_rand_score(x, y))


def test_demo_runs_and_is_cached(tmp_path):
    steps = []
    s = settings(tmp_path, methods=pl.Methods(walls=True, robustness=True))
    res = pl.run(s, progress=lambda step, done, total: steps.append(step))
    assert {"simulate", "describe", "methods"} <= set(steps)
    assert set(res.tables) == {"hand"} and len(res.runs) == 18
    row = res.scores.iloc[0]
    assert (row.describer, row.method) == ("hand", "walls") and row.points == 9 and "agree" in res.scores
    for fig in (res.phase_map("hand"), res.robustness_plot("hand"), res.snapshots("hand")):
        assert isinstance(fig, Figure)
    steps.clear()
    again = pl.run(s, progress=lambda step, done, total: steps.append(step))
    assert "simulate" not in steps and "describe" not in steps
    pd.testing.assert_frame_equal(again.scores, res.scores)


def test_blind_methods_and_probe(tmp_path):
    pytest.importorskip("sklearn")
    m = pl.Methods(walls=True, mixture=True, kmeans=True, hdbscan=(5,), probe=True)
    res = pl.run(settings(tmp_path, methods=m))
    assert res.scores.method.tolist() == ["walls", "mixture", "kmeans", "hdbscan_5"]
    assert 0 <= res.probe["hand"] <= 1
    assert isinstance(res.methods_map(), Figure)


def test_table_and_ensemble_sources(tmp_path):
    first = pl.run(settings(tmp_path / "a"))
    table = first.tables["hand"].assign(state=first.labels.loc[first.tables["hand"].run].to_numpy())
    table.to_csv(tmp_path / "table.csv", index=False)
    src = pl.Source(example=None, table=str(tmp_path / "table.csv"), parameters=("eps", "g"), label="state")
    res = pl.run(settings(tmp_path / "b", source=src))
    assert set(res.tables) == {"table"} and res.scores.agree.iloc[0] == first.scores.agree.iloc[0]
    toml = next((tmp_path / "a").glob("ensemble-*/phasewall.toml"))
    res = pl.run(settings(tmp_path / "c", source=pl.Source(example=None, config=str(toml), label="reference")))
    assert res.scores.agree.iloc[0] == first.scores.agree.iloc[0]


def test_cropped_frame(tmp_path):
    first = pl.run(settings(tmp_path))
    crop = 24 * first.ensemble.dx
    res = pl.run(settings(tmp_path, frame=pl.FrameSettings(crop=crop, max_shift=2 * first.ensemble.dx)))
    assert res.ensemble.crop == pytest.approx(crop)
    assert len(res.tables["hand"]) == len(first.tables["hand"])
    assert not res.tables["hand"].columns.equals(first.tables["hand"].columns)   # other S(k) shells


def test_learned_and_combined(tmp_path):
    pytest.importorskip("torch")
    d = pl.Describers(hand=True, learned=True, combined=True, epochs=1, device="cpu")
    res = pl.run(settings(tmp_path, describers=d, methods=pl.Methods(learning_curve=(1, 2))))
    assert set(res.tables) == {"hand", "learned seed 0", "combined seed 0"}
    combined = res.tables["combined seed 0"]
    assert "cnn.h.d000" in combined and "u.shape.k00" in combined
    assert set(res.learning_curve.runs_per_point) == {1, 2} and "agree" in res.learning_curve
    assert isinstance(res.training_plot(), Figure) and isinstance(res.learning_curve_plot(), Figure)


def test_estimate(tmp_path):
    e = pl.estimate(settings(tmp_path, describers=pl.Describers(learned=True)))
    assert set(e) == {"simulate", "describe", "train"} and all(v > 0 for v in e.values())
    assert pl.estimate(settings(tmp_path, source=pl.Source(example=None, table="x.csv", parameters=("a",)))) == {}


def test_blocks_chosen_per_field(tmp_path):
    s = settings(tmp_path, methods=pl.Methods(walls=False))
    full = pl.run(s).tables["hand"]
    s.describers.blocks = {"u": ("amp", "topo")}
    few = pl.run(s).tables["hand"]
    assert {c.split(".")[1] for c in few.columns if c.startswith("u.")} == {"amp", "topo"}
    assert len(few.columns) < len(full.columns)
    s.describers.blocks = {"u": ("flow",)}
    with pytest.raises(ValueError, match="do not apply"):
        pl.run(s)
