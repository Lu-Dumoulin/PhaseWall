"""Contract tests: every registered encoder and method (registry.py) is checked here, so a new
entry is tested by registering it. Methods: one integer group per run (-1 noise); a method on
the grid finds one phase in pure noise and two phases on a step; a blind method separates two
far clouds. Encoders: one row per snapshot with finite descriptors named name.block.column; an
encoder declared invariant gives the same description of a rotated and mirrored ensemble.
Also the registry itself: registration, settings for the interface, and a method and an encoder
added from outside, run through the pipeline."""

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from phasewall import pipeline as pl
from phasewall import registry
from phasewall.fields import Field, transform
from phasewall.io import Ensemble, write_snapshot

NA, NB, RUNS, SNAPS, DIM = 8, 6, 2, 4, 5


def table(level_of, noise: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, run = [], 0
    for a in range(NA):
        for b in range(NB):
            for _ in range(RUNS):
                m = level_of(a, b)
                for t in range(SNAPS):
                    x = m + noise * rng.standard_normal(DIM)
                    rows.append({"run": run, "time": float(t), "a": float(a), "b": float(b)}
                                | {f"f.blk.x{j}": x[j] for j in range(DIM)})
                run += 1
    return pd.DataFrame(rows)


def available(entries):
    return [n for n, e in entries.items() if registry.available(e)]


METHODS = available(registry.methods())
ENCODERS = available(registry.encoders())


def run_method(name, t):
    e = registry.methods()[name]
    return e.run(t, ["a", "b"], e.settings(), registry.Context())


@pytest.mark.parametrize("name", METHODS)
def test_method_gives_one_group_per_run(name):
    t = table(lambda a, b: np.zeros(DIM), seed=1)
    out = run_method(name, t)
    assert set(out.groups.index) == set(t.run)
    assert out.groups.dtype.kind == "i" and out.groups.min() >= -1


@pytest.mark.parametrize("name", [n for n in METHODS if registry.methods()[n].uses_grid])
def test_grid_method_finds_one_phase_in_noise_and_two_on_a_step(name):
    noise = run_method(name, table(lambda a, b: np.zeros(DIM), seed=2)).groups
    assert noise[noise >= 0].nunique() <= 1
    t = table(lambda a, b: np.full(DIM, 6.0 * (a >= NA / 2)), seed=3)
    g = run_method(name, t).groups
    side = t.groupby("run").a.first() >= NA / 2
    placed = g >= 0                       # boundary regions (fewer than MIN_PHASE points) are allowed
    assert placed.mean() >= 0.9 and g[placed].nunique() == 2
    assert g[side & placed].nunique() == 1 and g[~side & placed].nunique() == 1
    assert g[side & placed].iloc[0] != g[~side & placed].iloc[0]


@pytest.mark.parametrize("name", [n for n in METHODS if not registry.methods()[n].uses_grid])
def test_blind_method_separates_two_far_clouds(name):
    t = table(lambda a, b: np.full(DIM, 20.0 * ((a + b) % 2)), noise=0.5, seed=4)
    g = run_method(name, t).groups
    cloud = (t.groupby("run").a.first() + t.groupby("run").b.first()) % 2 == 1
    placed = g >= 0
    assert placed.mean() > 0.8
    assert set(g[cloud & placed]).isdisjoint(set(g[~cloud & placed]))


def make_ensemble(folder, n: int = 32) -> Ensemble:
    """2 points x 2 runs x 4 snapshots of a scalar u and a velocity v (smooth random fields)."""
    rows, run = [], 0
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n, 1 / n), np.fft.fftfreq(n, 1 / n), indexing="ij"))

    def smooth(rng, shape):
        f = np.fft.fft2(rng.standard_normal(shape), axes=(0, 1))
        kk = k[..., None] if len(shape) == 3 else k
        return np.fft.ifft2(f * (kk <= 4), axes=(0, 1)).real

    for a in (0.0, 1.0):
        for _ in range(2):
            rng = np.random.default_rng(run)
            for t in range(4):
                write_snapshot(folder / f"r{run}" / "Data" / f"{t:04d}.h5",
                               {"u": smooth(rng, (n, n)) * (1 + a), "v": smooth(rng, (n, n, 2))})
            rows.append({"run": run, "path": str(folder / f"r{run}"), "a": a})
            run += 1
    return Ensemble(runs=pd.DataFrame(rows), parameters=["a"],
                    fields=[Field("u", ("u",)), Field("v", ("v",), "velocity")], dx=0.5, snapshots="Data/*.h5",
                    time_regex=r"(\d+)\.h5$", last=3, pair_dt=1.0)


def describe(name, ens, folder):
    e = registry.encoders()[name]
    return e.run(ens, e.test_settings or e.settings(), registry.Context(cache=str(folder)))


@pytest.mark.parametrize("name", ENCODERS)
def test_encoder_tables(name, tmp_path):
    ens = make_ensemble(tmp_path / "ens")
    out = describe(name, ens, tmp_path)
    assert out.tables
    for t in out.tables.values():
        cols = [c for c in t.columns if c.count(".") >= 2]
        assert {"run", "time", "a"} <= set(t.columns) and cols
        assert set(t.columns) == {"run", "time", "a", *cols}
        assert np.isfinite(t[cols].to_numpy()).all()
        assert set(t.run) == set(ens.runs.run) and not t[["run", "time"]].duplicated().any()


@pytest.mark.parametrize("name", [n for n in ENCODERS if registry.encoders()[n].invariant])
def test_invariant_encoder_ignores_the_symmetries_of_the_box(name, tmp_path):
    from phasewall.io import read_snapshot, run_snapshots
    ens = make_ensemble(tmp_path / "ens")
    turned = dataclasses.replace(ens, runs=ens.runs.assign(path=[str(tmp_path / "turned" / f"r{r}") for r in ens.runs.run]))
    for r, path in zip(ens.runs.run, ens.runs.path):
        for t, file in run_snapshots(ens, path):
            snap = read_snapshot(file, ens.fields)
            write_snapshot(tmp_path / "turned" / f"r{r}" / "Data" / file.name,
                           transform(snap, ens.fields, turns=1, mirror=True, shift=(5, 9)))
    a = describe(name, ens, tmp_path / "c1").tables
    b = describe(name, turned, tmp_path / "c2").tables
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------- the registry itself
def test_settings_describe_themselves_and_are_checked():
    spec = {f["name"]: f for f in registry.settings_spec(registry.methods()["walls"].settings)}
    assert spec["ratio"]["range"] == (1.0, 100.0) and spec["ratio"]["help"]
    assert spec["merge"]["type"] == "bool" and spec["merge"]["default"] is True
    with pytest.raises(ValueError, match="outside"):
        registry.check_settings(registry.methods()["walls"].settings(ratio=0.5))
    for entries in (registry.methods(), registry.encoders()):
        for e in entries.values():
            assert e.help and dataclasses.is_dataclass(e.settings)
            registry.check_settings(e.settings())                  # every default is valid


@dataclass
class HalfSettings:
    column: str = field(default="eps", metadata={"help": "parameter that is cut", "choices": ("eps", "g")})


@pytest.fixture
def plugins():
    """A method and an encoder registered from outside, removed afterwards."""
    @registry.method("half", HalfSettings, help="two groups: the lower and upper half of a parameter",
                     uses_grid=True)
    def half(t, parameters, settings, context):
        p = t.groupby("run")[settings.column].first()
        return registry.MethodResult((p >= p.median()).astype(int))

    @dataclass
    class MeanSettings:
        pass

    @registry.encoder("means", MeanSettings, help="mean of every field", invariant=True)
    def means(ens, settings, context):
        from phasewall.describe import describe_ensemble
        t, _ = describe_ensemble(ens)
        cols = [c for c in t.columns if c.endswith((".mom.mean", ".amp.log10var"))]
        return registry.EncoderResult({"means": t[["run", "time", *ens.parameters, *cols]]})
    yield
    registry.METHODS.pop("half")
    registry.ENCODERS.pop("means")


def test_registration_errors():
    with pytest.raises(TypeError, match="dataclass"):
        registry.method("bad", dict)
    with pytest.raises(ValueError, match="already registered"):
        registry.method("walls", HalfSettings)(lambda *a: None)


def test_plugins_run_through_the_pipeline(plugins, tmp_path):
    src = pl.Source(example="swift-hohenberg", grid={"eps": [-0.1, 0.2, 0.5], "g": [0.0, 0.7, 1.4]}, seeds=2,
                    settings={"n": 32, "waves": 4, "n_save": 3, "t_relax": 500.0})
    s = pl.Settings(source=src, describers=pl.Describers(hand=False, others={"means": registry.encoders()["means"].settings()}),
                    methods=pl.Methods(walls=False, others={"half": HalfSettings("eps")}),
                    execution=pl.Execution(workers=1, cache=str(tmp_path)))
    res = pl.run(s)
    assert list(res.tables) == ["means"]
    row = res.scores.iloc[0]
    assert (row.describer, row.method, row.groups) == ("means", "half", 2)
    with pytest.raises(ValueError, match="choose one of"):
        pl.run(dataclasses.replace(s, methods=pl.Methods(walls=False, others={"half": HalfSettings("c")})))
