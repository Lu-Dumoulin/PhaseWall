"""A whole analysis in one call, with cached steps: the programming interface of the notebook
interface (apps/, see docs/plan_interface.md), also usable on its own from Python or a script.

    from phasewall import pipeline as pl
    settings = pl.Settings(source=pl.Source(example="gray-scott", quick=True),
                           describers=pl.Describers(hand=True, learned=True, combined=True),
                           methods=pl.Methods(walls=True, mixture=True, hdbscan=(10,), probe=True))
    print(pl.estimate(settings))                  # rough cost of each step, in seconds
    res = pl.run(settings, progress=print)        # progress(step, done, total)
    res.scores                                    # one row per description and method
    res.phase_map("hand")                         # matplotlib Figure

Steps, each cached under Execution.cache in a file or folder named by a hash of the settings
it depends on (so that changing a method recomputes only the methods):
  ensemble   <- Source                            simulated for a demo; read for an ensemble or a table
  hand       <- ensemble, FrameSettings           describe.py
  learned    <- ensemble, FrameSettings, epochs, seed, device-independent   nn (encoder + table of h)
  combined   <- hand, learned                     describe.combine_tables (not cached, cheap)
  methods    <- each description, Methods         walls, robustness, blind grouping, probe (seconds)
  learning curve <- learned settings, runs per point (cached)
Nothing here changes the other modules: it only calls them.
"""

import dataclasses
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import registry, report
from .describe import combine_tables, descriptor_columns
from .examples import example as example_module
from .io import MIN_WAVELENGTHS, Ensemble, load_ensemble, load_table
from .parallel import Progress
from .scores import point_scores
from .walls import ALPHA, MERGE, MIN_PHASE, RATIO, Walls, phases_against_ratio

QUICK_SEEDS = 2                         # runs per point of a quick demo (as the command line)
# rough costs for estimate(), measured on a laptop (Apple M4 Max) on the demos (choices of display only)
DESCRIBE_SECONDS_PER_SNAPSHOT_POINT = 1.0e-7   # per snapshot, per grid point, per field, one process
TRAIN_SECONDS_PER_ITEM_POINT = 9.0e-8          # per training item (two views), per grid point, Apple GPU


# ---------------------------------------------------------------- settings
@dataclass
class Source:
    """Where the runs come from: a demo (`example`), an ensemble (`config`, a TOML file) or a
    table of descriptors (`table`, a CSV file; see io.load_table)."""

    example: str | None = "swift-hohenberg"
    config: str | None = None
    table: str | None = None
    parameters: tuple[str, ...] | None = None   # table: parameter columns
    label: str | None = None                    # table or ensemble: column of labels (optional)
    prefix: str | None = None                   # table: descriptor columns starting with this
    grid: dict | None = None                    # demo: {parameter: values}; default the demo's GRID
    seeds: int | None = None                    # demo: runs per point; default the demo's SEEDS
    quick: bool = False                         # demo: its small QUICK grid and settings
    settings: dict = field(default_factory=dict)  # demo: numerical settings (n, dx or waves, t_relax, ...)

    def kind(self) -> str:
        given = [k for k, v in (("table", self.table), ("config", self.config), ("example", self.example)) if v]
        if not given:
            raise ValueError("Source needs an example, a config or a table")
        return given[0]


@dataclass
class FrameSettings:
    """The frame described in every run (frames.py); None keeps the ensemble's own value."""

    crop: float | None = None
    dx_target: float | None = None
    max_shift: float | None = None
    min_wavelengths: float = MIN_WAVELENGTHS
    periodic: bool | None = None


@dataclass
class Describers:
    """Which descriptions: hand-made, learned (one per seed), combined (hand-made + learned)."""

    hand: bool = True
    learned: bool = False
    combined: bool = False
    epochs: int = 40
    seeds: tuple[int, ...] = (0,)
    device: str | None = None                   # "cpu", "mps", "cuda"; None: the fastest available
    blocks: dict = field(default_factory=dict)  # hand-made: {field name: blocks}; others keep their own or default
    others: dict = field(default_factory=dict)  # other registered encoders: {name: settings} (registry.py)


@dataclass
class Methods:
    """Which methods, on every description."""

    walls: bool = True
    alpha: float = ALPHA
    ratio: float = RATIO
    min_phase: int = MIN_PHASE
    merge: bool = MERGE                         # walls: join separated parts of one phase
    tree: bool = False                          # unsupervised threshold tree (its default settings)
    robustness: bool = False
    mixture: bool = False
    kmeans: bool = False
    hdbscan: tuple[int, ...] = ()               # minimum sizes; empty: not run
    probe: bool = False                         # needs labels
    learning_curve: tuple[int, ...] = ()        # runs per point; empty: not run (needs learned)
    others: dict = field(default_factory=dict)  # other registered methods: {name: settings} (registry.py)


@dataclass
class Execution:
    workers: int = 4
    cache: str = "data/cache"


@dataclass
class Settings:
    source: Source = field(default_factory=Source)
    frame: FrameSettings = field(default_factory=FrameSettings)
    describers: Describers = field(default_factory=Describers)
    methods: Methods = field(default_factory=Methods)
    execution: Execution = field(default_factory=Execution)


# ---------------------------------------------------------------- results
@dataclass
class Results:
    settings: Settings
    ensemble: Ensemble | None                   # with the frame settings applied (None for a table)
    runs: pd.DataFrame                          # run, parameters (and reference for a demo)
    labels: pd.Series | None                    # label of each run (index run), if known
    parameters: list[str]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)       # description -> table
    walls: dict[str, Walls] = field(default_factory=dict)
    robustness: dict[str, pd.DataFrame] = field(default_factory=dict)
    groups: dict[str, pd.DataFrame] = field(default_factory=dict)       # description -> groups of runs
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)          # description x method
    probe: dict[str, float] = field(default_factory=dict)
    training: pd.DataFrame | None = None                                # logs, column name
    extras: dict = field(default_factory=dict)                          # (description, method) -> its extra output
    learning_curve: pd.DataFrame | None = None

    def reference(self, name: str) -> pd.Series | None:
        if self.labels is None or name not in self.walls:
            return None
        runs = self.runs.assign(reference=self.labels.loc[self.runs.run].to_numpy())
        return report.node_labels(self.walls[name], runs)

    def phase_map(self, name: str):
        return report.make_phase_map(self.walls[name], self.reference(name))

    def snapshots(self, name: str):
        return report.make_snapshots(self.walls[name], self.ensemble)

    def robustness_plot(self, name: str):
        return report.make_robustness(self.robustness[name], self.settings.methods.ratio)

    def methods_map(self):
        cols = [c for c in methods_list(self.settings.methods)]
        return report.make_methods_map(self.groups, self.parameters, cols, self.scores)

    def training_plot(self):
        return report.make_training(self.training)

    def learning_curve_plot(self):
        return report.make_learning_curve(self.learning_curve)


def methods_list(m: Methods) -> list[str]:
    """Names of the grouping methods asked for (columns of Results.groups)."""
    return [col for col, _, _ in method_calls(m)]


def method_calls(m: Methods, table: pd.DataFrame | None = None) -> list[tuple[str, str, object]]:
    """(column of Results.groups, registered method, its settings) for every method asked for.
    With a table, the tree splits on principal components when every descriptor is learned."""
    from .builtins import HDBSCANSettings, KMeansSettings, MixtureSettings, TreeSettings, WallsSettings
    out = []
    if m.walls:
        out.append(("walls", "walls", WallsSettings(m.alpha, m.ratio, m.min_phase, m.merge)))
    if m.tree:
        learned_only = table is not None and all(c.startswith("cnn.") for c in descriptor_columns(table))
        out.append(("tree", "tree", TreeSettings(components=learned_only)))
    if m.mixture:
        out.append(("mixture", "mixture", MixtureSettings()))
    if m.kmeans:
        out.append(("kmeans", "kmeans", KMeansSettings()))
    out += [(f"hdbscan_{size}", "hdbscan", HDBSCANSettings(size)) for size in m.hdbscan]
    out += [(name, name, settings) for name, settings in m.others.items()]
    return out


# ---------------------------------------------------------------- cache
def key(*parts) -> str:
    """A short hash of settings (dataclasses, dicts, values): registry.cache_key."""
    return registry.cache_key(*parts)


def _demo_arguments(source: Source) -> tuple[object, dict, tuple, dict]:
    module = example_module(source.example)
    quick = dict(module.QUICK) if source.quick else {}
    grid = dict(source.grid or {p: quick.get(p, v) for p, v in module.GRID.items()})
    settings = {k: v for k, v in quick.items() if k not in module.GRID} | dict(source.settings)
    n = source.seeds or (QUICK_SEEDS if source.quick else len(module.SEEDS))
    return module, grid, tuple(range(n)), settings


def ensemble_for(source: Source, cache: Path, workers: int = 1, progress: Progress | None = None
                 ) -> tuple[Ensemble, pd.DataFrame, pd.Series | None, str]:
    """The ensemble of a demo (simulated once, then read from the cache) or of a TOML file;
    with its runs, labels and cache key."""
    if source.kind() == "config":
        ens = load_ensemble(source.config)
        labels = ens.runs.set_index("run")[source.label] if source.label else None
        return ens, ens.runs, labels, key(Path(source.config).resolve(), Path(source.config).stat().st_mtime)
    module, grid, seeds, settings = _demo_arguments(source)
    # the key holds the full effective settings, so that a change of a demo's defaults is not
    # hidden by an ensemble cached with the old ones
    k = key(source.example, {p: list(map(float, v)) for p, v in grid.items()}, seeds, module.SETTINGS | settings)
    folder = cache / f"ensemble-{source.example}-{k}"
    if not (folder / "phasewall.toml").exists():
        tmp = cache / f"tmp-{k}"
        shutil.rmtree(tmp, ignore_errors=True)
        module.make_ensemble(tmp, seeds=seeds, workers=workers, progress=progress, **grid, **settings)
        tmp.rename(folder)
    ens = load_ensemble(folder / "phasewall.toml")
    runs = pd.read_csv(folder / "runs.csv")
    return ens, runs, runs.set_index("run")["reference"], k


def framed(ens: Ensemble, f: FrameSettings) -> Ensemble:
    """The ensemble with the frame settings applied."""
    return dataclasses.replace(ens, crop=f.crop, dx_target=f.dx_target, max_shift=f.max_shift,
                               min_wavelengths=f.min_wavelengths,
                               periodic=ens.periodic if f.periodic is None else f.periodic)


def with_blocks(ens: Ensemble, blocks: dict) -> Ensemble:
    """The ensemble with the blocks of some fields replaced ({field name: blocks})."""
    from .builtins import _with_blocks
    return _with_blocks(ens, blocks)


# ---------------------------------------------------------------- estimate
def estimate(settings: Settings) -> dict[str, float]:
    """Rough seconds of each step before running (demos only for the simulation; zero for a
    step already in the cache is not detected)."""
    s = settings
    out = {}
    if s.source.kind() != "example":
        return out
    module, grid, seeds, kw = _demo_arguments(s.source)
    runs = len(seeds) * int(pd.Series([len(v) for v in grid.values()]).prod())
    workers = max(1, s.execution.workers)
    out["simulate"] = runs * module.seconds_per_run(**kw) / workers
    n = (module.SETTINGS | kw)["n"]
    snaps = runs * (module.SETTINGS | kw)["n_save"]
    out["describe"] = snaps * n * n * module.N_FIELDS * DESCRIBE_SECONDS_PER_SNAPSHOT_POINT / workers
    if s.describers.learned or s.describers.combined:
        items = runs * 4 * s.describers.epochs           # 4 draws per run and epoch
        out["train"] = items * n * n * TRAIN_SECONDS_PER_ITEM_POINT * len(s.describers.seeds)
        if s.methods.learning_curve:
            out["learning curve"] = out["train"] * len(s.methods.learning_curve)
    return out


# ---------------------------------------------------------------- run
def run(settings: Settings, progress: Progress | None = None) -> Results:
    """The whole analysis (see the module docstring)."""
    s = settings
    cache = Path(s.execution.cache)
    cache.mkdir(parents=True, exist_ok=True)
    w = s.execution.workers

    if s.source.kind() == "table":
        params = list(s.source.parameters or [])
        table, labels = load_table(s.source.table, params, s.source.label, s.source.prefix)
        runs = table.groupby("run")[params].first().reset_index()
        res = Results(s, None, runs, labels, params, tables={"table": table})
    else:
        ens0, runs, labels, ek = ensemble_for(s.source, cache, w, progress)
        ens = framed(ens0, s.frame)
        res = Results(s, ens, runs, labels, list(ens.parameters))
        d = s.describers
        ctx = registry.Context(workers=w, cache=str(cache), key=ek, frame=s.frame, progress=progress,
                               device=d.device)
        encoders = registry.encoders()
        hand = None
        if d.hand or d.combined:
            from .builtins import HandSettings
            hand = encoders["hand"].run(ens, HandSettings(d.blocks), ctx).tables["hand"]
            if d.hand:
                res.tables["hand"] = hand
        if d.learned or d.combined:
            _learned(res, ens, ctx, hand)
        for name, enc_settings in d.others.items():
            registry.check_settings(enc_settings)
            out = encoders[name].run(ens, enc_settings, ctx)
            res.tables |= out.tables
            if out.log is not None:
                res.training = pd.concat([res.training, out.log], ignore_index=True) if res.training is not None \
                    else out.log

    for name, table in res.tables.items():
        _methods(res, name, table, progress)
    if res.scores is not None and len(res.scores):
        res.scores = res.scores.reset_index(drop=True)
    return res


def _learned(res: Results, ens: Ensemble, ctx: registry.Context, hand) -> None:
    from .builtins import LearnedSettings
    from .nn.train import Settings as TrainSettings

    d = res.settings.describers
    out = registry.encoders()["learned"].run(ens, LearnedSettings(d.epochs, tuple(d.seeds)), ctx)
    for name, table in out.tables.items():
        if d.learned:
            res.tables[name] = table
        if d.combined and hand is not None:
            res.tables[name.replace("learned", "combined")] = combine_tables(hand, table)
    res.training = out.log
    m = res.settings.methods
    if m.learning_curve:
        f = Path(ctx.cache) / f"curve-{key(ctx.key, res.settings.frame, d.epochs, d.seeds, m.learning_curve)}.pkl"
        if f.exists():
            res.learning_curve = pd.read_pickle(f)
        else:
            from .nn.curve import learning_curve
            from .nn.data import load_frames
            frames = load_frames(ens, ctx.workers, progress=ctx.progress)
            res.learning_curve = learning_curve(ens, frames, m.learning_curve, d.seeds,
                                                TrainSettings(epochs=d.epochs), res.labels, d.device, ctx.progress)
            res.learning_curve.to_pickle(f)


def _methods(res: Results, name: str, table: pd.DataFrame, progress) -> None:
    m, params = res.settings.methods, res.parameters
    rows = []
    groups = table.groupby("run")[params].first().reset_index()
    ctx = registry.Context(workers=res.settings.execution.workers, cache=res.settings.execution.cache,
                           progress=progress)
    entries = registry.methods()
    for col, method_name, settings in method_calls(m, table):
        registry.check_settings(settings)
        out = entries[method_name].run(table, params, settings, ctx)
        groups[col] = out.groups.loc[groups.run].to_numpy()
        res.extras[(name, col)] = out.extra
        if col == "walls":
            res.walls[name] = out.extra
            if m.robustness:
                res.robustness[name] = phases_against_ratio(table, params, alpha=m.alpha, min_phase=m.min_phase)
    if m.probe and res.labels is not None:
        from . import baselines  # needs the extra [baselines] (scikit-learn)
        res.probe[name] = baselines.probe(table, params, res.labels)
    for col in [c for c in methods_list(m) if c in groups]:
        rows.append({"describer": name, "method": col, **point_scores(groups, col, params, res.labels)})
    res.groups[name] = groups
    res.scores = pd.concat([res.scores, pd.DataFrame(rows)], ignore_index=True)
    if progress:
        progress("methods", len(res.groups), len(res.tables))