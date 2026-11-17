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
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import report
from .describe import combine_tables, describe_ensemble
from .examples import example as example_module
from .graph import parameter_nodes
from .io import MIN_WAVELENGTHS, Ensemble, load_ensemble, load_table
from .parallel import Progress
from .scores import point_scores
from .walls import ALPHA, MIN_PHASE, RATIO, Walls, find_walls, phases_against_ratio

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


@dataclass
class Methods:
    """Which methods, on every description."""

    walls: bool = True
    alpha: float = ALPHA
    ratio: float = RATIO
    min_phase: int = MIN_PHASE
    robustness: bool = False
    mixture: bool = False
    kmeans: bool = False
    hdbscan: tuple[int, ...] = ()               # minimum sizes; empty: not run
    probe: bool = False                         # needs labels
    learning_curve: tuple[int, ...] = ()        # runs per point; empty: not run (needs learned)


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
    out = ["walls"] if m.walls else []
    out += ["mixture"] * m.mixture + ["kmeans"] * m.kmeans + [f"hdbscan_{s}" for s in m.hdbscan]
    return out


# ---------------------------------------------------------------- cache
def key(*parts) -> str:
    """A short hash of settings (dataclasses, dicts, values)."""
    def plain(x):
        return asdict(x) if dataclasses.is_dataclass(x) else x
    text = json.dumps([plain(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


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
        hand = None
        if s.describers.hand or s.describers.combined:
            f = cache / f"hand-{key(ek, s.frame)}.pkl"
            if f.exists():
                hand = pd.read_pickle(f)
            else:
                hand, errors = describe_ensemble(ens, w, progress)
                if errors:
                    raise RuntimeError(f"{len(errors)} runs could not be described, e.g. {errors[0]}")
                hand.to_pickle(f)
            if s.describers.hand:
                res.tables["hand"] = hand
        if s.describers.learned or s.describers.combined:
            _learned(res, ens, ek, hand, cache, w, progress)

    for name, table in res.tables.items():
        _methods(res, name, table, progress)
    if res.scores is not None and len(res.scores):
        res.scores = res.scores.reset_index(drop=True)
    return res


def _learned(res: Results, ens: Ensemble, ek: str, hand, cache: Path, workers: int, progress) -> None:
    from .nn.data import load_frames
    from .nn.train import Settings as TrainSettings
    from .nn.train import embed, h_table, save, train

    d = res.settings.describers
    frames, logs = None, []
    for seed in d.seeds:
        folder = cache / f"learned-{key(ek, res.settings.frame, d.epochs, seed)}"
        if (folder / "table.pkl").exists():
            table = pd.read_pickle(folder / "table.pkl")
            log = pd.read_csv(folder / "log.csv")
        else:
            frames = frames or load_frames(ens, workers, progress=progress)
            ts = TrainSettings(epochs=d.epochs, seed=seed)
            enc, log = train(frames, ts, dev=d.device, verbose=False, progress=progress)
            save(folder, enc, frames, ts, log)
            table = h_table(frames, embed(enc, frames), ens)
            table.to_pickle(folder / "table.pkl")
        logs.append(log.assign(name=f"learned seed {seed}"))
        if d.learned:
            res.tables[f"learned seed {seed}"] = table
        if d.combined and hand is not None:
            res.tables[f"combined seed {seed}"] = combine_tables(hand, table)
    res.training = pd.concat(logs, ignore_index=True)
    m = res.settings.methods
    if m.learning_curve:
        f = cache / f"curve-{key(ek, res.settings.frame, d.epochs, d.seeds, m.learning_curve)}.pkl"
        if f.exists():
            res.learning_curve = pd.read_pickle(f)
        else:
            from .nn.curve import learning_curve
            frames = frames or load_frames(ens, workers, progress=progress)
            res.learning_curve = learning_curve(ens, frames, m.learning_curve, d.seeds,
                                                TrainSettings(epochs=d.epochs), res.labels, d.device, progress)
            res.learning_curve.to_pickle(f)


def _methods(res: Results, name: str, table: pd.DataFrame, progress) -> None:
    m, params = res.settings.methods, res.parameters
    rows = []
    groups = table.groupby("run")[params].first().reset_index()
    if m.walls:
        w = find_walls(table, params, alpha=m.alpha, ratio=m.ratio, min_phase=m.min_phase)
        res.walls[name] = w
        node, _ = parameter_nodes(groups[params].to_numpy())
        groups["walls"] = w.nodes["phase"].to_numpy()[node]
        if m.robustness:
            res.robustness[name] = phases_against_ratio(table, params, alpha=m.alpha, min_phase=m.min_phase)
    blind = m.mixture or m.kmeans or m.hdbscan
    if blind or m.probe:
        from . import baselines  # needs the extra [baselines] (scikit-learn)
        if blind:
            g, _ = baselines.blind_groups(table, params, sizes=tuple(m.hdbscan))
            for col in [c for c in methods_list(m) if c != "walls"]:
                groups[col] = g.set_index("run").loc[groups.run, col].to_numpy()
        if m.probe and res.labels is not None:
            res.probe[name] = baselines.probe(table, params, res.labels)
    for col in [c for c in methods_list(m) if c in groups]:
        rows.append({"describer": name, "method": col, **point_scores(groups, col, params, res.labels)})
    res.groups[name] = groups
    res.scores = pd.concat([res.scores, pd.DataFrame(rows)], ignore_index=True)
    if progress:
        progress("methods", len(res.groups), len(res.tables))
