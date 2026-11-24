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
from typing import ClassVar

import numpy as np
import pandas as pd

from . import registry, report
from .describe import combine_tables, descriptor_columns
from .examples import example as example_module
from .io import MIN_WAVELENGTHS, Ensemble, dims, load_any, load_table
from .nn.architecture import Network
from .parallel import Progress
from .scores import point_scores
from .walls import ALPHA, MERGE, MIN_PHASE, RATIO, Walls, phases_against_ratio

QUICK_SEEDS = 2                         # runs per point of a quick demo (as the command line)
# rough costs for estimate(), measured on a laptop (Apple M4 Max) on the demos (choices of display only)
DESCRIBE_SECONDS_PER_SNAPSHOT_POINT = 1.0e-7   # per snapshot, per grid point, per field, one process
TRAIN_SECONDS_PER_ITEM_POINT = 9.0e-8          # per training item (two views), per grid point, Apple GPU
CPU_TRAIN_SLOWER = 30.0      # training on the CPU instead of the Apple GPU: 0.55 against 0.018 s per step of the
                             # default network, batch 128, 48 x 48 input (Apple M4 Max, 12 threads, 2026-11-24)
JEPA_SECONDS_PER_SEED = (70, 370)   # JEPA training on the Apple GPU, per seed, on the 2D examples (not modelled)


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
    """The frame described in every run (frames.py; 2D) or the windows of every kymograph
    (oned/kymo.py; 1D); None keeps the ensemble's own value."""

    crop: float | None = None
    dx_target: float | None = None
    max_shift: float | None = None
    min_wavelengths: float = MIN_WAVELENGTHS
    periodic: bool | None = None
    share: float | None = None                  # 1D: share of each run described, counted from its end
    windows: int | None = None                  # 1D: windows cut from that share
    OPTIONAL_KEYS: ClassVar[tuple] = ("share", "windows")   # registry.cache_key: left out while None


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
    network: Network = field(default_factory=Network)   # learned: shape of the encoder (nn/architecture.py)


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
        """Snapshots (2D) or kymographs (1D) of one run in each phase."""
        if dims(self.ensemble) == 1:
            from .oned.report1d import make_kymographs
            return make_kymographs(self.walls[name], self.ensemble)
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
        ens = load_any(source.config)
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
    ens = load_any(folder / "phasewall.toml")
    runs = pd.read_csv(folder / "runs.csv")
    return ens, runs, runs.set_index("run")["reference"], k


def framed(ens: Ensemble, f: FrameSettings) -> Ensemble:
    """The ensemble with the frame settings applied; a setting left None keeps the ensemble's
    own value (from its TOML file)."""
    def keep(new, old):
        return old if new is None else new
    if dims(ens) == 1:
        return dataclasses.replace(ens, share=keep(f.share, ens.share), windows=keep(f.windows, ens.windows),
                                   periodic=keep(f.periodic, ens.periodic))
    return dataclasses.replace(ens, crop=keep(f.crop, ens.crop), dx_target=keep(f.dx_target, ens.dx_target),
                               max_shift=keep(f.max_shift, ens.max_shift), min_wavelengths=f.min_wavelengths,
                               periodic=keep(f.periodic, ens.periodic))


def fields_for(source: Source) -> list:
    """The fields of a source before anything is run: a demo's fields, those of a TOML file, or
    none for a table of descriptors."""
    kind = source.kind()
    if kind == "example":
        from .examples import fields_of
        return fields_of(source.example)
    if kind == "config":
        return list(load_any(source.config).fields)
    return []


def dims_for(source: Source) -> int:
    """1 for a 1D source (a 1D example or a [kymograph] file), 2 otherwise (snapshots, tables)."""
    kind = source.kind()
    if kind == "example":
        from .examples import dims_of
        return dims_of(source.example)
    if kind == "config":
        return dims(load_any(source.config))
    return 2


@dataclass
class FrameGeometry:
    """The box of a 2D source, known before running: grid of a snapshot, spacing, boundaries, and the
    frame the ensemble asks for itself (None: not set)."""

    shape: tuple[int, ...]
    dx: float
    periodic: bool
    crop: float | None = None
    dx_target: float | None = None
    max_shift: float | None = None

    @property
    def side(self) -> int:
        """Points per side of the largest square in the box."""
        return int(min(self.shape[:2]))


def frame_geometry(source: Source) -> FrameGeometry | None:
    """The box of a 2D example or 2D TOML file (from its settings, or its first snapshot); None for a
    table, a 1D source, or when it cannot be read (the run itself then reports why)."""
    try:
        if dims_for(source) == 1:
            return None
        if source.kind() == "example":
            module, _, _, kw = _demo_arguments(source)
            st = module.SETTINGS | kw
            n = int(st["n"])
            dx = float(st["dx"]) if "dx" in st else 2 * np.pi * st["waves"] / n
            return FrameGeometry((n, n), dx, True)
        if source.kind() == "config":
            from .io import read_snapshot, run_snapshots
            own = load_any(source.config)
            first = read_snapshot(run_snapshots(own, own.runs.path.iloc[0])[-1][1], own.fields[:1], own.julia_order)
            dx = own.dx if own.dx is not None else float(own.runs["dx"].iloc[0])
            return FrameGeometry(tuple(first[own.fields[0].name].shape), dx, bool(own.periodic), own.crop,
                                 own.dx_target, own.max_shift)
    except (ValueError, KeyError, IndexError, FileNotFoundError, OSError):
        return None
    return None


def frame_of(geo: FrameGeometry, f: FrameSettings):
    """frames.Frame of a geometry with the settings f (None in f: the ensemble's own value)."""
    from .frames import frame_for
    return frame_for(geo.shape, geo.dx, geo.periodic if f.periodic is None else f.periodic,
                     f.crop if f.crop is not None else geo.crop,
                     f.dx_target if f.dx_target is not None else geo.dx_target,
                     f.max_shift if f.max_shift is not None else geo.max_shift)


SHARE_TOLERANCE = 0.025   # interface sliders step by 0.05: a share within half a step of the ensemble's own
                          # is taken as the ensemble's own (a choice)


def frame_from_shares(geo: FrameGeometry, crop_share: float, factor: int, shift_share: float,
                      periodic: bool | None = None) -> tuple[float | None, float | None, float | None]:
    """(crop, dx_target, max_shift) in physical units from the interface's sliders: the share of the box
    side described (1: the whole box), the coarsening factor (points merged per side) and the largest
    shift as a share of the box side. Each is None when it equals what the ensemble does already, so
    that the cache keys of the default frame do not change. The crop is rounded down to whole points of
    the coarse grid, and shrunk to fit with its margin in a non-periodic box."""
    from .frames import MAX_SHIFT_FRACTION
    L = geo.side * geo.dx
    own_factor = 1 if geo.dx_target is None else max(1, round(geo.dx_target / geo.dx))
    dx_target = None if factor == own_factor else factor * geo.dx
    own_shift = MAX_SHIFT_FRACTION * L if geo.max_shift is None else geo.max_shift
    shift = shift_share * L
    max_shift = None if abs(shift - own_shift) <= SHARE_TOLERANCE * L else shift
    own_crop_share = 1.0 if geo.crop is None else geo.crop / L
    if abs(crop_share - own_crop_share) <= SHARE_TOLERANCE:
        return None, dx_target, max_shift
    per = geo.periodic if periodic is None else periodic
    margin = 0 if per else 2 * int(np.floor(shift / (geo.dx * factor) + 1e-9)) * factor
    n = int(np.floor(min(crop_share * geo.side, geo.side - margin) / factor + 1e-9))
    return max(n, 1) * factor * geo.dx, dx_target, max_shift


def frame_text(geo: FrameGeometry, f: FrameSettings) -> str:
    """One line on the frame the encoders will see, for the interface."""
    fr = frame_of(geo, f)
    side = (geo.side if fr.crop is None else fr.crop) // fr.factor
    where = "the whole box" if fr.crop is None else f"a centred square of {fr.crop * geo.dx:.3g}"
    return (f"box: {geo.shape[0]} x {geo.shape[1]} points, dx = {geo.dx:.3g}, side {geo.side * geo.dx:.3g}"
            f"{', periodic' if geo.periodic else ''}. Frame: {where}, {side} x {side} points, dx = {fr.dx:.3g}"
            f"{'' if fr.periodic else ' (not periodic)'}; moving correlation: "
            + ("every circular shift." if fr.periodic else f"shifts up to {fr.shift} points."))


def frame_side(settings: Settings) -> int | None:
    """Points per side of the frame the encoders will see (after crop and coarsening), for checks
    made before running (the shape of the learned encoder); None for a table, or when it cannot
    be known (a file missing, a frame that does not fit: the run itself reports why)."""
    geo = frame_geometry(settings.source)
    if geo is None:
        return None
    try:
        fr = frame_of(geo, settings.frame)
    except ValueError:
        return None
    return (geo.side if fr.crop is None else fr.crop) // fr.factor


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
    st = module.SETTINGS | kw
    n = st["n"]
    if getattr(module, "DIMS", 2) == 1:                 # values described per run, and per training item
        values, item = st["rows"] * n * module.N_FIELDS / 2, st["rows"] * n / 16
    else:
        values, item = st["n_save"] * n * n * module.N_FIELDS, n * n
    out["describe"] = runs * values * DESCRIBE_SECONDS_PER_SNAPSHOT_POINT / workers
    if s.describers.learned or s.describers.combined:
        items = runs * 4 * s.describers.epochs           # 4 draws per run and epoch
        out["train"] = items * item * TRAIN_SECONDS_PER_ITEM_POINT * len(s.describers.seeds)
        if s.methods.learning_curve:
            out["learning curve"] = out["train"] * len(s.methods.learning_curve)
    return out


def training_device(device: str | None) -> str:
    """The device a training will use: the one asked for, else the fastest available ("cuda", "mps",
    "cpu"); "cpu" without PyTorch."""
    if device is not None:
        return device
    from importlib.util import find_spec
    if find_spec("torch") is None:
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


DEVICE_NAMES = {"cuda": "the CUDA GPU", "mps": "the Apple GPU (mps)", "cpu": "the CPU"}


def estimate_details(settings: Settings, cores: int | None = None) -> list[dict]:
    """estimate() with, for each step, where it runs and what an alternative would cost: rows
    {step, seconds (None: not estimated), where, alternative}. cores: the cores of this computer."""
    import os
    s, est = settings, estimate(settings)
    cores = cores or os.cpu_count() or 1
    w = max(1, s.execution.workers)
    dev = training_device(s.describers.device)
    rows = []
    for step in ("simulate", "describe"):
        if step in est:
            faster = seconds_text(est[step] * w / cores)
            alt = (f"with all {cores} cores: {'about ' if faster != 'under 1 s' else ''}{faster}" if w < cores
                   else "already on every core")
            if step == "simulate":
                alt += "; kept in the cache, seconds the next time"
            rows.append({"step": step, "seconds": est[step], "where": f"{w} CPU cores in parallel (CPU only)", "alternative": alt})
    for step in ("train", "learning curve"):
        if step in est:
            seconds = est[step] * (CPU_TRAIN_SLOWER if dev == "cpu" else 1.0)
            if dev == "cpu":
                alt = (f"on the Apple GPU: about {seconds_text(seconds / CPU_TRAIN_SLOWER)}"
                       if training_device(None) == "mps" else "no GPU here")
            else:
                alt = f"on the CPU: about {seconds_text(seconds * CPU_TRAIN_SLOWER)}"
            where = DEVICE_NAMES[dev] + (" (GPU possible)" if dev == "cpu" else "") + \
                ("" if dev != "cuda" else " (time calibrated on an Apple GPU)")
            rows.append({"step": step, "seconds": seconds, "where": where,
                         "alternative": alt + f"; {len(s.describers.seeds)} seed(s), one after the other"})
    if "jepa" in s.describers.others:
        lo, hi = JEPA_SECONDS_PER_SEED
        rows.append({"step": "train (JEPA)", "seconds": None,
                     "where": DEVICE_NAMES[dev] + (" (GPU possible)" if dev == "cpu" else ""),
                     "alternative": f"not estimated: {seconds_text(lo)} to {seconds_text(hi)} per seed on the Apple GPU "
                                    "on the 2D examples; about 30 times longer on the CPU"})
    if est:
        rows.append({"step": "methods", "seconds": None, "where": "one CPU core (CPU only)",
                     "alternative": "walls, tree, clustering: usually seconds; the robustness curve repeats the walls"})
    return rows


def seconds_text(seconds: float) -> str:
    """A duration in words: under 1 s, 12 s, 3.4 min, 1.2 h (as the interface)."""
    if seconds < 1:
        return "under 1 s"
    return (f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min" if seconds < 5400
            else f"{seconds / 3600:.1f} h")


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
    out = registry.encoders()["learned"].run(ens, LearnedSettings(d.epochs, tuple(d.seeds), d.network), ctx)
    for name, table in out.tables.items():
        if d.learned:
            res.tables[name] = table
        if d.combined and hand is not None:
            res.tables[name.replace("learned", "combined")] = combine_tables(hand, table)
    res.training = out.log
    m = res.settings.methods
    if m.learning_curve and dims(ens) == 1:
        raise ValueError("the learning curve is not available for 1D ensembles yet")
    if m.learning_curve:
        extra = [] if d.network == Network() else [d.network]
        f = Path(ctx.cache) / f"curve-{key(ctx.key, res.settings.frame, d.epochs, d.seeds, m.learning_curve, *extra)}.pkl"
        if f.exists():
            res.learning_curve = pd.read_pickle(f)
        else:
            from .nn.curve import learning_curve
            from .nn.data import load_frames
            frames = load_frames(ens, ctx.workers, progress=ctx.progress)
            res.learning_curve = learning_curve(ens, frames, m.learning_curve, d.seeds,
                                                TrainSettings.from_network(d.network, epochs=d.epochs), res.labels,
                                                d.device, ctx.progress)
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