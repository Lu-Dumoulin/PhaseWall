"""The built-in encoders and methods, registered with the same interface as any extension
(registry.py): they are thin wrappers around describe.py, nn/, walls.py, trees.py and
baselines.py, which keep working on their own.

Encoders: "hand" (hand-made descriptors), "learned" (contrastive CNN, extra [nn]).
Methods: "walls", "tree" (on the grid); "mixture", "kmeans", "hdbscan" (blind, extra
[baselines]). The settings below are the defaults of the pipeline; each field says what it
does, and its range or choices, for the interface.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .describe import describe_ensemble
from .graph import parameter_nodes
from .nn.architecture import Network, check
from .registry import Context, EncoderResult, MethodResult, cache_key, encoder, method
from .trees import AGREE, SYMMETRIC
from .walls import ALPHA, MERGE, MIN_PHASE, RATIO, find_walls


def _groups_of_points(table: pd.DataFrame, parameters: list[str], phase_of_point: np.ndarray) -> pd.Series:
    """Group of every run from the phase of its parameter point (points numbered as
    graph.parameter_nodes numbers them)."""
    runs = table.groupby("run")[parameters].first()
    run_node, _ = parameter_nodes(runs.to_numpy())       # the same numbering as for the snapshots
    return pd.Series(phase_of_point[run_node], index=runs.index, name="group")


# ---------------------------------------------------------------- encoders
@dataclass
class HandSettings:
    blocks: dict = field(default_factory=dict, metadata={
        "help": "blocks of observables per field, {field name: (block, ...)}; empty: the defaults of each kind"})


def _with_blocks(ens, blocks: dict):
    unknown = sorted(set(blocks) - {f.name for f in ens.fields})
    if unknown:
        raise ValueError(f"blocks given for unknown fields {unknown}")
    fields = [dataclasses.replace(f, blocks=tuple(blocks[f.name])) if f.name in blocks else f for f in ens.fields]
    return dataclasses.replace(ens, fields=fields)


@encoder("hand", HandSettings, help="hand-made descriptors per field (describe.py)", invariant=True,
         test_settings=HandSettings())
def hand(ens, settings: HandSettings, context: Context) -> EncoderResult:
    ens = _with_blocks(ens, settings.blocks)
    f = None
    if context.cache:
        Path(context.cache).mkdir(parents=True, exist_ok=True)
        f = Path(context.cache) / f"hand-{cache_key(context.key, context.frame, *([settings.blocks] if settings.blocks else []))}.pkl"
        if f.exists():
            return EncoderResult({"hand": pd.read_pickle(f)})
    table, errors = describe_ensemble(ens, context.workers, context.progress)
    if errors:
        raise RuntimeError(f"{len(errors)} runs could not be described, e.g. {errors[0]}")
    if f is not None:
        table.to_pickle(f)
    return EncoderResult({"hand": table})


@dataclass
class LearnedSettings:
    epochs: int = field(default=40, metadata={"help": "training epochs", "range": (1, 1000)})
    seeds: tuple = field(default=(0,), metadata={"help": "one training per seed", "range": (0, 10_000)})
    network: Network = field(default_factory=Network, metadata={
        "help": "shape of the encoder and its head (nn/architecture.py), checked against the frame"})


@encoder("learned", LearnedSettings, help="contrastive convolutional encoder trained on the ensemble (nn/)",
         needs=("torch",), test_settings=LearnedSettings(epochs=1, network=Network(blocks=2, width=8, dim_h=32,
                                                                                   hidden=32, projection=16)))
def learned(ens, settings: LearnedSettings, context: Context) -> EncoderResult:
    from .nn.data import load_frames
    from .nn.train import Settings as TrainSettings
    from .nn.train import embed, h_table, save, train

    net = settings.network
    # the network (with its views) is always part of the cache name: trainings cached before the
    # default views changed (2026-11-19) must not be read as trainings with the new ones
    extra = [net]
    frames, tables, logs = None, {}, []
    for seed in settings.seeds:
        folder = Path(context.cache) / f"learned-{cache_key(context.key, context.frame, settings.epochs, seed, *extra)}" \
            if context.cache else None
        if folder is not None and (folder / "table.pkl").exists():
            table, log = pd.read_pickle(folder / "table.pkl"), pd.read_csv(folder / "log.csv")
        else:
            frames = frames or load_frames(ens, context.workers, progress=context.progress)
            check(net, frames.x.shape[-1], frames.periodic)
            ts = TrainSettings.from_network(net, epochs=settings.epochs, seed=seed)
            enc, log = train(frames, ts, dev=context.device, verbose=False, progress=context.progress)
            table = h_table(frames, embed(enc, frames), ens)
            if folder is not None:
                save(folder, enc, frames, ts, log)
                table.to_pickle(folder / "table.pkl")
        tables[f"learned seed {seed}"] = table
        logs.append(log.assign(name=f"learned seed {seed}"))
    return EncoderResult(tables, pd.concat(logs, ignore_index=True))


# ---------------------------------------------------------------- methods on the grid
@dataclass
class WallsSettings:
    alpha: float = field(default=ALPHA, metadata={"help": "Mann-Whitney threshold of a wall", "range": (1e-9, 0.5)})
    ratio: float = field(default=RATIO, metadata={"help": "crossing jumps / inner jumps needed", "range": (1.0, 100.0)})
    min_phase: int = field(default=MIN_PHASE, metadata={"help": "smallest phase, in points", "range": (1, 1000)})
    merge: bool = field(default=MERGE, metadata={"help": "join separated parts of one phase"})


@method("walls", WallsSettings, help="walls between neighbouring points: spectral cuts, tested (walls.py)",
        uses_grid=True)
def walls(table, parameters, settings: WallsSettings, context: Context) -> MethodResult:
    w = find_walls(table, parameters, alpha=settings.alpha, ratio=settings.ratio, min_phase=settings.min_phase,
                   merge=settings.merge)
    return MethodResult(_groups_of_points(table, parameters, w.nodes["phase"].to_numpy()), w)


@dataclass
class TreeSettings:
    halves_by: str = field(default="auto", metadata={
        "help": "independent halves: of the runs of each point, or a checkerboard of the points (one run per "
                "point); auto chooses", "choices": ("auto", "runs", "grid")})
    symmetric: bool = field(default=SYMMETRIC, metadata={"help": "keep a split only if both halves find it"})
    merge: bool = field(default=MERGE, metadata={"help": "join separated leaves that are one phase"})
    swap: bool = field(default=False, metadata={"help": "show the rule found by the second half (stability check)"})
    components: bool = field(default=False, metadata={
        "help": "split on principal components (for a learned description) instead of single descriptors"})
    alpha: float = field(default=ALPHA, metadata={"help": "Mann-Whitney threshold of a split", "range": (1e-9, 0.5)})
    ratio: float = field(default=RATIO, metadata={"help": "crossing / inner jump of the descriptor needed",
                                                  "range": (1.0, 100.0)})
    agree: float = field(default=AGREE, metadata={"help": "share of points on the same side with the other half",
                                                  "range": (0.5, 1.0)})


@method("tree", TreeSettings, help="unsupervised threshold tree: phases as rules on single descriptors (trees.py)",
        uses_grid=True)
def tree(table, parameters, settings: TreeSettings, context: Context) -> MethodResult:
    from .trees import fit_tree
    t = fit_tree(table, parameters, use_components=settings.components, swap=settings.swap, alpha=settings.alpha,
                 ratio=settings.ratio, agree_min=settings.agree, symmetric=settings.symmetric, merge=settings.merge,
                 halves_by=settings.halves_by)
    return MethodResult(_groups_of_points(table, parameters, t.nodes["phase"].to_numpy()), t)


# ---------------------------------------------------------------- blind methods (scikit-learn)
def _reduced(table, parameters):
    from .baselines import reduce, run_matrix
    M, runs = run_matrix(table, parameters)
    return reduce(M), runs["run"].to_numpy()


@dataclass
class MixtureSettings:
    k_max: int = field(default=10, metadata={"help": "largest number of Gaussians tried (from 1)", "range": (1, 50)})


@method("mixture", MixtureSettings, help="Gaussian mixture of the runs, number by BIC", needs=("sklearn",))
def mixture(table, parameters, settings: MixtureSettings, context: Context) -> MethodResult:
    from .baselines import fit_mixture
    Z, runs = _reduced(table, parameters)
    r = fit_mixture(Z, ks=tuple(range(1, settings.k_max + 1)))
    return MethodResult(pd.Series(r["labels"], index=runs, name="group"), {"k": r["k"], "bic": r["bic"]})


@dataclass
class KMeansSettings:
    k_max: int = field(default=10, metadata={"help": "largest k tried (from 2)", "range": (2, 50)})


@method("kmeans", KMeansSettings, help="k-means of the runs, k by silhouette", needs=("sklearn",))
def kmeans(table, parameters, settings: KMeansSettings, context: Context) -> MethodResult:
    from .baselines import fit_kmeans
    Z, runs = _reduced(table, parameters)
    r = fit_kmeans(Z, ks=tuple(range(2, settings.k_max + 1)))
    return MethodResult(pd.Series(r["labels"], index=runs, name="group"), {"k": r["k"], "silhouette": r["silhouette"]})


@dataclass
class HDBSCANSettings:
    min_size: int = field(default=10, metadata={"help": "smallest group, in runs", "range": (2, 1000)})


@method("hdbscan", HDBSCANSettings, help="HDBSCAN: dense groups of runs, the rest noise", needs=("sklearn",))
def hdbscan(table, parameters, settings: HDBSCANSettings, context: Context) -> MethodResult:
    from .baselines import fit_hdbscan
    Z, runs = _reduced(table, parameters)
    return MethodResult(pd.Series(fit_hdbscan(Z, (settings.min_size,))[settings.min_size], index=runs, name="group"))
