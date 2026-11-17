"""Learning curve of the learned description: does it change with the amount of training data?

The encoder is trained on the first m runs of every parameter point (m in runs_per_point),
with the same number of gradient steps whatever m (draws scaled), and evaluated on all the
pairs of the ensemble: the walls are searched on its description, and the phases, walls and
(with labels) the agreement are reported. On the Swift-Hohenberg demo (2026-11-17) 2, 4 and 8
runs per point gave the same result.
"""

import dataclasses

import numpy as np
import pandas as pd

from ..io import Ensemble
from ..report import agreement, node_labels
from ..walls import find_walls
from .data import FrameSet
from .train import Settings, embed, h_table, train


def first_runs(ens: Ensemble, m: int) -> np.ndarray:
    """The first m runs (by run number) of every parameter point."""
    key = ens.runs[ens.parameters].round(9).apply(tuple, axis=1)
    return ens.runs.assign(key=key).sort_values("run").groupby("key").head(m)["run"].to_numpy()


def learning_curve(ens: Ensemble, frames: FrameSet, runs_per_point=(2, 4, 8), seeds=(0,),
                   settings: Settings | None = None, labels: pd.Series | None = None, dev=None,
                   progress=None) -> pd.DataFrame:
    """One row per (runs per point, seed): training runs, last loss, phases, walls and, with
    labels (Series indexed by run), the points agreeing with them."""
    base = settings or Settings()
    full = max(runs_per_point)
    rows = []
    for i, m in enumerate(runs_per_point):
        keep = np.isin(frames.run, first_runs(ens, m))
        sub = dataclasses.replace(frames, x=frames.x[keep], run=frames.run[keep], time=frames.time[keep])
        for seed in seeds:
            s = dataclasses.replace(base, seed=seed, draws=max(1, round(base.draws * full / m)))
            enc, log = train(sub, s, dev=dev, verbose=False)
            w = find_walls(h_table(frames, embed(enc, frames), ens), ens.parameters)
            row = {"runs_per_point": m, "seed": seed, "train_runs": len(set(sub.run)),
                   "loss": float(log.loss.iloc[-1]), "phases": w.n_phases, "walls": int(w.edges.wall.sum())}
            if labels is not None:
                row["agree"] = agreement(w, node_labels(w, ens.runs.assign(reference=labels.loc[ens.runs.run].to_numpy())))[1]
            rows.append(row)
        if progress:
            progress("learning curve", i + 1, len(runs_per_point))
    return pd.DataFrame(rows)
