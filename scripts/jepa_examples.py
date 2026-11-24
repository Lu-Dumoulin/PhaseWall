"""The predictive learned description (nn/jepa.py) on the five 2D examples, next to the hand-made one:
walls, threshold tree, HDBSCAN, the probe and the hand-made + predictive table; scored against the
reference labels (used only here). The contrastive learned description is read from
docs/examples/scores.csv for comparison.

    python scripts/jepa_examples.py [--only barkley ...] [--seeds 0 1] [--out data/jepa]

Needs the cached ensembles and hand-made tables of scripts/run_examples.py (data/cache). The JEPA
tables are kept in the out folder (<example>_jepa_seed<k>.csv) and read back on a second run.
Writes <out>/scores.csv, <out>/training.csv and a phase map per example and seed.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_examples import JOBS

from phasewall import baselines, report, trees
from phasewall import pipeline as pl
from phasewall.describe import combine_tables
from phasewall.nn.data import load_frames
from phasewall.nn.jepa import SettingsJEPA, h_table_jepa, train_jepa
from phasewall.scores import point_scores
from phasewall.walls import find_walls

warnings.simplefilter("ignore", RuntimeWarning)
METHODS = ["walls", "tree", "hdbscan_10", "hdbscan_20"]


def groups_of(table, params, learned: bool):
    w = find_walls(table, params)
    groups, _ = baselines.blind_groups(table, params, sizes=(10, 20))
    key = pd.MultiIndex.from_frame(groups[params].round(9))
    node_of = pd.Series(w.nodes.index, index=pd.MultiIndex.from_frame(w.nodes[params].round(9)))
    pos = node_of.reindex(key).to_numpy()
    ok = ~np.isnan(pos)
    groups["walls"] = -1
    groups.loc[ok, "walls"] = w.nodes.phase.to_numpy()[pos[ok].astype(int)]
    tr = trees.fit_tree(table, params, use_components=learned)
    t_idx = {tuple(r): i for i, r in enumerate(tr.nodes[params].round(9).to_numpy())}
    groups["tree"] = tr.nodes.phase.to_numpy()[[t_idx[tuple(r)] for r in groups[params].round(9).to_numpy()]]
    return groups, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(JOBS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--out", default="data/jepa")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, logs = [], []
    for name in args.only:
        st = pl.Settings(source=JOBS[name], describers=pl.Describers(hand=True, learned=False, combined=False),
                         methods=pl.Methods(walls=False, robustness=False, mixture=False, kmeans=False, hdbscan=(),
                                            probe=False),
                         execution=pl.Execution(workers=args.workers, cache="data/cache"))
        res = pl.run(st)
        params, labels = list(res.parameters), res.labels
        tables = {"hand": res.tables["hand"]}
        frames = None
        for seed in args.seeds:
            path = out / f"{name}_jepa_seed{seed}.csv"
            if path.exists():
                tables[f"jepa seed {seed}"] = pd.read_csv(path)
                continue
            frames = frames or load_frames(res.ensemble, args.workers)
            t0 = time.time()
            enc, log = train_jepa(frames, SettingsJEPA(seed=seed), verbose=False)
            logs.append(log.assign(example=name, seed=seed))
            print(f"{name} seed {seed}: {len(frames.x)} pairs {frames.x.shape[2:]}, trained in {time.time() - t0:.0f} s, "
                  f"loss {log.loss.iloc[-1]:.4f}, h std {log.h_std.iloc[-1]:.4f}", flush=True)
            tables[f"jepa seed {seed}"] = h_table_jepa(enc, frames, res.ensemble)
            tables[f"jepa seed {seed}"].to_csv(path, index=False)
        tables["hand + jepa seed 0"] = combine_tables(tables["hand"], tables[f"jepa seed {args.seeds[0]}"])
        for desc, table in tables.items():
            groups, w = groups_of(table, params, desc.startswith("jepa"))
            probe = baselines.probe(table, params, labels)
            for m in METHODS:
                rows.append({"example": name, "describer": desc, "method": m, "probe": probe,
                             **point_scores(groups, m, params, labels)})
            runs = res.runs.assign(reference=labels.loc[res.runs.run].to_numpy())
            if desc != "hand":
                report.make_phase_map(w, report.node_labels(w, runs)).savefig(
                    out / f"phase_map_{name}_{desc.replace(' ', '_').replace('+', 'and')}.png", dpi=report.DPI,
                    facecolor=report.SURFACE)
            print(f"{name} {desc}: walls {w.n_phases} phases, ARI {rows[-4]['ari']:.3f}; probe {probe:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
        if logs:
            pd.concat(logs).to_csv(out / "training.csv", index=False)


if __name__ == "__main__":
    main()
