"""The unsupervised threshold tree (phasewall.trees, default settings) on the hand-made description
of every example of scripts/run_examples.py, scored like the other methods against the reference
labels (used only for the score); the tree with the halves swapped as a stability check; and the
same with one run per point (the first run of every point), where the tree uses checkerboard halves.

Usage: python scripts/tree_examples.py [--only gray-scott ...] [--workers 8]

Reads the cached ensembles and hand-made tables (data/cache; run scripts/run_examples.py first).
Writes docs/examples/<name>/tree.txt (rules, splits tried, merges), phase_map_tree.png and
docs/examples/trees.csv.
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_examples import JOBS, OUT

from phasewall import pipeline as pl
from phasewall import report, trees
from phasewall.scores import point_scores

warnings.simplefilter("ignore", RuntimeWarning)


def score(t, table, params, labels) -> dict:
    point = pd.Series(t.nodes["phase"].to_numpy(), index=pd.MultiIndex.from_frame(t.nodes[params].round(9)))
    groups = table[["run", *params]].drop_duplicates("run").reset_index(drop=True)
    groups["tree"] = point.loc[pd.MultiIndex.from_frame(groups[params].round(9))].to_numpy()
    return point_scores(groups, "tree", params, labels)


def run_one(name: str, workers: int) -> list[dict]:
    settings = pl.Settings(source=JOBS[name],
                           describers=pl.Describers(hand=True, learned=False, combined=False),
                           methods=pl.Methods(walls=False, robustness=False, mixture=False, kmeans=False, hdbscan=(),
                                              probe=False),
                           execution=pl.Execution(workers=workers, cache="data/cache"))
    res = pl.run(settings)
    table, params = res.tables["hand"], list(res.parameters)
    labels = res.labels
    first = table.groupby(params).run.transform("min")
    rows, text = [], []
    for data, tab in (("all runs", table), ("one run per point", table[table.run == first])):
        base = None
        for variant, kw in (("default", {}), ("halves swapped", {"swap": True})):
            t = trees.fit_tree(tab, params, **kw)
            base = base if base is not None else t
            same = bool((t.nodes.phase.to_numpy() == base.nodes.phase.to_numpy()).all())
            s = score(t, tab, params, labels)
            rows.append({"example": name, "data": data, "variant": variant, "halves_by": t.halves_by,
                         "leaves": t.nodes.leaf.nunique(), "same_as_default": same, **s})
            if variant == "default":
                text += [f"## {data}: {t.n_phases} phases ({t.halves_by} halves), ARI {s['ari']:.3f}", "",
                         trees.rules_text(t), "", "Splits tried:", "", report._markdown(t.splits.round(4)), ""]
                if len(t.merges):
                    text += ["Separated leaves compared:", "", report._markdown(t.merges.round(4)), ""]
                if data == "all runs":
                    runs = res.runs.assign(reference=labels.loc[res.runs.run].to_numpy())
                    w = t.as_walls()
                    report.make_phase_map(w, report.node_labels(w, runs)).savefig(
                        OUT / name / "phase_map_tree.png", dpi=report.DPI, facecolor=report.SURFACE)
    (OUT / name / "tree.txt").write_text(f"# Unsupervised threshold tree: {name}\n\n" + "\n".join(text))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(JOBS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = [r for name in args.only for r in run_one(name, args.workers)]
    new = pd.DataFrame(rows)
    f = OUT / "trees.csv"
    if f.exists():                          # keep the examples not run this time (same columns only)
        old = pd.read_csv(f)
        if set(new.columns) <= set(old.columns):
            new = pd.concat([old[~old.example.isin(new.example)][new.columns], new], ignore_index=True)
    new.to_csv(f, index=False)
    print(new.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
