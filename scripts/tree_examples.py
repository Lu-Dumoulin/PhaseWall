"""The unsupervised threshold tree (phasewall.trees) on the hand-made description of every
example of scripts/run_examples.py, with the halves as given and swapped, scored like the other
methods against the reference labels (used only for the score).

Usage: python scripts/tree_examples.py [--only gray-scott ...] [--workers 8]

Reads the cached ensembles and hand-made tables (data/cache; run scripts/run_examples.py first).
Writes docs/examples/<name>/tree.txt (rules and tried splits) and docs/examples/trees.csv.
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


def run_one(name: str, workers: int) -> list[dict]:
    settings = pl.Settings(source=JOBS[name],
                           describers=pl.Describers(hand=True, learned=False, combined=False),
                           methods=pl.Methods(walls=False, robustness=False, mixture=False, kmeans=False, hdbscan=(),
                                              probe=False),
                           execution=pl.Execution(workers=workers, cache="data/cache"))
    res = pl.run(settings)
    table, params = res.tables["hand"], list(res.parameters)
    labels = res.runs.set_index("run")["reference"]
    rows, text = [], []
    for swap in (False, True):
        t = trees.fit_tree(table, params, swap=swap)
        point = pd.Series(t.nodes["phase"].to_numpy(), index=pd.MultiIndex.from_frame(t.nodes[params].round(9)))
        groups = table[["run", *params]].drop_duplicates("run").reset_index(drop=True)
        groups["tree"] = point.loc[pd.MultiIndex.from_frame(groups[params].round(9))].to_numpy()
        s = point_scores(groups, "tree", params, labels)
        rows.append({"example": name, "swap": swap, "splits_tried": len(t.splits),
                     "splits_accepted": int(t.splits.accepted.sum()) if len(t.splits) else 0, **s})
        text += [f"## halves {'swapped' if swap else 'as given'}: {t.n_phases} phases, ARI {s['ari']:.3f}", "",
                 trees.rules_text(t), "", report._markdown(t.splits.round(4)), ""]
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
    if f.exists():                          # keep the examples not run this time
        old = pd.read_csv(f)
        new = pd.concat([old[~old.example.isin(new.example)], new], ignore_index=True)
    new.to_csv(f, index=False)
    print(new.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
