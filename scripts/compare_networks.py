"""Several shapes of the learned encoder (nn/architecture.py) trained on one example, compared by
their label-free diagnostics, as one would choose a network without labels; the scores against
the reference labels are written in a separate table, for the documentation only.

Usage: python scripts/compare_networks.py [--example swift-hohenberg] [--epochs 40] [--workers 8]

Label-free diagnostics of the last epoch: loss over the loss at collapse ln(2N - 1) (lower: the
views of a run are told apart from the others), alignment of the two views (near 1), uniformity
(lower: z spread over the sphere), effective rank of h over dim_h (the share of the dimensions
that are used). Writes docs/networks/<example>/diagnostics.csv, scores.csv and summary.md.
Trainings are cached by the pipeline (data/cache), so a rerun only trains new shapes.
"""

import argparse
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_examples import JOBS

from phasewall import pipeline as pl
from phasewall import report
from phasewall.nn.architecture import Network, summary

warnings.simplefilter("ignore", RuntimeWarning)

NETWORKS = {   # the default and one change at a time (a choice of the comparison)
    "default": Network(),
    "3 blocks": Network(blocks=3),
    "5 blocks": Network(blocks=5, dim_h=512, hidden=512),
    "narrow": Network(width=8, dim_h=64, hidden=64, projection=32),
    "wide h": Network(dim_h=512, hidden=512, projection=128),
    "T = 0.5": Network(temperature=0.5),
    "d4 views": Network(views="d4", normalise=False),     # the views used before 2026-11-19
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="swift-hohenberg", choices=list(JOBS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path("docs/networks") / args.example
    out.mkdir(parents=True, exist_ok=True)
    diag, scores = [], []
    for name, net in NETWORKS.items():
        t0 = time.time()
        s = pl.Settings(source=JOBS[args.example],
                        describers=pl.Describers(hand=False, learned=True, epochs=args.epochs, seeds=(0,), network=net),
                        methods=pl.Methods(walls=True, hdbscan=(20,), probe=True),
                        execution=pl.Execution(workers=args.workers, cache="data/cache"))
        res = pl.run(s)
        last = res.training.iloc[-1]
        from phasewall.io import read_snapshot, run_snapshots
        ens = res.ensemble
        first = read_snapshot(run_snapshots(ens, ens.runs.path.iloc[0])[-1][1], ens.fields, ens.julia_order)
        side = first[ens.fields[0].name].shape[0]
        channels = 2 * sum(1 if f.symmetry == "scalar" else 2 for f in ens.fields)   # pairs of frames
        info = summary(net, side, in_channels=channels)
        diag.append({"network": name, **{k: v for k, v in asdict(net).items()}, "channels": str(net.channels),
                     "receptive_field": info.get("receptive_field"), "weights_f": info.get("weights_f"),
                     "loss / collapse": last.loss / last.collapse_loss, "alignment": last.alignment,
                     "uniformity": last.uniformity, "rank of h / dim_h": last.effective_rank_h / net.dim_h,
                     "rank of h": last.effective_rank_h, "minutes": (time.time() - t0) / 60})
        for r in res.scores.itertuples():
            scores.append({"network": name, "method": r.method, "groups": r.groups, "agree": r.agree, "ari": r.ari,
                           "pieces": r.pieces, "probe": res.probe.get("learned seed 0")})
        print(f"{name}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    d, sc = pd.DataFrame(diag), pd.DataFrame(scores)
    d.to_csv(out / "diagnostics.csv", index=False)
    sc.to_csv(out / "scores.csv", index=False)
    cols = ["network", "channels", "hidden", "projection", "temperature", "receptive_field", "weights_f",
            "loss / collapse", "alignment", "uniformity", "rank of h", "rank of h / dim_h"]
    (out / "summary.md").write_text(
        f"# Networks compared on {args.example} ({args.epochs} epochs, seed 0)\n\n"
        "## Label-free diagnostics (last epoch): what a choice without labels can use\n\n"
        + report._markdown(d[cols].round(3)) + "\n\n"
        "## Scores against the reference labels (documentation only; not used to choose)\n\n"
        + report._markdown(sc.round(3)) + "\n")
    print(d[cols].round(3).to_string(index=False))
    print(sc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
