"""Batch run of every example: simulated to steady state, checked for steady state, described
(hand-made, learned with seeds 0 and 1, combined) and compared (walls with robustness, Gaussian
mixture, k-means, HDBSCAN 10 and 20, logistic probe), with phasewall.pipeline.

Usage: python scripts/run_examples.py [--only barkley gray-scott ...] [--workers 8]

Resumable: simulations, descriptions and trainings are cached by the pipeline (data/cache), so a
rerun after a stop starts where it stopped. Outputs per example in docs/examples/<name>/
(summary.md, scores.csv, steady.csv, figures); all examples in docs/examples/scores.csv and
overview.md; log in data/night/log.txt.

Steady-state check: STEADY_RUNS runs (first those at points whose seeds end in different
reference states, then at random) are simulated again to 2 and 4 times the run time; their
reference label and the change of the first field between two snapshots `every` apart are
compared with the original run.
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from phasewall import pipeline as pl
from phasewall import report
from phasewall.examples import example

OUT, LOG = Path("docs/examples"), Path("data/night/log.txt")
STEADY_RUNS = 24          # runs rerun longer for the steady-state check (a choice)
STEADY_FACTORS = (2, 4)
EPOCHS, TRAIN_SEEDS = 40, (0, 1)  # learned description (as in the Swift-Hohenberg comparison)

GS_GRID = {"F": np.round(np.arange(0.010, 0.0701, 0.0025), 4), "k": np.round(np.arange(0.045, 0.0701, 0.00125), 5)}
JOBS = {   # in order: quick ones first
    "barkley": pl.Source(example="barkley"),
    "swift-hohenberg": pl.Source(example=None, config="data/demo/phasewall.toml", label="reference"),
    "fitzhugh-nagumo": pl.Source(example="fitzhugh-nagumo"),
    "gray-scott": pl.Source(example="gray-scott", grid=GS_GRID, settings={"t_relax": 40_000.0}),
    "sh-localized": pl.Source(example="sh-localized"),
}
MODULE_OF_CONFIG = {"swift-hohenberg": "swift-hohenberg"}   # example module of an ensemble given by config


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class Progress:
    """Logs every 10% of each step."""

    def __init__(self, name):
        self.name, self.last = name, {}

    def __call__(self, step, done, total):
        tenth = int(10 * done / max(total, 1))
        if self.last.get(step) != tenth:
            self.last[step] = tenth
            log(f"{self.name}: {step} {done}/{total}")


def _label_and_change(module, params: tuple, seed: int, settings: dict, t: float) -> tuple[str, float]:
    kw = settings | {"t_relax": t, "n_save": 2}
    snaps = module.simulate(*params, seed, **kw)
    (_, s0), (_, s1) = snaps
    if isinstance(s1, dict):
        label = module.reference_label([s0, s1])
        f0, f1 = s0["u"], s1["u"]
    else:
        label = module.reference_label(s1)
        f0, f1 = s0, s1
    a, b = f0 - f0.mean(), f1 - f1.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return label, float(1 - np.sum(a * b) / den) if den > 0 else 0.0


def _one_steady(args):
    module_name, params, seed, settings, times = args
    module = example(module_name)
    return [(t, *_label_and_change(module, params, seed, settings, t)) for t in times]


def steady_check(name: str, source: pl.Source, runs: pd.DataFrame, parameters: list[str], workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    module_name = source.example or MODULE_OF_CONFIG[name]
    module = example(module_name)
    settings = module.SETTINGS | dict(source.settings)
    key = runs[parameters].round(9).apply(tuple, axis=1)
    split = runs.groupby(key).reference.transform("nunique") > 1
    rng = np.random.default_rng(0)
    chosen = list(runs[split].sample(min(STEADY_RUNS // 2, int(split.sum())), random_state=0).index)
    rest = [i for i in runs.index if i not in chosen]
    chosen += list(rng.choice(rest, size=min(STEADY_RUNS - len(chosen), len(rest)), replace=False))
    times = [settings["t_relax"] * f for f in STEADY_FACTORS]
    jobs = [(module_name, tuple(float(runs.loc[i, p]) for p in parameters), int(runs.loc[i, "seed"]), settings, times)
            for i in chosen]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_one_steady, jobs))
    rows = []
    for i, res in zip(chosen, results):
        row = {"run": int(runs.loc[i, "run"]), **{p: runs.loc[i, p] for p in parameters}, "seed": int(runs.loc[i, "seed"]),
               "at_boundary": bool(split.loc[i]), "label": runs.loc[i, "reference"]}
        for t, lab, ch in res:
            row[f"label_x{round(t / settings['t_relax'])}"] = lab
            row[f"change_x{round(t / settings['t_relax'])}"] = ch
        rows.append(row)
    return pd.DataFrame(rows)


def run_one(name: str, source: pl.Source, workers: int) -> pd.DataFrame:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    settings = pl.Settings(
        source=source,
        describers=pl.Describers(hand=True, learned=True, combined=True, epochs=EPOCHS, seeds=TRAIN_SEEDS),
        methods=pl.Methods(walls=True, robustness=True, mixture=True, kmeans=True, hdbscan=(10, 20), probe=True),
        execution=pl.Execution(workers=workers, cache="data/cache"))
    if source.example:
        log(f"{name}: estimate {pl.estimate(settings)}")
    # the ensemble first (simulated once, cached), then the steady-state check, then the rest
    ens, runs, _labels, _ = pl.ensemble_for(source, Path("data/cache"), workers, Progress(name))
    log(f"{name}: ensemble ready, {len(runs)} runs, reference {runs.reference.value_counts().to_dict()}")
    steady_file = out / "steady.csv"
    if not steady_file.exists():
        steady = steady_check(name, source, runs, list(ens.parameters), workers)
        steady.to_csv(steady_file, index=False)
        log(f"{name}: steady-state check done")
    steady = pd.read_csv(steady_file)
    res = pl.run(settings, progress=Progress(name))
    log(f"{name}: analysis done in {(time.time() - t0) / 60:.0f} min")

    res.scores.to_csv(out / "scores.csv", index=False)
    for d in res.tables:
        slug = d.replace(" ", "_")
        res.phase_map(d).savefig(out / f"phase_map_{slug}.png", dpi=report.DPI, facecolor=report.SURFACE)
        res.robustness_plot(d).savefig(out / f"robustness_{slug}.png", dpi=report.DPI, facecolor=report.SURFACE)
    res.snapshots("hand").savefig(out / "snapshots_hand.png", dpi=report.DPI, facecolor=report.SURFACE)
    res.methods_map().savefig(out / "methods_map.png", dpi=report.DPI, facecolor=report.SURFACE)
    res.training_plot().savefig(out / "training.png", dpi=report.DPI, facecolor=report.SURFACE)
    _summary(name, res, steady, out, time.time() - t0)
    return res.scores.assign(example=name, probe=[res.probe.get(d) for d in res.scores.describer])


def _summary(name, res, steady, out, seconds) -> None:
    last = res.training.groupby("name").tail(1)[["name", "loss", "collapse_loss", "alignment", "uniformity",
                                                  "effective_rank_h"]]
    cols = [c for c in steady.columns if c.startswith("label_x")]
    changed = {c: int((steady[c] != steady.label).sum()) for c in cols}
    moving = {c.replace("label", "change"): float(steady[c.replace("label", "change")].max()) for c in cols}
    lines = [f"# {name}", "",
             (f"{len(res.runs)} runs on {res.walls['hand'].nodes.shape[0]} parameter points "
              f"({', '.join(res.parameters)}); analysis {seconds / 60:.0f} min."), "",
             "Reference labels (runs): " + ", ".join(f"{k} {v}" for k, v in res.runs.reference.value_counts().items()), "",
             "## Steady state", "",
             f"{len(steady)} runs ({int(steady.at_boundary.sum())} at points whose seeds end in different states) rerun "
             f"to {', '.join(c.replace('label_x', '') + 'x' for c in cols)} the run time: label changed in "
             + ", ".join(f"{v} runs at {k.replace('label_x', '')}x" for k, v in changed.items())
             + "; largest change of the field between two snapshots at the longest time: "
             + ", ".join(f"{v:.1e}" for v in list(moving.values())[-1:]) + ".", "",
             report._markdown(steady.round(6)), "",
             "## Scores", "",
             ("Each row: a description and a method; agree = points carrying the most common reference label of "
              "their group, ari = adjusted Rand index with the reference, pieces = connected pieces on the grid."), "",
             report._markdown(res.scores.round(3)), "",
             "Probe (logistic regression of the reference labels, held-out points, balanced accuracy): "
             + ", ".join(f"{k} {v:.3f}" for k, v in res.probe.items()), "",
             "## Trainings (last epoch)", "", report._markdown(last.round(3)), ""]
    (out / "summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(JOBS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    log(f"start: {args.only}")
    all_scores = []
    for name in args.only:
        try:
            all_scores.append(run_one(name, JOBS[name], args.workers))
        except Exception:   # noqa: BLE001 (one failing example must not stop the night)
            log(f"{name}: FAILED\n{traceback.format_exc()}")
    if all_scores:
        s = pd.concat(all_scores, ignore_index=True)
        s.to_csv(OUT / "scores.csv", index=False)
        cols = ["example", "describer", "method", "groups", "agree", "points", "ari", "ari_all", "pieces", "noise_runs", "probe"]
        (OUT / "overview.md").write_text("# All examples\n\n" + report._markdown(s[[c for c in cols if c in s]].round(3)) + "\n")
    log("end")


if __name__ == "__main__":
    main()
