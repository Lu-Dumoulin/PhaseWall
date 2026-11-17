"""The command-line tool `phasewall` (installed by pip from pyproject.toml).

    phasewall toy FOLDER               write the Swift-Hohenberg demo ensemble
    phasewall describe CONFIG.toml     descriptors of every snapshot -> descriptors.csv
    phasewall walls TABLE.csv          walls and phases -> a report folder
    phasewall demo FOLDER              the three steps on the demo, checked against its reference labels

`phasewall <command> --help` lists the options. argparse (standard library) reads the
command line; each command is one function below, so it can also be called from Python
with run(["walls", ...]), as the tests do.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import report, toy
from .describe import describe_ensemble
from .graph import DECIMALS
from .io import load_ensemble
from .walls import ALPHA, MIN_PHASE, RATIO, Walls, find_walls, phases_against_ratio

# --quick: a small demo to try the commands in seconds; too coarse to find all three phases
QUICK = {"n": 32, "waves": 4}
QUICK_EPS, QUICK_G, QUICK_SEEDS = toy.EPS[::3], toy.G[::3], (0, 1)


def node_labels(w: Walls, runs: pd.DataFrame, column: str = "reference") -> pd.Series:
    """The most common label of the runs at each parameter point, indexed like w.nodes."""
    key = runs[w.parameters].round(DECIMALS).apply(tuple, axis=1)
    by_point = runs[column].groupby(key).agg(lambda s: s.value_counts().idxmax())
    nodes = w.nodes[w.parameters].round(DECIMALS).apply(tuple, axis=1)
    return pd.Series(by_point.loc[nodes].to_numpy(), index=w.nodes.index, name=column)


def agreement(w: Walls, reference: pd.Series) -> tuple[pd.DataFrame, int]:
    """Table of reference labels against regions found, and the number of points whose
    label is the most common label of their region."""
    table = pd.crosstab(reference, w.nodes.region)
    return table, int(table.max(axis=0).sum())


def cmd_toy(args) -> pd.DataFrame:
    kw = QUICK if args.quick else {}
    eps, g = (QUICK_EPS, QUICK_G) if args.quick else (toy.EPS, toy.G)
    seeds = QUICK_SEEDS if args.quick else tuple(range(args.seeds))
    t0 = time.time()
    runs = toy.make_ensemble(args.folder, eps=eps, g=g, seeds=seeds, workers=args.workers, **kw)
    print(f"{len(runs)} runs written to {args.folder} in {time.time() - t0:.0f} s; "
          f"configuration {Path(args.folder) / 'phasewall.toml'}")
    return runs


def cmd_describe(args) -> Path:
    ens = load_ensemble(args.config)
    out = Path(args.out) if args.out else Path(args.config).parent / "descriptors.csv"
    t0 = time.time()
    table, errors = describe_ensemble(ens, workers=args.workers)
    table.to_csv(out, index=False)
    print(f"{len(table)} snapshots of {table.run.nunique() if len(table) else 0} runs described "
          f"in {time.time() - t0:.0f} s -> {out}")
    if errors:
        (out.parent / "describe_errors.txt").write_text("\n".join(errors) + "\n")
        print(f"{len(errors)} runs could not be read: see {out.parent / 'describe_errors.txt'}")
    return out


def cmd_walls(args) -> Path:
    ens = load_ensemble(args.config) if args.config else None
    parameters = args.parameters or (ens.parameters if ens else None)
    if not parameters:
        sys.exit("phasewall walls: give --parameters, or --config to read them from the ensemble")
    table = pd.read_csv(args.table)
    out = Path(args.out) if args.out else Path(args.table).parent / "walls"
    w = find_walls(table, parameters, alpha=args.alpha, ratio=args.ratio, min_phase=args.min_phase)
    curve = phases_against_ratio(table, parameters, alpha=args.alpha, min_phase=args.min_phase)
    report.write_report(w, out, curve, ens=ens)
    print(f"{w.n_phases} phases, {int(w.edges.wall.sum())} wall links, {int(w.nodes.mixed.sum())} mixed points "
          f"-> {out / 'summary.md'}")
    return out


def cmd_demo(args) -> Path:
    folder = Path(args.folder)
    runs = cmd_toy(args)
    descriptors = cmd_describe(argparse.Namespace(config=folder / "phasewall.toml", out=None,
                                                  workers=args.workers))
    ens = load_ensemble(folder / "phasewall.toml")
    table = pd.read_csv(descriptors)
    w = find_walls(table, ens.parameters)
    curve = phases_against_ratio(table, ens.parameters)
    reference = node_labels(w, runs)
    counts, agree = agreement(w, reference)
    notes = "\n".join([
        "## Check against the reference labels (demo only)",
        "",
        ("The demo knows the state of every run from direct measurements (toy.reference_label); "
         "phasewall never sees these labels. Points (rows: reference label, columns: region found):"),
        "",
        report._markdown(counts.reset_index()),
        "",
        f"{agree} of {len(w.nodes)} points carry the most common label of their region.",
    ])
    out = folder / "walls"
    report.write_report(w, out, curve, reference=reference, ens=ens, notes=notes)
    print(f"{w.n_phases} phases, {agree} of {len(w.nodes)} points agree with the reference labels "
          f"-> {out / 'summary.md'}")
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phasewall", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def toy_options(s):
        s.add_argument("folder", help="folder to write the runs into")
        s.add_argument("--workers", type=int, default=4, help="processes running in parallel (default 4)")
        s.add_argument("--seeds", type=int, default=len(toy.SEEDS),
                       help=f"runs per parameter point (default {len(toy.SEEDS)})")
        s.add_argument("--quick", action="store_true",
                       help="small grid and box, seconds instead of a minute (too coarse for all three phases)")

    s = sub.add_parser("toy", help="write the Swift-Hohenberg demo ensemble")
    toy_options(s)
    s.set_defaults(func=cmd_toy)

    s = sub.add_parser("describe", help="descriptors of every snapshot of an ensemble")
    s.add_argument("config", help="TOML configuration of the ensemble (see phasewall.io)")
    s.add_argument("--out", help="CSV file to write (default: descriptors.csv next to the configuration)")
    s.add_argument("--workers", type=int, default=2,
                   help="processes reading runs in parallel (default 2; keep it small over a network share)")
    s.set_defaults(func=cmd_describe)

    s = sub.add_parser("walls", help="walls and phases from a table of descriptors")
    s.add_argument("table", help="CSV with one row per snapshot: run, time, the parameters, the descriptors")
    s.add_argument("--parameters", nargs="+", help="parameter columns (default: from --config)")
    s.add_argument("--config", help="TOML configuration of the ensemble: parameters, and example snapshots")
    s.add_argument("--out", help="report folder (default: walls/ next to the table)")
    s.add_argument("--alpha", type=float, default=ALPHA, help=f"Mann-Whitney threshold (default {ALPHA})")
    s.add_argument("--ratio", type=float, default=RATIO, help=f"crossing / inner jump needed (default {RATIO})")
    s.add_argument("--min-phase", type=int, default=MIN_PHASE, help=f"smallest phase in points (default {MIN_PHASE})")
    s.set_defaults(func=cmd_walls)

    s = sub.add_parser("demo", help="toy + describe + walls, checked against the reference labels")
    toy_options(s)
    s.set_defaults(func=cmd_demo)
    return p


def run(argv: list[str] | None = None):
    """Run one command and return what it made (a path or a table), for use from Python."""
    args = parser().parse_args(argv)
    with np.errstate(divide="ignore", invalid="ignore"):
        return args.func(args)


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `phasewall` command. It returns None so that the exit status is 0."""
    run(argv)


if __name__ == "__main__":
    main()
