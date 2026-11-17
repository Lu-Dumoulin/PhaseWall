"""The command-line tool `phasewall` (installed by pip from pyproject.toml).

    phasewall toy FOLDER               write a demo ensemble (Swift-Hohenberg, or --example gray-scott)
    phasewall describe CONFIG.toml     descriptors of every snapshot -> descriptors.csv
    phasewall learn CONFIG.toml        train the learned description (CNN, extra [nn]) -> its table
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
from .examples import EXAMPLES, example
from .io import load_ensemble
from .walls import ALPHA, MIN_PHASE, RATIO, Walls, find_walls, phases_against_ratio

QUICK_SEEDS = (0, 1)    # --quick: runs per point


def node_labels(w: Walls, runs: pd.DataFrame, column: str = "reference") -> pd.Series:
    """See report.node_labels (kept here for existing scripts)."""
    return report.node_labels(w, runs, column)


def agreement(w: Walls, reference: pd.Series) -> tuple[pd.DataFrame, int]:
    """See report.agreement (kept here for existing scripts)."""
    return report.agreement(w, reference)


def cmd_toy(args) -> pd.DataFrame:
    module = example(args.example)
    kw = dict(module.QUICK) if args.quick else {}
    if args.quick:
        seeds = QUICK_SEEDS
    else:
        seeds = tuple(range(args.seeds)) if args.seeds else module.SEEDS
    t0 = time.time()
    runs = module.make_ensemble(args.folder, seeds=seeds, workers=args.workers, **kw)
    print(f"{len(runs)} runs written to {args.folder} in {time.time() - t0:.0f} s; "
          f"configuration {Path(args.folder) / 'phasewall.toml'}")
    return runs


def learned_table(ens, encoder_dir, workers: int) -> pd.DataFrame:
    """h of every pair of an ensemble with a saved encoder (fields, frame kind and scales of
    its training)."""
    from .nn.data import load_frames
    from .nn.train import embed, h_table, load
    enc, meta = load(encoder_dir)
    if [(f.name, f.kind) for f in meta["fields"]] != [(f.name, f.kind) for f in ens.fields]:
        sys.exit(f"phasewall: the encoder in {encoder_dir} was trained on other fields")
    frames = load_frames(ens, workers, scale=np.array(meta["scale"]))
    if frames.periodic != meta["periodic"] or frames.in_channels != meta["in_channels"]:
        sys.exit(f"phasewall: the frames do not match the training of {encoder_dir} (periodic, channels)")
    return h_table(frames, embed(enc, frames), ens)


def cmd_describe(args) -> Path:
    ens = load_ensemble(args.config)
    default = "descriptors_cnn.csv" if args.encoder else "descriptors.csv"
    out = Path(args.out) if args.out else Path(args.config).parent / default
    t0 = time.time()
    if args.encoder:
        table, errors = learned_table(ens, args.encoder, args.workers), []
    else:
        table, errors = describe_ensemble(ens, workers=args.workers)
    table.to_csv(out, index=False)
    print(f"{len(table)} snapshots of {table.run.nunique() if len(table) else 0} runs described "
          f"in {time.time() - t0:.0f} s -> {out}")
    if errors:
        (out.parent / "describe_errors.txt").write_text("\n".join(errors) + "\n")
        print(f"{len(errors)} runs could not be read: see {out.parent / 'describe_errors.txt'}")
    return out


def cmd_learn(args) -> Path:
    from .nn.data import load_frames
    from .nn.train import Settings, embed, h_table, save, train
    ens = load_ensemble(args.config)
    out = Path(args.out) if args.out else Path(args.config).parent / "encoder"
    frames = load_frames(ens, workers=args.workers)
    print(f"{len(frames.x)} pairs of {len(set(frames.run))} runs, {frames.in_channels} channels, "
          f"frame {frames.x.shape[2]} x {frames.x.shape[3]}, {'periodic' if frames.periodic else 'not periodic'}")
    settings = Settings(epochs=args.epochs, seed=args.seed)
    t0 = time.time()
    encoder, log = train(frames, settings)
    save(out, encoder, frames, settings, log)
    h_table(frames, embed(encoder, frames), ens).to_csv(out / "descriptors_cnn.csv", index=False)
    print(f"trained in {time.time() - t0:.0f} s -> {out / 'encoder.pt'}, {out / 'descriptors_cnn.csv'}")
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
                                                  workers=args.workers, encoder=None))
    ens = load_ensemble(folder / "phasewall.toml")
    table = pd.read_csv(descriptors)
    w = find_walls(table, ens.parameters)
    curve = phases_against_ratio(table, ens.parameters)
    reference = node_labels(w, runs)
    counts, agree = agreement(w, reference)
    notes = "\n".join([
        "## Check against the reference labels (demo only)",
        "",
        (f"The demo knows the state of every run from direct measurements ({args.example}: reference_label); "
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
        s.add_argument("--example", choices=list(EXAMPLES), default="swift-hohenberg",
                       help="which demo (default swift-hohenberg)")
        s.add_argument("--seeds", type=int, default=None,
                       help=f"runs per parameter point (default: {len(toy.SEEDS)} for Swift-Hohenberg, "
                            "4 for Gray-Scott)")
        s.add_argument("--quick", action="store_true",
                       help="small grid and box, seconds instead of minutes (too coarse for all states)")

    s = sub.add_parser("toy", help="write a demo ensemble")
    toy_options(s)
    s.set_defaults(func=cmd_toy)

    s = sub.add_parser("describe", help="descriptors of every snapshot of an ensemble")
    s.add_argument("config", help="TOML configuration of the ensemble (see phasewall.io)")
    s.add_argument("--out", help="CSV file to write (default: descriptors.csv next to the configuration)")
    s.add_argument("--workers", type=int, default=2,
                   help="processes reading runs in parallel (default 2; keep it small over a network share)")
    s.add_argument("--encoder", help="folder of an encoder trained by `phasewall learn`: write its h instead")
    s.set_defaults(func=cmd_describe)

    s = sub.add_parser("learn", help="train the learned description (CNN, needs the extra [nn])")
    s.add_argument("config", help="TOML configuration of the ensemble (pair_dt is needed)")
    s.add_argument("--out", help="folder for the encoder and its table (default: encoder/ next to the configuration)")
    s.add_argument("--epochs", type=int, default=40, help="training epochs (default 40)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=2, help="processes reading runs (default 2)")
    s.set_defaults(func=cmd_learn)

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
