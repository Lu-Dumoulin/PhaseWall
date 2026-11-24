"""The command-line tool `phasewall` (installed by pip from pyproject.toml).

    phasewall inspect SNAPSHOT.h5      list the datasets of a file and suggest the [[fields]] blocks
    phasewall toy FOLDER               write a demo ensemble (Swift-Hohenberg, or --example gray-scott,
                                       ginzburg-landau-1d, ...)
    phasewall describe CONFIG.toml     descriptors of every snapshot (2D) or window (1D) -> descriptors.csv
    phasewall learn CONFIG.toml        train the learned description (CNN, extra [nn]) -> its table
    phasewall walls TABLE.csv          walls and phases -> a report folder
    phasewall demo FOLDER              the three steps on the demo, checked against its reference labels

A CONFIG.toml with an [ensemble] section describes 2D runs (snapshots), one with a [kymograph]
section 1D runs (kymographs, phasewall.oned); every command takes either (io.load_any).

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
from .io import dims, fields_toml, inspect_snapshot, load_any, suggest_fields
from .walls import ALPHA, MIN_PHASE, RATIO, Walls, find_walls, phases_against_ratio

QUICK_SEEDS = (0, 1)    # --quick: runs per point


def node_labels(w: Walls, runs: pd.DataFrame, column: str = "reference") -> pd.Series:
    """See report.node_labels (kept here for existing scripts)."""
    return report.node_labels(w, runs, column)


def agreement(w: Walls, reference: pd.Series) -> tuple[pd.DataFrame, int]:
    """See report.agreement (kept here for existing scripts)."""
    return report.agreement(w, reference)


def cmd_inspect(args) -> str:
    """Datasets of one snapshot file and the suggested [[fields]] blocks (to check and copy
    into the configuration; nothing is applied)."""
    datasets = inspect_snapshot(args.file, julia_order=not args.not_julia)
    print(f"{args.file}: {len(datasets)} datasets (shape as read by phasewall, indexed [x, y, component])")
    print(datasets.to_string(index=False))
    text = fields_toml(suggest_fields(datasets))
    print("\nSuggested fields (check every kind before use):\n")
    print(text)
    return text


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
    ens = load_any(args.config)
    default = "descriptors_cnn.csv" if args.encoder else "descriptors.csv"
    out = Path(args.out) if args.out else Path(args.config).parent / default
    t0 = time.time()
    if args.encoder and dims(ens) == 1:
        sys.exit("phasewall describe: --encoder is for 2D ensembles; `phasewall learn` writes the 1D table")
    if args.encoder:
        table, errors = learned_table(ens, args.encoder, args.workers), []
    elif dims(ens) == 1:
        from .oned.describe1d import describe_ensemble as describe_1d
        table, errors = describe_1d(ens, workers=args.workers)
    else:
        table, errors = describe_ensemble(ens, workers=args.workers)
    table.to_csv(out, index=False)
    print(f"{len(table)} {'windows' if dims(ens) == 1 else 'snapshots'} of {table.run.nunique() if len(table) else 0} "
          "runs described "
          f"in {time.time() - t0:.0f} s -> {out}")
    if errors:
        (out.parent / "describe_errors.txt").write_text("\n".join(errors) + "\n")
        print(f"{len(errors)} runs could not be read: see {out.parent / 'describe_errors.txt'}")
    return out


def cmd_learn(args) -> Path:
    from .nn.data import load_frames
    from .nn.train import Settings, embed, h_table, save, train
    ens = load_any(args.config)
    out = Path(args.out) if args.out else Path(args.config).parent / "encoder"
    if dims(ens) == 1 or args.objective == "jepa":
        return _learn_by_encoder(ens, out, args)
    frames = load_frames(ens, workers=args.workers)
    print(f"{len(frames.x)} pairs of {len(set(frames.run))} runs, {frames.in_channels} channels, "
          f"frame {frames.x.shape[2]} x {frames.x.shape[3]}, {'periodic' if frames.periodic else 'not periodic'}")
    from .nn.architecture import Network, check, summary
    net = Network(blocks=args.blocks, width=args.width, dim_h=args.dim_h, hidden=args.hidden,
                  projection=args.projection, temperature=args.temperature, views=args.views, zoom=args.zoom,
                  crop=args.crop, normalise=not args.no_normalise,
                  keep_vector_mean=not args.centre_vectors)
    try:
        check(net, frames.x.shape[-1], frames.periodic)
    except ValueError as err:
        sys.exit(f"phasewall: {err}")
    info = summary(net, frames.x.shape[-1], frames.in_channels, periodic=frames.periodic)
    print(f"views {net.views}{' normalised' if net.normalise else ''}, input {info['input']} points; "
          f"network: channels {info['channels']}, last map {info['last_map']} points, receptive field "
          f"{info['receptive_field']} points, {info['weights_f']} weights in f, about {info['memory_gb_rough']:.1f} GB "
          "per step (rough)")
    settings = Settings.from_network(net, epochs=args.epochs, seed=args.seed)
    t0 = time.time()
    encoder, log = train(frames, settings)
    save(out, encoder, frames, settings, log)
    h_table(frames, embed(encoder, frames), ens).to_csv(out / "descriptors_cnn.csv", index=False)
    print(f"trained in {time.time() - t0:.0f} s -> {out / 'encoder.pt'}, {out / 'descriptors_cnn.csv'}")
    return out


def _learn_by_encoder(ens, out: Path, args) -> Path:
    """learn for a 1D ensemble, or with the predictive objective: through the registered encoders
    (builtins.py), which give the table of h and the log (no encoder file is written)."""
    from .builtins import JepaSettings, LearnedSettings
    from .nn.architecture import Network
    from .registry import Context, encoders
    out.mkdir(parents=True, exist_ok=True)
    net = Network(blocks=args.blocks, width=args.width, dim_h=args.dim_h, hidden=args.hidden,
                  projection=args.projection, temperature=args.temperature)
    name, settings = ("jepa", JepaSettings(args.epochs, (args.seed,))) if args.objective == "jepa" else \
        ("learned", LearnedSettings(args.epochs, (args.seed,), net))
    t0 = time.time()
    res = encoders()[name].run(ens, settings, Context(workers=args.workers, cache=""))
    table = next(iter(res.tables.values()))
    table.to_csv(out / "descriptors_cnn.csv", index=False)
    res.log.to_csv(out / "log.csv", index=False)
    print(f"{name} description of {len(table)} rows, trained in {time.time() - t0:.0f} s -> "
          f"{out / 'descriptors_cnn.csv'}")
    return out


def cmd_walls(args) -> Path:
    ens = load_any(args.config) if args.config else None
    parameters = args.parameters or (ens.parameters if ens else None)
    if not parameters:
        sys.exit("phasewall walls: give --parameters, or --config to read them from the ensemble")
    table = pd.read_csv(args.table)
    out = Path(args.out) if args.out else Path(args.table).parent / "walls"
    w = find_walls(table, parameters, alpha=args.alpha, ratio=args.ratio, min_phase=args.min_phase)
    curve = phases_against_ratio(table, parameters, alpha=args.alpha, min_phase=args.min_phase)
    report.write_report(w, out, curve, ens=ens, table=table)
    print(f"{w.n_phases} phases, {int(w.edges.wall.sum())} wall links, {int(w.nodes.mixed.sum())} mixed points "
          f"-> {out / 'summary.md'}")
    return out


def cmd_demo(args) -> Path:
    folder = Path(args.folder)
    runs = cmd_toy(args)
    descriptors = cmd_describe(argparse.Namespace(config=folder / "phasewall.toml", out=None,
                                                  workers=args.workers, encoder=None))
    ens = load_any(folder / "phasewall.toml")
    table = pd.read_csv(descriptors)
    w = find_walls(table, ens.parameters)
    curve = phases_against_ratio(table, ens.parameters)
    reference = node_labels(w, runs)
    counts, agree = agreement(w, reference)
    notes = "\n".join([
        "## Check against the reference labels (demo only)",
        "",
        (f"The demo knows the state of every run from direct measurements ({args.example}: reference_label); "
         "PhaseWall never sees these labels. Points (rows: reference label, columns: region found):"),
        "",
        report._markdown(counts.reset_index()),
        "",
        f"{agree} of {len(w.nodes)} points carry the most common label of their region.",
    ])
    out = folder / "walls"
    report.write_report(w, out, curve, reference=reference, ens=ens, notes=notes, table=table)
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

    s = sub.add_parser("inspect", help="list the datasets of a snapshot file and suggest the fields")
    s.add_argument("file", help="one snapshot file (HDF5 or JLD2)")
    s.add_argument("--not-julia", action="store_true",
                   help="the arrays were not written by Julia (no transposition)")
    s.set_defaults(func=cmd_inspect)

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
    s.add_argument("--blocks", type=int, default=4, help="convolution blocks, each halving the grid (default 4)")
    s.add_argument("--width", type=int, default=32, help="channels of the first block (default 32)")
    s.add_argument("--dim-h", type=int, default=256, help="dimension of h, channels of the last block (default 256)")
    s.add_argument("--hidden", type=int, default=256, help="hidden layer of the head (default 256)")
    s.add_argument("--projection", type=int, default=64, help="dimension of z (default 64)")
    s.add_argument("--temperature", type=float, default=0.2, help="temperature of the loss (default 0.2)")
    s.add_argument("--views", choices=["rotzoom", "d4"], default="rotzoom",
                   help="views of a run: rotzoom (any angle, zoom, crop; default) or d4 (90 degrees, whole frame)")
    s.add_argument("--zoom", type=float, default=1.25, help="rotzoom: zoom factors in [1/zoom, zoom] (default 1.25)")
    s.add_argument("--crop", type=float, default=0.75, help="rotzoom: crop side / frame side (default 0.75)")
    s.add_argument("--no-normalise", action="store_true",
                   help="keep the amplitude of each view (default: each field divided by its own std)")
    s.add_argument("--centre-vectors", action="store_true",
                   help="normalise: centre vectors and nematics too (default: they keep their mean)")
    s.add_argument("--objective", choices=["contrastive", "jepa"], default="contrastive",
                   help="contrastive (default) or jepa: predict hidden parts of a sample (being tested)")
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
