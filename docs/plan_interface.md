# Plan: an interactive interface to explore phasewall

Status: plan (2026-11-17), not implemented. The interface will be written in a separate
session; this file is its contract with the package. Part of the development plan of the
README, private like it.

## Goal

A notebook that also runs as a dashboard, to explore first the Swift-Hohenberg demo and then
Gray-Scott, and later the user's own runs or a table of descriptors (as in hfml). The user
chooses:

- the source: a demo (Swift-Hohenberg, Gray-Scott), an ensemble (TOML configuration), or a
  CSV table of descriptors;
- the simulation of a demo: parameter grid, runs per point, grid size $N$, box size (or
  $\Delta x$), times, number of processes;
- the frame: whole periodic box or crop; crop side, `dx_target` (coarsening), `max_shift`,
  `min_wavelengths`;
- the description, three checkboxes: hand-made, learned, combined (both as separate blocks);
  for the learned one: epochs, seeds, device (CPU, Apple GPU, CUDA);
- the methods to run and compare: walls (with `ALPHA`, `RATIO`, `MIN_PHASE`), Gaussian
  mixture, k-means, HDBSCAN (sizes), logistic probe (needs reference labels), robustness
  curve, learning curve (learned description only);
- execution: number of processes, device, a cache folder.

Outputs: snapshots of each phase, phase maps per description and method side by side, the
comparison table, robustness curve, training curves, learning curve, probe scores; export of
the tables and figures.

## Which tool

**marimo** (https://marimo.io), chosen by Ludo on 2026-11-17; the Python counterpart of Pluto:

| Pluto (PlutoUI), as in HydraFluids | marimo |
| --- | --- |
| reactive cells, notebook saved as a `.jl` file | reactive cells, notebook saved as a plain `.py` file |
| `@bind x Switch()` | `x = mo.ui.switch()`, then `x.value` in other cells |
| `TextField`, `Select`, `MultiCheckBox` | `mo.ui.text` / `mo.ui.number`, `mo.ui.dropdown`, `mo.ui.multiselect` or `mo.ui.checkbox` |
| `Button`, `confirm(...)` | `mo.ui.run_button()`, `mo.ui.form(...)`; `mo.stop(not button.value)` holds a cell |
| `Slider` | `mo.ui.slider` |
| HTML / Markdown cells | `mo.md(...)`, `mo.hstack`, `mo.vstack`, `mo.ui.tabs`, `mo.accordion` |
| | `mo.status.progress_bar`, `mo.cache` / `mo.persistent_cache` |
| Pluto as a static page | `marimo run app.py` (dashboard), `marimo edit app.py` (notebook) |

Alternatives considered: Jupyter with ipywidgets (not reactive, hidden state), Streamlit
(dashboard only, re-runs the whole script on each change), Panel or Voila (heavier set-up for
the same result). marimo keeps the habits of Pluto, is reviewable as code, and serves both as
notebook and dashboard.

Expensive steps (simulation: about 1 min for the demo; learned description: about 2 min per
training; blind grouping: seconds) run only on a button, with a progress bar, and are cached so
that changing a method does not recompute the descriptions.

## Principle: the package does not depend on the interface

phasewall stays complete without the interface: Python modules, the `phasewall` command and
the scripts keep working as they do. The interface is a layer on top, in `apps/`, with
`marimo` only in an optional extra `[app]`. Changes to existing code for it are additions
whose defaults keep the present behaviour (for example `progress=None`); existing functions,
command-line options and outputs do not change, and every existing test must pass unchanged.
The new `pipeline` module is useful on its own (one call, cached steps, from Python or a
script), not only for the interface.

## Status of the package side (2026-11-17): READY, the interface can start

Implemented and tested (108 tests pass; `pip install -e ".[dev,app]"` installs everything,
including marimo). The interface calls only what is listed here.

```python
from phasewall import pipeline as pl
from phasewall.examples import EXAMPLES, example     # demo modules by name

s = pl.Settings(
    source=pl.Source(example="gray-scott",           # or "swift-hohenberg"; or config="x.toml"; or table="x.csv"
                     grid=None,                      # {param: values}; default example(name).GRID
                     seeds=None,                     # runs per point; default len(example(name).SEEDS)
                     quick=False,                    # example(name).QUICK grid and settings
                     settings={},                    # numerical: n, dx (GS) or waves (SH), dt, t_relax, every, n_save
                     parameters=None, label=None, prefix=None),   # for a table (io.load_table) / labels of an ensemble
    frame=pl.FrameSettings(crop=None, dx_target=None, max_shift=None, min_wavelengths=3, periodic=None),
    describers=pl.Describers(hand=True, learned=False, combined=False, epochs=40, seeds=(0,), device=None),
    methods=pl.Methods(walls=True, alpha=1e-3, ratio=1.5, min_phase=3, robustness=False,
                       mixture=False, kmeans=False, hdbscan=(), probe=False, learning_curve=()),
    execution=pl.Execution(workers=4, cache="data/cache"))

pl.estimate(s)                    # {"simulate": s, "describe": s, "train": s, "learning curve": s}, rough seconds
res = pl.run(s, progress=cb)      # cb(step: str, done: int, total: int); steps "simulate", "describe",
                                  # "frames", "train", "learning curve", "methods"
```

`Results` (all already computed; figures are matplotlib `Figure`, shown with `mo.mpl.interactive`
or as images):

| attribute or method | content |
| --- | --- |
| `res.tables` | description name -> table (one row per snapshot); names `"hand"`, `"learned seed 0"`, `"combined seed 0"`, or `"table"` |
| `res.scores` | one row per (describer, method): groups, points, pieces, noise_points, noise_runs and, with labels, agree, ari, ari_all |
| `res.walls[name]` | `walls.Walls`: `.nodes`, `.edges`, `.cuts`, `.n_phases`, `.settings` |
| `res.groups[name]` | groups of every run per method (columns walls, mixture, kmeans, hdbscan_m; -1 = noise) |
| `res.probe[name]` | balanced accuracy of the logistic probe (with labels) |
| `res.robustness[name]` | phases against RATIO |
| `res.training`, `res.learning_curve` | training logs (column `name`), learning curve table |
| `res.labels`, `res.runs`, `res.ensemble`, `res.parameters` | labels per run (demos: reference), runs, the framed ensemble |
| `res.phase_map(name)`, `res.snapshots(name)`, `res.robustness_plot(name)` | figures |
| `res.methods_map()`, `res.training_plot()`, `res.learning_curve_plot()` | figures |

Notes for the interface: the steps are cached on disk (`Execution.cache`), so re-running with
another method or threshold is fast; the simulation and the training run only when their
settings change. `learned`, `combined` and `learning_curve` need the extra `[nn]`; `mixture`,
`kmeans`, `hdbscan` and `probe` need `[baselines]`. Errors are raised as `ValueError` with a
message meant for the user (unknown keys, crop that does not fit, frames of different sizes,
missing columns of a table): show them as they are. Gray-Scott takes about 4 min on 8
processes for its default grid; Swift-Hohenberg about 1 min; `quick=True` takes seconds.

## What the package must provide first

The interface must only call the package; no computation in the notebook. Needed before the
interface session starts (to be written in the main session, which owns `src/` and `tests/`):

1. **`phasewall.pipeline`**: settings dataclasses (`Source`, `FrameSettings`, `Describers`,
   `Methods`, `Execution`) and `run(settings, progress=None) -> Results`. Each step (simulate,
   describe, train, group, walls) is cached on disk under a key made of the settings it
   depends on, so a change of a method re-runs only that method. `Results` holds the tables
   (descriptions, nodes, edges, cuts, groups, scores) and the figures as matplotlib `Figure`
   objects.
2. **`phasewall.examples`**: a registry of demos with one interface (name, parameters and
   default grid, fields, `simulate`, `reference_label`, `make_ensemble(folder, grid, seeds,
   size, workers)`, cost estimate). Swift-Hohenberg moves there from `toy.py` (`toy` kept as an
   alias); Gray-Scott is added there.
3. **CSV input**: a documented format, one row per snapshot or per run, columns `run`,
   optional `time`, the parameters, the descriptors named `block.name` or any numeric columns
   selected by prefix, an optional label column for the probe.
4. **Combined description**: hand-made and learned tables joined on (run, time) with distinct
   block names (planned after Gray-Scott; the third checkbox depends on it).
5. **Plot functions in the package** (`phasewall.report` or `phasewall.plots`): phase maps of
   several methods, comparison tables, training and learning curves; now in
   `scripts/compare_describers.py` and `scripts/doc_figures.py`.
6. **Learning curve in the package** (now a scratch script): train on subsets of runs per
   point with the same number of steps, evaluate on all runs.
7. **Progress and device**: a `progress(step, done, total)` callback in `describe_ensemble`,
   `load_frames`, `train` and `make_ensemble` (instead of printing); a `device` argument for
   the training.
8. **Cost estimates**: runs x time per run for a demo, epochs x steps for a training, shown
   before the button is pressed.
9. **Extra `[app]`**: `marimo` (and the extras `[nn]`, `[baselines]` it uses).

## Files and ownership

- Interface session: `apps/explore.py` (marimo notebook), `apps/README.md`, tests of the
  notebook's helpers if any; nothing in `src/` without agreement.
- Main session: `src/phasewall/` (items 1 to 8), `tests/`, documentation.
- The interface starts once items 1, 2, 5 and 7 exist; 3, 4, 6 and 8 can follow.

## Layout (proposal)

Sidebar (or top accordion): Source, Simulation, Frame, Description, Methods, Execution, and a
run button with an estimate of the cost. Main area, tabs: Snapshots (one per phase and per
method), Phase maps (methods x descriptions grid), Comparison (table: groups, agreement, ARI,
pieces, noise, probe), Walls (jumps, cuts, robustness curve), Learned description (training
curves, learning curve), Export.
