# PhaseWall explorer

An interactive interface to PhaseWall, written as a [marimo](https://marimo.io) notebook
(`explore.py`). The same file opens as a notebook, with the code of every cell, or as a
dashboard, with only the controls and the results.

```
./explore.sh                         # from the root of the repository: dashboard in the browser
./explore.sh --edit                  # notebook: cells visible and editable
```

`explore.sh` makes the environment `.venv` and installs the package with the extra `[app]`
(marimo, PyTorch and scikit-learn) the first time, then starts marimo from it. By hand, in an
activated environment: `pip install -e ".[app]"`, then `marimo run apps/explore.py` or
`marimo edit apps/explore.py`.

## What it does

At the top, the dimension of the system: 2D (snapshots on a plane) or 1D (kymographs on a line;
`marimo run apps/explore.py -- --1d` opens there). It decides the sources (1D: an example, a
`[kymograph]` TOML file or a table), the fields and their blocks, the frame or windows, and whether the
results show snapshots or kymographs. Then the page is a column of sections, one per step of an analysis (1 System, 2 Fields, 3 Encoders,
4 Methods, 5 Frame of each snapshot (2D) or Windows of each kymograph (1D), folded), then the run bar:
execution on the left, the cost estimate (each step with where it runs, CPU cores or GPU, and what an
alternative would cost) and **Run** on the right (greyed out while a run goes on), then
the results:

- **System**: an example (Swift-Hohenberg, Gray-Scott, FitzHugh-Nagumo, Barkley, localised
  Swift-Hohenberg, or a quick version of any that runs in seconds), **your runs** (a CSV table of
  runs, one column `path` to each run folder and one per parameter, with the snapshot files of
  each run), an ensemble described by a TOML file, or a CSV table of descriptors made elsewhere.
  For an example: the parameter grid, typed as `a, b, c`, `start:step:stop` or
  `start:*factor:stop` (geometric), the runs per point and the numerical settings.
- **Fields** (your runs): the first snapshot of the first run is inspected; its datasets are
  listed with their shapes, and fields are suggested from the names: a dataset with two
  components, or two datasets that are the components of one vector or tensor (`vx`, `vy`;
  `Qxx`, `Qxy`), grouped into one field. Tick the fields to use and confirm the kind of each
  (scalar, velocity, polarity, vector when the role is not known, nematic); other fields can be
  typed one per line (`name kind dataset[,dataset] [relative] [single]`). A component alone is
  refused as a scalar, since it is not invariant under rotations. The app writes the TOML
  configuration into the cache folder (`phasewall.io.write_config`) and reads the runs from it.
- **Frame**: whole box or a centred crop, coarsening, the largest shift of the moving
  correlation, periodic or not. Empty fields keep the settings of the ensemble.
- **Encoders**: hand-made descriptors, with the blocks of observables in a grid of checkboxes (one
  row per field, one column per block, a box only where the block fits the kind of the field, its
  defaults ticked; the definition of each block on hover and below the grid, from
  `phasewall.fields.BLOCK_HELP`); learned (a CNN per training seed) with epochs,
  seeds and the shape of the network (blocks, width, dimension of h, head, temperature),
  checked against the frame before any training, with the receptive field, the number of
  weights and a rough memory of one training step shown; combined (both sets of columns as separate blocks); and any encoder registered
  by another package.
- **Methods**: walls (`ALPHA`, `RATIO`, `MIN_PHASE`, joining of separated parts of one phase), the
  robustness curve, the unsupervised threshold tree (its settings), a Gaussian mixture, k-means,
  HDBSCAN (minimum sizes as checkboxes), and any method registered by another package. **Checks**: the logistic probe (needs
  labels; a ceiling, never a method) and the learning curve.
- **Frame of each snapshot** (2D, folded): sliders for the part of the box described (a centred square, as a
  share of the side), the coarsening (grid points merged per side), the largest shift of the moving
  correlation (share of the side) and the wavelengths kept in S(k) for non-periodic frames; they start at the
  frame of the TOML file, and a line below says the frame you get (points, dx, periodic or not).
- **Execution** (in the run bar): CPU cores used in parallel (simulations, hand-made descriptions), the device
  that trains the learned encoders (the fastest available by default: the Apple GPU or a CUDA GPU), and the
  cache folder (default `data/cache_app`, not tracked; separate from the cache of scripts that may run at the
  same time).

The controls of the tree, of the network and of registered entries are not written here: they
are built from the settings dataclass of each entry (`phasewall.registry.settings_spec`: type,
default, choices, range, help), so a new encoder or method appears in the panel without any
change to this file.

An estimate of the cost is shown above the button. Each step (simulation, descriptions,
trainings) is cached on disk under a key made of the settings it depends on, so a second run
that only changes a method or a threshold takes seconds. The results stay on screen while the
settings change, with a note until Run is pressed again.

Results, for the description chosen above the tabs:

| tab | content |
| --- | --- |
| Phase map | jumps across the links, the phases (of the walls or of the tree) with their walls, and the labels if known; below, what changes across each wall and the candidate order parameters (the descriptors that tell two phases, or one phase and the others, apart with one threshold, with the % of points on the right side) |
| Comparison | one row per description and method (groups, pieces on the grid, agreement, ARI, noise), probe scores, and the groups of every method side by side |
| Snapshots | one snapshot from the middle of each phase (walls or tree), per field, on a common colour scale |
| Rules (tree) | the rule of each phase, the meaning of each descriptor in the rules (`phasewall.describe.explain`), every split tried (with its meaning and the agreement of the two halves), the separated leaves compared |
| Walls | summary of the cuts tried, the robustness curve, the tables of points, links, cuts and separated phases compared |
| Learned description | the network, training curves (loss, effective rank of $h$) and the learning curve |
| Export | the tables as CSV, the rules of the tree and the phase map as PNG |

Command line, without a browser (a smoke test; also used by the tests):
`python apps/explore.py --run [--quick] [--tree] [--runs runs.csv --dx 0.5 --pair-dt 25] [--cache DIR]`.

## How it is built

The look comes from `explore.css` (loaded with `marimo.App(css_file=...)`): the accent colour, the
frames of the sections and the step labels of the header are variables at its top, with a dark
mode version; the page works without it.

The notebook only reads the controls and calls the package: `phasewall.pipeline` (settings,
`estimate`, `run`, cached steps, `fields_for`, `frame_side`), `phasewall.registry` (the entries and
their settings), `phasewall.io` (inspection of a snapshot, `write_config`) and the figures of
`phasewall.report`. It computes nothing
itself, and the package does not depend on it.

For a reader who knows Pluto: each cell is a function whose arguments are the names it reads
from other cells, and every cell that reads a control runs again when the control changes.
`mo.ui.checkbox`, `mo.ui.text`, `mo.ui.dropdown` play the role of PlutoUI's `@bind` widgets
(read with `.value`); `mo.stop(...)` holds the run cell until the button is pressed; `mo.state`
keeps the last results while the settings change.

## Without a browser

```
python apps/explore.py --run --quick --cache /tmp/pw                     # runs once, no output
marimo export html apps/explore.py -o report.html -- --run --quick       # a static page with every tab
```

`--run` runs the default settings once (`--quick` for the quick demo, `--cache DIR` for
another cache). `tests/test_app.py` does the same on the quick demo and checks that every tab
is drawn without error; it also tests the parsing of the typed values. It is skipped when
marimo is not installed.
