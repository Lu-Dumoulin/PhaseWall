# phasewall explorer

An interactive interface to phasewall, written as a [marimo](https://marimo.io) notebook
(`explore.py`). The same file opens as a notebook, with the code of every cell, or as a
dashboard, with only the controls and the results.

```
pip install -e ".[app]"              # marimo, PyTorch and scikit-learn

marimo run apps/explore.py           # dashboard in the browser
marimo edit apps/explore.py          # notebook: cells visible and editable
```

## What it does

The panel on the left sets every choice of an analysis, in the order of the analysis, then **Run**
starts it:

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
- **Encoders**: hand-made descriptors, with the blocks of observables of each field (those that
  fit its kind are offered, its defaults ticked); learned (a CNN per training seed) with epochs,
  seeds, device and the shape of the network (blocks, width, dimension of h, head, temperature),
  checked against the frame before any training, with the receptive field and the number of
  weights shown; combined (both sets of columns as separate blocks); and any encoder registered
  by another package.
- **Methods**: walls (`ALPHA`, `RATIO`, `MIN_PHASE`, joining of separated parts of one phase), the
  robustness curve, the unsupervised threshold tree (its settings), a Gaussian mixture, k-means,
  HDBSCAN, and any method registered by another package. **Checks**: the logistic probe (needs
  labels; a ceiling, never a method) and the learning curve.
- **Execution**: processes and the cache folder (default `data/cache_app`, not tracked;
  separate from the cache of scripts that may run at the same time).

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
| Phase map | jumps across the links, the phases (of the walls or of the tree) with their walls, and the labels if known |
| Comparison | one row per description and method (groups, pieces on the grid, agreement, ARI, noise), probe scores, and the groups of every method side by side |
| Snapshots | one snapshot from the middle of each phase (walls or tree), per field, on a common colour scale |
| Rules (tree) | the rule of each phase, every split tried (with the agreement of the two halves), the separated leaves compared |
| Walls | summary of the cuts tried, the robustness curve, the tables of points, links, cuts and separated phases compared |
| Learned description | the network, training curves (loss, effective rank of $h$) and the learning curve |
| Export | the tables as CSV, the rules of the tree and the phase map as PNG |

Command line, without a browser (a smoke test; also used by the tests):
`python apps/explore.py --run [--quick] [--tree] [--runs runs.csv --dx 0.5 --pair-dt 25] [--cache DIR]`.

## How it is built

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
