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

The panel on the left sets every choice of an analysis, then **Run** starts it:

- **Source**: a demo (Swift-Hohenberg or Gray-Scott, or a quick version of either that runs in
  seconds), an ensemble described by a TOML file (see the main README), or a CSV table of
  descriptors made elsewhere (columns `run`, the parameters, optionally `time` and a label).
- **Simulation** (demos): the parameter grid, typed as `a, b, c`, `start:step:stop` or
  `start:*factor:stop` (geometric: start, start × factor, ...), the runs per point and the
  numerical settings, pre-filled with the defaults of the demo. A grid field left as shown
  gives the demo's own values exactly.
- **Frame**: whole box or a centred crop, coarsening, the largest shift of the moving
  correlation, periodic or not. Empty fields keep the settings of the ensemble.
- **Description**: hand-made descriptors, learned (a CNN per training seed) and combined (both
  sets of columns as separate blocks), with epochs and device for the training.
- **Methods**: walls (`ALPHA`, `RATIO`, `MIN_PHASE`), the robustness curve, and for comparison
  a Gaussian mixture, k-means, HDBSCAN, a logistic probe (needs labels) and a learning curve.
- **Execution**: processes and the cache folder (default `data/cache_app`, not tracked;
  separate from the cache of scripts that may run at the same time).

An estimate of the cost is shown above the button. Each step (simulation, descriptions,
trainings) is cached on disk under a key made of the settings it depends on, so a second run
that only changes a method or a threshold takes seconds. The results stay on screen while the
settings change, with a note until Run is pressed again.

Results, for the description chosen above the tabs:

| tab | content |
| --- | --- |
| Phase map | jumps across the links, the phases with their walls, and the labels if known |
| Comparison | one row per description and method (groups, pieces on the grid, agreement, ARI, noise), probe scores, and the groups of every method side by side |
| Snapshots | one snapshot from the middle of each phase, per field, on a common colour scale |
| Walls | summary of the cuts tried, the robustness curve, and the tables of points, links and cuts |
| Learned description | training curves (loss, effective rank of $h$) and the learning curve |
| Export | the tables as CSV and the phase map as PNG |

## How it is built

The notebook only reads the controls and calls the package: `phasewall.pipeline` (settings,
`estimate`, `run`, cached steps) and the figures of `phasewall.report`. It computes nothing
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
