# Development

## Code layout

- `src/phasewall/`: the package. It does all the work and does not depend on an interface.
  - `fields.py`, `io.py`, `frames.py`: fields and their symmetries, files, crops.
  - `describe.py`: the hand-made description. `oned/`: the 1D parts (kymographs, 1D observables,
    1D views, the Ginzburg-Landau example).
  - `nn/`: the learned description (contrastive and JEPA).
  - `walls.py`, `graph.py`: the walls, and the links between points (grid steps, or the k
    nearest neighbours of scattered points). `trees.py`: the threshold tree. `orderparams.py`:
    candidate order parameters. `baselines.py`: blind methods and probe. `scores.py`: scores
    against labels.
  - `registry.py`, `builtins.py`: the registry of encoders and methods.
  - `pipeline.py`: settings in, results out, with a disk cache. The scripts, the command line and
    the interface call it.
  - `cli.py`, `report.py`: the command line and the figures and tables.
- `apps/`: the marimo page. It only fills settings and shows results.
- `scripts/`: the batch runs that make the figures in `docs/`.
- `tests/`: the tests.

Labels never enter a method. They are used only for scores and the probe.

## Add an encoder or a method

Each encoder and method, the built-in ones too, is a settings dataclass and one function. A new
one is one file. Then it runs in the pipeline, the scripts, the scores and the interface:

```python
from dataclasses import dataclass, field
from phasewall import pipeline as pl, registry

@dataclass
class HalfSettings:
    column: str = field(default="eps", metadata={"help": "parameter cut", "choices": ("eps", "g")})

@registry.method("half", HalfSettings, help="lower and upper half of a parameter", uses_grid=True)
def half(table, parameters, settings, context):
    p = table.groupby("run")[settings.column].first()
    return registry.MethodResult((p >= p.median()).astype(int))   # one group per run

res = pl.run(pl.Settings(methods=pl.Methods(others={"half": HalfSettings()})))
```

- An encoder is `run(ensemble, settings, context)` and returns tables with one row per snapshot.
- The metadata of the settings (`help`, `choices`, `range`) make the controls of the interface and
  are checked before a run.
- `tests/test_contracts.py` checks each registered entry: output format, one phase in noise, two
  phases on a step, invariance under rotations, mirrors and shifts.
- Other packages can register entries through the entry points `phasewall.encoders` and
  `phasewall.methods`.

## Tests

```bash
pip install -e ".[dev]"
pytest -q            # 308 tests, about two minutes
ruff check .
```

The continuous integration runs `ruff` and `pytest` on each push and pull request
(`.github/workflows/tests.yml`). Each test uses small synthetic data with a known answer:

- **Fields and files**: symmetry rules of vector and nematic fields, checked against the gradient
  of a scalar and a director field; round trips in Julia layout; components grouped; the
  suggestions of `phasewall inspect`.
- **Descriptors**: $S(k)$ of one wave; domains of spots against stripes; flows, vortex cores,
  winding defects, Frank energy; frozen, travelling and changing patterns; invariance of each
  descriptor under the eight symmetries of the square and periodic shifts.
- **Frames**: $S(k)$ of crops against the exact one; finite against spectral derivatives;
  rotation and mirror invariance; the same result after coarsening.
- **Walls**: no wall in noise or under a smooth drift; walls on the correct links for two and three
  phases; ended runs, mixed points, joined parts, reassigned points; the observable that jumps.
- **Scattered points**: grid or not; scaling by the step and in log; $k = \max(6, 2d)$; no link across a hole; no wall
  in noise or drift, the right sides for two and three phases; the length correction; a grid keeps
  its grid links; eight parameters (no wall in noise, one wall with the right sides); the tree and the
  wall drawing on scattered points.
- **Order parameters**: the column that jumps ranked first with its threshold; noise near chance;
  a combination of two columns for a middle phase.
- **Tree**: no split in noise or drift; the correct rules for two and three phases; the
  checkerboard for one run per point.
- **Learned description**: views and their symmetries; the loss at collapse; a short training that
  separates stripes from spots; saving and loading; JEPA without collapse.
- **1D**: kymograph files in both orders; windows; each observable unchanged by shifts and the
  mirror; known answers (travelling waves, drift speed, phase slips); one file per run; the
  Ginzburg-Landau regimes.
- **Pipeline, command line and interface**: each command; the interface runs the quick demos (2D
  and 1D) and your own runs with every tab drawn.
