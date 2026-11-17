# phasewall

Work in progress.

phasewall finds the phases of a simulated system from an ensemble of runs that were already
computed on a grid of parameters. Each snapshot is described by generic numbers computed from
its fields (density, velocity, polarity, nematic order, ...). Neighbouring parameter points are
compared, and a wall is placed wherever the description changes more than it varies within one
point. The regions enclosed by the walls are the candidate phases of the scanned range. No
labels and no number of phases are given in advance.

## Example: the Swift-Hohenberg equation

The demo uses a model with a known answer, the Swift-Hohenberg equation
du/dt = eps u - (1 + laplacian)^2 u + g u^2 - u^3, the textbook model of pattern formation.
In two dimensions it has three states: uniform (eps < 0), hexagons of spots (small eps > 0
with g > 0) and stripes. `phasewall demo` simulates 16 x 15 parameter points (eps, g) with 8
runs each from different random initial conditions (1920 runs, about one minute on 8 cores),
describes the last 6 snapshots of every run and searches for walls.

One snapshot from the middle of each phase found, on a common colour scale:

![One snapshot from each phase found](docs/figures/snapshots.png)

The phase map. Left: how much the description changes across each link of the parameter grid.
Middle: the three regions found and the walls between them; the ring marks a point whose runs
disagree with each other. Right: the state of each point measured directly (spot or stripe
symmetry from the skewness, amplitude), which phasewall never sees.

![Phase map of the demo](docs/figures/phase_map.png)

234 of the 240 points carry the most common reference label of their region. The 6 others are
hexagon points next to the hexagon-stripe boundary, where runs at the same parameters can end
in either state. The number of phases stays at 3 when the threshold RATIO (how much more the
crossing links must jump than the inner ones) goes from 1.2 to 2, and drops to 2 (uniform
against patterned) from 2.5:

<img src="docs/figures/robustness.png" alt="Number of phases against the threshold" width="420">

## How it works

1. **Describe.** Every snapshot becomes a vector of numbers that do not change when the box is
   rotated, mirrored or shifted: amplitude, the shape of the structure factor S(k), moments of
   the values, the number of connected domains above and below a few levels, global order,
   divergence and curl of vector fields, and how the pattern changes between two snapshots
   (frozen, travelling or changing). Vector and nematic fields are transformed with their
   components, so the invariance holds for all field types (`describe.py`, `fields.py`).
2. **Compare neighbours.** The runs at one parameter point form a sample. Two points that
   differ by one step in one parameter are linked. For each link the jump is the energy
   distance between the two samples; for each point the spread is the same distance between
   two halves of its own sample (`walls.py`, `graph.py`).
3. **Cut.** A random walk on the grid, which moves easily along small jumps and rarely across
   large ones, stays trapped in regions bounded by walls. The slowest mode of this diffusion
   (the Fiedler vector of the graph Laplacian) orders the points across the strongest barrier,
   and the region is split where the normalised cut is smallest.
4. **Test.** A cut is kept only if its crossing links jump more than the inner links: a
   one-sided Mann-Whitney test and a minimum ratio of medians. The cut is placed with the jumps
   between the first halves of the points and tested with the jumps between the second halves,
   so the test does not reuse the data that chose the cut. Both sides are then cut again until
   no cut passes.

Points whose two halves differ far more than usual are flagged as mixed: next to a
first-order transition, runs at the same parameters can end in different states.

## Usage

```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

phasewall demo data/demo                 # the example above (about one minute)
phasewall demo data/quick --quick        # a small version in a few seconds
```

For your own runs, write a TOML file that describes the ensemble (see `src/phasewall/io.py`):

```toml
[ensemble]
runs = "runs.csv"              # one row per run: a column `path`, one column per parameter
parameters = ["rho0", "zeta"]
snapshots = "Data/*.jld2"      # snapshot files inside each run folder
time_regex = '(\d+)\.jld2$'    # the time is the number in the file name
julia_order = true             # arrays written by Julia (column-major)
dx = 0.01
last = 10                      # the last 10 snapshots of every run are described
pair_dt = 1000.0               # time between the two snapshots used for the dynamics

[[fields]]
name = "rho"
keys = ["C", "rho"]            # names it may have in the files
kind = "scalar"                # scalar, vector or nematic
relative = true                # describe (rho - <rho>) / <rho>

[[fields]]
name = "p"
keys = ["P"]
kind = "vector"
```

then

```
phasewall describe ensemble.toml --workers 4      # -> descriptors.csv, one row per snapshot
phasewall walls descriptors.csv --config ensemble.toml
```

`walls` writes a folder with `summary.md`, the tables `nodes.csv` (one row per parameter point,
with its region and phase), `edges.csv` (one row per link, with its jump and whether it is a
wall), `cuts.csv` and `robustness.csv`, and the figures shown above. It also accepts any table
with one row per snapshot and descriptor columns named `field.block.name`, for example
descriptions made by a neural network, with `--parameters` instead of `--config`.

## Tests

`pytest` runs 70 tests on small synthetic fields with known answers, in about 8 seconds:
the symmetry rules of vector and nematic fields against the gradient of a scalar and the
tensor of a director field; file round trips in Julia layout; the structure factor of a single
wave, domain counts of spots against stripes, curl-free and divergence-free flows, frozen,
travelling and changing patterns, and the invariance of every descriptor under the eight
symmetries of the square and periodic shifts; the links of a refined parameter grid; no wall in
pure noise or under a smooth drift, walls on the right links for two and three phases, a
bistable point flagged as mixed; the Swift-Hohenberg runs (state, and stripe amplitude
sqrt(2 eps / 3) from weakly nonlinear theory); and every command of the command-line tool.
