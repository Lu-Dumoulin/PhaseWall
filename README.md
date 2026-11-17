# phasewall

Work in progress.

phasewall finds the phases of a simulated system from an ensemble of runs that were already
computed on a grid of parameters. Each snapshot is described by generic numbers computed from
its fields (density, velocity, polarity, nematic order, ...). Neighbouring parameter points are
compared, and a wall is placed wherever the description changes more than it varies within one
point. The regions enclosed by the walls are the candidate phases of the scanned range. No
labels and no number of phases are given in advance.

The method, its mathematics, a physics parallel for each step and the tests are described in
[docs/documentation.pdf](docs/documentation.pdf).

## Example: the Swift-Hohenberg equation

The demo uses a model with a known answer, the Swift-Hohenberg equation, the textbook model of
pattern formation:

$$\partial_t u = \varepsilon u - (1 + \nabla^2)^2 u + g u^2 - u^3 .$$

In two dimensions it has three states: uniform ($\varepsilon < 0$), hexagons of spots (small
$\varepsilon > 0$ with $g > 0$) and stripes. `phasewall demo` simulates $16 \times 15$ parameter
points $(\varepsilon, g)$ with 8 runs each from different random initial conditions (1920 runs,
about one minute on 8 cores), describes the last 6 snapshots of every run and searches for walls.

One snapshot from the middle of each phase found, on a common colour scale:

![One snapshot from each phase found](docs/figures/snapshots.png)

The phase map. Left: how much the description changes across each link of the parameter grid.
Middle: the three regions found and the walls between them; the ring marks a point whose runs
disagree with each other. Right: the state of each point measured directly (spot or stripe
symmetry from the skewness, amplitude), which phasewall never sees.

![Phase map of the demo](docs/figures/phase_map.png)

234 of the 240 points carry the most common reference label of their region. The 6 others are
hexagon points next to the hexagon-stripe boundary, where runs at the same parameters can end
in either state. The number of phases stays at 3 when the threshold `RATIO` (how much more the
crossing links must jump than the inner ones) goes from 1.2 to 2, and drops to 2 (uniform
against patterned) from 2.5:

<img src="docs/figures/robustness.png" alt="Number of phases against the threshold" width="420">

## How it works

1. **Describe.** Every snapshot becomes a vector of numbers that do not change when the box is
   rotated, mirrored or shifted: amplitude, the shape of the structure factor $S(k)$, moments of
   the values, the number of connected domains above and below a few levels, global order
   $|\langle \mathbf a \rangle| / \langle |\mathbf a| \rangle$, divergence and curl of vector
   fields, and how the pattern changes between two snapshots (frozen, travelling or changing).
   Vector and nematic fields are transformed with their components (a rotation by $90^\circ$
   sends $(a_x, a_y) \to (-a_y, a_x)$ and $\mathsf Q \to -\mathsf Q$), so the invariance holds
   for all field types (`describe.py`, `fields.py`).
2. **Compare neighbours.** The runs at one parameter point form a sample. Two points that
   differ by one step in one parameter are linked. For each link the jump $J(a, b)$ is the
   energy distance between the two samples,

   $$D^2(X, Y) = 2\langle |x - y| \rangle - \langle |x - x'| \rangle - \langle |y - y'| \rangle ,$$

   which vanishes only when the two distributions are equal; for each point the spread
   $s(a) = D(a_1, a_2)$ is the same distance between two halves of its own sample
   (`walls.py`, `graph.py`).
3. **Cut.** Each link gets a weight $w = e^{-(J/\tilde J)^2}$, with $\tilde J$ the median
   jump. A random walk on the grid, which moves easily along small jumps and rarely across
   large ones, stays trapped in regions bounded by walls. The slowest mode of this diffusion
   (the Fiedler vector, second eigenvector of $L = 1 - d^{-1/2} W d^{-1/2}$, with $W$ the weights
   and $d$ the diagonal matrix of the summed weights of each point) orders the points
   across the strongest barrier, and the region is split where the normalised cut
   $\mathrm{cut}(S, \bar S)/\mathrm{vol}(S) + \mathrm{cut}(S, \bar S)/\mathrm{vol}(\bar S)$ is
   smallest.
4. **Test.** A cut is kept only if its crossing links jump more than the inner links: a
   one-sided Mann-Whitney test with $p <$ `ALPHA` $= 10^{-3}$ and a ratio of medians of at least
   `RATIO` $= 1.5$. The cut is placed with the jumps between the first halves of the points,
   $J_1 = D(a_1, b_1)$, and tested with the jumps between the second halves,
   $J_2 = D(a_2, b_2)$, so the test does not reuse the data that chose the cut. Both sides are
   then cut again until no cut passes.

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
periodic = true                # periodic boundary conditions
# optional: the frame described in every run (needed when boxes or grids differ between runs)
crop = 5.0                     # side of a centred square crop
dx_target = 0.02               # coarsen to this grid spacing (an integer multiple of dx)
max_shift = 1.0                # moving correlation: shifts up to this in x and y

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

Unknown keys are errors. The grid spacing can also be given per run, as a column `dx` of
the table of runs. A whole periodic box is described with FFTs and periodic shifts. A crop,
or a non-periodic box ($L$ its side, $s$ = `max_shift`, by default $L/5$), is described with
rules that need no periodicity: $S(k)$ of the field times a Tukey window, for wavelengths up
to a third of the frame only; divergence and curl by finite differences; domains not joined
across the edges; and the moving correlation by sliding the frame at $t$ over the frame at
$t - \Delta t$ by up to $\pm s$ in $x$ and $y$ (a crop of a non-periodic box stays $s$ away from
the walls, so by default it has side $L - 2s$). These frames are invariant under rotations and
mirrors only.

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

`pytest` runs 83 tests on small synthetic fields with known answers, in about 8 seconds:
the symmetry rules of vector and nematic fields against the gradient of a scalar and the
tensor of a director field; file round trips in Julia layout; the structure factor of a single
wave, domain counts of spots against stripes, curl-free and divergence-free flows, frozen,
travelling and changing patterns, and the invariance of every descriptor under the eight
symmetries of the square and periodic shifts; the links of a refined parameter grid; no wall in
pure noise or under a smooth drift, walls on the right links for two and three phases, a
bistable point flagged as mixed; the Swift-Hohenberg runs (state, and the stripe amplitude
$\mathrm{std}(u) = \sqrt{2\varepsilon/3}$ of weakly nonlinear theory); crops and non-periodic frames ($S(k)$ of
crops against the exact one, finite against spectral derivatives, the windowed moving
correlation, rotation and mirror invariance, the same descriptors at $\Delta x$ and
$\Delta x/2$ after coarsening, mixed grids refused without a common frame); and every command
of the command-line tool.

## Development plan

This section is for development and will be revised before any public release.

| Step | Content | Status |
| --- | --- | --- |
| 1 | Package: fields and their symmetries, descriptors, parameter graph, walls method, Swift-Hohenberg demo | done |
| 2 | Tests on synthetic data with known answers (83), continuous integration (ruff, pytest) | done |
| 2b | Frames: crops, coarsening, per-run grid spacing, non-periodic boxes; errors for unknown settings | done |
| 3 | Command-line tool `phasewall toy \| describe \| walls \| demo`, report (summary, tables, figures) | done |
| 4 | Documentation: method, mathematics, physics parallels, tests ([PDF](docs/documentation.pdf)) | done |
| 5 | Check on real simulations of an active fluid, from their descriptors and from a learned description; compare the phases found with published phase diagrams. The data stay outside this repository. | next |
| 6 | Optional self-supervised encoder (extra `[nn]`, PyTorch): contrastive training on snapshot pairs, with augmentations from the field symmetries | planned |
| 7 | README figures of the method itself (graph, jumps, cut, test) and of the encoder | planned |
| 8 | Licence and public release, once step 5 shows that the method works on real data | planned |

Open questions on the method:

- Power of the split-sample test: the demo needs 6 to 8 runs per point to find the weak
  hexagon-stripe wall. Cross-fitting (place with one half, test with the other, then swap)
  could use all runs.
- Mixed points: a split of 7 runs against 1 at a boundary point is not flagged; a test per
  point (for example bimodality of its runs) could replace the robust threshold.
- Scans that are not grids: links between nearest neighbours instead of grid neighbours.
- Bistable regions that runs from one kind of initial condition never reach (Section 8 of the
  documentation): runs from several initial conditions per point.

