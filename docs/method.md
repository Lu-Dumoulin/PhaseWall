# Method

This page explains each step in short. The [documentation (PDF)](documentation.pdf) gives the
mathematics, a physics parallel for each step, and every choice.

## 1. Describe each snapshot

Each snapshot becomes a vector $h$ of numbers. Steps 2 to 4 are the same for any description.

### Hand-made description

A fixed list of observables per field (`describe.py`, `fields.py`). Nothing is learned. A rotation,
a mirror or a shift of the box does not change them. The observables come in blocks:

| block | what it measures |
| --- | --- |
| `amp` | $\log_{10}$ of the variance: how strong the pattern is |
| `shape` | $S(k)$ on rings of $\lvert k \rvert$, divided by its sum: which length scales are present |
| `mom` | mean, standard deviation, skewness, kurtosis |
| `topo` | number of domains above and below three levels |
| `dyn` | change between two snapshots: frozen, travelling or changing |
| `order` | global order $\lvert\langle \mathbf a \rangle\rvert / \langle \lvert \mathbf a \rvert \rangle$ (vectors, nematics) |
| `flow` | divergence and curl (vectors) |
| `vortex` | vortex cores, Okubo-Weiss criterion (velocity) |
| `defects` | winding defects $\pm 1$ (polarity), $\pm \frac12$ (nematic) |
| `frank` | Frank energy of the orientation: splay, bend, change over time (polarity, nematic) |
| `parts` | two states side by side in one box (optional) |

- Each kind of field has default blocks. You can choose the blocks per field.
- Vector and nematic components turn with the box: a rotation by $90^\circ$ gives
  $(a_x, a_y) \to (-a_y, a_x)$ and $\mathsf Q \to -\mathsf Q$.
- Each block gets the same weight in the distances (columns standardised, block divided by
  $\sqrt{n}$).
- A crop or a non-periodic box uses a Tukey window for $S(k)$, finite differences, and no periodic
  domains.

### Learned description

A convolutional network $f$ (`phasewall.nn`, extra `[nn]`) turns a pair of frames
$(t - \Delta t, t)$ into $h$ (256 numbers). Four blocks halve the grid, then the mean over the frame
gives $h$.

- **Training without labels (contrastive, SimCLR).** Two views of one run must be recognised among
  the views of the other runs (InfoNCE loss on $z = g(h)/\lvert g(h)\rvert$; the head $g$ is dropped
  after training).
- **Views.** A view is a pair of frames turned by any angle, mirrored, zoomed by 0.8 to 1.25, cut
  to 3/4 of the frame, and normalised. Orientation, wavelength, position and amplitude change
  between views, so the loss can only use the shape of the pattern.
- **Why normalise.** Without it, $h$ followed the amplitude. The amplitude grows like
  $\sqrt\varepsilon$ inside a phase, so it changes smoothly, and the walls found 2 phases of 3.
- **Checks before training.** The network shape is checked against the frame: the last map has at
  least 2 points, the channels never decrease. The interface shows the receptive field, the
  number of weights and the memory of one training step.
- **Predictive description (JEPA), being tested.** Hide four squares of a frame and predict their
  description from the rest, with a target encoder that slowly follows the main one. There are no
  negatives, so nothing rewards the recognition of a run. See the next steps in the
  [README](../README.md#next-steps).

A **combined** description puts the hand-made and the learned blocks side by side.

## 2. Compare neighbouring points

- The runs at one point form a sample. Each point is split into two halves (its runs, or the early
  and late snapshots of a single run).
- On a grid, two points one step apart along one parameter are linked. Scattered points share no
  grid line: see [Scattered points](#scattered-points-k-nn-links) below.
- The jump of a link is the energy distance between the two samples:

$$D^2(X, Y) = 2\langle \lvert x - y \rvert \rangle - \langle \lvert x - x' \rvert \rangle - \langle \lvert y - y' \rvert \rangle .$$

  It is zero only for equal distributions, so it also sees a bistable point.
- $J_1$ uses the first halves of the two points, $J_2$ the second halves.

## 3. Cut

- Each link gets the weight $w = e^{-(J_1/\tilde J_1)^2}$ ($\tilde J_1$: median jump).
- A random walk on these weights rarely crosses a large jump. Its slowest mode, the Fiedler vector
  of $L = 1 - d^{-1/2} W d^{-1/2}$, orders the points across the strongest barrier.
- PhaseWall sorts the points by this vector and cuts where the normalised cut
  $\mathrm{cut}(S, \bar S)/\mathrm{vol}(S) + \mathrm{cut}(S, \bar S)/\mathrm{vol}(\bar S)$ is smallest
  (at least 3 points per side).

## 4. Test

- The cut is a wall when its crossing links jump more than the inner links, measured with $J_2$:
  one-sided Mann-Whitney $p <$ `ALPHA` $= 10^{-3}$ and ratio of medians $\ge$ `RATIO` $= 1.5$.
- $J_2$ did not place the cut. A test with $J_1$ would find walls in noise (24 of 40 tables); with
  $J_2$, none (0 of 40).
- PhaseWall cuts each side again until no cut passes. The regions are the phases (at least 3
  points).

## Scattered points: k-NN links

PhaseWall uses grid links when at least 90 % of the points have one and they join all the points.
Otherwise (random or space-filling samples):

1. Each parameter is divided by the median gap between its values. A positive parameter spread more
   evenly in $\log_{10}$ than in linear scale is taken in $\log_{10}$ first.
2. Each point is linked to its $k = \max(6, 2d)$ nearest points ($d$: number of parameters; 6 up to
   3 parameters, 8 for 4, 10 for 5). A link is kept if either point chose it.
3. Links longer than 3 times the median link are dropped: they cross a hole in the sampling.
4. Inside a phase a longer link jumps more. So $\log J$ is corrected by the Theil-Sen slope $\beta$
   of $\log J$ against $\log(\ell/\tilde\ell)$: $\log J - \beta \log(\ell/\tilde\ell)$.
5. Cut and test as above, on every link. "Grid steps" below become median link lengths.

| example (same points and runs as the grid) | grid | random points |
| --- | --- | --- |
| Swift-Hohenberg (240 x 8) | 3 phases, ARI 0.94 | 3 phases, ARI 0.94 |
| FitzHugh-Nagumo (99 x 4) | 1 phase | 2 phases, ARI 0.36 |
| Barkley (165 x 4) | 2 phases, ARI 0.58 | 3 phases, ARI 0.59 |
| localised Swift-Hohenberg (105 x 4) | 3 phases, ARI 1.00 | 2 phases, ARI 0.80 |
| Gray-Scott (525 x 4) | 8 phases, ARI 0.95 | 7 phases, ARI 0.79 |

Losses come from thin phases: the localised band holds 11 of 105 random points. On synthetic noise
tables of scattered points, 2 of 100 get a false wall (grid: 0 of 100): links that share a point are
not independent, and $k$-NN gives more links per point. Tried and not kept: testing on the 4 nearest
links only (it hides real walls), several candidate cuts per step.

**Many parameters.** Synthetic tables: 500 or 2000 random points in $[0, 1]^d$, 2 runs per point,
a wall at $x_0 = 0.5$ (and a second at $x_1 = 0.5$ on one side), 3 noise widths.

| parameters | $k = 6$ | $k = 2d$ |
| --- | --- | --- |
| 3, 4, 5 | right, except 1 false wall in 10 noise tables (4 parameters) | all right |
| 8 | all right | all right |
| 16 | 1 false wall in 10 noise tables; 2 of 20 wall tables wrong | all right |

Hence $k = \max(6, 2d)$. Limits of this test: strong walls, and the description depends only on the
parameters that carry the walls. Weak walls and drifts along many parameters are not tested.

## After the walls

- **Runs that ended early.** A run whose last snapshot comes before 0.1 of the median end time is
  left out. A point where all runs ended early is *unstable* (a cross on the map).
- **Mixed points** (a ring on the map). Points whose runs differ much more than at the other points
  with several runs: runs at the same parameters end in different states.
- **One phase in two places.** Two phases that share no link are joined when their cores (points at
  least 2 grid steps from another phase) are as close as the two halves of one core.
- **Points near the walls** (a square on the map). A point within 2 grid steps of another phase
  joins it when the core of that phase is 1.2 times closer than its own core.

## What changes across a wall

For each pair of neighbouring phases, PhaseWall lists the descriptors with the largest jump across
the wall, compared with their jumps inside the two phases (`wall_observables.csv`). These describe
the walls; they play no part in finding them.

## Candidate order parameters

For each pair of phases, and for each phase against all the others, PhaseWall looks for one
descriptor and one threshold that separate the two groups (`order_parameters.csv`).

- The threshold is chosen on one half of the runs. The score (% of points on the correct side,
  balanced between the two groups, 50 % = chance) is counted on the other half. A descriptor that
  separates a few points by chance does not rank high.
- A combination of 2 or 3 descriptors (Fisher discriminant) is listed only when it beats the best
  single descriptor by 3 points.
- Swift-Hohenberg: hexagons against stripes by the skewness of $u$ (98 %); uniform against the rest
  by the variance of $u$ (100 %).

## Threshold tree

An alternative to the walls that gives each phase as a rule (`trees.py`). It never uses labels.

1. In a region, the value of a point is the median of a descriptor.
2. Each threshold gets a score: the share of variance between the two sides (Otsu) times their
   coherence on the grid (normalised cut).
3. The other half of the runs tests the best split: the two sides differ (Mann-Whitney), the
   descriptor jumps across the split (ratio $\ge$ `RATIO`, along each parameter, so a drift fails),
   and 90 % of the points stay on the same side.
4. Both halves must find the split. Then the tree splits each side again.

With one run per point, the two halves of a run are not independent. The tree then uses a
checkerboard of the points: it places the split on one colour and tests it on the other.

Swift-Hohenberg: phase 2 (uniform) is "variance of $u$ at its floor"; hexagons and stripes differ
by the number of small domains below the mean.

## Blind methods and the probe

These methods do not use the grid. They group the runs by their description only
(`baselines.py`, extra `[baselines]`). Each run is the mean of its snapshots, reduced to the
principal components that keep 95 % of the variance.

- **Gaussian mixture**: 1 to 10 Gaussians, number chosen by BIC.
- **k-means**: $k$ = 2 to 10, chosen by the silhouette.
- **HDBSCAN**: dense groups; runs in sparse regions are noise.

The **probe** is a logistic regression of the known states, tested on held-out parameter points.
It sees the labels, so it is a ceiling, not a method: it says how well a linear reading of $h$
could separate the states.

Blind methods need separated clouds of runs. The walls need only a jump along the grid. On
Swift-Hohenberg the hexagon and stripe clouds touch and drift with the parameters, so only the
walls give three connected phases.

## One-dimensional systems

A 1D run is a kymograph $a(x, t)$ per field. PhaseWall cuts the last part of each run (half by
default) into time windows (8 by default). Each window is one row of the table, so steps 2 to 4
do not change.

- **Fields**: scalar, velocity, polarity $p_x$ or $(p_x, p_y)$, complex amplitude. The only
  symmetries are shifts and the mirror $x \to -x$.
- **Observables**: the 2D blocks in 1D form, and four blocks that need the dynamics: the
  space-time spectrum $S(k, \omega)$, the correlation at many lags (decorrelation time, drift
  speed), kinks (zeros of the field, their births and deaths), and a trend (for runs still
  relaxing).
- **Complex fields** are described through $\lvert A \rvert$: the global phase does not count.
- **Learned description**: the same CNN on the $(x, t)$ window. A $3 \times 3$ kernel can hold
  $\partial_x$, $\partial_t$ and their products. The views are crops and the mirror, without
  rotation or zoom (they would mix $x$ with $t$).
