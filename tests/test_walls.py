"""Tests of walls.py on synthetic descriptor tables with a known answer: one smooth phase (no
wall), two and three phases (walls on the right links), a bistable point (mixed), mixed judged
only among points with several runs, a drifting run not mixed, one phase in
two places joined across the phase between them, a small piece (three points) joined or not, runs that
ended early left out (unstable points), points near a wall moved to the phase they belong to, and the
building blocks (energy distance, halves, cuts on a chain)."""

import numpy as np
import pandas as pd
import pytest

from phasewall import walls as w

NA, NB = 10, 8          # grid of NA x NB parameter points
RUNS, SNAPS = 2, 5      # runs per point, snapshots per run
DIM = 6                 # descriptor columns


def table(mean_of, noise: float = 1.0, seed: int = 0, run_mean_of=None) -> pd.DataFrame:
    """One row per snapshot on the NA x NB grid (a, b in 0, 1, ...). The descriptors are
    Gaussian with mean mean_of(a, b) (a vector of length DIM) and standard deviation `noise`;
    run_mean_of(a, b, r), if given, sets the mean of run r instead."""
    rng = np.random.default_rng(seed)
    rows, run = [], 0
    for a in range(NA):
        for b in range(NB):
            for r in range(RUNS):
                m = run_mean_of(a, b, r) if run_mean_of else mean_of(a, b)
                for t in range(SNAPS):
                    x = m + noise * rng.standard_normal(DIM)
                    rows.append({"run": run, "time": float(t), "a": float(a), "b": float(b)}
                                | {f"f.blk.x{j}": x[j] for j in range(DIM)})
                run += 1
    return pd.DataFrame(rows)


def walls_between(result: w.Walls) -> set:
    """The walls as pairs of parameter points."""
    n, e = result.nodes, result.edges[result.edges.wall]
    return {(tuple(n.loc[a, ["a", "b"]]), tuple(n.loc[b, ["a", "b"]])) for a, b in zip(e.a, e.b)}


def test_energy_distance():
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((200, 3)), rng.standard_normal((200, 3))
    assert w.energy_distance(A, B) < 0.15
    far = w.energy_distance(A, B + [3, 0, 0])
    # for a large shift s, 2 <|x - y|> - <|x - x'|> - <|y - y'|> tends to 2 s - 2 * 1.6
    assert far == pytest.approx(np.sqrt(2 * 3 - 2 * 1.6), rel=0.15)
    assert w.energy_distance(A, B + [6, 0, 0]) > far


def test_halves_by_run_or_by_time():
    run = np.array([4, 4, 7, 7, 9, 9])
    time = np.array([0, 1, 0, 1, 0, 1.0])
    assert w.halves(run, time).tolist() == [True, True, False, False, True, True]
    one_run = np.zeros(4, dtype=int)
    assert w.halves(one_run, np.array([3, 0, 2, 1.0])).tolist() == [False, True, False, True]
    assert w.halves(one_run, np.zeros(4)).tolist() == [True, False, True, False]


def test_feature_matrix():
    t = pd.DataFrame({"run": 0, "time": 0.0, "f.one.x": [1.0, 2.0, np.nan, 3.0],
                      "f.four.a": [1.0, 2, 3, 4], "f.four.b": [4.0, 3, 2, 1],
                      "f.four.c": [0.0, 1, 0, 1], "f.four.d": [1.0, 1, 2, 2], "f.flat.x": 5.0})
    X, names = w.feature_matrix(t)
    assert names == ["f.one.x", "f.four.a", "f.four.b", "f.four.c", "f.four.d"]
    assert X[2, 0] == pytest.approx((2 - 2) / np.std([1, 2, 2, 3]))      # NaN -> median 2
    assert np.allclose(X.mean(axis=0), 0)
    # a block of 4 columns is divided by 2, so it weighs as much as a block of 1
    assert np.allclose(X.std(axis=0), [1, 0.5, 0.5, 0.5, 0.5])
    X, _ = w.feature_matrix(t, weight_blocks=False)
    assert np.allclose(X.std(axis=0), 1)


def test_cut_of_a_chain_with_one_large_jump():
    """Chain of 10 points; the link 5-6 jumps 10 times more than the others."""
    a, b = np.arange(9), np.arange(1, 10)
    log_jump = np.log(np.where(a == 5, 10.0, 1.0) * (1 + 0.05 * np.sin(a)))
    region, tried, wall = w.split_regions(10, a, b, log_jump, min_phase=2, alpha=0.2)
    assert wall.tolist() == [i == 5 for i in range(9)]
    assert region.tolist() == [0] * 6 + [1] * 4      # region 0 is the larger side
    assert tried[0]["accepted"] and tried[0]["crossing_links"] == 1


@pytest.mark.parametrize("seed", range(5))
def test_no_wall_in_pure_noise(seed):
    """The same distribution everywhere. Before the cuts were tested on other halves than
    those that placed them, 2 or 3 phases came out here for about half of the seeds."""
    res = w.find_walls(table(lambda a, b: np.zeros(DIM), seed=seed), ["a", "b"])
    assert res.n_phases == 1
    assert not res.edges.wall.any()


def test_no_wall_in_one_smooth_phase():
    """The mean drifts smoothly across the grid, by less than the noise between neighbours."""
    res = w.find_walls(table(lambda a, b: np.full(DIM, 0.3 * a + 0.2 * b)), ["a", "b"])
    assert res.n_phases == 1
    assert not res.edges.wall.any()
    assert not res.nodes.mixed.any()
    assert len(res.cuts) >= 1 and not res.cuts.accepted.any()


def test_two_phases():
    """The mean jumps by 3 noise widths between a = 5 and a = 6."""
    res = w.find_walls(table(lambda a, b: np.full(DIM, 3.0 * (a > 5))), ["a", "b"])
    assert res.n_phases == 2
    assert walls_between(res) == {((5.0, b), (6.0, b)) for b in range(NB)}
    left = res.nodes.a <= 5
    assert res.nodes.loc[left, "phase"].nunique() == 1
    assert res.nodes.loc[~left, "phase"].nunique() == 1
    assert res.nodes.loc[left, "phase"].iloc[0] != res.nodes.loc[~left, "phase"].iloc[0]
    assert (res.nodes.runs == RUNS).all() and (res.nodes.snapshots == RUNS * SNAPS).all()


def test_three_phases_on_a_smooth_background():
    """A smooth drift everywhere, plus a jump at a = 5 | 6 and, for a >= 6, another at b = 3 | 4
    along other descriptors."""
    def mean(a, b):
        m = np.full(DIM, 0.2 * a + 0.1 * b)
        m[:3] += 3.0 * (a > 5)
        m[3:] += 3.0 * ((a > 5) & (b > 3))
        return m

    res = w.find_walls(table(mean), ["a", "b"])
    assert res.n_phases == 3
    expected = ({((5.0, b), (6.0, b)) for b in range(NB)}
                | {((a, 3.0), (a, 4.0)) for a in np.arange(6.0, NA)})
    assert walls_between(res) == expected


def test_bistable_point_is_flagged_mixed():
    """At (2, 2) one run is in each of two very different states; everywhere else both runs
    are in the same state."""
    def run_mean(a, b, r):
        return np.full(DIM, 8.0 if (a, b, r) == (2, 2, 1) else 0.0)

    res = w.find_walls(table(None, run_mean_of=run_mean), ["a", "b"])
    mixed = res.nodes[res.nodes.mixed]
    assert list(zip(mixed.a, mixed.b)) == [(2.0, 2.0)]


def test_phase_count_against_ratio():
    t = table(lambda a, b: np.full(DIM, 3.0 * (a > 5)))
    curve = w.phases_against_ratio(t, ["a", "b"], ratios=(1.5, 2.0, 1e6))
    assert curve.phases.tolist() == [2, 2, 1]
    assert curve.walls.tolist()[-1] == 0


def bands(levels: tuple) -> callable:
    """A mean that is levels[k] (every descriptor) in the k-th third of the grid along a."""
    def mean_of(a, b):
        return np.full(DIM, float(levels[min(int(3 * a / NA), 2)]))
    return mean_of


def test_one_phase_in_two_places_is_joined():
    t = table(bands((0.0, 8.0, 0.0)), noise=0.5, seed=3)
    apart = w.find_walls(t, ["a", "b"], merge=False)
    joined = w.find_walls(t, ["a", "b"], merge=True)
    assert apart.n_phases == 3 and joined.n_phases == 2
    left, right = joined.nodes[joined.nodes.a == 0].phase, joined.nodes[joined.nodes.a == NA - 1].phase
    assert left.nunique() == 1 and set(left) == set(right)
    assert walls_between(joined) == walls_between(apart)            # the walls stay
    assert joined.merges.merged.tolist() == [True]


def test_different_phases_in_two_places_stay_apart():
    t = table(bands((0.0, 8.0, 16.0)), noise=0.5, seed=3)
    r = w.find_walls(t, ["a", "b"], merge=True)
    assert r.n_phases == 3 and r.merges.merged.tolist() == [False]


def test_grid_coordinates_count_steps_by_rank():
    pts = np.array([[0.01, 1.0], [0.1, 1.0], [1.0, 2.0], [0.1, 5.0]])
    assert w.grid_coordinates(pts).tolist() == [[0, 0], [1, 0], [2, 1], [1, 2]]


def uneven_table(two_runs: set, run_noise: float = 2.0, seed: int = 0, odd=None, drift=None) -> pd.DataFrame:
    """NA x NB points, one run each except the points in two_runs (two runs). Every run has its own
    offset (Gaussian, run_noise per descriptor: two runs of one point differ much more than two
    time halves of one run), snapshots add noise 1. odd: a point whose second
    run is offset by 8 (two states); drift: a point whose single run drifts by 6 over its snapshots."""
    rng = np.random.default_rng(seed)
    rows, run = [], 0
    for a in range(NA):
        for b in range(NB):
            for r in range(2 if (a, b) in two_runs else 1):
                m = run_noise * rng.standard_normal(DIM) + (8.0 if (a, b, r) == odd else 0.0)
                for t in range(6):
                    x = m + rng.standard_normal(DIM) + (6.0 * t / 5 if (a, b) == drift else 0.0)
                    # the second run of a point covers later times (runs of different lengths)
                    rows.append({"run": run, "time": float(t + 100 * r), "a": float(a), "b": float(b)}
                                | {f"f.blk.x{j}": x[j] for j in range(DIM)})
                run += 1
    return pd.DataFrame(rows)


TWO_RUNS = {(a, b) for a in range(0, NA, 2) for b in range(0, NB, 3)}       # 15 points with two runs


def test_mixed_is_judged_among_points_with_several_runs():
    """Most points have one run (spread from time halves, small); 15 have two runs that differ by
    ordinary run-to-run noise. None of them is mixed. With a single baseline for all points (the
    rule before 2026-11-23) most of them were flagged."""
    res = w.find_walls(uneven_table(TWO_RUNS), ["a", "b"])
    assert not res.nodes.mixed.any()
    n = res.nodes
    assert set(n.halves[n.runs == 2]) == {"runs"} and set(n.halves[n.runs == 1]) == {"time"}
    old = w.robust_outliers(n.spread.to_numpy() ** 2, w.MIXED_K, min_scale=res.floor**2)
    assert old[n.runs.to_numpy() == 2].sum() >= 8


def test_two_states_at_one_point_is_mixed_and_a_drift_is_not():
    """Among the points with two runs, (2, 3) has one run in another state: mixed. The point (5, 4)
    has one run that drifts in time: not mixed (a single run says nothing about several states)."""
    res = w.find_walls(uneven_table(TWO_RUNS, odd=(2, 3, 1), drift=(5, 4)), ["a", "b"])
    n = res.nodes
    assert list(zip(n.a[n.mixed], n.b[n.mixed])) == [(2.0, 3.0)]


def test_too_few_points_with_several_runs_flag_nothing():
    """With fewer than MIN_MIXED_GROUP points of several runs there is no baseline: no mixed point."""
    res = w.find_walls(uneven_table({(2, 3), (6, 1)}, odd=(2, 3, 1)), ["a", "b"])
    assert not res.nodes.mixed.any()


def small_piece(level_of_piece: float):
    """Phases given by hand on the NA x NB grid: 0 (a <= 3, level 0), 1 (a = 4, 5, level 8), 3 (a >= 6,
    level 16) and a small phase 2 of three points (a = 9, b <= 2) at level_of_piece, which touches
    only phase 3. Ten snapshots per point, noise 0.5."""
    rng = np.random.default_rng(1)
    pts = np.array([(a, b) for a in range(NA) for b in range(NB)], dtype=float)
    phase = np.where(pts[:, 0] <= 3, 0, np.where(pts[:, 0] <= 5, 1, 3))
    phase[(pts[:, 0] == 9) & (pts[:, 1] <= 2)] = 2
    level = np.array([0.0, 8.0, level_of_piece, 16.0])[phase]
    X = np.repeat(level, 10)[:, None] + 0.5 * rng.standard_normal((len(pts) * 10, DIM))
    rows_of = [np.arange(10 * n, 10 * n + 10) for n in range(len(pts))]
    e = w.grid_edges(pts)
    return w.merge_phases(phase, pts, e.a.to_numpy(), e.b.to_numpy(), X, rows_of, floor=0.1, noise=0.5)


def test_a_small_piece_of_a_phase_is_joined():
    """A phase of three points (too small to split in two halves) is compared all the same: joined
    with the phase it belongs to, kept apart from a different one."""
    new, tests = small_piece(0.0)
    assert ((tests.phase_a == 0) & (tests.phase_b == 2) & tests.merged).any()
    assert len(np.unique(new)) == 3
    new, tests = small_piece(24.0)
    assert ((tests.phase_a == 0) & (tests.phase_b == 2)).any() and not tests.merged.any()
    assert len(np.unique(new)) == 4


def test_runs_that_ended_early_are_left_out():
    """Two runs per point end at t = 4 (ordinary); at (7, 2) both runs stop at t = 0 and carry a
    state far from the others (a crash: early transients), at (1, 1) one of the two. With the
    default END_FRACTION they are left out: (7, 2) is an unstable point, outside every phase,
    (1, 1) is described by its other run, and the walls are those of one phase."""
    t = table(lambda a, b: np.zeros(DIM), seed=2)
    run_at = t.groupby("run")[["a", "b"]].first()
    crashed = list(run_at.index[(run_at.a == 7) & (run_at.b == 2)]) + [run_at.index[(run_at.a == 1) & (run_at.b == 1)][0]]
    t = t[~(t.run.isin(crashed) & (t.time > 0))].copy()
    cols = [c for c in t if c.startswith("f.")]
    t.loc[t.run.isin(crashed), cols] += 20.0
    res = w.find_walls(t, ["a", "b"])
    assert sorted(res.ended.run) == sorted(crashed)
    assert list(zip(res.unstable.a, res.unstable.b)) == [(7.0, 2.0)] and res.unstable.runs.tolist() == [2]
    n = res.nodes
    assert len(n) == NA * NB - 1 and n.ended_runs[(n.a == 1) & (n.b == 1)].tolist() == [1]
    assert res.n_phases == 1 and not res.edges.wall.any()
    kept = w.find_walls(t, ["a", "b"], end_fraction=None)
    assert len(kept.ended) == 0 and len(kept.nodes) == NA * NB


def test_points_near_a_wall_join_the_phase_they_belong_to():
    """Two phases (level 0 for a <= 3, level 4 for a >= 6) and a band a = 4, 5 where the two states
    alternate from point to point (a checkerboard, as where both states are possible and each
    point has one run). No cut along the grid follows a checkerboard; after the reassignment every
    point of the band is in the phase of its own state. A point in the other state deep inside a
    phase (more than NEAR grid steps from the other phase) is not moved, and nothing moves in a
    clean two-phase table."""
    def state_b(a, b):
        return a >= 6 or (a in (4, 5) and (a + b) % 2 == 1) or (a, b) == (0, 6)
    t = table(lambda a, b: np.full(DIM, 4.0 if state_b(a, b) else 0.0), noise=0.5, seed=4)
    res = w.find_walls(t, ["a", "b"])
    n = res.nodes
    left, right = n.phase[n.a == 1].iloc[0], n.phase[n.a == 9].iloc[0]
    band = n[n.a.isin([4, 5])]
    expected = [right if state_b(a, b) else left for a, b in zip(band.a, band.b)]
    assert (band.phase_walls.to_numpy() != expected).sum() >= 4       # the cut alone gets several wrong
    assert band.phase.tolist() == expected
    assert n.phase[(n.a == 0) & (n.b == 6)].tolist() == [left] and n.reassigned.sum() > 0
    clean = w.find_walls(table(lambda a, b: np.full(DIM, 4.0 * (a > 4)), noise=0.5, seed=4), ["a", "b"])
    assert not clean.nodes.reassigned.any() and len(clean.reassigned) == 0
    off = w.find_walls(t, ["a", "b"], reassign=False)
    assert "phase_walls" not in off.nodes and not off.nodes.reassigned.any()
