"""Tests of report.py on the synthetic two-phase table of test_walls: wall segments in the
right place, the summary, and the files written."""

import numpy as np
import pandas as pd
import pytest
from test_walls import DIM, NB, table

from phasewall import report
from phasewall.walls import find_walls, phases_against_ratio


@pytest.fixture(scope="module")
def two_phases():
    t = table(lambda a, b: np.full(DIM, 3.0 * (a > 5)))
    return t, find_walls(t, ["a", "b"])


def test_wall_segments_cross_the_cut_links(two_phases):
    """The wall between a = 5 and a = 6 is drawn at a = 5.5, one step of b long per link."""
    _, w = two_phases
    segs = sorted(report.wall_segments(w), key=lambda s: s[0][1])
    assert len(segs) == NB
    for b, ((a0, b0), (a1, b1)) in enumerate(segs):
        assert (a0, a1) == (5.5, 5.5)
        assert (b0, b1) == (b - 0.5, b + 0.5)


def test_phase_extent_and_summary(two_phases):
    t, w = two_phases
    extent = report.phase_extent(w)
    assert sorted(extent.a) == ["0 to 5", "6 to 9"]
    assert extent.points.sum() == len(w.nodes)
    text = report.summary(w, phases_against_ratio(t, ["a", "b"], ratios=(1.5, 3.0)), notes="extra note")
    assert text.startswith("# PhaseWall result\n\n2 phases on 80 parameter points (a, b)")
    assert f"{NB} of {len(w.edges)} links are walls: {NB} across a." in text
    assert text.rstrip().endswith("extra note")


def test_markdown_table():
    md = report._markdown(pd.DataFrame({"x": [1, 2], "y": [0.5, 1.23456]}))
    assert md.splitlines() == ["| x | y |", "|---|---|", "| 1 | 0.5 |", "| 2 | 1.23 |"]


def test_write_report(two_phases, tmp_path):
    t, w = two_phases
    reference = pd.Series(np.where(w.nodes.a > 5, "right", "left"), index=w.nodes.index)
    out = report.write_report(w, tmp_path / "rep", phases_against_ratio(t, ["a", "b"], ratios=(1.5,)),
                              reference=reference)
    names = {p.name for p in out.iterdir()}
    assert names == {"nodes.csv", "edges.csv", "cuts.csv", "robustness.csv", "summary.md",
                     "phase_map.png", "robustness.png"}
    nodes = pd.read_csv(out / "nodes.csv")
    assert len(nodes) == len(w.nodes) and "phase" in nodes


def test_phase_map_needs_two_parameters(two_phases, tmp_path):
    t, _ = two_phases
    w1 = find_walls(t[t.b == 0], ["a"])
    with pytest.raises(ValueError):
        report.phase_map_figure(w1, tmp_path / "x.png")
    report.write_report(w1, tmp_path / "one")          # no phase map, no error
    assert not (tmp_path / "one" / "phase_map.png").exists()


def test_label_colours_are_one_to_one():
    """Two labels mostly in the same phase still get different colours (and so different
    legend entries)."""
    labels = pd.Series(["a", "a", "b", "b", "b", "c"])
    phase = pd.Series([0, 0, 0, 0, 0, 1])
    colour = report.label_colours(labels, phase)
    assert colour["b"] == 0 and colour["c"] == 1          # largest overlaps take their phase colour
    assert len(set(colour.values())) == 3                 # "a" gets a free colour


def test_phase_map_of_a_tree_has_no_jumps_panel(two_phases):
    """A tree places its splits by thresholds, not by jumps: its phase map shows the phases (and
    the labels), not the constant jumps of Tree.as_walls."""
    from phasewall.trees import fit_tree

    t, w = two_phases
    tree = fit_tree(t, ["a", "b"]).as_walls()
    labels = pd.Series(np.where(w.nodes.a > 5, "high", "low"), index=w.nodes.index)

    def titles(fig):
        return [t for ax in fig.axes for t in (ax.get_title(loc="left"), ax.get_title()) if t]

    assert any("Jump" in s for s in titles(report.make_phase_map(w, labels)))
    tree_titles = titles(report.make_phase_map(tree, labels))
    assert len(tree_titles) == 2 and not any("Jump" in s for s in tree_titles)
    assert len(titles(report.make_phase_map(tree))) == 1


def test_report_says_what_changes_across_the_wall(two_phases, tmp_path):
    t, w = two_phases
    report.write_report(w, tmp_path, table=t)
    obs = pd.read_csv(tmp_path / "wall_observables.csv")
    assert len(obs) == 3 and {"column", "ratio", "meaning"} <= set(obs.columns)
    assert "What changes across each wall" in (tmp_path / "summary.md").read_text()


def test_report_lists_candidate_order_parameters(two_phases, tmp_path):
    t, w = two_phases
    report.write_report(w, tmp_path, table=t)
    ops = pd.read_csv(tmp_path / "order_parameters.csv")
    first = ops[ops["rank"] == 1].iloc[0]
    assert first.column.startswith("f.blk.x") and first.held_out == 100 and first.shared_neighbour   # every column jumps
    assert {"rule", "placing", "value_a", "value_b", "meaning"} <= set(ops.columns)
    assert not ops.comparison.str.startswith("others").any()        # two phases: the pair is enough
    text = (tmp_path / "summary.md").read_text()
    assert "Candidate order parameters" in text and "| 1 (100 %) |" in text and "shared neighbour" in text
