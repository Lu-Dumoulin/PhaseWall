"""Tests of the explanations of blocks and descriptor columns (fields.BLOCK_HELP, describe.explain),
which the interface shows: every block has a definition, and every column the encoders make is
explained (never returned as its bare name)."""

import numpy as np

from phasewall.describe import describe_snapshot, explain
from phasewall.fields import BLOCK_HELP, BLOCKS, Field


def test_every_block_has_a_definition():
    assert set(BLOCK_HELP) == set(BLOCKS) and all(BLOCK_HELP.values())


def test_every_column_of_every_kind_is_explained():
    rng = np.random.default_rng(0)
    fields = [Field("u", ("u",)), Field("v", ("v",), "velocity"), Field("p", ("p",), "polarity"),
              Field("Q", ("Q",), "nematic")]
    snap = {"u": rng.normal(size=(16, 16))} | {f.name: rng.normal(size=(16, 16, 2)) for f in fields[1:]}
    row = describe_snapshot(snap, fields, 1.0, before=snap)
    for column in row:
        text = explain(column, fields)
        assert text != column and " " in text, column


def test_examples_of_explanations():
    fields = [Field("u", ("u",)), Field("p", ("p",), "polarity")]
    assert explain("u.topo.below-1", fields) == "domains per unit area where u is below its mean minus 1 standard deviation"
    assert explain("u.topo.above+0", fields) == "domains per unit area where u is above its mean"
    assert explain("p.mom.mean", fields) == "mean of |p|"
    assert explain("u.mom.skew") == "skewness of the values of u"
    assert explain("cnn.h.d007") == "component 7 of h, the learned description"
    assert explain("pc.3").startswith("principal component 3")
    assert explain("something") == "something"
