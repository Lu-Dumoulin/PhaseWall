"""Tests of nn/architecture.py: the channels of a network, its receptive field, the constraints
against the frame, the weight count against PyTorch's, a network of any allowed shape built and
trained, its shape kept on saving, and a different network kept apart in the pipeline cache."""

import numpy as np
import pytest

from phasewall.nn.architecture import MIN_FINAL, Network, check, max_blocks, receptive_field, summary, weights


def test_default_network_and_views():
    net = Network()
    assert net.channels == (32, 64, 128, 256)
    assert (net.hidden, net.projection, net.temperature) == (256, 64, 0.2)
    assert (net.views, net.zoom, net.crop, net.normalise) == ("rotzoom", 1.25, 0.75, True)


def test_channels_double_up_to_dim_h():
    assert Network(blocks=3, width=16, dim_h=256).channels == (16, 32, 256)
    assert Network(blocks=5, width=64, dim_h=128).channels == (64, 128, 128, 128, 128)
    assert Network(blocks=1, width=8, dim_h=64).channels == (64,)


def test_receptive_field():
    assert [receptive_field(b) for b in (1, 2, 3, 4)] == [7, 19, 43, 91]


def test_constraints():
    check(Network(), 64)
    assert max_blocks(64) == 5 and max_blocks(32) == 4 and max_blocks(MIN_FINAL) == 0
    with pytest.raises(ValueError, match="at most 4 blocks"):
        check(Network(blocks=5, views="d4"), 32)
    with pytest.raises(ValueError, match="at most 4 blocks"):
        check(Network(blocks=5), 32)            # rotzoom: the crop grows to the frame, no further
    check(Network(blocks=4), 32)                # rotzoom: crop 24 of 32 grown to 32 for 4 blocks
    # non-periodic: the turned and zoomed crop must stay inside (at most 32 / (1.25 sqrt 2) = 18)
    with pytest.raises(ValueError, match="sees 18 of the frame's 32: at most 3 blocks"):
        check(Network(blocks=4), 32, periodic=False)
    with pytest.raises(ValueError, match="width 128 is larger than dim_h 64"):
        check(Network(width=128, dim_h=64, projection=32), 64)
    with pytest.raises(ValueError, match="projection 128 is larger than dim_h 64"):
        check(Network(dim_h=64, projection=128), 64)
    with pytest.raises(ValueError, match="hidden layer 32"):
        check(Network(hidden=32, projection=64), 64)
    with pytest.raises(ValueError, match="choose one of"):
        check(Network(width=24), 64)
    with pytest.raises(ValueError, match="outside"):
        check(Network(blocks=9), 4096)


def test_summary():
    s = summary(Network(views="d4"), 64, in_channels=2, batch=128)
    assert s["input"] == 64 and s["last_map"] == 4 and s["receptive_field"] == 91 and s["dim_h"] == 256
    r = summary(Network(), 64, in_channels=2, batch=128)
    assert r["input"] == 48 and r["last_map"] == 3 and r["memory_gb_rough"] < s["memory_gb_rough"]
    small = summary(Network(blocks=2, width=8, dim_h=32), 64, in_channels=2, batch=128)
    assert small["weights_f"] < s["weights_f"] and small["memory_gb_rough"] < s["memory_gb_rough"]


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("net", [Network(), Network(blocks=2, width=8, dim_h=32, hidden=32, projection=16),
                                 Network(blocks=5, width=16, dim_h=512, hidden=512, projection=128)])
def test_weights_and_shapes_match_pytorch(net):
    from phasewall.nn.model import Encoder, ProjectionHead, count_weights
    enc, head = Encoder(4, net.channels, True), ProjectionHead(net.dim_h, net.hidden, net.projection)
    assert weights(net, 4) == {"f": count_weights(enc), "g": count_weights(head)}
    h = enc.eval()(torch.zeros(3, 4, 64, 64))
    assert h.shape == (3, net.dim_h) and head(h).shape == (3, net.projection)


def test_a_chosen_network_trains_and_is_saved(tmp_path):
    from test_nn import U, field

    from phasewall.nn.data import FrameSet
    from phasewall.nn.train import Settings, load, save, train
    net = Network(blocks=3, width=8, dim_h=64, hidden=64, projection=16, temperature=0.3)
    x = np.stack([np.stack([field(32, 2 * i), field(32, 2 * i + 1)]) for i in range(16)]).astype(np.float32)
    frames = FrameSet(x, np.arange(16), np.zeros(16), [U], True, np.ones(1))
    enc, log = train(frames, Settings.from_network(net, epochs=1, batch=8), dev="cpu", verbose=False)
    assert enc.dim == 64 and np.isfinite(log.loss).all()
    back, meta = load(save(tmp_path / "enc", enc, frames, Settings.from_network(net, epochs=1, batch=8), log))
    assert back.dim == 64 and meta["settings"]["projection"] == 16 and meta["settings"]["temperature"] == 0.3


def test_the_network_is_part_of_the_cache_name(tmp_path):
    from phasewall import pipeline as pl
    src = pl.Source(example="swift-hohenberg", grid={"eps": [-0.1, 0.5], "g": [0.0, 1.4]}, seeds=2,
                    settings={"n": 32, "waves": 4, "n_save": 3, "t_relax": 200.0})
    small = Network(blocks=2, width=8, dim_h=32, hidden=32, projection=16)

    def run(net):
        s = pl.Settings(source=src, describers=pl.Describers(hand=False, learned=True, epochs=1, network=net),
                        methods=pl.Methods(walls=False), execution=pl.Execution(workers=1, cache=str(tmp_path)))
        return pl.run(s).tables["learned seed 0"]
    a, b = run(small), run(Network(blocks=2, width=8, dim_h=64, hidden=64, projection=16))
    assert a.filter(like="cnn.h").shape[1] == 32 and b.filter(like="cnn.h").shape[1] == 64
    assert len(list(tmp_path.glob("learned-*"))) == 2
    with pytest.raises(ValueError, match="at most 4 blocks"):
        run(Network(blocks=5))


def test_learn_command_takes_a_shape_and_refuses_a_wrong_one(tmp_path):
    from phasewall import cli, toy
    toy.make_ensemble(tmp_path, eps=[-0.1, 0.3], g=[0.0, 1.0], seeds=(0, 1), n=32, waves=4)
    args = ["learn", str(tmp_path / "phasewall.toml"), "--epochs", "1", "--workers", "1"]
    out = cli.run([*args, "--blocks", "2", "--width", "8", "--dim-h", "32", "--hidden", "32", "--projection", "16"])
    import pandas as pd
    t = pd.read_csv(out / "descriptors_cnn.csv")
    assert "cnn.h.d031" in t and "cnn.h.d032" not in t
    with pytest.raises(SystemExit, match="at most 4 blocks"):
        cli.run([*args, "--blocks", "6", "--out", str(tmp_path / "bad")])
