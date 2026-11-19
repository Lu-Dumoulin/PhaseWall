"""The shape of the learned encoder f and of its head g, chosen within checked constraints.
No PyTorch needed here, so that an interface can check a choice before any training.

A network is given by five numbers (Network):
- blocks: convolution blocks of f; each halves the grid (3 x 3 convolution with stride 2, then a
  3 x 3 convolution, batch normalisation and ReLU after each);
- width: channels of the first block; each next block doubles them, up to dim_h;
- dim_h: channels of the last block, which is the dimension of h (the mean of each final map over
  the frame);
- hidden, projection: the head g, Linear(dim_h, hidden), ReLU, Linear(hidden, projection), then
  z on the unit sphere; only used in training;
- temperature of the InfoNCE loss;
- the views the network is trained on (nn/views.py): `views` "rotzoom" (default: rotation by any
  angle, mirror, zoom by a factor in [1/zoom, zoom], a crop of `crop` times the frame side at a
  random place) or "d4" (rotations by 90 degrees, mirror and, in a periodic frame, a shift of the
  whole frame); `normalise` (default on): every field of every view minus its own mean, divided by
  its own standard deviation, so that h describes the shape of the pattern, not its amplitude.
The default shape (4 blocks, 32 to 256 channels, head 256 and 64, T = 0.2) is a small network in
the style of SimCLR. The default views were chosen on 2026-11-19: with "d4" and no normalisation the learned
description found 2 phases on Swift-Hohenberg and 1 on the other examples, because h followed the
amplitude, which changes smoothly inside a phase; with "rotzoom" and the normalisation it finds the
3 phases of Swift-Hohenberg for three training seeds (README, plan step 18).

The network sees input_side() points per side: the frame side with "d4"; with "rotzoom" the crop:
in a periodic frame round(crop x side), but at least the smallest input the network needs
(MIN_FINAL x 2^blocks) and at most the frame side (the field is tiled, so a turned crop never
leaves it); in a non-periodic frame at most side / (zoom sqrt 2), so that the turned and zoomed
crop stays inside the frame.

Constraints (check), each refused with a message:
- the last map has at least MIN_FINAL points per side: blocks <= log2(input side / MIN_FINAL);
- width <= dim_h (channels never decrease), projection <= dim_h and projection <= hidden;
- the values within the choices or ranges of the metadata (registry.check_settings).
summary() also gives the receptive field (points of the input that one value of the last map
depends on: 7, 19, 43, 91, ... for 1, 2, 3, 4 blocks), compared with the frame side, the number
of weights, and a rough memory estimate of one training step, shown as a warning only (it is not
calibrated on a device).

Choosing a network without labels: compare trainings by their label-free diagnostics (final loss
against the loss at collapse ln(2N - 1), alignment, uniformity, effective rank of h against
dim_h), never by scores against reference labels, which would bring the labels back into the
method (scripts/compare_networks.py).
"""

import math
from dataclasses import dataclass, field

MIN_FINAL = 2               # smallest side of the last map, in points (a choice: h is a mean over at least 2 x 2 values)
BYTES = 4                   # float32
ACTIVATIONS_PER_CONV = 3    # memory estimate: output, normalised, activated (a choice, rough)
VIEWS = ("rotzoom", "d4")   # kinds of views (nn/views.py)


@dataclass
class Network:
    blocks: int = field(default=4, metadata={"help": "convolution blocks; each halves the grid", "range": (1, 8)})
    width: int = field(default=32, metadata={"help": "channels of the first block (doubled at each block)",
                                             "choices": (8, 16, 32, 64, 128)})
    dim_h: int = field(default=256, metadata={"help": "dimension of h (channels of the last block)",
                                              "choices": (16, 32, 64, 128, 256, 512, 1024)})
    hidden: int = field(default=256, metadata={"help": "hidden layer of the head g (training only)",
                                               "choices": (32, 64, 128, 256, 512, 1024)})
    projection: int = field(default=64, metadata={"help": "dimension of z, where the loss is computed",
                                                  "choices": (8, 16, 32, 64, 128, 256)})
    temperature: float = field(default=0.2, metadata={"help": "temperature T of the InfoNCE loss",
                                                      "range": (0.01, 1.0)})
    views: str = field(default="rotzoom", metadata={
        "help": "views of a run: rotzoom (any angle, zoom, crop) or d4 (90 degrees, whole frame)", "choices": VIEWS})
    zoom: float = field(default=1.25, metadata={"help": "rotzoom: zoom factors in [1/zoom, zoom] (1: no zoom)",
                                                "range": (1.0, 2.0)})
    crop: float = field(default=0.75, metadata={"help": "rotzoom: crop side / frame side (periodic frame)",
                                                "range": (0.25, 1.0)})
    normalise: bool = field(default=True, metadata={
        "help": "every field of every view divided by its own standard deviation (h without the amplitude)"})

    @property
    def channels(self) -> tuple[int, ...]:
        """Channels of the blocks: width doubled at each block, at most dim_h; the last is dim_h."""
        return tuple(min(self.width * 2**k, self.dim_h) for k in range(self.blocks - 1)) + (self.dim_h,)


def view_side(views: str, zoom: float, crop: float, blocks: int, side: int, periodic: bool = True) -> int:
    """Points per side of what a network of `blocks` blocks sees for a frame of `side` points with
    these views (module docstring)."""
    if views == "d4":
        return side
    want = round(crop * side)
    if periodic:
        return max(1, min(side, max(want, MIN_FINAL * 2**blocks)))
    return max(1, min(want, math.floor(side / (zoom * math.sqrt(2)))))


def input_side(net: Network, side: int, periodic: bool = True) -> int:
    """Points per side of what the network sees for a frame of `side` points."""
    return view_side(net.views, net.zoom, net.crop, net.blocks, side, periodic)


def receptive_field(blocks: int) -> int:
    """Points of the input (per side) that one value of the last map depends on."""
    r, step = 1, 1
    for _ in range(blocks):
        r += 2 * step          # 3 x 3 convolution with stride 2
        step *= 2
        r += 2 * step          # 3 x 3 convolution with stride 1
    return r


def max_blocks(side: int) -> int:
    """The most blocks a frame of `side` points allows (last map of at least MIN_FINAL points)."""
    return max(0, math.floor(math.log2(side / MIN_FINAL)))


def weights(net: Network, in_channels: int) -> dict[str, int]:
    """Number of weights of f and g (convolutions with bias, batch normalisation, linear layers)."""
    f, c_in = 0, in_channels
    for c in net.channels:
        f += (9 * c_in + 1) * c + 2 * c          # strided convolution and its normalisation
        f += (9 * c + 1) * c + 2 * c             # second convolution and its normalisation
        c_in = c
    g = (net.dim_h + 1) * net.hidden + (net.hidden + 1) * net.projection
    return {"f": f, "g": g}


def memory_estimate(net: Network, side: int, in_channels: int, batch: int) -> float:
    """Rough bytes of one training step: activations of the 2 batch views (kept for the backward
    pass, twice for their gradients), weights with their gradients and the two AdamW moments."""
    per_view, s = side * side * in_channels, side
    for c in net.channels:
        s = math.ceil(s / 2)
        per_view += 2 * ACTIVATIONS_PER_CONV * c * s * s
    w = sum(weights(net, in_channels).values())
    return float(BYTES * (2 * 2 * batch * per_view + 4 * w))


def check(net: Network, side: int, periodic: bool = True) -> None:
    """ValueError listing every constraint the network breaks for a frame of `side` points."""
    from ..registry import check_settings
    check_settings(net)
    problems = []
    seen = input_side(net, side, periodic)
    what = "the frame has" if seen == side else f"the network sees {seen} of the frame's {side}"
    if net.blocks > max_blocks(seen):
        problems.append(f"{net.blocks} blocks need an input of at least {MIN_FINAL * 2**net.blocks} points per side "
                        f"({what}: at most {max_blocks(seen)} blocks)")
    if net.width > net.dim_h:
        problems.append(f"width {net.width} is larger than dim_h {net.dim_h} (channels may not decrease)")
    if net.projection > net.dim_h:
        problems.append(f"projection {net.projection} is larger than dim_h {net.dim_h}")
    if net.projection > net.hidden:
        problems.append(f"projection {net.projection} is larger than the hidden layer {net.hidden}")
    if problems:
        raise ValueError("network: " + "; ".join(problems))


def summary(net: Network, side: int, in_channels: int, batch: int = 128, periodic: bool = True) -> dict:
    """What a network means for a frame: channels, input side, last map, receptive field, weights,
    memory."""
    w = weights(net, in_channels)
    seen = input_side(net, side, periodic)
    return {"channels": net.channels, "input": seen, "last_map": math.ceil(seen / 2**net.blocks),
            "receptive_field": receptive_field(net.blocks), "frame": side, "dim_h": net.dim_h,
            "weights_f": w["f"], "weights_g": w["g"],
            "memory_gb_rough": memory_estimate(net, seen, in_channels, batch) / 1e9}
