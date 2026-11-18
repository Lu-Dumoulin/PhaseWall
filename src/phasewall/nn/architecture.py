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
- temperature of the InfoNCE loss.
The default (4 blocks, 32 to 256 channels, head 256 and 64, T = 0.2) is the network of hfml Step 7.

Constraints (check), each refused with a message:
- the last map has at least MIN_FINAL points per side: blocks <= log2(frame side / MIN_FINAL);
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

    @property
    def channels(self) -> tuple[int, ...]:
        """Channels of the blocks: width doubled at each block, at most dim_h; the last is dim_h."""
        return tuple(min(self.width * 2**k, self.dim_h) for k in range(self.blocks - 1)) + (self.dim_h,)


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


def check(net: Network, side: int) -> None:
    """ValueError listing every constraint the network breaks for a frame of `side` points."""
    from ..registry import check_settings
    check_settings(net)
    problems = []
    if net.blocks > max_blocks(side):
        problems.append(f"{net.blocks} blocks need a frame of at least {MIN_FINAL * 2**net.blocks} points per side "
                        f"(the frame has {side}: at most {max_blocks(side)} blocks)")
    if net.width > net.dim_h:
        problems.append(f"width {net.width} is larger than dim_h {net.dim_h} (channels may not decrease)")
    if net.projection > net.dim_h:
        problems.append(f"projection {net.projection} is larger than dim_h {net.dim_h}")
    if net.projection > net.hidden:
        problems.append(f"projection {net.projection} is larger than the hidden layer {net.hidden}")
    if problems:
        raise ValueError("network: " + "; ".join(problems))


def summary(net: Network, side: int, in_channels: int, batch: int = 128) -> dict:
    """What a network means for a frame: channels, last map, receptive field, weights, memory."""
    w = weights(net, in_channels)
    return {"channels": net.channels, "last_map": math.ceil(side / 2**net.blocks),
            "receptive_field": receptive_field(net.blocks), "frame": side, "dim_h": net.dim_h,
            "weights_f": w["f"], "weights_g": w["g"],
            "memory_gb_rough": memory_estimate(net, side, in_channels, batch) / 1e9}
