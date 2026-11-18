"""Training of the encoder f (with the head g), and the description h of every pair.

Each step: a batch of N runs, two views per run (2N inputs), z = g(f(views)) on the unit
sphere, InfoNCE loss, back-propagation through g and f, AdamW update. The learning rate grows
linearly over the first `warmup` epochs, then decays to 0 as a cosine. Every epoch the mean
loss, alignment and uniformity of the batches, and the effective rank of h on all pairs, are
logged (they do not stop the training, which always runs the given number of epochs).

After training, f is frozen and g dropped: h_table gives one row per pair with the columns
run, time, the parameters and cnn.h.d000, cnn.h.d001, ... (the format of describe.py, so that
walls.find_walls reads it unchanged; the 256 columns form one block).
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..fields import Field
from ..io import Ensemble
from .architecture import Network
from .data import DRAWS, FrameSet, PairViews
from .loss import T, alignment, effective_rank, info_nce, uniformity
from .model import CHANNELS, HIDDEN, OUT, Encoder, ProjectionHead, count_weights

PREFIX = "cnn.h"     # columns of the learned description: cnn.h.d000, ...


@dataclass
class Settings:
    """Settings of a training (choices; the values of hfml Step 7 except epochs and draws,
    which are scaled to small ensembles)."""

    epochs: int = 40
    batch: int = 128          # runs per batch (2 views each)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: int = 3           # epochs of linear warm-up
    temperature: float = T
    draws: int = DRAWS
    channels: tuple = CHANNELS
    seed: int = 0
    hidden: int = HIDDEN      # head g: hidden layer
    projection: int = OUT     # head g: dimension of z

    @classmethod
    def from_network(cls, network: Network, **kw) -> "Settings":
        """Training settings for a network of architecture.py (channels, head, temperature)."""
        return cls(channels=network.channels, hidden=network.hidden, projection=network.projection,
                   temperature=network.temperature, **kw)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    """Factor of the learning rate at a step: linear warm-up, then cosine decay to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def embed(encoder: Encoder, frames: FrameSet, dev: torch.device | None = None, batch: int = 256) -> np.ndarray:
    """h of every pair (rows, dim), encoder in evaluation mode, no augmentation."""
    dev = dev or next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    out = [encoder(torch.from_numpy(frames.x[i:i + batch]).to(dev)).cpu()
           for i in range(0, len(frames.x), batch)]
    encoder.train(was_training)
    return torch.cat(out).numpy()


def h_table(frames: FrameSet, h: np.ndarray, ens: Ensemble) -> pd.DataFrame:
    """One row per pair: run, time, the parameters, then cnn.h.d000, ..."""
    cols = {f"{PREFIX}.d{j:03d}": h[:, j] for j in range(h.shape[1])}
    t = pd.DataFrame({"run": frames.run, "time": frames.time} | cols)
    t = ens.runs[["run", *ens.parameters]].merge(t, on="run")
    return t[["run", "time", *ens.parameters, *cols]]


def new_encoder(frames: FrameSet, settings: Settings) -> Encoder:
    torch.manual_seed(settings.seed)
    return Encoder(frames.in_channels, tuple(settings.channels), frames.periodic)


def train(frames: FrameSet, settings: Settings | None = None, dev: torch.device | str | None = None,
          verbose: bool = True, progress=None) -> tuple[Encoder, pd.DataFrame]:
    """Train f and g on the pairs of an ensemble; returns f (on the CPU) and the log.
    dev: "cpu", "mps", "cuda" or a torch.device (default: the fastest available).
    progress: optional callback progress("train", epoch, epochs) after each epoch."""
    dev = torch.device(dev) if isinstance(dev, str) else (dev or device())
    s = settings or Settings()
    np.random.seed(s.seed)
    encoder = new_encoder(frames, s).to(dev)
    head = ProjectionHead(encoder.dim, s.hidden, s.projection).to(dev)
    dataset = PairViews(frames, draws=s.draws, seed=s.seed)
    batch = min(s.batch, len(dataset.rows))
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(s.seed))
    opt = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=s.lr, weight_decay=s.weight_decay)
    steps = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: schedule(k, s.warmup * steps, s.epochs * steps))
    log = []
    for epoch in range(s.epochs):
        dataset.epoch = epoch
        t0, losses, aligns, unifs = time.perf_counter(), [], [], []
        for v1, v2 in loader:
            z = head(encoder(torch.cat([v1, v2]).to(dev)))
            loss = info_nce(z, s.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            aligns.append(alignment(z.detach()))
            unifs.append(uniformity(z.detach()))
        rank = effective_rank(torch.from_numpy(embed(encoder, frames, dev)))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "alignment": float(np.mean(aligns)),
               "uniformity": float(np.mean(unifs)), "effective_rank_h": rank,
               "collapse_loss": math.log(2 * batch - 1), "seconds": time.perf_counter() - t0}
        log.append(row)
        if progress:
            progress("train", epoch + 1, s.epochs)
        if verbose:
            print(f"epoch {epoch + 1}/{s.epochs}: loss {row['loss']:.3f} (collapse {row['collapse_loss']:.2f}), "
                  f"alignment {row['alignment']:.3f}, uniformity {row['uniformity']:.2f}, "
                  f"rank of h {rank:.1f}, {row['seconds']:.1f} s", flush=True)
    return encoder.cpu(), pd.DataFrame(log)


# ---------------------------------------------------------------- saving
def save(out: str | Path, encoder: Encoder, frames: FrameSet, settings: Settings, log: pd.DataFrame) -> Path:
    """encoder.pt, settings.json (training settings, fields, scales, periodic) and log.csv."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out / "encoder.pt")
    log.to_csv(out / "log.csv", index=False)
    meta = {"settings": asdict(settings), "in_channels": frames.in_channels, "periodic": frames.periodic,
            "scale": frames.scale.tolist(), "fields": [[f.name, list(f.keys), f.kind, f.relative, list(f.components)] for f in frames.fields],
            "weights_f": count_weights(encoder), "torch": torch.__version__}
    (out / "settings.json").write_text(json.dumps(meta, indent=1))
    return out


def load(out: str | Path) -> tuple[Encoder, dict]:
    """The encoder saved by save() (on the CPU, evaluation mode) and its settings.json."""
    out = Path(out)
    meta = json.loads((out / "settings.json").read_text())
    enc = Encoder(meta["in_channels"], tuple(meta["settings"]["channels"]), meta["periodic"])
    enc.load_state_dict(torch.load(out / "encoder.pt", map_location="cpu", weights_only=True))
    # fields saved before components existed have four entries
    meta["fields"] = [Field(f[0], tuple(f[1]), f[2], f[3], tuple(f[4]) if len(f) > 4 else ()) for f in meta["fields"]]
    return enc.eval(), meta
