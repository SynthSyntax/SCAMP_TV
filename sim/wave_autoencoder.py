# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "matplotlib"]
# ///
#
# torch comes from PyPI's default (CUDA) wheel so it runs on the GPU. The
# previous header pinned the CPU-only index (download.pytorch.org/whl/cpu),
# which silently forced device=cpu. On an RTX 5090 (Blackwell / sm_120) you
# need a CUDA >= 12.8 build; the current PyPI torch (cu130) covers it.
"""
Wave autoencoder: reconstruct an image from ONLY the border of the Kuramoto
lattice, read over time.

The premise
-----------
Reading the full 256x256 array off a SCAMP-5 is the expensive operation the
chip is designed to avoid. But the phase-oscillator lattice (src/scamp5_main.cpp)
turns the image into travelling waves: every pixel's frequency is set by its
intensity, and nearest-neighbour coupling propagates phase structure outward.
So the 4 border lines of the phase field, sampled every frame, are a TEMPORAL
code for the SPATIAL interior - the lattice itself is the encoder, for free,
in analog. This script trains the matching decoder:

    image --> [Kuramoto lattice (physics, frozen)] --> edge phase (T, 4N)
          --> [GRU over time] --> [ConvTranspose CNN head] --> image_hat

Per frame the readout is 4N of N^2 pixels (1.6% at N=256, 6% at N=64); the
compression story over a whole sequence depends on how few frames the decoder
can get away with (--frames, --readout-every).

The encoder here is a torch twin of the same dynamics as sim/kuramoto_sim.py
(same units, same order of operations, same clipping), batched so it can
generate training data on the fly - infinitely many (image, edge-trace) pairs,
no dataset on disk. The chip posts raw phase; the decoder lifts each sample
onto the unit circle as (cos, sin) before the RNN, because the sawtooth wrap
at +/-60 is an artifact of the representation, not of the signal.

Usage:
    uv run sim/wave_autoencoder.py train                 # train + checkpoint + figure
    uv run sim/wave_autoencoder.py train --steps 4000 --couple 2
    uv run sim/wave_autoencoder.py eval                  # figure from a saved checkpoint
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TURN = 120.0  # one full turn of phase, in register units
HALF = TURN / 2.0  # theta lives in [-60, +60)
RAIL = 127.0  # analog registers saturate here

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# scenes (stands in for get_image)
# --------------------------------------------------------------------------- #
def random_scene(n: int, rng: np.random.Generator, max_shapes: int = 2) -> np.ndarray:
    """1..max_shapes random filled polygons, bright (+110) on dark (-110).

    Same half-plane rasteriser as kuramoto_sim.polygon_image, with random side
    count, size, rotation and position, so the decoder can never memorise a
    fixed silhouette - it has to read the waves.
    """
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    img = np.full((n, n), -110.0)
    for _ in range(rng.integers(1, max_shapes + 1)):
        sides = int(rng.integers(3, 9))
        r = float(rng.uniform(0.12, 0.32)) * n
        rot = float(rng.uniform(0, 2 * np.pi))
        cy = (0.5 + float(rng.uniform(-0.22, 0.22))) * (n - 1)
        cx = (0.5 + float(rng.uniform(-0.22, 0.22))) * (n - 1)
        inside = np.ones((n, n), dtype=bool)
        for k in range(sides):
            a = rot + 2.0 * np.pi * k / sides
            inside &= (np.cos(a) * (xx - cx) + np.sin(a) * (yy - cy)) <= r * np.cos(np.pi / sides)
        img[inside] = 110.0
    return img


def scene_batch(batch: int, n: int, rng: np.random.Generator, max_shapes: int, device) -> torch.Tensor:
    imgs = np.stack([random_scene(n, rng, max_shapes) for _ in range(batch)])
    return torch.from_numpy(imgs).float().to(device)


# --------------------------------------------------------------------------- #
# the encoder: the lattice itself (physics, no learned parameters)
# --------------------------------------------------------------------------- #
class WaveEncoder:
    """Batched torch twin of the SCAMP kernel; one step() == one chip frame.

    Boundary handling for movx is selectable: 'periodic' matches the numpy twin
    (np.roll); 'replicate' is closer to a physical border where the edge PE has
    no neighbour to pull on it. The decoder learns whichever is used - what
    matters is that training matches deployment.
    """

    def __init__(self, base_freq=4, freq_gain=4, couple=2, noise=0.02,
                 readout_noise=1.0, boundary="periodic", seed=0, device="cpu"):
        self.base_freq, self.freq_gain, self.couple = base_freq, freq_gain, couple
        self.noise, self.readout_noise, self.boundary = noise, readout_noise, boundary
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    def _neighbours(self, a: torch.Tensor):
        if self.boundary == "periodic":
            return (torch.roll(a, 1, 1), torch.roll(a, -1, 1),
                    torch.roll(a, -1, 2), torch.roll(a, 1, 2))
        # replicate: the border pixel sees itself in the missing direction
        p = F.pad(a.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        return (p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, 2:], p[:, 1:-1, :-2])

    @staticmethod
    def _wrap(d: torch.Tensor) -> torch.Tensor:
        return (d + HALF) % TURN - HALF

    def _edges(self, theta: torch.Tensor) -> torch.Tensor:
        """The only thing that ever leaves the chip: [north, south, west, east].

        Quantised to whole register units, as scamp5_scan_areg's uint8 does,
        plus optional readout noise for the analog scan.
        """
        e = torch.cat([theta[:, 0, :], theta[:, -1, :], theta[:, :, 0], theta[:, :, -1]], dim=1)
        if self.readout_noise > 0:
            e = e + self.readout_noise * torch.randn(e.shape, generator=self.gen, device=e.device)
        return torch.round(torch.clamp(e, -RAIL, RAIL))

    @torch.no_grad()
    def encode(self, images: torch.Tensor, frames: int, readout_every: int = 1,
               keep_theta: bool = False):
        """images (B,N,N) in signed units -> edge traces (B,T,4N), T = frames//readout_every."""
        B, n, _ = images.shape
        omega = images / 2.0 ** self.freq_gain + (128 >> self.freq_gain) + self.base_freq
        omega = torch.clamp(omega, -RAIL, RAIL)
        if self.noise > 0:  # per-pixel analog mismatch, fixed for the run
            omega = omega * (1 + self.noise * torch.randn(omega.shape, generator=self.gen, device=self.device))

        theta = torch.zeros_like(images)  # res(A)
        traces, snaps = [], []
        for f in range(frames):
            theta = torch.clamp(theta + omega, -RAIL, RAIL)
            if self.couple > 0:
                acc = torch.zeros_like(theta)
                for nb in self._neighbours(theta):
                    acc = acc + self._wrap(nb - theta) / 4.0  # diva twice == /4
                theta = torch.clamp(theta + acc / 2.0 ** self.couple, -RAIL, RAIL)
            if self.noise > 0:
                theta = theta + self.noise * torch.randn(theta.shape, generator=self.gen, device=self.device)
            theta = self._wrap(theta)
            if (f + 1) % readout_every == 0:
                traces.append(self._edges(theta))
                if keep_theta:
                    snaps.append(theta.clone())
        edges = torch.stack(traces, dim=1)  # (B, T, 4N)
        return (edges, torch.stack(snaps, 1)) if keep_theta else edges


# --------------------------------------------------------------------------- #
# the decoder: GRU over the edge movie, CNN head back up to an image
# --------------------------------------------------------------------------- #
class WaveDecoder(nn.Module):
    """(B, T, 4N) raw edge phase -> (B, 1, N, N) image in [-1, 1].

    Phase is circular, so each sample is lifted to (cos, sin) - 8N inputs per
    frame - then a GRU integrates the movie into a single state vector, which a
    ConvTranspose stack (8x upsampling) renders back into the image. The GRU is
    the part that "plays with the temporal information": its final hidden state
    has seen every wavefront that reached the border.
    """

    def __init__(self, n: int, embed: int = 256, hidden: int = 256, layers: int = 2):
        super().__init__()
        assert n % 8 == 0, "n must be divisible by 8 (three 2x upsampling stages)"
        self.n = n
        self.seed_hw = n // 8
        self.embed = nn.Sequential(nn.Linear(8 * n, embed), nn.ReLU())
        self.rnn = nn.GRU(embed, hidden, num_layers=layers, batch_first=True)
        # the early synchronisation transient carries most of the interior
        # information; the mean over ALL timesteps preserves it, instead of
        # asking the last hidden state to remember 100+ frames back
        self.to_seed = nn.Linear(2 * hidden, 128 * self.seed_hw ** 2)

        def up(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                nn.GroupNorm(8, cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(8, cout), nn.ReLU(),
            )

        self.head = nn.Sequential(up(128, 64), up(64, 32), up(32, 16),
                                  nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        ang = edges * (2 * torch.pi / TURN)  # register units -> radians
        x = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
        out, h = self.rnn(self.embed(x))
        z = self.to_seed(torch.cat([out.mean(dim=1), h[-1]], dim=-1))
        return torch.tanh(self.head(z.view(-1, 128, self.seed_hw, self.seed_hw)))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def make_figure(images, edges, recon, path: str):
    """input | what left the chip (edge space-time) | reconstruction | error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(4, images.shape[0])
    fig, axes = plt.subplots(k, 4, figsize=(13, 3.1 * k), squeeze=False)
    for i in range(k):
        ax = axes[i][0]
        ax.imshow(images[i].cpu(), cmap="gray", vmin=-128, vmax=127)
        ax.set_ylabel(f"sample {i}")
        ax.set_title("input (never read out)" if i == 0 else "")
        ax = axes[i][1]
        ax.imshow(edges[i].cpu().T, cmap="twilight", vmin=-HALF, vmax=HALF, aspect="auto")
        ax.set_title("edge phase over time (the only readout)" if i == 0 else "")
        if i == k - 1:
            ax.set_xlabel("frame")
        ax.set_yticks([])
        ax = axes[i][2]
        ax.imshow(recon[i, 0].cpu() * 128, cmap="gray", vmin=-128, vmax=127)
        ax.set_title("reconstruction" if i == 0 else "")
        ax = axes[i][3]
        ax.imshow((recon[i, 0].cpu() * 128 - images[i].cpu()).abs(), cmap="magma", vmin=0, vmax=128)
        ax.set_title("|error|" if i == 0 else "")
    for ax in axes.flat:
        ax.set_xticks([]) if ax != axes[k - 1][1] else None
        ax.set_yticks([]) if ax not in [row[1] for row in axes] else None
    fig.suptitle("Kuramoto wave autoencoder: image -> travelling waves -> border readout -> RNN+CNN decoder")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    enc = WaveEncoder(args.base_freq, args.freq_gain, args.couple, args.noise,
                      args.readout_noise, args.boundary, args.seed, device)
    dec = WaveDecoder(args.n).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"device={device}  decoder params={n_params/1e6:.2f}M  "
          f"n={args.n} frames={args.frames} couple={args.couple}")
    print(f"readout per frame: {4*args.n} of {args.n**2} px "
          f"({400*args.n/args.n**2:.1f}% of the array)")

    # a fixed held-out batch: same physics, scenes the decoder never trains on
    val_rng = np.random.default_rng(args.seed + 999)
    val_imgs = scene_batch(8, args.n, val_rng, args.max_shapes, device)
    val_edges = enc.encode(val_imgs, args.frames, args.readout_every)

    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        imgs = scene_batch(args.batch, args.n, rng, args.max_shapes, device)
        edges = enc.encode(imgs, args.frames, args.readout_every)
        recon = dec(edges)
        target = (imgs / 128.0).unsqueeze(1)
        loss = F.mse_loss(recon, target) + 0.2 * F.l1_loss(recon, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == args.steps:
            dec.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(dec(val_edges), (val_imgs / 128.0).unsqueeze(1)).item()
            dec.train()
            print(f"step {step:5d}/{args.steps}  train {loss.item():.4f}  "
                  f"val mse {val_loss:.4f}  {(time.time()-t0)/step:.2f}s/step")
            if val_loss < best:
                best = val_loss
                torch.save({"model": dec.state_dict(),
                            "config": {k: getattr(args, k) for k in
                                       ("n", "frames", "readout_every", "base_freq", "freq_gain",
                                        "couple", "noise", "readout_noise", "boundary", "max_shapes")}},
                           args.ckpt)

    print(f"best val mse {best:.4f}  checkpoint: {args.ckpt}")
    evaluate(argparse.Namespace(ckpt=args.ckpt, fig=args.fig, seed=args.seed + 999))


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    dec = WaveDecoder(cfg["n"]).to(device)
    dec.load_state_dict(ck["model"])
    dec.eval()

    enc = WaveEncoder(cfg["base_freq"], cfg["freq_gain"], cfg["couple"], cfg["noise"],
                      cfg["readout_noise"], cfg["boundary"], args.seed, device)
    rng = np.random.default_rng(args.seed)
    imgs = scene_batch(4, cfg["n"], rng, cfg["max_shapes"], device)
    edges = enc.encode(imgs, cfg["frames"], cfg["readout_every"])
    with torch.no_grad():
        recon = dec(edges)
    mse = F.mse_loss(recon, (imgs / 128.0).unsqueeze(1)).item()
    print(f"eval mse (fresh scenes): {mse:.4f}")
    make_figure(imgs, edges, recon, args.fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the decoder against the lattice physics")
    t.add_argument("--n", type=int, default=64, help="lattice size (SCAMP is 256; 64 trains fast on CPU)")
    t.add_argument("--frames", type=int, default=128, help="chip frames simulated per scene")
    t.add_argument("--readout-every", type=int, default=1, help="read the border every k-th frame")
    t.add_argument("--base-freq", type=int, default=4)
    t.add_argument("--freq-gain", type=int, default=4)
    t.add_argument("--couple", type=int, default=2, help="coupling halvings; smaller = stronger waves")
    t.add_argument("--noise", type=float, default=0.02, help="analog mismatch/step noise")
    t.add_argument("--readout-noise", type=float, default=1.0, help="scan noise, register units")
    t.add_argument("--boundary", choices=["periodic", "replicate"], default="periodic")
    t.add_argument("--max-shapes", type=int, default=2)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--ckpt", default=str(HERE / "wave_decoder.pt"))
    t.add_argument("--fig", default=str(HERE / "wave_recon.png"))
    t.set_defaults(fn=train)

    e = sub.add_parser("eval", help="reconstruct fresh scenes with a saved checkpoint")
    e.add_argument("--ckpt", default=str(HERE / "wave_decoder.pt"))
    e.add_argument("--fig", default=str(HERE / "wave_recon.png"))
    e.add_argument("--seed", type=int, default=7)
    e.set_defaults(fn=evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
