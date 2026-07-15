"""
Option A: full-rate border readout, decoded as 4N parallel waveforms.

wave_autoencoder.py samples the border every k-th frame and lets a GRU treat
each snapshot as a token: spatial-first, temporal-second. Here the border is
read EVERY frame (readout_every=1) - for the discrete-time lattice that is
the complete, lossless time series; theta does not evolve between kernel
frames, so there is nothing "in between" left to capture - and the decoder is
temporal-first: each of the 4N border pixels is a (cos, sin) waveform of
length T, encoded by a Conv1d stack along TIME shared across pixels, pooled
to coarse time bins (so wavefront arrival time survives), then fused along
the BORDER by a second Conv1d stack. The ConvTranspose head is the same as
the GRU model's.

    image -> [Kuramoto lattice] -> border phase (T, 4N), T = frames
          -> shared Conv1d over time, per pixel  -> (4N, F)
          -> Conv1d along the border -> seed -> ConvTranspose head -> image_hat

Usage:
    pixi run python sim/wave_timeseries.py train
    pixi run python sim/wave_timeseries.py train --steps 4000 --device cuda
    pixi run python sim/wave_timeseries.py eval
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_autoencoder import TURN, WaveEncoder, make_figure, scene_batch

HERE = Path(__file__).resolve().parent


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout), nn.ReLU(),
    )


class TemporalDecoder(nn.Module):
    """(B, T, 4N) full-rate edge phase -> (B, 1, N, N) image in [-1, 1].

    Temporal-first: the model never sees a "frame" as a unit. Each border
    pixel's waveform is embedded independently (weights shared across pixels,
    like a matched filter bank for oscillation patterns), and only then do
    features mix spatially along the border. Border order is [N row, S row,
    W col, E col] - not a contiguous ring, but locality within each side is
    what the border convs exploit.
    """

    def __init__(self, n: int, t_feat: int = 96, pix_feat: int = 128,
                 time_bins: int = 8, border_bins: int = 8):
        super().__init__()
        assert n % 8 == 0, "n must be divisible by 8 (three 2x upsampling stages)"
        self.n = n
        self.seed_hw = n // 8
        # per-pixel waveform encoder, stride 8 overall then pooled to
        # time_bins coarse bins: arrival time survives at T/time_bins
        # granularity, oscillation shape is captured by the filters
        self.tenc = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, t_feat, 5, stride=2, padding=2), nn.GroupNorm(8, t_feat), nn.ReLU(),
            nn.AdaptiveAvgPool1d(time_bins),
        )
        self.pix = nn.Sequential(nn.Linear(t_feat * time_bins, pix_feat), nn.ReLU())
        self.benc = nn.Sequential(
            nn.Conv1d(pix_feat, pix_feat, 5, stride=2, padding=2),
            nn.GroupNorm(8, pix_feat), nn.ReLU(),
            nn.Conv1d(pix_feat, pix_feat, 5, stride=2, padding=2),
            nn.GroupNorm(8, pix_feat), nn.ReLU(),
            nn.AdaptiveAvgPool1d(border_bins),
        )
        self.to_seed = nn.Linear(pix_feat * border_bins, 128 * self.seed_hw ** 2)
        self.head = nn.Sequential(up(128, 64), up(64, 32), up(32, 16),
                                  nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        B, T, P = edges.shape
        ang = edges * (2 * torch.pi / TURN)  # register units -> radians
        w = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)  # (B, T, 2, P)
        w = w.permute(0, 3, 2, 1).reshape(B * P, 2, T)  # one waveform per pixel
        f = self.tenc(w).flatten(1)                     # (B*P, t_feat*time_bins)
        f = self.pix(f).view(B, P, -1).transpose(1, 2)  # (B, F, P)
        z = self.to_seed(self.benc(f).flatten(1))
        return torch.tanh(self.head(z.view(B, 128, self.seed_hw, self.seed_hw)))


# --------------------------------------------------------------------------- #
def train(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    enc = WaveEncoder(args.base_freq, args.freq_gain, args.couple, args.noise,
                      args.readout_noise, args.boundary, args.seed, device)
    dec = TemporalDecoder(args.n).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"device={device}  decoder params={n_params/1e6:.2f}M  "
          f"n={args.n} frames={args.frames} couple={args.couple}  readout_every=1 (full rate)")

    val_rng = np.random.default_rng(args.seed + 999)
    val_imgs = scene_batch(8, args.n, val_rng, args.max_shapes, device)
    val_edges = enc.encode(val_imgs, args.frames, 1)

    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        imgs = scene_batch(args.batch, args.n, rng, args.max_shapes, device)
        edges = enc.encode(imgs, args.frames, 1)
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
                                       ("n", "frames", "base_freq", "freq_gain", "couple",
                                        "noise", "readout_noise", "boundary", "max_shapes")}},
                           args.ckpt)

    print(f"best val mse {best:.4f}  checkpoint: {args.ckpt}")
    evaluate(argparse.Namespace(ckpt=args.ckpt, fig=args.fig, seed=args.seed + 999,
                                device=args.device))


def evaluate(args):
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    dec = TemporalDecoder(cfg["n"]).to(device)
    dec.load_state_dict(ck["model"])
    dec.eval()

    enc = WaveEncoder(cfg["base_freq"], cfg["freq_gain"], cfg["couple"], cfg["noise"],
                      cfg["readout_noise"], cfg["boundary"], args.seed, device)
    rng = np.random.default_rng(args.seed)
    imgs = scene_batch(4, cfg["n"], rng, cfg["max_shapes"], device)
    edges = enc.encode(imgs, cfg["frames"], 1)
    with torch.no_grad():
        recon = dec(edges)
    mse = F.mse_loss(recon, (imgs / 128.0).unsqueeze(1)).item()
    print(f"eval mse (fresh scenes): {mse:.4f}")
    make_figure(imgs, edges, recon, args.fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the temporal-first decoder")
    t.add_argument("--n", type=int, default=64)
    t.add_argument("--frames", type=int, default=256)
    t.add_argument("--base-freq", type=int, default=4)
    t.add_argument("--freq-gain", type=int, default=4)
    t.add_argument("--couple", type=int, default=1, help="coupling halvings; 1 = strongest (0 disables!)")
    t.add_argument("--noise", type=float, default=0.02)
    t.add_argument("--readout-noise", type=float, default=1.0)
    t.add_argument("--boundary", choices=["periodic", "replicate"], default="periodic")
    t.add_argument("--max-shapes", type=int, default=2)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--ckpt", default=str(HERE / "wave_ts_decoder.pt"))
    t.add_argument("--fig", default=str(HERE / "wave_ts_recon.png"))
    t.set_defaults(fn=train)

    e = sub.add_parser("eval", help="reconstruct fresh scenes with a saved checkpoint")
    e.add_argument("--ckpt", default=str(HERE / "wave_ts_decoder.pt"))
    e.add_argument("--fig", default=str(HERE / "wave_ts_recon.png"))
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.set_defaults(fn=evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()