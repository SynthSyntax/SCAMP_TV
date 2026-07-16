"""
Option B: event-based border readout - spike-time coding.

No analog scan at all. Each border pixel emits an EVENT at the frame its
phase wraps (theta crosses the +60 -> -60 discontinuity). On SCAMP-5 the wrap
is a threshold flip into a DREG and the sparse address-event readout streams
the flipped coordinates off-chip; the host stamps each with the frame counter.
What leaves the chip is a list of (border_index, t) pairs - orders of
magnitude fewer bits than scanning 4N analog values per frame.

The physics guarantee: a pixel with intrinsic frequency omega wraps every
TURN/omega frames, and a passing wavefront advances or retards its next wrap,
so ALL of the phase information lives in wrap timing. The decoder therefore
consumes ONLY spike times: per border pixel, the timestamps of its first K
wraps plus inter-spike intervals (masked where a pixel produced fewer than K
events), embedded by a shared MLP, fused along the border by a Conv1d stack,
rendered by the same ConvTranspose head as the other decoders.

    image -> [Kuramoto lattice] -> wrap events (border_index, t)
          -> per-pixel [t_1..t_K, isi_1..isi_K, mask] -> shared MLP -> (4N, F)
          -> Conv1d along the border -> seed -> ConvTranspose head -> image_hat

Usage:
    pixi run python sim/wave_events.py train
    pixi run python sim/wave_events.py train --steps 4000 --device cuda
    pixi run python sim/wave_events.py eval
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_autoencoder import HALF, WaveEncoder, scene_batch

HERE = Path(__file__).resolve().parent

ISI_SCALE = 32.0  # ~ the background wrap period, for feature normalisation


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def feats_from_times(t_k: torch.Tensor, frames: int) -> torch.Tensor:
    """(B, P, K) first-K wrap times -> (B, P, 3K) decoder features.

    Absent events carry a sentinel > frames. This is the ONE place the
    spike-timing features are computed: the sim generator below and the
    real-chip log loader (hw/scamp_log.py) both call it, so the decoder
    cannot tell where its events came from.
    """
    big = float(frames + 1)
    mask = (t_k < big).float()
    isi = torch.diff(t_k, dim=2, prepend=torch.zeros_like(t_k[:, :, :1]))
    t_k = t_k * mask   # zero out sentinel entries
    isi = isi * mask
    return torch.cat([t_k / frames, isi / ISI_SCALE, mask], dim=2)


@torch.no_grad()
def encode_events(enc: WaveEncoder, images: torch.Tensor, frames: int, K: int):
    """Run the lattice, detect border wrap events, build timing features.

    Returns
        feats  (B, 4N, 3K)  per pixel: [t_k/frames, isi_k/ISI_SCALE, mask_k]
        spikes (B, T-1, 4N) boolean raster (for figures/stats only)

    The wrap detector needs the full-rate border trace internally, but that
    trace never "leaves the chip" - on hardware the comparison happens in-pixel
    and only the events are read out.
    """
    edges = enc.encode(images, frames, readout_every=1)  # (B, T, 4N)
    d = edges[:, 1:] - edges[:, :-1]
    # a wrap shows as a fall of ~a full turn; noise and coupling perturb by a
    # few units at most, so -HALF is a safe threshold
    spikes = d < -HALF
    B, Tm1, P = spikes.shape

    big = float(frames + 1)  # sentinel sorted past every real timestamp
    t_idx = torch.arange(1, frames, device=edges.device, dtype=torch.float32)
    t_all = torch.where(spikes, t_idx.view(1, -1, 1).expand(B, Tm1, P),
                        torch.full_like(d, big))
    t_sorted, _ = torch.sort(t_all, dim=1)
    t_k = t_sorted[:, :K, :].transpose(1, 2)  # (B, P, K) first K wrap times
    return feats_from_times(t_k, frames), spikes


def up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout), nn.ReLU(),
    )


class EventDecoder(nn.Module):
    """Per-pixel spike-timing features (B, 4N, 3K) -> (B, 1, N, N) in [-1, 1].

    There is no dense time axis anymore, so nothing recurrent: a shared MLP
    reads each pixel's timing vector (the K wrap times + intervals ARE the
    signal), then a Conv1d stack mixes along the border. Border order is
    [N row, S row, W col, E col].
    """

    def __init__(self, n: int, K: int = 20, pix_feat: int = 128, border_bins: int = 8):
        super().__init__()
        assert n % 8 == 0, "n must be divisible by 8 (three 2x upsampling stages)"
        self.n = n
        self.seed_hw = n // 8
        self.pix = nn.Sequential(
            nn.Linear(3 * K, 128), nn.ReLU(),
            nn.Linear(128, pix_feat), nn.ReLU(),
        )
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

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        f = self.pix(feats).transpose(1, 2)  # (B, F, 4N)
        z = self.to_seed(self.benc(f).flatten(1))
        return torch.tanh(self.head(z.view(B, 128, self.seed_hw, self.seed_hw)))


# --------------------------------------------------------------------------- #
def make_figure(images, spikes, recon, path: str):
    """input | event raster (the only readout) | reconstruction | error."""
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
        ax.imshow(spikes[i].cpu().T, cmap="gray_r", aspect="auto", interpolation="none")
        ax.set_title("wrap events (the only readout)" if i == 0 else "")
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
    fig.suptitle("Kuramoto wave autoencoder: image -> travelling waves -> border WRAP EVENTS -> spike-timing decoder")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
def train(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    enc = WaveEncoder(args.base_freq, args.freq_gain, args.couple, args.noise,
                      args.readout_noise, args.boundary, args.seed, device)
    dec = EventDecoder(args.n, args.k_events).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"device={device}  decoder params={n_params/1e6:.2f}M  "
          f"n={args.n} frames={args.frames} couple={args.couple} K={args.k_events}")

    val_rng = np.random.default_rng(args.seed + 999)
    val_imgs = scene_batch(8, args.n, val_rng, args.max_shapes, device)
    val_feats, val_spikes = encode_events(enc, val_imgs, args.frames, args.k_events)

    ev = val_spikes.float().sum(dim=(1, 2)).mean().item()
    dense = args.frames * 4 * args.n
    print(f"events/sample: {ev:.0f} (vs {dense} border samples at full analog "
          f"rate -> {100 * ev / dense:.1f}%); events/pixel: {ev / (4 * args.n):.1f}")

    best = float("inf")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        imgs = scene_batch(args.batch, args.n, rng, args.max_shapes, device)
        feats, _ = encode_events(enc, imgs, args.frames, args.k_events)
        recon = dec(feats)
        target = (imgs / 128.0).unsqueeze(1)
        loss = F.mse_loss(recon, target) + 0.2 * F.l1_loss(recon, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == args.steps:
            dec.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(dec(val_feats), (val_imgs / 128.0).unsqueeze(1)).item()
            dec.train()
            print(f"step {step:5d}/{args.steps}  train {loss.item():.4f}  "
                  f"val mse {val_loss:.4f}  {(time.time()-t0)/step:.2f}s/step")
            if val_loss < best:
                best = val_loss
                torch.save({"model": dec.state_dict(),
                            "config": {k: getattr(args, k) for k in
                                       ("n", "frames", "k_events", "base_freq", "freq_gain",
                                        "couple", "noise", "readout_noise", "boundary",
                                        "max_shapes")}},
                           args.ckpt)

    print(f"best val mse {best:.4f}  checkpoint: {args.ckpt}")
    evaluate(argparse.Namespace(ckpt=args.ckpt, fig=args.fig, seed=args.seed + 999,
                                device=args.device))


def evaluate(args):
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    dec = EventDecoder(cfg["n"], cfg["k_events"]).to(device)
    dec.load_state_dict(ck["model"])
    dec.eval()

    enc = WaveEncoder(cfg["base_freq"], cfg["freq_gain"], cfg["couple"], cfg["noise"],
                      cfg["readout_noise"], cfg["boundary"], args.seed, device)
    rng = np.random.default_rng(args.seed)
    imgs = scene_batch(4, cfg["n"], rng, cfg["max_shapes"], device)
    feats, spikes = encode_events(enc, imgs, cfg["frames"], cfg["k_events"])
    with torch.no_grad():
        recon = dec(feats)
    mse = F.mse_loss(recon, (imgs / 128.0).unsqueeze(1)).item()
    print(f"eval mse (fresh scenes): {mse:.4f}")
    make_figure(imgs, spikes, recon, args.fig)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the spike-timing decoder")
    t.add_argument("--n", type=int, default=64)
    t.add_argument("--frames", type=int, default=256)
    t.add_argument("--k-events", type=int, default=20, help="wrap events kept per border pixel")
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
    t.add_argument("--ckpt", default=str(HERE / "wave_ev_decoder.pt"))
    t.add_argument("--fig", default=str(HERE / "wave_ev_recon.png"))
    t.set_defaults(fn=train)

    e = sub.add_parser("eval", help="reconstruct fresh scenes with a saved checkpoint")
    e.add_argument("--ckpt", default=str(HERE / "wave_ev_decoder.pt"))
    e.add_argument("--fig", default=str(HERE / "wave_ev_recon.png"))
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.set_defaults(fn=evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()