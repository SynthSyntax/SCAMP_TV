"""
Visualize wave propagation in the Kuramoto lattice BEFORE any readout.

Raw phase is dominated by the global oscillation (every pixel spinning at the
background rate), so the shape's contribution is invisible to the naked eye.
The fix: run the identical encoder (same seed -> same noise realization) on
the scene AND on a blank background, and plot the wrapped phase difference.
That difference is precisely the information the shape injected into the
lattice - where it is nonzero, the travelling wave has arrived. When the
colored region reaches the border, the edge readout carries signal.

Outputs:
    <out>.png  - snapshot grid (raw phase | injected signal) at log-spaced
                 frames, plus border-signal-vs-time curve against the
                 readout noise floor
    <out>.gif  - the same two panels animated over every frame

Usage:
    pixi run python sim/wave_viz.py                        # training defaults
    pixi run python sim/wave_viz.py --couple 0 --frames 256 --out sim/wave_prop_strong
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wave_autoencoder import HALF, TURN, WaveEncoder, random_scene

HERE = Path(__file__).resolve().parent


def wrap(d: torch.Tensor) -> torch.Tensor:
    return (d + HALF) % TURN - HALF


def simulate(img: np.ndarray, args) -> torch.Tensor:
    """(T, n, n) phase snapshots; fresh encoder each call so the same seed
    gives the identical noise realization for scene and blank."""
    enc = WaveEncoder(args.base_freq, args.freq_gain, args.couple, args.noise,
                      args.readout_noise, args.boundary, args.seed, "cpu")
    t = torch.from_numpy(img).float()[None]
    _, snaps = enc.encode(t, args.frames, 1, keep_theta=True)
    return snaps[0]


def border_mean_abs(sig: torch.Tensor) -> np.ndarray:
    """Mean |signal| over the 4 border lines, per frame."""
    b = torch.cat([sig[:, 0, :], sig[:, -1, :], sig[:, :, 0], sig[:, :, -1]], dim=1)
    return b.abs().mean(dim=1).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--frames", type=int, default=128)
    p.add_argument("--base-freq", type=int, default=4)
    p.add_argument("--freq-gain", type=int, default=4)
    p.add_argument("--couple", type=int, default=2)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--readout-noise", type=float, default=1.0)
    p.add_argument("--boundary", choices=["periodic", "replicate"], default="periodic")
    p.add_argument("--max-shapes", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=str(HERE / "wave_prop"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    img = random_scene(args.n, rng, args.max_shapes)
    blank = np.full((args.n, args.n), -110.0)

    theta = simulate(img, args)            # (T, n, n)
    theta_blank = simulate(blank, args)
    signal = wrap(theta - theta_blank)     # what the shape injected

    curve = border_mean_abs(signal)
    T = signal.shape[0]

    # ---- snapshot grid ---------------------------------------------------- #
    picks = sorted(set(np.unique(np.geomspace(1, T, 6).astype(int)) - 1))
    rows = len(picks)
    fig, axes = plt.subplots(rows, 3, figsize=(9.5, 3.0 * rows), squeeze=False)
    for r, f in enumerate(picks):
        ax = axes[r][0]
        ax.imshow(img, cmap="gray", vmin=-128, vmax=127)
        ax.set_ylabel(f"frame {f + 1}")
        ax.set_title("input" if r == 0 else "")
        ax = axes[r][1]
        ax.imshow(theta[f], cmap="twilight", vmin=-HALF, vmax=HALF)
        ax.set_title("raw phase (what stripes look like)" if r == 0 else "")
        ax = axes[r][2]
        ax.imshow(signal[f], cmap="coolwarm", vmin=-10, vmax=10)
        ax.set_title("injected signal: theta - theta_blank" if r == 0 else "")
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout(rect=(0, 0.14, 1, 1))

    # border signal vs time, against the readout noise floor
    ax = fig.add_axes((0.10, 0.035, 0.85, 0.09))
    ax.plot(np.arange(1, T + 1), curve, lw=1.5, label="mean |signal| on border")
    if args.readout_noise > 0:
        ax.axhline(args.readout_noise * np.sqrt(2 / np.pi), color="r", ls="--", lw=1,
                   label=f"readout noise floor ({args.readout_noise:g} units)")
    ax.set_xlabel("frame"), ax.set_ylabel("register units")
    ax.legend(fontsize=8, loc="upper left")

    png = args.out + ".png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"border signal: frame 1 = {curve[0]:.2f}, peak = {curve.max():.2f} "
          f"(at frame {curve.argmax() + 1}), last = {curve[-1]:.2f} register units")

    # ---- animation -------------------------------------------------------- #
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.2))
    im1 = a1.imshow(theta[0], cmap="twilight", vmin=-HALF, vmax=HALF)
    a1.set_title("raw phase")
    im2 = a2.imshow(signal[0], cmap="coolwarm", vmin=-10, vmax=10)
    a2.set_title("injected signal")
    for ax in (a1, a2):
        ax.set_xticks([]), ax.set_yticks([])
    sup = fig.suptitle("frame 1")
    fig.tight_layout()

    def update(f):
        im1.set_data(theta[f])
        im2.set_data(signal[f])
        sup.set_text(f"frame {f + 1}/{T}")
        return im1, im2, sup

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    gif = args.out + ".gif"
    anim.save(gif, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
