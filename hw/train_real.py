"""Train the option-B event decoder on REAL SCAMP-5 episodes.

Input is the episodes.pt written by scamp_log.py: a list of dicts with
    image  (256,256) int8   exposure-matched ground-truth capture
    events (M,2)     int64  [frame, border_index] wrap events on the readout ring
    config dict             n_frames, couple, inset, ...

Differences from the sim trainer (sim/wave_events.py):
  * Data is finite (hundreds of episodes, not infinite generator) -> fixed
    train/val split, and the script always reports the mean-image baseline
    MSE so mode collapse is immediately visible (val mse ~ baseline = the
    decoder learned nothing scene-specific).
  * Episodes are LONG (4096 frames). A pixel can wrap ~170 times, so instead
    of the sim's first-K events the K kept per pixel are spread EVENLY over
    the pixel's whole event list - late wave arrivals carry most of the
    information at n=256. Features still go through the shared
    wave_events.feats_from_times.
  * The target is the ground truth downsampled to --n-out (default 64):
    the border code does not carry 256x256 of detail, and a smaller head
    resists overfitting the small dataset. Targets are per-image normalized
    (real captures are low-contrast); the normalization is stored in the
    checkpoint.

Usage:
    pixi run python hw/scamp_log.py episodes night.bin --out episodes.pt --swap-xy
    pixi run python hw/train_real.py train --data episodes.pt
    pixi run python hw/train_real.py eval  --data episodes.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
from wave_events import EventDecoder, feats_from_times  # noqa: E402

P = 1024  # border positions (4 x 256, sim order [N,S,W,E])


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_feats(events: torch.Tensor, frames: int, K: int) -> torch.Tensor:
    """(M,2) [t, border_index] -> (P, 3K) via the shared feature function.

    Pixels with more than K events keep K EVENLY SPACED ones (first-K would
    discard the late arrivals that carry the interior information)."""
    ev = events.numpy()
    t_k = np.full((P, K), float(frames + 1), np.float32)
    order = np.lexsort((ev[:, 0], ev[:, 1]))
    ev = ev[order]
    pix, start = np.unique(ev[:, 1], return_index=True)
    bounds = np.append(start, len(ev))
    for j, p in enumerate(pix):
        ts = ev[bounds[j]:bounds[j + 1], 0].astype(np.float32)
        if len(ts) > K:
            ts = ts[np.round(np.linspace(0, len(ts) - 1, K)).astype(int)]
        t_k[p, :len(ts)] = ts
    return feats_from_times(torch.from_numpy(t_k).unsqueeze(0), frames)[0]


def load_dataset(path: str, K: int, n_out: int):
    eps = torch.load(path, weights_only=False)
    if not eps:
        sys.exit(f"{path}: no episodes")
    feats, imgs, counts = [], [], []
    for ep in eps:
        frames = ep["config"]["n_frames"]
        feats.append(build_feats(ep["events"], frames, K))
        img = ep["image"].float().view(1, 1, 256, 256)
        imgs.append(F.avg_pool2d(img, 256 // n_out).view(n_out, n_out))
        counts.append(len(ep["events"]))
    X = torch.stack(feats)                       # (B, P, 3K)
    Y = torch.stack(imgs)                        # (B, n, n) raw units
    # per-image normalization: real captures are low-contrast and their
    # brightness depends on exposure; the decoder should learn structure
    mu = Y.mean(dim=(1, 2), keepdim=True)
    sd = Y.std(dim=(1, 2), keepdim=True) + 1e-4
    Yn = ((Y - mu) / (3 * sd)).clamp(-1, 1)
    print(f"{path}: {len(eps)} episodes, events/episode "
          f"{np.mean(counts):.0f} +- {np.std(counts):.0f}, frames={frames}")
    return X, Yn.unsqueeze(1), Y


def split(n: int, seed: int, val_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(2, int(n * val_frac))
    return idx[n_val:], idx[:n_val]


def make_figure(gt, recon, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(6, gt.shape[0])
    fig, axes = plt.subplots(3, k, figsize=(2.6 * k, 8), squeeze=False)
    for i in range(k):
        axes[0][i].imshow(gt[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[0][i].set_title("ground truth" if i == 0 else "")
        axes[1][i].imshow(recon[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1][i].set_title("reconstruction" if i == 0 else "")
        axes[2][i].imshow((recon[i, 0] - gt[i, 0]).abs().cpu(), cmap="magma", vmin=0, vmax=1)
        axes[2][i].set_title("|error|" if i == 0 else "")
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("REAL SCAMP-5: border wrap events -> reconstruction (val episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def train(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    X, Y, _ = load_dataset(args.data, args.k_events, args.n_out)
    tr, va = split(X.shape[0], args.seed)
    Xtr, Ytr = X[tr].to(device), Y[tr].to(device)
    Xva, Yva = X[va].to(device), Y[va].to(device)

    base = F.mse_loss(Ytr.mean(0, keepdim=True).expand_as(Yva), Yva).item()
    print(f"train {len(tr)} / val {len(va)}  |  mean-image baseline val mse {base:.4f} "
          f"(the number to beat)")

    dec = EventDecoder(args.n_out, args.k_events).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"device={device}  decoder params={sum(p.numel() for p in dec.parameters())/1e6:.2f}M")

    best, t0 = float("inf"), time.time()
    for step in range(1, args.steps + 1):
        i = torch.randint(0, len(tr), (min(args.batch, len(tr)),), device=device)
        recon = dec(Xtr[i])
        loss = F.mse_loss(recon, Ytr[i]) + 0.2 * F.l1_loss(recon, Ytr[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            dec.eval()
            with torch.no_grad():
                val = F.mse_loss(dec(Xva), Yva).item()
            dec.train()
            print(f"step {step:5d}/{args.steps}  train {loss.item():.4f}  val {val:.4f}  "
                  f"(baseline {base:.4f})  {(time.time()-t0)/step:.2f}s/step")
            if val < best:
                best = val
                torch.save({"model": dec.state_dict(),
                            "config": {"n_out": args.n_out, "k_events": args.k_events,
                                       "data": args.data}}, args.ckpt)
    print(f"best val mse {best:.4f} vs baseline {base:.4f}  "
          f"({'LEARNED scene structure' if best < 0.8 * base else 'NOT clearly better than mean image'})")
    evaluate(argparse.Namespace(data=args.data, ckpt=args.ckpt, fig=args.fig,
                                device=args.device, seed=args.seed))


def evaluate(args):
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    dec = EventDecoder(cfg["n_out"], cfg["k_events"]).to(device)
    dec.load_state_dict(ck["model"])
    dec.eval()
    X, Y, _ = load_dataset(args.data, cfg["k_events"], cfg["n_out"])
    _, va = split(X.shape[0], args.seed)
    with torch.no_grad():
        recon = dec(X[va].to(device))
    mse = F.mse_loss(recon, Y[va].to(device)).item()
    print(f"val mse: {mse:.4f}")
    make_figure(Y[va], recon, args.fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", default=str(HERE / "episodes.pt"))
    t.add_argument("--k-events", type=int, default=32, help="events kept per border pixel (evenly spaced)")
    t.add_argument("--n-out", type=int, default=64, help="reconstruction resolution (GT is downsampled to this)")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--ckpt", default=str(HERE / "wave_real.pt"))
    t.add_argument("--fig", default=str(HERE / "wave_real_recon.png"))
    t.set_defaults(fn=train)

    e = sub.add_parser("eval")
    e.add_argument("--data", default=str(HERE / "episodes.pt"))
    e.add_argument("--ckpt", default=str(HERE / "wave_real.pt"))
    e.add_argument("--fig", default=str(HERE / "wave_real_recon.png"))
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.set_defaults(fn=evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
