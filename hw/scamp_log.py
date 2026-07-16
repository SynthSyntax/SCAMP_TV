"""Parse wave_logger binary dumps from the real SCAMP-5 into training episodes.

The chip program (src/scamp5_main.cpp) posts int32 packets which the host
logger (hw/host_logger) appends to a binary file as records:

    { int32 magic 'SC5L', int32 channel, int32 rows, int32 cols, payload }

Channels:
    42  analog edge scan   { frame_id, 256 x packed 4 uint8 }   (calibration)
    43  wrap events        { frame_id, count, count x ((x<<8)|y) }
                           frame_id -1 = episode header
                                    {-1, n_frames, base_freq, freq_gain, couple, 0}
                           frame_id -2 = end marker
    44  ground truth image { row_id, 64 x packed 4 uint8 } per row, posted at
                           BOTH episode start and end; episodes whose two
                           captures differ (scene moved mid-run) are dropped.

Orientation gotchas, all fixable here without re-recording:
  * scamp5_scan_areg scans right-to-left and returns analog+128: every GT row
    and every edge line is reversed and offset (undone by default,
    --no-flip-scan to disable).
  * scamp5_scan_events' (x,y) convention is UNVERIFIED on silicon. Record one
    episode with BOTH "edge readout" and "event readout" on, then run
    `calibrate`: it extracts wraps from the analog trace and searches the 16
    orientation combos (edge line reversal x event swap/flip) for the one
    where events and analog wraps coincide. Feed the winning flags to
    `episodes` via --swap-xy/--flip-x/--flip-y.

Usage:
    pixi run python hw/scamp_log.py summary  log.bin
    pixi run python hw/scamp_log.py calibrate log.bin
    pixi run python hw/scamp_log.py episodes log.bin --out episodes.pt \
        [--swap-xy] [--flip-x] [--flip-y] [--k-events 20] [--keep-bad]

`episodes` writes a torch file with a list of dicts
    {image (n,n) int8, events (M,2) int64 [t, border_index], config, feats}
whose `feats` come from the SAME function as the sim generator
(sim/wave_events.py feats_from_times), so sim-pretrained decoders load
directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x4C354353
CH_EDGE, CH_EVENT, CH_GT = 42, 43, 44
N = 256                      # chip array size
P = 4 * N                    # border positions, sim order [north, south, west, east]
HALF = 60                    # wrap threshold in register units (sim TURN/2)


# --------------------------------------------------------------------------- #
# raw records
# --------------------------------------------------------------------------- #
def read_records(path: str):
    data = np.fromfile(path, dtype="<i4")
    i = 0
    while i + 4 <= len(data):
        if data[i] != MAGIC:
            raise ValueError(f"bad magic at word {i} (0x{data[i] & 0xFFFFFFFF:08X}) "
                             f"- truncated or non-logger file?")
        ch, rows, cols = int(data[i + 1]), int(data[i + 2]), int(data[i + 3])
        n = rows * cols
        if i + 4 + n > len(data):
            print(f"warning: truncated final record (channel {ch}), dropped")
            return
        yield ch, data[i + 4:i + 4 + n]
        i += 4 + n


def unpack4(words: np.ndarray) -> np.ndarray:
    """int32 words packed MSB-first on the chip -> uint8 array, 4x longer."""
    w = words.astype(np.int64) & 0xFFFFFFFF
    out = np.empty(4 * len(w), np.uint8)
    out[0::4] = (w >> 24) & 0xFF
    out[1::4] = (w >> 16) & 0xFF
    out[2::4] = (w >> 8) & 0xFF
    out[3::4] = w & 0xFF
    return out


def to_signed(u: np.ndarray, flip_scan: bool, line: int = 256) -> np.ndarray:
    """Undo scan_areg: uint8 = analog+128, each `line`-long scan right-to-left."""
    s = u.astype(np.int16) - 128
    if flip_scan:
        s = s.reshape(-1, line)[:, ::-1].reshape(s.shape)
    return s


# --------------------------------------------------------------------------- #
# episode assembly
# --------------------------------------------------------------------------- #
def parse_episodes(path: str, flip_scan: bool = True):
    """Returns (episodes, n_bad). Each episode dict:
    config {n_frames, base_freq, freq_gain, couple},
    raw_events list of (frame, x, y),
    gt0 / gt1 (n,n) int16 signed images (orientation-corrected),
    edges (T, 1024) int16 or None, ok bool, why str."""
    episodes, bad = [], 0
    cur, gt_rows, gt_slot = None, None, None

    def finalize():
        nonlocal cur, bad
        if cur is None:
            return
        ep = cur
        cur = None
        if ep["gt1"] is None:
            ep["ok"], ep["why"] = False, "no end marker / end ground truth"
        elif len(ep["raw_events"]) == 0 and ep["n_ev_frames"] == 0:
            ep["ok"], ep["why"] = False, "no event packets"
        elif ep["n_ev_frames"] < ep["config"]["n_frames"]:
            ep["ok"], ep["why"] = False, (f"only {ep['n_ev_frames']}/"
                                          f"{ep['config']['n_frames']} frames")
        else:
            drift = float(np.abs(ep["gt0"].astype(np.float32)
                                 - ep["gt1"].astype(np.float32)).mean())
            ep["gt_drift"] = drift
            if drift > 6.0:
                ep["ok"], ep["why"] = False, f"scene changed (gt drift {drift:.1f})"
            else:
                ep["ok"], ep["why"] = True, ""
        if not ep["ok"]:
            bad += 1
        episodes.append(ep)

    for ch, flat in read_records(path):
        if ch == CH_EVENT:
            fid = int(flat[0])
            if fid == -1:                       # episode header
                finalize()
                cur = dict(config=dict(n_frames=int(flat[1]), base_freq=int(flat[2]),
                                       freq_gain=int(flat[3]), couple=int(flat[4])),
                           raw_events=[], n_ev_frames=0, gt0=None, gt1=None,
                           edges={}, ok=False, why="", gt_drift=float("nan"))
                gt_rows, gt_slot = np.zeros((N, N), np.uint8), "gt0"
            elif fid == -2:                     # end marker: end-GT follows
                if cur is not None:
                    gt_rows, gt_slot = np.zeros((N, N), np.uint8), "gt1"
            elif cur is not None and 0 <= fid < cur["config"]["n_frames"]:
                cnt = int(flat[1])
                for w in flat[2:2 + cnt]:
                    cur["raw_events"].append((fid, (int(w) >> 8) & 0xFF, int(w) & 0xFF))
                cur["n_ev_frames"] += 1
                if cur["gt1"] is not None:      # events after end = stray, new ep missing
                    cur["why"] = "events after end marker"
        elif ch == CH_GT and cur is not None and gt_slot is not None:
            row = int(flat[0])
            gt_rows[row] = unpack4(flat[1:65])
            if row == N - 1:
                img = to_signed(gt_rows.reshape(-1), flip_scan, N).reshape(N, N)
                cur[gt_slot] = img
                if gt_slot == "gt1":
                    finalize()
                gt_slot = None
        elif ch == CH_EDGE and cur is not None:
            fid = int(flat[0])
            cur["edges"][fid] = to_signed(unpack4(flat[1:257]), flip_scan, N)

    finalize()
    for ep in episodes:                          # dict of frames -> dense array
        if ep["edges"]:
            T = max(ep["edges"]) + 1
            dense = np.zeros((T, P), np.int16)
            for fid, tr in ep["edges"].items():
                dense[fid] = tr
            ep["edges"] = dense
        else:
            ep["edges"] = None
    return episodes, bad


# --------------------------------------------------------------------------- #
# event coordinates -> sim border index
# --------------------------------------------------------------------------- #
def map_coord(x: int, y: int, swap_xy: bool, flip_x: bool, flip_y: bool):
    if swap_xy:
        x, y = y, x
    if flip_x:
        x = N - 1 - x
    if flip_y:
        y = N - 1 - y
    return x, y   # interpreted as (row, col) after correction


def border_indices(r: int, c: int) -> list[int]:
    """Sim order: [north row, south row, west col, east col]; corners appear
    in BOTH lines they belong to, exactly as WaveEncoder._edges duplicates
    them. Interior coords return [] (event-mask failure upstream)."""
    idx = []
    if r == 0:
        idx.append(c)
    if r == N - 1:
        idx.append(N + c)
    if c == 0:
        idx.append(2 * N + r)
    if c == N - 1:
        idx.append(3 * N + r)
    return idx


def event_array(ep, swap_xy=False, flip_x=False, flip_y=False):
    """raw (frame,x,y) -> (M,2) int64 [t, border_index]; counts interior hits."""
    out, interior = [], 0
    for t, x, y in ep["raw_events"]:
        r, c = map_coord(x, y, swap_xy, flip_x, flip_y)
        idx = border_indices(r, c)
        if not idx:
            interior += 1
        out.extend((t, p) for p in idx)
    return np.array(out, np.int64).reshape(-1, 2), interior


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_summary(args):
    eps, bad = parse_episodes(args.log, args.flip_scan)
    print(f"{args.log}: {len(eps)} episodes ({bad} discarded)")
    for i, ep in enumerate(eps):
        c = ep["config"]
        ev = len(ep["raw_events"])
        status = "ok " if ep["ok"] else f"BAD ({ep['why']})"
        edge = f", analog edges {ep['edges'].shape[0]}f" if ep["edges"] is not None else ""
        print(f"  ep{i:3d} [{status}] {c['n_frames']}f couple={c['couple']} "
              f"gain={c['freq_gain']} base={c['base_freq']}  {ev} events "
              f"({ev / max(c['n_frames'], 1):.1f}/frame){edge} "
              f"gt drift {ep['gt_drift']:.2f}")


def cmd_calibrate(args):
    """Find scan_events' orientation from an episode recorded with BOTH
    readouts on: wraps in the analog border trace must coincide with events."""
    eps, _ = parse_episodes(args.log, args.flip_scan)
    eps = [e for e in eps if e["edges"] is not None and len(e["raw_events"])]
    if not eps:
        sys.exit("no episode has both analog edges and events - record one "
                 "with both GUI switches on")
    ep = eps[0]
    d = ep["edges"][1:].astype(np.int32) - ep["edges"][:-1].astype(np.int32)
    raster_a = d < -HALF                       # (T-1, P), wrap between t-1 and t
    T = raster_a.shape[0] + 1

    print(f"analog wraps: {raster_a.sum()}  events: {len(ep['raw_events'])}")
    best = []
    for swap in (False, True):
        for fx in (False, True):
            for fy in (False, True):
                ev, interior = event_array(ep, swap, fx, fy)
                raster_e = np.zeros_like(raster_a)
                for t, p in ev:
                    if 1 <= t < T:
                        raster_e[t - 1, p] = True
                for shift in (-1, 0, 1):
                    a = raster_a if shift == 0 else np.roll(raster_a, shift, axis=0)
                    inter = float(np.logical_and(a, raster_e).sum())
                    union = float(np.logical_or(a, raster_e).sum())
                    iou = inter / union if union else 0.0
                    best.append((iou, swap, fx, fy, shift, interior))
    best.sort(reverse=True)
    print("top orientation candidates (IoU of analog-wrap vs event rasters):")
    for iou, swap, fx, fy, shift, interior in best[:5]:
        print(f"  IoU {iou:.3f}  --swap-xy={swap} --flip-x={fx} --flip-y={fy} "
              f"(frame shift {shift:+d}, {interior} interior hits)")
    iou = best[0][0]
    print("=> confident match" if iou > 0.8 else
          "=> LOW confidence - check flip_scan / recording", f"(IoU {iou:.3f})")


def cmd_episodes(args):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
    from wave_events import feats_from_times

    eps, bad = parse_episodes(args.log, args.flip_scan)
    keep = [e for e in eps if e["ok"] or args.keep_bad]
    print(f"{len(eps)} episodes, {bad} bad, keeping {len(keep)}")
    out, K = [], args.k_events
    for ep in keep:
        frames = ep["config"]["n_frames"]
        ev, interior = event_array(ep, args.swap_xy, args.flip_x, args.flip_y)
        if interior:
            print(f"  warning: {interior} interior events (wrong orientation "
                  f"flags? run `calibrate`)")
        big = float(frames + 1)
        t_k = np.full((P, K), big, np.float32)
        for p in range(P):
            ts = np.sort(ev[ev[:, 1] == p, 0])[:K]
            t_k[p, :len(ts)] = ts
        feats = feats_from_times(torch.from_numpy(t_k).unsqueeze(0), frames)[0]
        out.append(dict(image=torch.from_numpy(ep["gt0"].astype(np.int8)),
                        events=torch.from_numpy(ev), feats=feats,
                        config=ep["config"]))
    torch.save(out, args.out)
    print(f"wrote {len(out)} episodes -> {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-flip-scan", dest="flip_scan", action="store_false",
                   help="do NOT reverse each scan line (default reverses: "
                        "scan_areg reads right-to-left)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="list episodes in a log")
    s.add_argument("log")
    s.set_defaults(fn=cmd_summary)

    c = sub.add_parser("calibrate", help="pin down scan_events orientation")
    c.add_argument("log")
    c.set_defaults(fn=cmd_calibrate)

    e = sub.add_parser("episodes", help="export decoder-ready episodes")
    e.add_argument("log")
    e.add_argument("--out", default="episodes.pt")
    e.add_argument("--k-events", type=int, default=20)
    e.add_argument("--swap-xy", action="store_true")
    e.add_argument("--flip-x", action="store_true")
    e.add_argument("--flip-y", action="store_true")
    e.add_argument("--keep-bad", action="store_true")
    e.set_defaults(fn=cmd_episodes)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
