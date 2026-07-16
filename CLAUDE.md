# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Travelling-wave experiments on the SCAMP-5 vision chip (256x256 pixel-processor array). Every pixel runs a Kuramoto phase oscillator in analog, on-sensor: phase advances at an intensity-set frequency and is pulled toward its 4 neighbours, so the image turns into travelling waves. The waves carry interior structure to the array border, so reading ONLY the border over time (1.6% of the array per read) is a temporal code for the whole image; a learned decoder inverts it. Everything currently runs against a simulated twin of the chip physics — no hardware in the loop. After the simulation tests, actual reconstructions with the SCAMP5 chip will follow.

## Commands

Python environment is managed by **pixi** (conda-forge `pytorch-gpu`, CUDA 13):

```sh
pixi run python sim/wave_autoencoder.py train    # train GRU baseline decoder + checkpoint + figure
pixi run python sim/wave_autoencoder.py eval     # figure from saved checkpoint, fresh scenes
pixi run python sim/wave_viz.py                  # wave-propagation diagnostic (PNG + GIF), no training
pixi run python sim/wave_timeseries.py train     # option A (d-kuramoto-timeseries branch)
pixi run python sim/wave_events.py train         # option B (d-kuramoto-events branch)
pixi run python hw/scamp_log.py summary  log.bin     # inspect a recorded chip log
pixi run python hw/scamp_log.py calibrate log.bin    # pin scan_events (x,y) orientation, once
pixi run python hw/scamp_log.py episodes log.bin --out episodes.pt --swap-xy ...  # decoder-ready episodes
```

`hw/host_logger/` is the C++ packet logger that produces `log.bin`; it builds against the scamp5d_interface library from the SCAMP-5 devkit (not present on this machine — see its Makefile), connects over USB or TCP, and appends every int32 data packet to the file.

All train/eval scripts accept `--device {auto,cpu,cuda}` (except the baseline, which auto-selects), `--n` (lattice size), `--frames`, `--couple`, `--steps`, `--batch`. Defaults are n=64 (fast); the real chip is n=256. Quick smoke test: `--n 32 --frames 96 --batch 4 --steps 2 --log-every 1 --device cpu`.

There are no tests or linters. The chip program (`src/scamp5_main.cpp`) is built and flashed from MCUXpresso IDE against `scamp5.hpp` — it cannot be built here.

The GPU is on a shared machine: check `nvidia-smi` before launching long runs.

## Architecture

Three parts that must stay in sync:

1. **Chip kernel** (`src/scamp5_main.cpp`) — the Kuramoto lattice in SCAMP-5 analog registers. A=theta, B=intensity (reused as wrap threshold), C-F scratch, NEWS clobbered by every analog macro. One frame = capture, omega = base + intensity/2^gain (kept > 0 so phase always ramps forward), theta += omega, optional wrapped-difference coupling, manual two-sided wrap. The "edge readout" GUI switch scans the 4 border lines of A each frame and posts them on raw channel 42: int32 frame id + 4x256 int8, rows [north, south, west, east]. `scamp5_scan_areg` returns analog value +128 as uint8 and scans right-to-left; a host loader must undo both. The "event readout" switch (option B) instead latches the wrap-step FLAG into DREG R11, masks it to the border (interior mask R10, rebuilt every frame — DREGs leak), and posts `scamp5_scan_events` coordinates on channel 43 (int32 frame id + int32 count + count x,y uint8 pairs; frame id −1 = episode header, −2 = end marker). "record episode" / "auto episodes" reset theta to 0 and stream one fixed-length episode, posting the captured intensity as ground truth on channel 44 (row id + 256 bytes per row) at BOTH episode start and end so the host can reject scenes that changed mid-run. Silicon facts (measured, not from docs): DREG ops do NOT honour FLAG — conditional DREG masking must be pure logic (`NOT`/`NOR`), a `CLR` under `WHERE` clears the whole array; `scamp5_scan_events` zero-fills every unused buffer slot, so counts are recovered by trimming trailing (0,0) pairs (sentinel prefills do not survive); this chip has at least one stuck DREG cell, interior, at scan-coords (184,25) — removed by the border mask.

2. **Torch twin + decoders** (`sim/`) — `wave_autoencoder.py` holds `WaveEncoder`, a batched no-grad torch reimplementation of the same dynamics (same units, order of operations, clipping), which generates unlimited (image, edge-trace) training pairs from random polygon scenes. All other sim scripts import the encoder/scenes/constants from it.

3. **Hardware data path** (`hw/`) — `host_logger/` (C++, devkit) dumps chip packets to a binary log; `scamp_log.py` parses it into episodes and builds decoder features through `wave_events.feats_from_times`, the single shared feature function, so sim-pretrained event decoders load onto real data unchanged. Episodes whose start/end ground-truth captures differ (scene moved mid-run) are auto-discarded; `calibrate` recovers the `scan_events` coordinate convention from one episode recorded with both readouts on (verified: recovers a planted convention at IoU 1.0 on a synthetic log).

**Shared physics constants** (must match the chip): one turn = 120 register units (`TURN`), theta in [-60, +60) (`HALF`), rails at ±127 (`RAIL`). Turn is 120, not 128, because constants at ±128 sit on the DAC limit and mis-load.

**Branches are experiments**, each adding a decoder for a different readout scheme:
- `main` / `d-kuramoto` — GRU baseline: border sampled every k-th frame, GRU over frame snapshots + ConvTranspose head (`wave_autoencoder.py`).
- `d-kuramoto-timeseries` — option A (`wave_timeseries.py`): full-rate border readout, temporal-first decoder (shared Conv1d along time per border pixel, then Conv1d along the border).
- `d-kuramoto-events` — option B (`wave_events.py`): no analog scan; each border pixel emits an event at its phase wrap, decoder consumes only (position, timestamp) spike features. ~4% of the analog scan volume.
- `digital-firefly` — unrelated older experiment (pulse-coupled oscillators, digital registers).

## Physics gotchas (these have burned time before)

- **`--couple 0` disables coupling entirely** (guard is `if couple > 0`); it does NOT mean strongest. Smaller values = stronger coupling; `--couple 1` is the strongest usable and the value that makes the whole scheme work. Defaults of newer scripts use 1; the baseline's default is 2, which at n=64/128 frames leaves border signal 3x BELOW the readout noise floor → decoder collapses to the dataset-mean blob.
- **Wave propagation is diffusive**: time for interior information to reach the border scales roughly with distance². Scaling n=64 → n=256 needs ~16x the frames (measured: signal crosses the noise floor at ~frame 75 for n=64/couple=1; use ~4096 frames at n=256), not 4x.
- **`base_freq` adds a uniform rotation and carries zero information** — raising it does not speed up information transfer, and large omega risks rail-clipping in `clamp(theta + omega)` before the wrap. The information levers are coupling strength and frequency *contrast* (`--freq-gain`, lower = stronger).
- **`--readout-every 8` aliases the bright-pixel oscillation** (period ~6.4 frames at defaults). Fine for the slow wavefront envelope, but full-rate (`1`) is the lossless choice — the lattice is discrete-time, so nothing exists between frames.
- Checkpoints save **only when val loss improves**; a checkpoint timestamp older than the end of training is normal (best-so-far semantics).
- `sim/wave_viz.py` visualizes the injected signal (theta minus a blank-scene run with identical noise) and reports when the border signal crosses the readout noise floor — run it before training whenever physics parameters change; it is much cheaper than a failed training run.

## Workflow

- Never run `git commit` (or `git push`) automatically. if feels like the code should be committed, instruct the user to do so. Stage/inspect diffs if useful, then tell the user what to run and let them commit it themselves.

## Repo conventions

- Training outputs (`sim/*.pt`, `sim/*.png`, `sim/*.gif`) are gitignored on purpose — every train/eval overwrites them, and tracked figures previously blocked branch switching. Copy keeper figures into a tracked `docs/` folder instead. Never force-add weights: a 275 MB checkpoint once had to be rewritten out of history (GitHub's 100 MB limit).
- `.gitignore` does not end predictably — append with a real editor, not `>>` (a missing trailing newline once glued two patterns into one broken line).
