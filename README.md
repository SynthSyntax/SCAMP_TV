# travelling_waves

Oscillator-lattice experiments for the **SCAMP-5** vision chip. Every pixel of the
256x256 processor array runs its own oscillator entirely on the sensor, coupled
only to its four nearest neighbours. The goal is emergent behaviour - travelling
waves, synchronisation, and oscillatory-correlation segmentation - computed in the
focal plane with no host in the loop.

## Platform

- **Hardware:** SCAMP-5 pixel-processor array (SIMD analog + digital per-pixel).
- **Toolchain:** MCUXpresso IDE (Eclipse-based), built as a standard SCAMP-5
  application against `scamp5.hpp`.
- Programs are written as `scamp5_kernel_begin() ... scamp5_kernel_end()` blocks
  (one SIMD instruction stream broadcast to every PE) plus host-side GUI setup.

## Layout

```
src/scamp5_main.cpp          the active program
src/MISC/MISC_FUNCS.hpp       helper kernels (e.g. 4-bit display via DNEWS)
scamp5_main.oscillator.cpp.bak   earlier wave/spring lattice, kept for reference
```

Only these source files are tracked; MCUXpresso project files (`.project`,
`.cproject`, `.settings/`) and build output are ignored.

## Experiments (branches)

Each branch is a self-contained version of `src/scamp5_main.cpp`.

- **`main` / `d-kuramoto`** - Kuramoto phase-oscillator lattice, fully analog.
  Each pixel stores only a phase `theta` (a rotating phasor of fixed length), so
  the oscillation can never decay. Phase advances by an intensity-set frequency
  each frame and is weakly pulled toward its neighbours' phase, so similar-
  brightness regions lock into synchrony. `d-kuramoto` carries the wrap-safe
  (circular) coupling fix.

- **`digital-firefly`** - pulse-coupled "firefly" oscillators using digital
  per-pixel storage. Each pixel charges to a threshold, fires, and nudges its
  neighbours' phase, driving them toward synchronised flashing.

- **`.bak` (wave/spring lattice)** - the original experiment: a position/velocity
  oscillator per pixel coupled like a mass-spring mesh, seeded at one pixel to
  launch a travelling wave. Digital-storage version.

## Running

Build and flash `src/scamp5_main.cpp` from MCUXpresso onto the SCAMP-5 host. The
program opens GUI displays and sliders (via `vs_gui_*`) and then loops forever:

1. capture light intensity,
2. set each pixel's oscillator frequency from that intensity,
3. advance the oscillator,
4. apply nearest-neighbour coupling (optional, slider-controlled),
5. keep the state in range and output the fields.

Displays show the phase/state field, the intensity map, and a live scope of one
probe pixel. Sliders tune base frequency, intensity-to-frequency gain, coupling
strength, and the probe location at runtime.

## Wave autoencoder (edge readout + learned decoder)

Reading the full 256x256 array off the chip is the expensive operation SCAMP is
built to avoid. The travelling waves offer a way around it: coupling propagates
interior phase structure outward, so the **border of the phase field, read over
time, is a temporal code for the whole image**. The lattice is the encoder -
free, analog, in the focal plane - and a small neural network is the decoder.

```
image -> [Kuramoto lattice on-chip]        (physics, no parameters)
      -> border phase, 4x256 px/frame      (1.6% of the array per read)
      -> [GRU over the frame sequence]     (integrates the wave history)
      -> [ConvTranspose CNN head]          (renders the image back)
```

**Chip side** (`src/scamp5_main.cpp`): the *edge readout* GUI switch scans the
four border lines of theta each frame (`scamp5_scan_areg`) and posts them on
raw channel 42: one `int32` frame id, then a `4x256 int8` array, rows =
[north, south, west, east]. Note `scamp5_scan_areg` returns the analog value
+128 as uint8, and scans rows right-to-left (column-major from top right) - the
host loader must undo both.

**Host side** (`sim/wave_autoencoder.py`): a batched torch twin of the lattice
generates unlimited (image, edge-trace) training pairs with random polygon
scenes, analog mismatch noise, and 8-bit readout quantisation; the decoder
(GRU + CNN, ~1.5M params) trains against it on CPU in minutes. Raw phase is a
sawtooth, so each edge sample is lifted to (cos, sin) before the RNN.

```sh
uv run sim/wave_autoencoder.py train            # train, checkpoint, figure
uv run sim/wave_autoencoder.py eval             # reconstruct fresh scenes
uv run sim/wave_autoencoder.py train --couple 1 --frames 256   # stronger waves, longer code
```

Knobs that shape the code: `--couple` (smaller = stronger coupling = waves
carry farther), `--frames` (how long the border listens), `--readout-every`
(temporal subsampling), `--boundary` (periodic matches the numpy twin;
replicate is closer to a physical border). Training on real chip logs instead
of the twin only requires replacing `WaveEncoder.encode` with a loader for the
channel-42 packets.

### Reading the output (Kuramoto version)

- **theta (phase)** - the oscillator field. Synchronised regions settle into a
  shared shade; the whole field cycles as the phasors rotate.
- **intensity** - the captured image, which also sets the per-pixel frequency map.
- **theta @ probe** - the probe pixel over time: a **sawtooth** (ramp up, snap
  down) is one phasor rotating. With coupling off, pixels free-run and each traces
  its own sawtooth; with coupling on, neighbours pull into step.
