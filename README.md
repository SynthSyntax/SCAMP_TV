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

### Reading the output (Kuramoto version)

- **theta (phase)** - the oscillator field. Synchronised regions settle into a
  shared shade; the whole field cycles as the phasors rotate.
- **intensity** - the captured image, which also sets the per-pixel frequency map.
- **theta @ probe** - the probe pixel over time: a **sawtooth** (ramp up, snap
  down) is one phasor rotating. With coupling off, pixels free-run and each traces
  its own sawtooth; with coupling on, neighbours pull into step.
