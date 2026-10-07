# StillWedge measurements on this PC

Measured on 2026-10-07 with Julia 1.12.7, CUDA.jl 6.4.2, and an NVIDIA
GeForce RTX 5080. The case has 3,027 generated particles and advances 4 seconds
of physical time in 8,500 symplectic steps. Positions remain Float64; force,
velocity, and density calculations remain Float32. Both particle and cell-grid
outputs are written every 0.01 seconds.

| Measurement | Before | After |
| --- | ---: | ---: |
| Fresh process, including package load and geometry setup | 13.070 s | 7.487 s |
| First simulation call in that process | 5.078 s | 0.507 s |
| Warm simulation call, median of five runs | 0.399 s | 0.346 s |
| Warm stepping loop, median of five runs | 0.305 s | 0.282 s |
| Output queue budget | 256 MB | 8 MB |

These timings use `-t 8,0`. With Julia's default `1,1` threads the final warm
median was 0.343 s, and the stepping loop was 0.283 s. Timing varies with GPU
clocks, disk activity, and garbage collection. The one-time package cache
rebuild is excluded from fresh-process measurements.

Changes include cached CUDA graphs of up to eight steps, fused reciprocal-density
and cell-position preparation for DBC, precompilation of the StillWedge double
position configuration, and the smaller example output queue. Importing the
geometry generator no longer writes files or repeats particle sampling.

Final particle positions, velocities, densities, elapsed physical time, and
step counts matched the saved original four-second run exactly in all twelve
final runs across both thread configurations. Tuning alternative lane counts,
block sizes, and cell subdivision did not justify changing the automatic
settings.

Validation:

- All 72 new integration and saved-frame regression checks passed, covering
  DBC, mDBC, moving walls, both timestep schemes, and both position precisions.
- All 15,317 checks in the remaining GPU solver testset passed.
- Measurement, example generation, ParaView, and StillWedge geometry tests passed.
- The full GPU suite still stops at three existing middle-square geometry export
  failures: 22 rather than 23 cells, different connectivity offsets, and square
  area 0.1 rather than 0.2. These also occurred before the optimization.
- CPU-reference comparisons were skipped by the existing test harness because
  this repository's CPU package lacks the required metadata mode API.

Commands run from `gpu_version`:

```powershell
julia -t 8,0 --project=. benchmark/still_wedge_speed.jl
julia -t 8,0 --project=. test/runtests.jl
```

Temporary Julia harnesses additionally recorded and compared the complete
four-second particle state, measured fresh-process and warm runs, swept output
queue budgets, and ran the main GPU testset separately after the existing
middle-square failures stopped the full suite.
