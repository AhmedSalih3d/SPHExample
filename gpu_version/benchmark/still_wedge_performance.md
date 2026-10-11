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

## Historical scheme comparison (2026-10-11)

An earlier attempt changed the example to single-neighbor integration.
That default has been reverted: `run_still_wedge()` again uses the original
symplectic scheme. The following comparison is retained as evidence of a
different integrator, **not** an equivalent-workload optimization. No
approximate-math or experimental fused kernels were enabled.

On an RTX A1000 Laptop GPU with Julia 1.12.7, five alternating warmed runs of
the exact four-second generated example measured about 0.796 s symplectic and
0.521 s single-neighbor, or 1.52x throughput. This is approximately 34% less
time, not half the runtime. The measurements include input generation and
cleanup, upload, graphs, rebuild handling, log writing, all particle/grid
frames, and closing. Loading Julia and the package, ParaView and the log
viewer are excluded.

Both schemes took 8,500 steps and wrote 401 particle frames and 400 nonempty
grid frames. They are not numerically
identical: maximum final differences in position, velocity, density and
pressure were approximately 0.021 m, 0.0094 m/s, 0.29 kg/m^3 and 521 Pa.
The single-neighbor final time was about 4.000348 s versus 4.000393 s
symplectic.

From `gpu_version/`, reproduce the complete comparison and explicitly check
the 1.5x throughput target:

```powershell
julia --project=. -t 4,0 benchmark/still_wedge_speed.jl --compare-schemes --assert-1.5x-throughput
```

The per-run measurements are saved in `still_wedge_scheme_results.csv`.
GPU integration and generated-example tests cover the symplectic default and the
explicit single-neighbor alternative.

Output cleanup now attempts normal deletion before collecting unreachable file
handles and retrying a blocked deletion. It no longer forces a full GC for
an empty folder or already-closed outputs.

## Fair same-workload attempt (2026-10-11)

The original symplectic scheme, all constants, 3,027 particles, Float32 physics,
Float64 positions, four-second duration, boundary-force evaluation and output
schedule were held fixed. No reduced-resolution, approximate-math or
single-neighbor result was accepted.

Independent prototypes measured on the RTX A1000:

| Candidate | Median paired total-time ratio to unchanged solver | Validation |
| --- | ---: | --- |
| Buffered per-lane neighbors, inlined full-traversal fallback | 1.045 | Exact final state and all saved datasets |
| Buffered neighbors, separate fast/fallback kernels | 1.060 | Exact final state and all saved datasets |
| Cell-cooperative shared-memory tiles | 1.230 | Exact final state and all saved datasets |
| Compile-time specialization of invariant constants | 1.002 | Exact final state and all saved datasets |
| Specialization plus read-only input loads | 0.999 | Exact final state and all saved datasets |

A full-warp arithmetic prototype was also rejected in a shorter screen: it
was slower and changed rounding. These prototypes remain session experiments,
not installed solver changes. They do not establish that all implementations
of these algorithms would be slower.

The final comparison uses the actual original example, cleanup, interaction
kernels and driver from `2167212` in isolated modules, alongside the current
example. Unchanged geometry/physics helpers and dependencies are shared. Each
variant is warmed; five rounds alternate the order of original, identical-original
control and current. Timing includes preparation, cleanup, upload, simulation,
log writing, all particle/grid frames and closing; package loading, warmup,
output hashing and external viewers are excluded.

| Variant | Median total seconds |
| --- | ---: |
| Original symplectic example | 1.004407 |
| Current symplectic example | 0.808445 |

The median paired current/original ratio was **0.8050**, or **19.50% less time**;
the identical-original control ratio was 0.9999. This is the benefit of the
retained cleanup change from the earlier attempt, not a newly faster pair kernel.
All 8,500 steps, final physical state and every exported dataset matched exactly:
401 particle frames and 400 nonempty grid frames.

**The 50% less runtime target was not met.** The assertion below requires
current/original <= 0.5 and does not reinterpret that target as 1.5x throughput:

```powershell
julia --project=. -t 4,0 benchmark/still_wedge_same_work.jl
julia --project=. -t 4,0 benchmark/still_wedge_same_work.jl --assert-half-runtime
```

Raw paired results: [still_wedge_same_work_results.csv](still_wedge_same_work_results.csv).
