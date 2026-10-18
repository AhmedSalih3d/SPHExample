# Fixed-grid validation and timing

Measured on 2026-10-18 with Julia 1.12.7, CUDA.jl 6.4.2 and an NVIDIA
GeForce RTX 5080. These timings predate frozen-particle handling and do not
measure the current batch-level escape checks. Benchmark the current code
from the repository root:

```bash
julia --project=gpu_version -t 2,0 gpu_version/benchmark/fixed_grid.jl
```

The MovingSquare2D case has 33,020 particles, Float32 physics, Float64
positions, LaminarSPS viscosity, planar shifting and single-neighbor time
stepping. Both variants use compact buffers, deterministic sorting, cell
subdivision 1, CUDA step graphs and a maximum batch size of 32. Only
`fixed_grid` differs; all expansion sides are locked.

Each variant warms for five 0.01 s simulation intervals. Nine measured pairs
then advance five intervals each, alternating variant order. Wall times
include the complete solver intervals, natural cell-list rebuilds and timestep
readbacks. Setup, output and verification are excluded. After every pair,
positions, velocities, densities, pressures and accelerations sorted by ID
must match exactly, together with simulation time and work counts.

| Pair | Dynamic (ms) | Fixed (ms) | Dynamic / fixed |
| --- | ---: | ---: | ---: |
| 1 | 6.846 | 6.590 | 1.0389 |
| 2 | 6.737 | 6.888 | 0.9780 |
| 3 | 6.737 | 6.174 | 1.0911 |
| 4 | 6.363 | 6.675 | 0.9532 |
| 5 | 8.351 | 8.044 | 1.0381 |
| 6 | 8.691 | 8.684 | 1.0009 |
| 7 | 7.750 | 8.191 | 0.9462 |
| 8 | 7.772 | 8.327 | 0.9334 |
| 9 | 7.259 | 7.140 | 1.0167 |

The median paired ratio is **1.0009**, with a range of 0.9334–1.0911.
This run shows no clear elapsed speedup. Measured counts per variant match:
1,120 steps, 93 rebuilds, 182 timestep readbacks and zero dedicated grid
readbacks. No memory-saving claim is made: retaining empty cells can increase
dense-grid storage and scan work when the dynamic domain would shrink.

Focused GPU regressions cover initial bounds, locked-side failures before
reordering, every expansion direction in 2D/3D, retained bounds, simultaneous
growth, capacity limits, nonfinite positions, timestep-header consumption and
graph invalidation after growth. Complete 2D mDBC and 3D simulations compare
fixed/batched-graph runs against dynamic/direct runs for both time-stepping
schemes and Float64 or mixed Float32/Float64 precision. Grid-output regressions
also pass; the full package test suite was not completed.

## Frozen-particle follow-up

The simulation now freezes particles crossing locked limits at the next batch
check, before subsequent steps or output. Frozen particles retain position,
density and pressure, have zero velocity/acceleration and are excluded from
interactions. Floating bodies freeze as a whole. Events are logged once with
IDs, positions, time, step and crossed limits, including with `NoLog`.

Interaction exclusions activate only after the first freeze, at which point
cached CUDA graphs are invalidated. Runs without frozen particles retain the
original interaction path. Accepted-state derivatives and timestep reductions
are refreshed after freezing.

Validation on the same GPU passed:

- `test/frozen_grid.jl`: 279 checks, including small final-frame escapes,
  asynchronous output, numerical errors, frozen mDBC owners and floating bodies
  sharing a group marker with other particles.
- `test/cell_list_rebuild.jl`: 7,604 checks.
- `test/fixed_grid.jl`: initial bounds, directional growth and exact simulation comparisons.
- `test/run_floating.jl`: 90 floating-body checks plus five single-neighbor graph checks.
- `test/updated_mdbc.jl`, `test/logging.jl`, `test/stationary_predictors.jl`
  and `test/grid_export.jl`: all passed.
