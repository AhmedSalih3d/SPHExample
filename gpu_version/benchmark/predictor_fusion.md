# Gather/predictor fusion

`GPUFusePredictor` combines the first symplectic interaction gather and the
half-step update on the stationary, nonperiodic DBC path. The normal driver
enables it by default with interaction fast math. It retains separate kernels
for strict interaction arithmetic, single-neighbor stepping, mDBC, periodic
sources, moving boundaries, floating bodies, external forces, rotations, open
boundaries and variable resolution. The resolved setting is exposed through
`result.configuration.effective.fuse_predictor`.

The integration helper is shared with the separate half-step kernel. Inside
the gather it inherits the interaction compiler options, so fusion can change
rounding and accumulated trajectories. The gather inputs remain immutable:
predictor pressure/reciprocal caches and mixed-precision cell-relative positions
have separate destinations. Reorder scratch still supplies dense predictor
positions and velocities when compact buffers are enabled. The second gather,
final integration/reduction and commit retain their parallel launch layouts.

With cached Float32 physics and Float64 positions in 2D, compact support adds
20 live device bytes per particle: two Float32 cache entries and one 12-byte
`PosCell`. This is 60,540 bytes for 3,027 particles, separate from allocator
reservations and total solver memory. Uncached runs add only the needed
cell-relative buffer; full-buffer runs already have separate cell-relative
storage. Low-level constructors retain their previous allocation defaults.

## Reproduce

From the repository root:

```powershell
julia --project=gpu_version -t 4,0 gpu_version/test/predictor_fusion.jl
julia --project=gpu_version -t 4,0 gpu_version/benchmark/predictor_fusion.jl --rounds=9 --duration=4
```

The benchmark compares the installed solver with `GPUFusePredictor = false`
and `true`. Each variant is warmed for the complete duration, then variant
order alternates across rounds. Timings include geometry generation, host
allocation, upload, the whole solver, logging, particle and occupied-cell
output, and closing. Package loading, warmup, a collection/synchronization
before each measurement, output extraction and validation are excluded.
Optional PDF reports and external viewers are outside this workload.
The benchmark is unprofiled and requires matching accepted-step, rebuild and
loop-state readback counts. Fewer launches alone are not its speed evidence.

All saved frames are sorted by stable particle ID. The checks require finite
positions, velocities and pressure; positive finite density; matching particle
identities/types/groups and frame shapes; and unchanged fixed-wall positions
with zero velocity. The default position limit is a maximum Euclidean
difference of 5 mm across every saved particle and frame. Velocity, density,
pressure and frame-time differences are reported separately without silently
assigning them the position tolerance. Use `--position-mm=N` to change that
limit, `--cache=false` for uncached fields, `--dx=N` for spacing, and
`--output=path` for the CSV destination. This validates the selected workload;
it does not establish a universal trajectory-error bound.

`--validate-only` compares the two warmed trajectories and reports work counts
without timing rounds. It permits different rebuild/readback counts, which
disqualify a paired speed comparison but can follow from small numerical changes.

## Measurements

Measured on 2026-11-11 with Julia 1.12.7, `-t 4,0`, an NVIDIA GeForce RTX 5080,
Windows WDDM driver 616.92, and source based on `075800d`. This was a desktop
session. Each timing experiment ran after this task's other test/benchmark
processes had exited. Results
are specific to this workload and GPU; they do not establish speed or an error
bound for every supported configuration or longer trajectories.

Both four-second cases use 3,027 generated particles, `dx = 0.02`, Float32
physics, Float64 positions, compact buffers, radial interactions, symplectic
integration, `R = 1`, and particle plus occupied-cell output every 0.01 seconds.
Both variants take 8,499 steps, three rebuilds and 404 readbacks. The baseline
and fused variants differ only in `GPUFusePredictor` within each comparison.

| Interaction fields | Baseline median total | Fused median total | Median paired ratio | Paired elapsed reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cached | 0.277639 s | 0.265256 s | 0.95706 | 4.29% | 9/9 |
| Uncached | 0.269505 s | 0.259568 s | 0.96758 | 3.24% | 9/9 |

Individual ratios range from 0.94484 to 0.98257 (cached), and 0.90601 to
0.97347 (uncached). Compare variants within each paired experiment; the two
rows were collected in separate sessions. Raw timings and saved-frame metrics
are in [cached results](predictor_fusion_results.csv) and
[uncached results](predictor_fusion_uncached_results.csv).

| Difference across all saved particles/frames | Cached | Uncached |
| --- | ---: | ---: |
| Maximum Euclidean position | 1.76032 mm | 0.86239 mm |
| RMS Euclidean position | 0.05709 mm | 0.02719 mm |
| Maximum Euclidean velocity | 0.004166 m/s | 0.002056 m/s |
| Maximum density | 0.173523 kg/m^3 | 0.055542 kg/m^3 |
| Maximum pressure | 318.585 Pa | 100.549 Pa |
| RMS pressure | 6.7124 Pa | 2.7192 Pa |
| Maximum corresponding frame-time difference | 1.1023 microseconds | 0.4741 microseconds |

A larger cached pilot (`dx = 0.005`, 47,790 particles, 0.5 seconds) passed
the same numerical/identity/fixed-wall checks with maximum/RMS position
differences of 3.22986/0.02343 mm. Its maximum velocity, density and pressure
differences were 0.04288 m/s, 1.20349 kg/m^3 and 2,206.062 Pa; RMS pressure
difference was 24.8132 Pa and maximum frame-time difference 0.3367 microseconds.
Both variants took 4,250 steps, but baseline/fused rebuilds were 3/2 and
readbacks 152/151. The timing harness rejected this comparison; no larger-case
speedup is claimed. Numerical validation can be reproduced separately:

```powershell
julia --project=gpu_version -t 4,0 gpu_version/benchmark/predictor_fusion.jl --rounds=9 --duration=4 --cache=false --output=gpu_version/benchmark/predictor_fusion_uncached_results.csv
julia --project=gpu_version -t 4,0 gpu_version/benchmark/predictor_fusion.jl --validate-only --duration=0.5 --dx=0.005
```

## Regression coverage

The focused fusion test covers 2D/3D, Float32/Float64, native and mixed position
precision, compact/full support, cached/uncached fields, gather lanes 1/4/16/32,
graph preparation and replay, two particle reorderings, and fixed/frozen
particles. It checks the extra live-buffer footprint and rejects overlapping
gather inputs/predictor outputs. A wider custom-model derivative is converted
to the stored working precision before prediction.

The full package suite was attempted. It stops in `test/simulation_result.jl`
with eight missing `VTKHDF/Steps/Values` errors (352 assertions pass).
A clean temporary source snapshot of the original commit `075800d` reproduces
the same eight errors and 352 passing assertions. These output-path failures
predate this change. Solver regressions are also run separately so this early
failure does not prevent their execution. Exact compact/full allocation tests
and the accepted-state-cache benchmark disable predictor fusion to keep their
arithmetic path constant; fusion has its own numerical checks.
The floating-body portion of `test/gpu_execution.jl` similarly expects a root
`Points` dataset and errors once; the original snapshot reproduces that error
too. Its preceding 72 graph/integration/saved-frame checks pass in both versions.

Focused checks passed: predictor fusion (901 assertions), compact allocation
and output (2,868), stationary predictors (2,993), accepted-state caches (1,320),
and sorting fields (1,008), as well as effective configuration and Updated-mDBC
checks. Device-step state, grid export, damping, external-force, floating-force
and periodic regressions also pass. The final-source fusion rerun passes all
901 assertions. The commands for these standalone files use the same project/thread
options shown above.
