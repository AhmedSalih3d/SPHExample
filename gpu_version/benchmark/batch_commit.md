# Commit/timestep fusion inside CUDA graph batches

For multiple reduction blocks, a graph batch now combines each interior
step's commit with the next step's timestep selection. The first selection
and last commit stay complete, so graph batches can still be replayed or
followed by direct execution independently. A batch of `B` steps removes
`B - 1` launches. Single-block final integration retains its existing fused
commit, and direct/timed launches retain their separate control kernels.

The combined kernel commits on thread one, then synchronizes the block
before reading the clock, phase or stop flags for the next selection. All
preceding particle, floating-body and open-boundary updates complete in
earlier kernels on the same stream. Timestep formulas, reductions, particle
integration, neighbor traversal and precision are unchanged. Both control
specializations are compiled/linked against a stopped state before capture.

## Reproduce

From the repository root:

```powershell
julia --project=gpu_version -t 4,0 gpu_version/test/batch_commit.jl
julia --project=gpu_version -t 4,0 gpu_version/benchmark/batch_commit.jl --rounds=15 --duration=20
```

The benchmark loads the original kernel/driver from git revision `c90337d`
into separate modules and compares them with the installed package. Two
complete warmup runs per variant precede alternating paired measurements.
Timings cover `RunSimulation`, including device upload, graph preparation and
capture, solver execution, particle/grid output and completion, followed by
CUDA synchronization. Package loading, geometry generation, host particle
allocation, logger creation/closing, explicit pre-run collection/synchronization,
output hashing and verification are excluded.

Each pair requires identical accepted time, step/rebuild/state-readback
counts, stable-ID-sorted final positions, velocities, densities, accelerations,
IDs, types and groups, and hashes/types/shapes of every exported particle/grid
dataset. Optional timestep diagnostic CSVs
must also match exactly. Nonfinite states, nonpositive density, graph capture
failures and graph-enabled runs with no actual graph batches are rejected.

Use `--duration=N`, `--dx=N`, `--float64`, `--single-neighbor`, `--no-graph`,
`--diagnostics`, `--clip-output`, `--reference-interactions`, or `--output=path`
to select other workloads. The default uses Float32 physics, Float64
positions, symplectic integration, cached fields, predictor fusion, radial
interactions, `R = 1`, and particle plus occupied-cell output every 0.01 s.
Baseline and optimized runs use the same settings within each comparison.
The radial diffusion operator is an existing opt-in setting, separate from
this launch-sequence optimization.

## Measurements

Measured on 2026-11-11 with Julia 1.12.7, CUDA.jl 6.4.2, `-t 4,0`, an NVIDIA
GeForce RTX 5080 and Windows WDDM driver 616.92. Each timing experiment ran
after this task's other test/benchmark processes had exited; desktop
applications remained open. These are workload/hardware-specific observations.

| Physical duration | Particles | Baseline median | Fused median | Median paired ratio | Paired elapsed reduction | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 s | 3,027 | 1.177911 s | 1.149200 s | 0.98186 | 1.81% | 10/15 |
| 4 s | 3,027 | 0.252027 s | 0.251973 s | 0.99713 | 0.29% | 9/15 |
| 0.5 s | 47,790 | 0.474174 s | 0.473761 s | 0.99301 | 0.70% | 7/9 |

The 20-second ratios range from 0.93990 to 1.02322; four-second ratios range
from 0.92581 to 1.02905. The short-case difference is within the observed
timing noise. The larger-particle case ranges from 0.98029 to 1.01937 and also
shows a small change within the observed noise. The longer case shows a modest
median reduction with overlapping individual timings; this does not establish
a consistent gain for every run.
Raw results are in [20-second results](batch_commit_results.csv) and
[four-second results](batch_commit_short_results.csv), plus
[larger-case results](batch_commit_large_results.csv).

Every pair retains exact accepted particle fields and particle/grid output,
including output timestamps and frame shapes. Both variants take 42,488
steps, six rebuilds and 2,007 readbacks in the 20-second case; the four-second
case takes 8,499 steps, three rebuilds and 404 readbacks. Both use 6,008 and
1,203 graph batches respectively, with zero direct fallback batches or capture
failures. Full adaptive steps reach exactly the same accepted physical times
in each comparison. The larger case uses `dx = 0.005` and takes 4,250 steps,
three rebuilds, 151 readbacks and 255 graph batches in both variants, also
with exact particle/grid output and zero capture failures/direct fallbacks.

Additional timing commands, using the same project/thread options above:

```powershell
julia --project=gpu_version -t 4,0 gpu_version/benchmark/batch_commit.jl --rounds=15 --duration=4 --output=gpu_version/benchmark/batch_commit_short_results.csv
julia --project=gpu_version -t 4,0 gpu_version/benchmark/batch_commit.jl --rounds=9 --duration=0.5 --dx=0.005 --output=gpu_version/benchmark/batch_commit_large_results.csv
```

## Regression coverage

`test/batch_commit.jl` compares the combined control launch directly with
separate commit/selection over active, stopped, output-deadline and rebuild
transitions in both precisions and with output clipping on/off. Its complete
simulations compare direct execution against batched graphs in 2D/3D,
Float32/Float64, both steppers, compact/full buffers, cached/uncached fields,
moving walls, floating placement and timestep diagnostic output. Every full
simulation fixture uses multiple reduction blocks. The control comparisons
include up to 512 partial reduction blocks.

The production source passed all 488 new assertions. Focused existing checks
also passed for effective configuration/capture fallback, device-step state,
timestep diagnostics, predictor fusion (901), stationary predictors (2,993),
accepted-state caches (1,320), sorting fields (1,008), Updated mDBC, damping,
floating forces, periodicity, grid export, open boundaries, motion-stage
sampling and floating springs. The 72 DBC/mDBC/moving-wall integration and
saved-frame comparisons from the first testset of `test/gpu_execution.jl`
passed separately.

Two existing test sections stop their containing files:

- `test/output_deadlines.jl`: its first 80 assertions pass, then the saved-frame
  reader errors because `VTKHDF/Steps/Values` is absent.
- `test/periodic_images.jl`: its initial configuration testset passes eight
  assertions and errors twice because `only(configuration.fallbacks)` receives
  two fallbacks, including the interaction-type-preloading fallback.

Both failures were reproduced using the original modules from `c90337d`
loaded under separate names (including the original `GPUConfiguration` for
the latter check). Unchanged shared package code remains common to the two
versions. These are baseline failures, and a passing full suite is not claimed.
The other selected solver test files were run separately after these early failures.
