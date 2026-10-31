# Variable-resolution 2D dam break

This native Julia/CUDA example follows the geometry and physical settings of
DualSPHysics 5.4 `vresolution/01_DamBreak2D`: a 4 m × 3 m open tank, a 1 m × 2 m
water column and a refined rectangle from `(2.5, -0.2)` to `(4.1, 0.6)` m.
Coarse spacing is 0.01 m; fine spacing is 0.005 m. Each domain has its own
particle mass and Wendland kernel (`h = 1.5dp`). Five wall layers use mDBC.
The reference settings are gamma 1, artificial viscosity 0.05, CFL 0.2 and
Fourtakas full density diffusion 0.1. Initial water density is hydrostatic.

Run from the repository root with an NVIDIA GPU:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/example/VResolutionDamBreak2d.jl
```

To change the run, include the definitions without starting a simulation:

```julia
include("gpu_version/example/VResolutionDamBreak2d/Setup.jl")
case = vresolution_dambreak_case()  # geometry and settings only; no GPU allocation
result = run_vresolution_dambreak(duration = 1.0, output_interval = 0.01,
    save_location = "C:/TestSimulations/DamBreak2D_VRes")
```

Each run writes `SimulationOutput.log` in its output directory. It contains
the usual `PART`, physical time, total/interval steps, elapsed runtime,
runtime per simulated second and estimated finish time. The final report
separates initialization, solver stepping and output/download time, and records
throughput, rebuild counts and particle transfer/exclusion totals. Progress is
flushed after every frame so the log can be read while the simulation runs.

The clock starts before device initialization and includes JIT compilation
performed inside the driver. Package loading/precompilation and caller-side
geometry construction/export are outside that clock. CUDA synchronization at
initialization/output boundaries makes the phase totals include completed GPU
work; no per-kernel profiling or additional per-step synchronization is added.
Caught failures and interruptions finalize the log before being rethrown.
The returned result provides `log_path` and `timings` for programmatic access.

The default `execution=:gpu` keeps all four timestep criteria, the previous
predictor limit, simulation clock, particle allocation decisions and routine
grid checks on the GPU. `batch_steps=32` and `use_graphs=true` replay a fixed
kernel sequence, with one small control readback per batch. A batch stops at
the exact output deadline; remaining launches leave particle, grid and flux
state unchanged. Capacity growth pauses at the pending rebuild, allocates on
the host, then resumes that phase without repeating integration or transfer.
Errors (including exhausted particle reserves) stop device work and are
reported at the next batch check, before another frame is written.

Live particles are compacted into a dense prefix, with new particles appended
after that prefix. Dormant capacity does not enter grid histograms, cell sorts,
reorder copies, predictor metadata copies or timestep reductions. Kernels still
launch against the allocated capacity and reject indices beyond the device
count. Predictor integration writes its dense evaluation state directly,
avoiding the separate position/velocity/density copy. Boundary-owner mappings
are refreshed as part of reordering. Full output downloads remain at frames.

`execution=:host` retains the previous host-controlled reference driver.
`use_graphs=false` runs the device phase machine through direct launches;
`batch_steps=1` is useful for debugging. The result configuration and log record
control readbacks and graph/direct batch counts. Grid-status reads for initial
allocation and exceptional capacity growth are additional, not per-step reads.
Float64 positions and MLS arithmetic, the timestep formula, and cell subdivision
`R=1` are unchanged. New allocation order can change within-cell summation order
after particle creation, so long-run comparisons use physical tolerances.

The output directory must be empty. Open `DamBreak2D_VRes.vtkhdf.series` in
ParaView for the numbered frames with changing particle counts. `Resolution`
is 0 for coarse particles and 1 for fine particles; `Spacing` and `Mass` carry
their physical values. Output selects one domain per position, so overlap and
buffer particles do not appear twice. Use `include_buffers = true` to inspect
both domains with `Owned` and `IsBuffer`. `History.csv` records particle counts,
represented fluid mass, created and retired particles, and density exclusions.
Finite fluid densities outside 70–130% of the reference density are excluded,
as in the reference case; their counts remain visible in the history. A
non-finite particle state stops the GPU driver at the next batch check (the
host reference reports it at the next output deadline).
Input IDs are remapped into separate odd/even namespaces so IDs remain unique
across both domains and subsequent creation events.

Two domains advance with a common adaptive symplectic timestep. Predictor
states have independent neighbor sorting before the corrector force evaluation;
derivatives return to the accepted particle order afterwards. Their buffer
states exchange density and velocity through first-order moving least squares
with a Shepard fallback when the support is singular. Face-integrated mass
flux accumulates until a receiving particle can be created; particles leaving
a domain are retired. Dormant GPU slots are reused and excluded from all
neighbor sums. Reserve exhaustion raises an error; increase `particle_reserve`
to provide more slots per domain.

This is an experimental first implementation for one stationary 2D rectangle
and fixed walls. It uses MLS instead of the reference's default Liu–Liu
reconstruction and a cell-centred initial lattice. It does not claim matching
DualSPHysics trajectories or exact global mass conservation across the overlap;
use the mass history and uniform-resolution runs when assessing accuracy.
Moving refinement regions, nested levels, periodicity and floating bodies are
not supported. `GPUCellSubdivision` changes search cells and is unrelated to
particle resolution.

Run focused geometry, reconstruction, transfer and output checks from the
repository root:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/test/variable_resolution.jl
julia --project=gpu_version -t 2,0 gpu_version/test/vresolution_logging.jl
```

The complete 1 s dam-break validation uses the default 0.01 m coarse spacing,
checks non-finite failures, density-excluded mass below 0.1% and represented
mass deviation below 1%, and retains its output for
inspection:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/test/run_vresolution_dambreak.jl
```

The generator accepts spacings that divide the 1 m water width exactly, such
as 0.01, 0.02, 0.025 or 0.05 m. Creation slots use GPU atomics; repeated runs
are not guaranteed to produce bitwise-identical trajectories.

The original host-controlled case was checked on Julia 1.12.7 with an RTX 5080, Float32 state
and Float64 positions. It completed 43,397 steps and 401 frames through 1 s.
There were 33,280 fine-particle and 2,482 coarse-particle creation events,
confirming transfer in both directions. Maximum represented-mass deviation
was 0.055%, with no density exclusions or non-finite states. All 401 exported
frames passed field-length, finite-state, ownership and ID checks, and the
series opened in ParaView 6.1.0. The 67 focused checks and existing mDBC,
sorting, stationary-predictor, frozen-particle and periodic-source regression
suites also passed.

For a controlled performance comparison, run:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/benchmark/variable_resolution.jl 0.01 0.02 3
```

The benchmark warms both variants, alternates their order, checks matching
steps, frames, rebuilds, events and accepted states, and reports paired elapsed
solver time separately from initialization and output. Its short default run
measures the initial dam-break stage; it is not a full-run DualSPHysics comparison.

On an RTX 5080 with Julia 1.12.7 and CUDA.jl 6.4.2, three paired runs of that
0.02 s case (592 steps, nine frames) measured a median **12.02× solver speedup**,
with paired ratios 12.01–12.34×. Median stepping time was 10.311 s for the host
reference and 0.859 s for device control. Both performed 593 accepted rebuilds
and 592 predictor rebuilds per domain and passed the benchmark's state checks.
The device driver used 25 control readbacks, including initialization; initial
grid allocation reads and output downloads are separate. These measurements
cover the combined driver/pool changes, not their individual contributions.

The optimized driver also passed the complete 1 s, 0.01 m validation on the
same device: 43,430 steps, 401 frames, **75.194 s stepping**, 19.303 s
initialization/JIT and 7.370 s output (102.45 s total driver wall time). Maximum
represented-mass deviation was 0.05375%, with no density exclusions or
non-finite states. It created 2,477 coarse and 33,283 fine particles and used
1,596 control readbacks. These are a single full-run validation's timings,
not a paired speedup against the historical full run or DualSPHysics.
