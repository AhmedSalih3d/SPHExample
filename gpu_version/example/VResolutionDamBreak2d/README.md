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

The output directory must be empty. Open `DamBreak2D_VRes.vtkhdf.series` in
ParaView for the numbered frames with changing particle counts. `Resolution`
is 0 for coarse particles and 1 for fine particles; `Spacing` and `Mass` carry
their physical values. Output selects one domain per position, so overlap and
buffer particles do not appear twice. Use `include_buffers = true` to inspect
both domains with `Owned` and `IsBuffer`. `History.csv` records particle counts,
represented fluid mass, created and retired particles, and density exclusions.
Finite fluid densities outside 70–130% of the reference density are excluded,
as in the reference case; their counts remain visible in the history. A
non-finite particle state stops the run explicitly at the next output deadline.
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

The default case was checked on Julia 1.12.7 with an RTX 5080, Float32 state
and Float64 positions. It completed 43,397 steps and 401 frames through 1 s.
There were 33,280 fine-particle and 2,482 coarse-particle creation events,
confirming transfer in both directions. Maximum represented-mass deviation
was 0.055%, with no density exclusions or non-finite states. All 401 exported
frames passed field-length, finite-state, ownership and ID checks, and the
series opened in ParaView 6.1.0. The 67 focused checks and existing mDBC,
sorting, stationary-predictor, frozen-particle and periodic-source regression
suites also passed.
