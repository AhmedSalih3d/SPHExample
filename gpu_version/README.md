# SPHExampleGPU – CUDA version of SPHExample

`gpu_version/` is a copy of `src/` ported to NVIDIA GPUs with
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). It is a self contained Julia
package (`SPHExampleGPU`) with the same public API as the CPU package: the
example scripts only differ in `using SPHExampleGPU` instead of
`using SPHExample`. All physics (Wendland/cubic kernels, artificial, laminar
and laminar+SPS viscosity, all density diffusion models, dynamic and mDBC
boundary conditions, particle shifting, prescribed motion of bodies) is
supported in 2D and 3D, in `Float64` or `Float32`.

## Requirements

* An NVIDIA GPU with a recent driver (CUDA 12 or newer). The CUDA toolkit is
  downloaded automatically by CUDA.jl, `nvcc` is not needed.
* Julia 1.10 or newer.

## Getting started

```bash
# from the repository root
julia --project=gpu_version -e 'using Pkg; Pkg.instantiate()'
julia --project=gpu_version gpu_version/example/Dambreak2dMDBC.jl
```

The examples in `gpu_version/example/` mirror those in `example/`. Choose the
precision with `FloatType = Float32` or `Float64` at the top of a script.

### Which precision?

* `Float64` reproduces the CPU results to round-off (the test suite checks
  that densities, positions and velocities agree to about `1e-14` relative
  after a few dozen steps).
* `Float32` is where consumer and laptop GPUs are fast. GeForce/RTX-A/Quadro
  cards execute double precision at 1/32 (Ampere, Ada) or 1/64 (Blackwell) of
  their single precision rate, so `Float64` on such a GPU is not faster than a
  modern multi core CPU. Datacenter GPUs (A100, H100) have fast `Float64`.
  `Float32` is the precision DualSPHysics uses by default as well; densities
  stay well within the ~1 % variation of weakly compressible SPH.

### Startup time and precompilation

The package ships a [PrecompileTools](https://github.com/JuliaLang/PrecompileTools.jl)
workload (`src/PrecompileWorkload.jl`). When the package is precompiled it
writes tiny versions of the example cases (2D/3D, mDBC and DBC, a moving body
with planar shifting) and runs them. The host code (CSV input, logging, VTKHDF
output, the time loop) is then cached in the package image. If a working GPU is
present during precompilation, so are the CUDA kernels. On an RTX A1000 laptop
GPU the first 2D dam break run in a new session drops from about 45 s to about
1 s. Loading the package (`using SPHExampleGPU`, about 5 s, mostly CUDA.jl)
stays the same, and no sysimage is needed.

This has a cost and some limits:

* Precompilation takes longer: about 2 minutes with a GPU and 30 s without one
  on the machine above. It runs once per package or dependency change, not once
  per session.
* Kernels are cached only for the GPU architecture present during
  precompilation. Without a GPU (for example on an HPC login node) only the host
  code is cached, and kernels compile on the first run. Precompile on a GPU node
  to cache them as well.
* Only the precompiled types are cached. Other types still work but compile on
  first use. By default that means `Float32`, the `WendlandC2` kernel,
  `SymplecticTimeStepping`, and the viscosity and density diffusion models of
  the examples. Keep `FloatType` and the models consistent between runs to
  benefit.

The following preferences, stored in `LocalPreferences.toml`, change the
workload. Restart Julia after setting them; the package then precompiles again.

```julia
using Preferences, SPHExampleGPU
# also cache Float64 (the test suite and CPU parity runs use it)
set_preferences!(SPHExampleGPU, "precompile_float_types" => ["Float32", "Float64"])
# lanes per particle compiled for the gather kernels (default [1, 2, 4, 8, 16, 32])
set_preferences!(SPHExampleGPU, "precompile_gpu_lanes" => [1])
# turn the workload off, e.g. while editing the package source
set_preferences!(SPHExampleGPU, "precompile_workload" => false; force = true)
```

## What the GPU does differently

The CPU code loops over cells, evaluates each particle pair once and scatters
the two halves into per thread arrays that are reduced afterwards. On the GPU
that would need atomics or large reductions, so the GPU version instead uses
the standard *gather* formulation:

* **One thread (or a few warp lanes) per particle.** The thread of particle
  `i` loops over the particles of the 9 (2D) or 27 (3D) surrounding cells and
  accumulates `dρ/dt`, the acceleration and the optional kernel and shifting
  sums in registers, then writes them once. No atomics, no per thread arrays,
  no reduction step, no zeroing of arrays. For small particle counts the
  neighbour loop is split over several lanes of a warp (each lane takes every
  `K`-th candidate) and the partial sums are merged with warp shuffles, so a
  3 000 particle 2D case still keeps the whole GPU busy.
* **Exact CPU semantics.** The CPU's density diffusion uses `Dⱼ = -Dᵢ` as a
  short cut that is not symmetric in the particle roles. The gather kernel
  reconstructs which particle of a pair was the CPU's `i` (lower index inside
  a cell, higher index across cells) and calls the very same
  `compute_viscosity`/`compute_density_diffusion` functions with the same
  signature, so the GPU reproduces the CPU results instead of a "close"
  variant. As on the CPU, the models receive the densities of the state being
  evaluated (`ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹`) as arguments, so the corrector loop sees the
  predictor density and velocity, and the reciprocals are computed once per
  particle instead of once per pair.
* **Fluid gating of density diffusion by select, not early exit.** The
  `LinearDensityDiffusion` and `ComplexDensityDiffusion` terms act only between
  two fluid particles. The CPU code returns early when either particle is not
  fluid (about 6% faster there). On the GPU that `return` is a divergent branch
  inside the pair loop: it pays off for warps of boundary particles and costs
  in warps of fluid particles with boundary neighbours. Measured on an RTX
  A1000 (Float32, pair kernel time, 9 interleaved rounds, median ratio to the
  old MotionLimiter multiply): early exit 1.00 (DamBreak2D), 1.02
  (StillWedge2D), 1.01 (DamBreak3D), 0.99 (Duckling3D), a wash; a branch
  free select of the finished term 0.98, 0.99, 0.90, 0.96. The models
  therefore compute the term unconditionally and select it. A multiply by one
  is exact, so this is equivalent to the old form; the only differences seen
  in tests are ulp level and come from FMA contraction in `dot`, which can
  change with any recompile.
* **Dense uniform grid + counting sort.** Particles are binned into cells of
  edge `H` on a grid covering their bounding box (plus a one cell margin).
  A histogram (atomics), a prefix scan and a scatter reorder all particle
  arrays by cell in a single fused gather kernel; the three cells of one row
  form one contiguous index range so a 3D particle scans 9 ranges instead of
  27 cells. `GPUCellSubdivision = 2` bins at `H/2` with a 5x5(x5) stencil
  instead (25 row ranges in 3D), which scans 42 % less volume in 3D and 31 %
  in 2D; see the next point for what that buys. The reorder is followed by an in-cell insertion sort, which makes
  runs bitwise reproducible (`GPUDeterministicSort = false` skips it). The
  cell list is only rebuilt when the accumulated displacement exceeds `h`,
  exactly like the CPU version. Gravity and motion factors are derived from
  particle type instead of stored or reordered as separate arrays.
* **Half width cells (`GPUCellSubdivision = 2`).** Candidates per particle
  drop from 30.5 to 21.8 (DamBreak2D), 34.3 to 23.7 (MovingSquare2D), 520 to
  322 (DamBreak3D) and 262 to 159 (Duckling3D); the share of candidates
  inside the support rises from 20-31 % to 32-46 %. The pairs found are the
  same (tested with the density diffusion off, where every term is
  antisymmetric: both grids agree to 1e-14). The price is 25 instead of 9 row
  ranges per particle in 3D (iterated lazily; a 25-tuple would unroll the
  pair body 25 times), a grid with 4-8 times more cells, and an mDBC kernel
  that visits the fluid ranges of 125 instead of 27 cells. Measured on an RTX
  A1000 (Float32, device time per step, 9 interleaved rounds, median ratio to
  `GPUCellSubdivision = 1`): with one lane per particle the pair kernel takes
  0.84 (DamBreak2D mDBC), 0.73 (MovingSquare2D), 0.77 (DamBreak3D), 0.65
  (Duckling3D mDBC) and the whole step 0.94, 0.76, 0.77, 0.73. With the
  automatic lane split of the small cases (32, 4, 8, 4 lanes) the pair kernel
  takes 1.41, 0.85, 0.94, 0.81 and the step 1.33, 0.87, 0.93, 0.94: the lanes
  stride each of the 25 short row ranges separately, and the mDBC kernel
  time roughly doubles. The default therefore stays 1, the grid whose pair
  orientation reproduces the CPU results; use 2 for large cases that run with
  one lane per particle. Splitting the rows rather than the candidates of a
  row over the lanes, and merging the fluid ranges of a row in the mDBC
  kernel, are the obvious follow ups. A full run of the 3D dam break at
  `dp = 0.0085` (1.6 s simulated, `GPUCellSubdivision = 2`,
  `GPULanesPerParticle = 1`) finished 40 s sooner than with the `H` grid.
* **Lanes per particle (`GPULanesPerParticle`).** With one lane, particle
  `i` gets thread `i`, and that thread visits every candidate itself. With
  `K` lanes, particle `i` gets `K` consecutive threads of a warp instead:
  lane `k` starts at candidate `k` of each row range and steps by `K`, so the
  `K` lanes together visit every candidate exactly once, each accumulating
  its own partial sums. At the end the partial sums are merged across the
  lanes with warp shuffles (`lanes_sum`) and lane 0 writes the result; the
  mDBC kernel does the same for its ghost node matrix. The option exists
  because a small case has too few particles to fill the GPU: an RTX A1000
  holds roughly 16 000 resident threads per pass, so a 3 000 particle 2D
  case with one lane leaves most of the chip idle. The automatic choice
  (`0`) picks `K` such that at least four times the resident thread capacity
  is launched, which gives 32 lanes to a 6 700 particle 2D case, 4 to a
  55 000 particle 3D case and 1 above roughly 100 000 particles. Lanes
  interact badly with half width cells: they stride within one row range and
  only finish together when the range holds many candidates. With `H/2`
  cells there are 25 rows per particle, each about five times shorter, so
  lanes sit idle while the loop overhead is paid 25 times instead of 9 (the
  1.41 above). Even on the `H` grid, 32 lanes took 2.6 times the device time
  of 1 lane on the pair kernel of the 6 700 particle 2D dam break; many lanes
  only help when the GPU would otherwise be idle, and they cost real work in
  shuffles and duplicated loop control. The heuristic was tuned on wall time
  on Windows, where launch overhead dominates the small cases, so device
  time alone is not the whole picture, but a sweep with `benchmark/tune.jl`
  on the small cases is worth doing. Rule of thumb: large cases run with 1
  lane and `GPUCellSubdivision = 2`; small cases keep both defaults unless
  measured otherwise.
* **Fused element-wise kernels.** Half step, density limiting, prescribed
  motion and pressure are one kernel; the final step also computes the
  pressure for the next step and the block wise reduction of the time step
  limits and the maximum displacement. The corrector advances the position
  with the half step velocity times `dt`, the same symplectic scheme as the
  CPU `FullTimeStep`. A step therefore consists of 6 kernel launches (8
  with mDBC and moving bodies), two of which are single block kernels that
  do the loop control.
* **Device resident time step.** The time step, the simulated time, the
  displacement bound that triggers a cell list rebuild and the loop flags
  live on the device. A single block kernel turns the block wise reduction
  of the previous step into `dt` and decides whether the step may run; the
  other kernels read `dt` from device memory and exit at once when it may
  not. The host enqueues a batch of steps (its size estimated from the last
  read back, at most `GPUMaxStepsPerSync`), reads the state back once, and
  only then rebuilds the cell list or writes an output. Because nothing in
  the launch sequence of a step changes from step to step, it is captured
  once as a CUDA graph and replayed (`GPUUseGraph`), which removes most of
  the per launch overhead that bounds the small 2D cases on Windows.
* **mDBC on the GPU.** One thread per boundary particle gathers the fluid
  neighbours of its ghost node, assembles the `(D+1)×(D+1)` system in
  registers and solves it with StaticArrays inside the kernel.
* **Asynchronous output.** Particle data is downloaded when an output is due
  and written to `vtkhdf` by a Julia task while the GPU already integrates the
  next output interval (`GPUAsyncOutput = false` disables this).

Data stays on the GPU for the whole run; the host `StructArray` passed to
`RunSimulation` holds the state of the last written output (reordered by
cell, like the CPU version) when the function returns.

### Modes are type parameters

As in the CPU package, the optional features of a run are type parameters of
`SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}` rather
than boolean flags, so the fused kernels are specialised at compile time:

| Parameter | Types | Meaning |
|-----------|-------|---------|
| `SMode` | `NoShifting`, `PlanarShifting` | particle shifting |
| `KMode` | `NoKernelOutput`, `StoreKernelOutput` | store the kernel and kernel gradient sums for output |
| `BMode` | `NoMDBC`, `SimpleMDBC` | mDBC boundary condition (needs `ParticleNormalsPath`) |
| `LMode` | `NoLog`, `StoreLog` | write the simulation log file |

Trailing parameters can be omitted, `SimulationMetaData{2, Float32}(...)`
selects all defaults. `RunSimulation` takes the time stepping scheme as
`SimTimeStepping`: `SymplecticTimeStepping()` evaluates two neighbour loops per
step and is the scheme validated against the CPU; `SingleNeighborTimeStepping()`
reuses the corrector derivative of the previous step as the next predictor and
evaluates one neighbour loop per step. Like the CPU scheme it applies the mDBC
correction (with `SimpleMDBC`) to the boundary densities before every half
step, and it evaluates the derivative at the accepted full state (mDBC,
pressure, neighbour loop) before the first step and after every cell list
rebuild. Unlike the CPU it does not re-evaluate the carried derivative every
20 steps. Because both packages share these types
and the `AllocateDataStructures(SimGeometry, SimMetaData)` form, the case
definitions in `benchmark/cases.jl` construct against either package.

### GPU specific `SimulationMetaData` options

| Option | Default | Meaning |
|--------|---------|---------|
| `GPUSyncTimers` | `false` | Synchronize after every phase so the `TimerOutputs` table shows real per kernel times (costs a few percent). |
| `GPUDeterministicSort` | `true` | Sort particles inside each cell after the counting sort; results become bitwise reproducible between runs. |
| `GPUMaxCells` | `50_000_000` | Abort with a clear message if the neighbour grid would need more cells (a particle escaped). |
| `GPUInteractionThreads` | `128` | Threads per block of the interaction and mDBC kernels. |
| `GPULanesPerParticle` | `0` (auto) | Warp lanes that share one particle's neighbour loop (1, 2, 4, ... 32). Small cases cannot fill the GPU with one thread per particle, so the automatic choice launches at least four times the resident thread capacity of the device and lets several lanes scan alternating neighbours, combined with warp shuffles. Use `1` for large cases and together with `GPUCellSubdivision = 2` (see "Lanes per particle" above). |
| `GPUBoundaryForces` | `true` | Evaluate the momentum equation for boundary particles too, as the CPU does (their acceleration only enters the force based time step limit). `false` skips it and saves 10-20 % in cases with many boundary particles, at the price of a slightly different adaptive time step. |
| `GPUAsyncOutput` | `true` | Write output files on a task while the GPU continues. |
| `GPUMaxStepsPerSync` | `32` | Upper bound on the steps enqueued between two host read backs of the device resident step state. The actual batch is the estimated number of steps until the next cell list rebuild or output. `1` reproduces a synchronization per step. |
| `GPUUseGraph` | `true` | Capture the launch sequence of a step as a CUDA graph and replay it. Disabled automatically with `GPUSyncTimers`. |
| `GPUCellSubdivision` | `1` | Cells per support radius `H` along each axis: `1` bins at `H` with a 3^D stencil (the CPU's cells), `2` at `H/2` with a 5^D stencil. Same neighbour pairs, fewer distance checks, more cell ranges per particle. Only `1` reproduces the CPU's orientation of the asymmetric density diffusion term (the results of the two grids differ by that term only). `2` pays off with one lane per particle (large cases); with many lanes it is slower. |

`OutputTimes` also accepts `Float64` values or vectors when `FloatType = Float32`.

## Layout

```
gpu_version/
├── Project.toml          # package SPHExampleGPU (adds CUDA.jl)
├── src/
│   ├── SPHExampleGPU.jl              # module glue, same exports as SPHExample
│   ├── GPUReductions.jl              # fused SVector block reductions
│   ├── GPUCellGrid.jl                # dense grid, counting sort, reorder kernel
│   ├── GPUKernels.jl                 # interaction, mDBC, half/final step kernels
│   ├── SPHCellList.jl                # device containers, time loop, RunSimulation
│   ├── PrecompileWorkload.jl         # tiny example runs cached at precompile time
│   └── ...                           # unchanged host side files from src/
├── example/              # GPU versions of the example scripts
├── benchmark/            # CPU/GPU benchmark harness (shared case list)
└── test/                 # unit tests and CPU vs GPU comparison
```

Files not listed under `src/` above are identical to the CPU version except
for small changes that make them precision generic (`Float32` literals), the
`Float32` variant of `Estimate7thRoot`, and the removal of the CPU only
`AllocateThreadedArrays`.

## Tests

```bash
julia --project=gpu_version gpu_version/test/runtests.jl
```

The suite checks the grid helpers and reductions, bitwise reproducibility,
`Float32` sanity, and runs four cases (2D mDBC wedge, 2D moving square with
shifting and SPS turbulence, 3D dam break, 3D duckling with mDBC) with the CPU
package in a separate process and compares the final state.

## Benchmarks

```bash
# CPU (use -t N,0 on Julia 1.12: `-t auto` adds an interactive thread which the
# CPU code does not account for and crashes)
julia -t 24,0 --project=. gpu_version/benchmark/benchmark_cpu.jl
# GPU
julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl --float32
julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl            # Float64
# kernel level profile of one case
julia --project=gpu_version gpu_version/benchmark/profile_steps.jl --float32 DamBreak3D_dp0.0085
```

Each case runs the same physics as the corresponding example script for a
few hundred steps after a warm up run that compiles everything. Results
measured on an Intel i7-12850HX (16 cores, 24 threads) and an NVIDIA RTX
A1000 Laptop GPU (16 SMs, 4 GB, 1/32 rate `Float64`) are listed in the table
below (time per step of the time loop, output excluded).

| Case | Dim | Particles | CPU 24 threads, Float64 [ms/step] | GPU Float32 [ms/step] | Speedup Float32 | GPU Float64 [ms/step] | Speedup Float64 |
|------|-----|----------:|------------:|------------:|--------:|------------:|--------:|
| StillWedge2D_MDBC_dp0.02 | 2D | 3,027 | 1.27 | 0.217 | **5.8x** | 3.16 | 0.4x |
| DamBreak2D_MDBC_dp0.01 | 2D | 6,678 | 1.21 | 0.245 | **4.9x** | 4.16 | 0.3x |
| StillWedge2D_DBC_dp0.01 | 2D | 11,246 | 1.83 | 0.302 | **6.1x** | 8.55 | 0.2x |
| MovingSquare2D_dp0.04 | 2D | 33,020 | 7.48 | 0.710 | **10.5x** | 14.24 | 0.5x |
| MovingSquare2D_dp0.02 | 2D | 128,775 | 29.20 | 1.598 | **18.3x** | 41.63 | 0.7x |
| DamBreak3D_dp0.02 | 3D | 17,446 | 8.23 | 1.367 | **6.0x** | 33.52 | 0.2x |
| Duckling3D_MDBC_dp0.01 | 3D | 54,817 | 20.45 | 2.747 | **7.4x** | 72.35 | 0.3x |
| DamBreak3D_dp0.0085 | 3D | 171,496 | 99.78 | 15.425 | **6.5x** | 478.41 | 0.2x |
| Duckling3D_MDBC_dp0.005 | 3D | 365,656 | 161.08 | 15.746 | **10.2x** | 476.85 | 0.3x |

Notes on reading the table:

* Times are the best of 2-3 runs of the whole time stepping loop (a few
  hundred adaptive steps each), file output excluded; measured 2026-09-25.
* The RTX A1000 Laptop GPU is a small 16 SM GPU whose `Float64` throughput is
  1/32 of `Float32` — that is why `Float64` loses to 24 CPU cores here. On a
  datacenter GPU (A100/H100, 1/2 rate) `Float64` scales like `Float32` does.
* The `Float32` speedup grows with the particle count (18x at 129k particles
  in 2D, 10x at 366k in 3D) because large launches amortize the fixed
  per-kernel overhead (~16 µs on Windows/WDDM); a desktop GPU with more SMs
  will extend this trend to even larger cases.
