# Agent Guide for SPHExample

This repository contains two related Julia packages for weakly-compressible
smoothed particle hydrodynamics (SPH): the multithreaded CPU package
`SPHExample` at the repository root and the CUDA package `SPHExampleGPU` in
`gpu_version/`. The GPU package has its own dependencies, examples, tests, and
benchmarks. Its API is designed to mirror the CPU package, with additional
GPU-specific geometry, particle generation, and simulation support.

The packages cover 2D and 3D simulations, configurable SPH kernels, boundary
conditions, viscosity and density-diffusion models, prescribed motion, and
simulation output and measurements. The GPU package also supports floating
bodies and polygon-based particle generation. Output includes HDF5/VTKHDF
data and ParaView workflows.

## Project layout

- `src/`: CPU package implementation and public `SPHExample` module.
- `example/`: CPU simulation examples.
- `input/`: particle layouts and other example inputs.
- `test/`: CPU package tests.
- `gpu_version/`: independent CUDA package, with its own `src/`, `example/`,
  `test/`, `benchmark/`, `Project.toml`, `Manifest.toml`, and `README.md`.
- `README.md`: repository overview and CPU package guidance.
- `gpu_version/README.md`: CUDA package usage, features, and benchmarks.
- `images/`: images used by the repository README.

## Coding conventions

- Use Julia for new implementation code and follow the surrounding patterns.
- Use four-space indentation. Prefer `snake_case` for new internal functions
  and variables, and CamelCase for types. Preserve established public API
  names.
- Do not impose a fixed source-code line-length limit, including the old
  92-character guideline. On wide displays, keep related code together when
  that improves readability; wrap lines when it makes the structure clearer.
- Keep comments focused on non-obvious logic. Document public APIs with
  docstrings where appropriate.
- Preserve intended CPU/GPU behavior and numerical results. Keep CUDA kernel
  code compatible with device execution and the package's supported numeric
  types.

## GPU optimization learnings

- **Mode-aware allocation:** Use metadata/device-particle constructors when
  omitting inactive GPU fields, while preserving full-buffer defaults for
  low-level APIs. Unavailable downloads and modes missing required buffers
  must fail explicitly, not return incomplete data.
- **Dense inputs and scratch lifetimes:** Keep interaction inputs dense;
  per-neighbor predictor indirection can erase the benefit of smaller buffers.
  Share reorder scratch only when predictor and rebuild lifetimes do not
  overlap. Refresh fixed entries after sorting and preserve updates for fluid,
  moving, velocity-only and floating particles.
- **Derived fields:** Reconstruct pressure with the appropriate EOS, including
  negative Updated-mDBC wall pressure, and preserve startup rounding. Only the
  simulation driver may discard acceleration when every entry is overwritten
  before any reader; standalone rebuilds must preserve reductions, floating
  forces and downloads. Compute reciprocal density from the state being
  evaluated, including predictor and mDBC-corrected densities.
- **Boundary ownership:** Keep compact mDBC auxiliaries in stable boundary
  slots and refresh both owner mappings after particle reordering. Asynchronous
  frames must retain their own owner-index snapshot; expand compact ghost data
  to full host fields with zeros for fluid particles when exporting.
- **Grid export contract:** Export only cells occupied by the current output
  frame's positions, with edge length `H / GPUCellSubdivision`, not stale
  neighbor-list occupancy or stencil margins. Adjacent occupied cells must
  share identical faces; do not fill genuinely empty cells to hide gaps.
  Preserve this contract for both single-file and numbered VTKHDF output.
- **Numerical tuning:** Cell subdivision and opt-in radial arithmetic are not
  interchangeable, accuracy-preserving speed knobs. Equal accepted-neighbor
  counts do not prove identical pairs or trajectories; reference density
  diffusion depends on pair orientation. Keep subdivision `R = 1` when
  reproducing CPU reference behavior and validate long-run behavior before
  changing search settings or operators.
- **Performance evidence:** Isolate one change, warm kernels/CUDA graphs,
  alternate variant order and compare paired, unprofiled whole-solver timings
  with matching step/rebuild/readback counts and particle-state checks.
  Kernel-duration sums and rejection rates do not establish elapsed solver
  speed or memory-bandwidth saturation. Report memory savings separately from
  speedups, especially when small laptop cases are within timing noise.
- **Memory accounting:** Distinguish CPU/host particle storage, live GPU/device
  buffers and CUDA allocator reservations. Count aliased scratch once, use
  actual element types and state whether support/reorder buffers, cell grids,
  output, motion, reductions and floating-body arrays are included. Comparisons
  with solvers such as DualSPHysics need particle/boundary counts, dimensionality,
  precision and CUDA/device information; auxiliary-buffer savings are not
  whole-solver VRAM savings.

See the [GPU README](gpu_version/README.md#compact-gpu-buffers) for allocation
contracts and the [memory benchmark](gpu_version/benchmark/memory_buffers.jl)
for alias-aware live-device-buffer accounting.

## Documentation and dependencies

- Update the README or example documentation for the package whose behavior
  or usage has changed.
- The root and CUDA packages have separate dependency files. Change a
  package's `Project.toml` or `Manifest.toml` only when the task requires a
  dependency change, and validate that package after the change.

## Testing

Run the CPU package tests from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run the CUDA package tests from the repository root:

```bash
julia --project=gpu_version gpu_version/test/runtests.jl
```

The full CUDA suite requires an NVIDIA GPU and compares selected behavior with
the CPU package in a subprocess. Some geometry tests can run without a GPU;
see `gpu_version/README.md` for focused commands. When dependencies change,
instantiate the affected package before testing.

Select focused GPU regressions for the changed contract:
[gpu_memory.jl](gpu_version/test/gpu_memory.jl) for allocation,
[stationary_predictors.jl](gpu_version/test/stationary_predictors.jl) for scratch
sharing, [sorting_shadows.jl](gpu_version/test/sorting_shadows.jl) for derived
fields and downloads, [updated_mdbc.jl](gpu_version/test/updated_mdbc.jl) for
boundary slots, and [grid_export.jl](gpu_version/test/grid_export.jl) for output.
See the [GPU README](gpu_version/README.md#tests) for commands and requirements.

On Julia 1.12, launch direct multithreaded CPU runs with an explicit compute
thread count and no interactive thread, for example `-t N,0`. Avoid `-t auto`:
the interactive thread can make `threadid()` exceed `nthreads()` in the
current threaded neighbor loop.

## Change and pull request guidance

- Keep changes focused and check both packages when shared behavior is affected.
- Do not amend or rewrite commits that have already been published.
- Include the relevant validation commands in pull request descriptions.
