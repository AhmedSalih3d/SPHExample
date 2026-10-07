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

On Julia 1.12, launch direct multithreaded CPU runs with an explicit compute
thread count and no interactive thread, for example `-t N,0`. Avoid `-t auto`:
the interactive thread can make `threadid()` exceed `nthreads()` in the
current threaded neighbor loop.

## Change and pull request guidance

- Keep changes focused and check both packages when shared behavior is affected.
- Do not amend or rewrite commits that have already been published.
- Include the relevant validation commands in pull request descriptions.
