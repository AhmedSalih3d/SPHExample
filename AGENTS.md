# Project Agents.md Guide for OpenAI Codex

This file describes how OpenAI Codex and other AI agents should work with this repository.
It may evolve over time, in shaa Allah.

## Project Structure for OpenAI Codex Navigation
- `/src`: main Julia source code for the SPH solver
- `/example`: example scripts demonstrating solver usage
- `/input`: sample input files used by the examples
- `/images`: images referenced by `README.md`
- `Project.toml`/`Manifest.toml`: dependency declarations
  (do not modify without instruction)
- `README.md`: high level project overview and instructions

## Common Code Patterns in This Repository
- Each file in `/src` typically defines a `module` with the same name as the file
  and explicitly `export`s the public API.
- Internal modules are referenced with relative `using ..ModuleName` imports.
- Mutating functions use a trailing `!` (e.g., `ResetArrays!`).
- Configuration types usually use `Parameters.@with_kw` structs with default
  values and inline `@assert` checks.
- Some configuration types are `mutable struct`s when state is updated during a
  simulation run (e.g., metadata tracking counters and timers).
- Particle data is frequently stored in `StructArray`s containing `SVector`
  fields from `StaticArrays`.
- Dimension-aware code uses type parameters like `{D, T}` and `Val(D)` dispatch.
- Numeric computations often use `@unpack` to pull parameters out of structs in
  inner loops.
- Unicode names (e.g., `ρ₀`, `Δt`) are common for physical quantities; keep
  naming consistent with existing symbols.
- Prefer small helper functions marked `@inline` when they are performance
  sensitive and widely reused.
- Numerical kernels frequently use `@inbounds` and, where safe, `@simd` to keep
  tight loops fast; only apply when bounds and data dependencies are correct.
- Time stepping and logging utilities expect fields to be updated in place; keep
  field names and types stable to avoid breaking downstream code.

## Coding Conventions for OpenAI Codex
- Use Julia for all new code additions and avoid introducing other languages
- Indent with four spaces
- Let lines flow out to greater lengths where needed for readability
- Use CapitalCase for functions and variables and CamelCase for types
- Document public functions with docstrings and comment complex logic
- Use `function` blocks for multi-line logic and keep `@inline` for short,
  hot-path helpers.
- Keep exported APIs grouped at the top of each module to match the existing
  file layout.
- Avoid unnecessary allocations in hot loops; prefer preallocated buffers and
  in-place updates when possible.

## Documentation Standards for OpenAI Codex
- Update `README.md` or example docs when behaviour changes
- Keep explanations concise and clear
- When adding new parameters or output fields, update the relevant docstrings
  in the same module.
- If new files are added to `/src`, mirror the module/file name and update the
  re-exports in `src/SPHExample.jl` when necessary.

## Pull Request Guidelines for OpenAI Codex
- Reference related issues when applicable
- Keep changes focused on a single concern
- List commands you executed (tests, scripts) in the PR description
- Follow commit message conventions: short imperative summary (≤50 chars)
  Provide details in the body if needed
- Do not amend or rebase pushed commits

## Evolution of Agents.md
These instructions may change as the project grows.
Feel free to open an issue or PR proposing improvements.
