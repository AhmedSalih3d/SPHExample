# Device-resident cell-list rebuilding

Status: Implemented and GPU-validated. See the [validation and measured performance report](../benchmark/cell_list_rebuild_performance.md) for the scan correctness repair, exact committed-baseline comparisons, full-suite results, transfer/allocation evidence, and limitations. No commit or publication was requested or performed.

## Objective

Remove the bounding-box device-to-host synchronization and CPU grid construction from normal cell-list rebuilds. Keep safe host handling for capacity growth and errors. Measure the resulting rebuild and whole-simulation performance without changing the physical configuration.

## Committed baseline behavior

- [update_cell_list!](../src/GPUCellGrid.jl) calls `reduce_svector` to obtain the bounding box.
- [finish_reduction](../src/GPUReductions.jl) copies per-block partial results to the host synchronously and completes the reduction on the CPU.
- The CPU computes cell bounds, dimensions, and cell count, validates them, ensures allocation capacity, and writes the grid back to the device.
- Histogram, prefix scan, scatter, deterministic sort, and particle gathering already execute on the GPU.
- [rebuild_cell_list!](../src/SPHCellList.jl) swaps persistent arrays and rebuilds ghost-owner indices. The timestep loop also has a separate stop-state host readback; removing the bounding-box transfer alone does not eliminate that readback.

## Implementation requirements

1. Inspect repository instructions and existing changes before editing. Preserve unrelated work.
2. Establish warmed timing baselines before changing the implementation.
3. Complete the bounding-box reduction and grid calculation on the device. Prefer preserving the existing dynamic grid semantics within preallocated capacity. If using a fixed conservative grid, demonstrate that cell mapping and diffusion orientation remain correct.
4. Provide a preallocated normal rebuild path without bounding-box host readback. Use a safe fallback for capacity growth, with device status reporting integrated into an existing necessary host synchronization where practical.
5. Detect nonfinite positions, cell-count overflow, and `GPUMaxCells` violations before unsafe kernels execute. Never silently clamp particles into an obsolete grid.
6. Make the normal rebuild path compatible with CUDA graph capture where practical. Address prefix-scan workspace, pointer stability, alternating particle buffers, graph invalidation on growth, and ghost/floating index compaction.
7. Preserve cell rounding, stencil margins, deterministic ordering, and asymmetric density-diffusion pair orientation. Retain 2D/3D, Float32/Float64/mixed-position storage, supported subdivisions, both timestep schemes, all boundary modes, and floating bodies.
8. Keep host grid metadata and exported grid geometry consistent with the particle snapshot, including asynchronous output.
9. Update the GPU README with the implementation, fallback behavior, supported graph path, and measured limitations. Avoid unnecessary dependency changes.

## Validation and acceptance

- Add focused tests for stationary and moving domains, capacity growth, invalid positions, overflow protection, deterministic sorting, and graph/non-graph execution.
- Compare neighbor membership, cell assignments, sorted particle identities, physical state, timestep decisions, and output geometry with the baseline. Explain any numerical differences rather than silently loosening tolerances.
- Run applicable GPU regressions and the full CUDA suite where available. Distinguish pre-existing failures from new ones.
- Benchmark representative small and large cases after warmup, with interleaved repetitions and identical precision, physics, duration, and output settings. Report rebuild time, full-run time, steps, rebuild counts, allocations, and synchronization behavior.
- Demonstrate that steady-state rebuilds no longer copy bounding-box partials to the host. State explicitly whether the existing timestep stop-state readback remains.
- Report any scan, graph, or allocation limitation and any measured regressions. Do not claim an unmeasured speedup.
- Do not commit or publish unless separately requested.

Useful starting points: [step profiler](../benchmark/profile_steps.jl), [StillWedge benchmark](../benchmark/still_wedge_speed.jl), [GPU tests](../test/runtests.jl).
