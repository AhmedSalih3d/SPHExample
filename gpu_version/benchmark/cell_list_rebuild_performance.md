# Device-resident cell-list rebuild validation

Measured on 2026-10-09 with Julia 1.12.7, CUDA.jl 6.4.2, an RTX 5080 (Windows/WDDM), and an AMD Ryzen 9 9950X3D. The baseline is the committed GPU package at `3f4aaca66a86f14667c3388c062fe11d9ccf9462`. The harness loads that source into a separate module without replacing working files. No dependencies were changed.

## Correctness repair and implementation

The incomplete implementation scanned cell counts **exclusively** into `CellStart[2:end]`, despite already reserving `CellStart[1] = 0`. For counts `[0, 2, 0, 3]`, this produced `[0, 0, 0, 2, 2]` instead of `[0, 0, 2, 2, 5]`. Scatter ranges consequently overlapped and some permutation entries remained uninitialized, allowing an invalid gather address. The cell-boundary scan is now inclusive; compaction also uses inclusive ranks. The original committed rebuild first passed fresh-process GPU checks independently of the candidate.

Two kernels reduce positions and finish/validate the dynamic grid on the device. Nonfinite flags are reduced per block, including later grid-stride iterations. Grid origins, dimensions, rounding, stencil margins, deterministic ordering, and pair orientation retain the original construction. Coordinate and cell-count overflow, invalid scales, and `GPUMaxCells` violations stop before histogram/scatter/gather.

After each timestep batch, preparation is guarded by `STOP_REBUILD`. It publishes status, origins, dimensions, and cell count into the existing integer timestep state. The necessary timestep readback returns the header; the driver validates it, grows capacity if needed, invalidates graphs when buffer pointers change, and enqueues sorting/reordering. There is no separate bounding-box or grid-status readback in this driver path.

Standalone `update_cell_list!` still waits for a small status/header copy. `prepared=true` skips that copy only after successful preparation and consumption. It is an internal enqueue contract: positions must remain unchanged between preparation and rebuilding.

## Method

From `gpu_version/`:

```powershell
julia --project=. --compiled-modules=existing -t 4,0 benchmark/cell_list_rebuild.jl --rounds 5 --rebuilds 100 --profile --output benchmark/cell_list_rebuild_results.csv
```

Each variant warms its kernels and grows buffers before isolated measurements. Five repetitions alternate baseline/candidate order (AB, BA, AB, BA, AB). Both variants use Float32 positions/arithmetic, deterministic sorting, subdivision 1, the symplectic scheme, CUDA step graphs, and at most 32 steps per host readback. Whole runs simulate 0.15 seconds with the same physics, spacing, prescribed motion, shifting, output times, asynchronous output, output fields, and exported grid cells. Complete wall time includes upload, graph construction, stepping, output, logging, and file closing; initial capacity growth remains included.

The isolated phase rebuilds preallocated lists. The shared phase additionally models the necessary timestep readback: the baseline reads timestep state then bounding-box partials, while the candidate prepares before reading timestep state. Neither timed rebuild phase grows capacity. Complete runs must rebuild more than once. A stationary StillWedge mDBC trial rebuilt only once at this duration and was rejected as evidence for frequent rebuilding.

Raw measurements: [CSV](cell_list_rebuild_results.csv). CUDA transfer/allocation evidence: [CUPTI report](cell_list_rebuild_results_profile.txt). Host allocations are Julia allocated bytes, including launch bookkeeping. Device allocations are measured separately with `CUDA.@allocated` after warmup. CPU enqueue timers do not measure GPU execution time; preparation and rebuilding are reported together where compared.

## Results

Times below are median milliseconds. The change column is the median of **paired candidate/baseline ratios**, so it need not equal the ratio of the two displayed medians.

| Case / particles | Measurement | Baseline | Device path | Paired change |
|---|---|---:|---:|---:|
| MovingSquare dp0.04 / 33,020 | One isolated rebuild | 0.1715 | 0.1279 | -27.0% |
| Same | One rebuild with timestep readback | 0.1915 | 0.1316 | -30.5% |
| Same | Complete simulation | 75.912 | 74.454 | -2.8% |
| Same | Simulation loop | 58.100 | 56.966 | -2.7% |
| MovingSquare dp0.02 / 128,775 | One isolated rebuild | 0.1813 | 0.1483 | -18.7% |
| Same | One rebuild with timestep readback | 0.2063 | 0.1617 | -22.6% |
| Same | Complete simulation | 209.282 | 202.456 | -3.3% |
| Same | Simulation loop | 172.156 | 164.806 | -4.1% |

| Whole-run counters | 33,020 particles | 128,775 particles |
|---|---:|---:|
| Accepted steps, both implementations | 373 | 745 |
| Rebuilds, both implementations | 24 | 53 |
| Timestep readback calls, both implementations | 45 | 94 |
| Dedicated bbox readbacks, baseline → candidate | 24 → 0 | 53 → 0 |
| Dedicated candidate grid-status readbacks | 0 | 0 |
| Candidate capacity growths, including initialization | 1 | 1 |

Median CPU time charged to whole-run rebuild orchestration plus preparation decreased from 5.404 to 4.067 ms for the smaller case and from 12.839 to 9.461 ms for the larger case. Preparation runs conditionally after every batch, so including its enqueue cost avoids overstating the improvement. Device work and waits can be charged to the subsequent timestep readback timer.

Median host allocation bytes per isolated rebuild decreased from about 9,468 to 7,000 for the smaller case and 9,496 to 7,003 for the larger case. With the timestep readback, they decreased from about 9,790/9,822 to 7,931. Complete-run host allocations were essentially unchanged: 17,262,071 → 17,275,648 bytes and 41,774,857 → 41,775,810 bytes. Median measured GC time was zero.

For ten warmed shared rebuilds, profiling recorded:

| Case | D2H copies, baseline → candidate | D2H bytes | Temporary device allocation bytes |
|---|---:|---:|---:|
| 33,020 particles | 30 → 20 | 20,960 → 640 | 720 → 0 |
| 128,775 particles | 30 → 20 | 80,960 → 640 | 2,600 → 0 |

The remaining two copies per timestep readback transfer floating and integer state. The integer state grew by 32 bytes to carry the grid header. CUDA.jl often waits by polling `cuStreamQuery`; counted `Synchronize` API calls alone therefore do **not** count host synchronization boundaries. The profiler's query counts are scheduling-dependent. The robust result is removal of one separate bounding-box copy/wait per rebuild, while retaining the timestep readback.

The whole-run improvement is modest. Paired wall ratios ranged from 0.9383 to 0.9918 in the smaller case and 0.9223 to 1.0422 in the larger case; individual larger-case repetitions regressed. These short desktop-GPU measurements establish the benefit for these configurations, not a universal speedup.

Additional interleaved comparisons also matched particle state, timestep decisions, and every exported dataset exactly:

| Configuration | Steps / rebuilds / timestep reads | Baseline → candidate complete median | Paired complete change |
|---|---:|---:|---:|
| DamBreak3D, 17,446 particles, Float64, direct launches, subdivision 2; 3 repetitions | 720 / 11 / 33 | 2,495.576 → 2,511.782 ms | **+0.4%** |
| Stationary StillWedge mDBC, 3,027 particles, Float32, graphs, subdivision 1; 5 repetitions | 319 / 1 / 15 | 23.930 → 23.549 ms | -0.4% |

The 3D rebuild shared with the timestep readback improved by 30.8%, but its complete simulation was slightly slower. The stationary control's standalone rebuild regressed by **3.3%**, with host allocations increasing from about 6,036 to 6,968 bytes per rebuild. Its shared rebuild improved by 7.0%; whole-run paired ratios ranged from 0.8326 to 1.0858. This explicitly labeled single-rebuild control is not evidence for frequent-rebuild performance. Preparation adds two guarded launches per batch, including batches that do not rebuild, and can cost more than it saves in some configurations. The 3D and stationary results support treating the whole-run impact as small and workload-dependent.

Raw data: [3D CSV](cell_list_rebuild_3d_results.csv), [stationary-control CSV](cell_list_rebuild_rare_results.csv). Reproduction commands:

```powershell
julia --project=. --compiled-modules=existing -t 4,0 benchmark/cell_list_rebuild.jl --small DamBreak3D_dp0.02 --large DamBreak3D_dp0.02 --float64 --no-graph --subdivision 2 --rounds 3 --rebuilds 50 --duration 0.15 --output benchmark/cell_list_rebuild_3d_results.csv
julia --project=. --compiled-modules=existing -t 4,0 benchmark/cell_list_rebuild.jl --small StillWedge2D_MDBC_dp0.02 --large StillWedge2D_MDBC_dp0.02 --allow-single-rebuild --rounds 5 --rebuilds 100 --duration 0.15 --output benchmark/cell_list_rebuild_rare_results.csv
```

## Validation and limits

```powershell
julia --project=. --compiled-modules=existing -t 4,0 test/cell_list_rebuild.jl
julia --project=. --compiled-modules=existing -t 4,0 test/runtests.jl
```

The full suite reported all executed test sets passing, including 5,774 focused rebuild checks and 15,209 checks in the main GPU test set. Focused checks cover 2D/3D, Float32/Float64/mixed position storage, subdivisions 1/2/3, repeated and translated domains, all persistent fields, exact prefix boundaries, neighbor candidates, ghost/floating compaction, multi-level and in-place scans, capacity growth, late-block/grid-stride nonfinite detection and recovery, overflow, prepared rebuild graph replay, both timestep schemes after graph-cache invalidation, and asynchronous output snapshots. Existing regressions cover boundary modes, prescribed motion, floating bodies, graph/direct execution, and saved frames. No tolerances were relaxed.

For every benchmark repetition, cell grids, boundaries, sorted identities, positions, final density/velocity/pressure/acceleration, ghost fields, accepted steps, final timestep/time, and all exported VTKHDF datasets matched the committed GPU baseline exactly. CPU-reference subprocess checks were skipped by the existing suite because the parent CPU checkout exposes `FlagSingleStepTimeStepping` rather than the mode API expected by those tests. No executed test failures were observed; this CPU comparison remains untested.

Step graphs remain supported across alternating particle buffers and are invalidated after growth. The production driver keeps conditional preparation, capacity handling, rebuilding, and index compaction outside step graph capture. The focused standalone capture test covers only the prepared enqueue phase with fixed dimensions/scan lengths and stable buffer pointers; it updates origins before replay. It does not support replay across dimension or capacity changes without recapture. Normal rebuild reduction/scan storage makes no temporary device allocations; host launch allocations and initialization/capacity allocations remain. Timestep readbacks, resume/output slot writes, output transfers, and final downloads remain.

The implementation is retained for its verified removal of the rebuild synchronization and device scan allocations, with measured benefits for repeated rebuilding. It is not a uniform whole-run speedup: the 3D complete run and stationary standalone rebuild regressed. Broader hardware, longer runs, mixed-position performance, and other physical cases should be measured before extrapolating these speedups.
