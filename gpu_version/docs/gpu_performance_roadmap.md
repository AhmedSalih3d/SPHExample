# GPU performance roadmap: 100 ideas to test

This is a list of experiments, not a proposal to change solver defaults.
The likely win depends on what costs time: the neighbor loop in large 3D
cases, wall correction in some mDBC cases, rebuilds in fast-moving flows,
or host scheduling and output in short runs. A smaller device buffer,
fewer candidate visits, and lower elapsed time are three different results.

## Where to start

- **Duckling:** Try independent mDBC launch settings (51), reuse stationary
  ghost queries (52), or carry accepted fluid caches into the next correction
  (57). In the [54,817-particle example](../example/DucklingMDBC.jl) on an RTX
  A1000 Laptop GPU, moving the fluid-type check ahead of the coordinate load
  was inconclusive, and a packed fluid mask took 4.6% longer. Both matched the
  full-run state and output exactly. The best tested grid setting,
  `(R, Rx, lanes) = (2, 4, 1)`, took 8.4% less elapsed time in seven
  one-second pairs with `Float32` physics and positions, writing particle and
  grid frames every 0.02 s. Both grids advanced 7,821 steps, but the results
  were **not** bit-identical: the largest saved-position difference was
  1.03 mm, and the grid file grew from about 78 MB to 529 MB.
- **Fine 3D DamBreak:** Retune the existing grid and lane settings (2–4)
  before introducing new traversal code. If the interaction pass still
  dominates, test cached row ranges (12) and profile register pressure (32).
- **Small StillWedge:** The [short-row walk](../benchmark/still_wedge_candidate_ranges.md)
  is already in use. A separate experiment removed about 25% of graph
  launches without a measurable whole-run gain. The
  [output mask](../README.md#stillwedge-performance) reduced frame transfer
  bytes by 0–25% across measured cases, but did not reliably reduce elapsed time.
  Look for CPU stalls at output (80) or startup (70) before doing more bit
  manipulation.
- **MovingSquare and other frequent rebuilders:** Measure rebuild time and
  call counts before testing a tighter displacement bound (21) or a faster
  deterministic sort (23). The existing
  [device-resident rebuild](../benchmark/cell_list_rebuild_performance.md)
  showed that a large isolated rebuild gain can still translate into only a
  few percent for a complete run.

Generic buffered neighbor lists and shared-memory position tiles are not good
first experiments: the [StillWedge tests](../benchmark/still_wedge_performance.md#fair-same-workload-attempt-2026-10-11)
already found regressions. The ideas below either target a different cost
or describe a narrower experiment.

## How to judge a candidate

Use the [shared cases](../benchmark/cases.jl) and compare one change at a time
against the current checkout, with matching physics, case duration, and output
settings. Warm both paths and alternate their order; use a same-source control
for small effects. Report paired unprofiled wall time separately from profiling,
along with steps, rebuilds, readbacks, output frames, graph fallbacks, live GPU
bytes, and output size where relevant.

An optimization that claims identical work must preserve the relevant stage
results, final particle fields by ID, and saved datasets. If scheduling changes
counts, explain why and verify the accepted states and output times. A change
to the grid, accumulation order, precision, or numerical method needs explicit
tolerances and longer checks of physical quantities such as pressure, forces,
mass balance, and free-surface motion. For CPU reference diffusion orientation,
use `R = 1` and `Rx = 1`; matching neighbor counts alone does not prove the
same pairs, order, or trajectory.

## Existing settings: start here

See the [GPU settings and traversal measurements](../README.md#neighbor-traversal-investigation).

1. **Use the measured Duckling grid as an opt-in preset.** Test `(2, 4, 1)` on the GPU and duration of interest. It narrows the search in the two transverse directions; use it only if the numerical differences and larger grid export are acceptable.
2. **Sweep `Rx` at fixed `R`.** Narrowing contiguous rows along `x` can remove distance rejects without increasing the number of transverse rows. Check the additional grid storage and the changed reference pair orientation.
3. **Tune lanes per particle for each workload.** More lanes can fill the GPU for a small particle count; one lane avoids shuffle and duplicated row overhead in dense cases. Record the resolved automatic value, and treat a change in reduction order as numerical.
4. **Tune interaction block size at fixed lane count.** Compare warp-aligned block sizes to see whether register usage or the number of resident blocks limits the interaction pass. Hold lane count constant to isolate the launch shape.
5. **Compare cached and computed interaction fields.** `GPUCacheInteractionFields` trades extra arrays and writes for fewer EOS and reciprocal evaluations per accepted pair. Test both paths on a dense DamBreak and a boundary-heavy Duckling; include cache preparation in the timing.
6. **Choose position precision independently of physics precision.** Native `Float32` positions avoid the mixed-precision position buffers and conversions. Keep `GPUDoublePosition` where offset domains or long integrations need it; test both translated and long-running cases before changing it.
7. **Try the existing scalar SPS contraction on SPS cases.** `GPUScalarSPSViscosity` replaces tensor work with a rank-one contraction without changing the closure. Check numerical differences on MovingSquare and longer shear flows; it is not a Duckling optimization.
8. **Tune the maximum steps per synchronization.** Sweep `GPUMaxStepsPerSync` for workloads where readbacks limit progress. Larger batches amortize host work only if rebuild and output stops allow enough active steps; count inactive work and fixed-grid escape-check boundaries.
9. **Disable boundary momentum only when wall forces are not needed.** `GPUBoundaryForces = false` can save force arithmetic in wall-heavy runs, but changes the boundary contribution to the adaptive timestep. Validate the new trajectory and do not treat it as equivalent wall-force output.
10. **Sweep compiler inlining on the actual models.** Use `GPUInteractionAlwaysInline` while keeping the fast-math setting fixed. Less inlining may reduce spills or instruction-cache pressure in heavy kernels, but can also make the common path slower.

## Neighbor search and traversal

Relevant code: [cell grid](../src/GPUCellGrid.jl) and [interaction gather](../src/GPUKernels.jl).

11. **Select short-row traversal from a measured row-length distribution.** Compare the existing stride and bit walk on short-row cases besides StillWedge. Consider a separate wider-row path only if enough work lands there; preserve each lane's original index sequence.
12. **Cache row endpoints once per occupied target cell.** Nearby targets often read the same two `CellStart` entries per row. A small per-cell table could amortize address and range lookups, but must pay for its rebuild and cache footprint.
13. **Reject entire row segments with conservative swept bounds.** A segment can be skipped if even its closest possible position is outside support. Include accepted and predictor excursions and rounding margins; retain the live cutoff and lane assignment on surviving segments.
14. **Choose which physical axis is contiguous in the cell index.** A different stride-one axis may make searches more uniform in elongated tanks and channels. Do not rotate the physics; validate the new ordering and any reference-orientation differences.
15. **Use a sparse brick index for mostly empty domains.** Keep dense row ranges inside allocated bricks, rather than scanning a dense histogram across gaps. Compare memory saved against directory lookups and test cross-brick pairs before using it on thin 3D domains.
16. **Precompute stencil offsets for stable grid dimensions.** Move repeated row-index arithmetic out of the hot loop when a small stencil can be described compactly. Compare the saved instructions against lookup traffic, especially on `R = 1`.
17. **Allow different transverse reaches in 3D.** The current grid already treats `x` separately; letting `y` and `z` vary independently could narrow the most useful axis without moving both to a finer grid. Validate support coverage, orientation, and storage growth.
18. **Test a one-axis rejection before fetching the remaining coordinates.** In a suitable layout, an initial component can prove a candidate is outside support. Use the same rounded coordinate model and an inclusive cutoff; branch divergence and extra memory transactions may cost more than they save.
19. **Decouple reference pair orientation from the search grid.** Carry enough reference-cell and owner information to select the original diffusion orientation on a finer grid. This could make a faster search usable for reference runs; it still does not recover the original floating-point accumulation order.
20. **Schedule very long rows separately.** A secondary kernel or block-per-target schedule could reduce the tail where a few exceptionally dense rows keep warps busy. Restrict it to real load imbalance and verify the effect of combining partial sums in a new order.

## Rebuilding and sorting the cell list

Relevant code: [rebuild kernels and workspace](../src/GPUCellGrid.jl) and [step/rebuild control](../src/SPHCellList.jl).

21. **Track net displacement from the build positions.** The accumulated maximum step displacement can be loose for oscillatory motion. A conservative bound on actual accepted and predictor excursions could avoid rebuilds, if the extra reference loads or reductions cost less than the rebuilds saved.
22. **Tune search skin and rebuild frequency together.** A wider conservative search region can permit longer reuse; a tighter one lowers rejected work at the cost of more rebuilds. Account for moving walls, ghost queries, and periodic wrapping before changing either side of the contract.
23. **Use a faster deterministic sort for crowded cells.** The current in-cell insertion sort is simple for short cells. A warp sort for large cells could retain exactly the same ascending index order while avoiding the long serial tail.
24. **Compare a stable key sort with histogram/scatter on awkward grids.** A radix sort by cell and original index may beat histogram/scatter plus per-cell sorting on highly skewed workloads. Include key construction, scratch bytes, and generation of the same range boundaries.
25. **Keep the permutation when a rebuild leaves cell membership unchanged.** If the bounds, bin scale, and per-particle cell IDs all match, the driver could skip gathering unchanged arrays. Reset rebuild state and refresh any position-derived fields that still need updating.
26. **Update only ranges affected by cell crossings.** For small migration fractions, merge incoming particles with the existing order and copy changed spans instead of rebuilding every array at once. Preserve dense interaction inputs, deterministic ties, and valid graph pointers.
27. **Aggregate contended histogram updates within a block.** Combine repeated cell keys before issuing global atomics when many threads bin into the same cell. Dense uniform grids may not benefit; count atomic contention and charge the local work.
28. **Tune prefix-scan launch geometry by grid size.** Benchmark the existing workspace scan against launch shapes suited to small or very large cell counts. Keep the same inclusive prefix boundaries and avoid adding a temporary allocation on every rebuild.
29. **Run deterministic sorting only on multi-particle cells.** A worklist built from the histogram could avoid launching mostly empty sort work in sparse grids. Include the cost of assembling the list and keep the dense path as a control.
30. **Reserve cell workspace capacity from a known case envelope.** A reliable capacity hint can avoid buffer growth, copy work, and graph recapture without locking the physical domain or freezing particles. Keep checked growth for runs that exceed the hint.

## Interaction arithmetic and compiler behavior

Relevant code: [GPU pair arithmetic](../src/GPUKernels.jl),
[SPH kernels](../src/SPHKernels.jl), and
[reference force fusion](../benchmark/reference_force_fusion.md).

31. **Reuse relative-velocity projections when the operands are identical.** The timestep signal, diffusion, and viscosity may request the same contraction. Check generated code before changing it, and do not share a projection when corrected wall velocity or reference orientation differs.
32. **Find and remove the cause of register spills.** Inspect register counts and local-memory traffic for the heavy 3D pair variants. Shortening live ranges or isolating a cold branch can allow more resident warps without changing the math; a blind register cap can make spills worse.
33. **Skip truly unused wall-wall visits on a narrow DBC path.** When boundary momentum is disabled, wall-wall continuity already contributes zero. Skip earlier only if viscosity-based timestep signals, kernel sums, shifting, and custom operators do not consume the visit.
34. **Cache pressure and inverse density independently.** A run may benefit from one derived field but not the other. Test the trade-off against the current combined cache, and refresh each field from the correct stage density and boundary mode.
35. **Reuse particle volume in models that repeat `m/ρ`.** If several gathers consume the same volume, a stage-local value can remove repeated arithmetic. Include the new load and cache writes, and respect different masses and updated wall densities.
36. **Compute kernel value and gradient together where both are used.** Share polynomial intermediates in mDBC or kernel-output modes. Restrict it to models that need both values; verify results near zero and the support edge.
37. **Keep high precision only in numerically sensitive steps.** In a full-`Float64` baseline, measure whether selected integration or reduction quantities require doubles while pair arithmetic can use `Float32`. This is a precision change and needs convergence checks on the target GPU.
38. **Try limited compiler math relaxations rather than blanket fast math.** A specialized FMA or reciprocal path may help specific operators. Keep timestep and error checks strict, including their `Inf` behavior, and check pressure and long-run stability before accepting rounding changes.
39. **Specialize explicitly disabled model terms.** Use or extend zero-term models where viscosity, diffusion, or another term is intentionally disabled. Removing dead work is preferable to approximating an active term; confirm the replacement still supplies the required timestep signals.
40. **Compute both pair directions once only for eligible operators.** A half-list prototype could avoid duplicated pair math, but needs conflict-free or deterministic accumulation and will add writes. Start with an actually reciprocal model; do not assume all diffusion, wall, or custom terms are antisymmetric.

## Device memory layout and data movement

See [compact GPU buffers](../README.md#compact-gpu-buffers) and the
[alias-aware memory benchmark](../benchmark/memory_buffers.jl).

41. **Test read-only loads for immutable inputs in a measured large case.** A different cache path might help if candidates compete with other loads. The StillWedge read-only-load experiment did not help; revisit only with evidence of cache or load stalls.
42. **Compare compact versus aligned position loads.** The existing 3D cell-relative path already has a vector load for `PosCell{3, Float32}`. In 2D, compare a packed layout against separate coordinate arrays or alignment padding and measure both transactions and bytes.
43. **Pack frequently co-loaded accepted-neighbor fields.** A 3D `Float32` velocity and density fit in 16 bytes. Loading them together might reduce instructions after the pair is accepted; measure possible extra stores and unwanted density or velocity traffic on other consumers.
44. **Reuse the cell index already loaded with mixed-precision positions.** Check whether a target loads both `PosCell.cell` and an equivalent `CellID` in the same gather. Remove the second load only where the two views describe the same current ordering.
45. **Use narrower local cell metadata where it fits losslessly.** A brick-local index or shorter grid index may make relative-coordinate data denser. Check all ranges after growth and retain the full-width path for large or changing domains.
46. **Store prescribed-motion parameters once per group.** A small dense table may be faster than reloading or duplicating parameters per particle in moving-wall workloads. Keep stable group mappings and compare against the already sampled motion paths.
47. **Avoid reordering metadata that no hot kernel reads.** Highly stable fields used only by output can remain in ID order and be gathered at output boundaries. Keep hot fields dense, and verify that open boundaries and user callbacks do not need a field on every step.
48. **Share scratch only across proven nonoverlapping phases.** Look for peak allocations in rebuilds, periodic sources, or optional modes beyond the existing predictor scratch reuse. Verify pointer stability under graph replay, and count aliased storage once.
49. **Reuse workspaces across same-shape runs.** Parameter sweeps can amortize allocation and grid provisioning if each run resets device control, state, and graph dependencies. This targets setup time and allocation peaks, not pair throughput.
50. **Remove accidental wide indexing in the hot loop.** Inspect generated code for `Int64` promotions of bounded cell or lane indices. Keep checked limits and wide address calculations where needed, and require an instruction or timing improvement before changing types.

## Boundaries, ghosts, and periodic search

Relevant code: [mDBC kernels](../src/GPUKernels.jl),
[Updated mDBC](../src/UpdatedMDBC.jl), and
[periodic search](../src/GPUPeriodicSearch.jl).

51. **Tune mDBC launch shape separately from the force gather.** Ghost owners and their 3×3 or 4×4 local solves have different occupancy and register needs. A separate tuning sweep can improve one pass without making the other pay for it.
52. **Cache fixed-ghost query geometry between rebuilds.** Reuse a stationary ghost's reference point and row ranges in both evaluations. Refresh after reordering, and never cache neighbor density, wall pressure, or a live-distance decision.
53. **Skip nonempty ghost rows that contain no fluid.** A prefix count or row-level hint built at rebuild time could avoid whole wall-only ranges without walking a fluid bitmask per candidate. Gate on stable particle types and measure the added metadata lookup against the existing pass.
54. **Share rigid-pose trigonometry across one body's ghosts.** Compute the transform coefficients once per body and evaluated stage instead of at each owner. Preserve the predictor and corrector poses and verify moving/floating ghost output.
55. **Reuse interpolation for identical ghost query points.** First count exact duplicate queries in real boundary geometries. If present, compute shared fluid interpolation once, but keep each owner's extrapolation, normals, and boundary state separate.
56. **Compare specialized small-matrix factorizations.** A single factorization might replace separate determinant and solve work. Compare against StaticArrays code generation, especially around the determinant threshold and the fallback used by weakly supported ghosts.
57. **Carry accepted fluid caches into the next mDBC pass.** Extend accepted-state cache maintenance only where the final step can write the next fluid reciprocals and pressures correctly. Keep wall corrections, motion changes, and rebuild invalidations explicit; the goal is one less full-domain preparation.
58. **Separate simple fixed-wall work from expensive moving-wall logic.** If moving walls cause divergence or large live temporary state for fixed owners, partition the owner list and specialize. Count the overhead of extra launches and preserve stage-specific ghost motion.
59. **Avoid scheduling inactive open-boundary reserves in more stages.** The neighbor traversal already skips the inactive cell. Try a live-target path for independent updates or reductions when reserves dominate, refreshing its membership on emission or freezing.
60. **Compare periodic image and translated-query paths on the intended geometry.** The image path trades duplicated sources and preparation for simpler gathers. Measure the live image count and verify the seam pairs; do not extend it to mDBC or short periods without checking their extra constraints.

## CUDA graphs and step orchestration

See the [batch-commit](../benchmark/batch_commit.md) and
[graph-preparation](../benchmark/graph_preparation.md) experiments.

61. **Pack floating and integer step-state readbacks into one record.** The current path batches two small transfers behind one wait. An aligned representation could remove one copy submission without changing the snapshot or adding a packing launch.
62. **Run grid preparation only when a batch actually requests a rebuild.** Explore device-conditional graph nodes where supported, so rare-rebuild cases do not schedule guarded bounding-box work on every batch. Keep the existing error and capacity checks on the taken path.
63. **Avoid speculative frame downloads that will be discarded.** First verify that discarded transfers matter. Compare a conditionally staged snapshot against the current optimistic copy, charging any extra device traffic or loss of overlap.
64. **Capture the most common non-power-of-two batches.** A frequently used size might replay as one graph instead of several. Check the distribution first and include capture time and retained graph memory; trimming graph tails alone already showed no clear gain.
65. **Test a persistent step loop only on small, launch-bound cases.** A resident kernel could avoid repeated launches if it can synchronize all participating work safely. Treat it as research: GPU occupancy, watchdogs, memory ordering, and strict timestep arithmetic constrain the design.
66. **Extend batch queueing to safe clipped schedules.** If the next exact deadline and accepted dependencies are known, queue behind the readback without the host bubble. Verify strict clipping and callback order; keep synchronous fallback for schedules that require host decisions.
67. **Overlap independent work in the step graph.** For example, reduce rigid-body loads while a disjoint particle update runs, when both read and write separate state. Completions must join before placement, timestep selection, or output, and contention can erase the benefit.
68. **Reuse graph templates across reorder buffer orientations.** Test supported parameter updates instead of capturing another copy when only known buffer addresses change. Measure first-use and rebuild overhead separately, and validate every patched dependency.
69. **Tune integration and reduction block sizes independently.** Corrector bandwidth, partial reduction count, and last-block commit costs differ from the pair kernel. Use a separate sweep and verify that reductions still publish one complete accepted step.
70. **Precompile the cold paths users actually run.** Add workload coverage only for a measured compile gap, such as a common mDBC launch or a reporting path. Measure startup in a fresh process separately from warm throughput and from package precompile cost.

## Output, diagnostics, and host work

See [output-interval benchmarking](../benchmark/bench_output_interval.jl) and
[grid export regressions](../test/grid_export.jl).

71. **Reduce full-frame output frequency when acceptable.** Fewer frames save readbacks, transfers, HDF5 work, and storage. Keep required timestamps and record whether the output schedule clips or otherwise changes the accepted steps.
72. **Use probes and force histories for high-rate diagnostics.** Keep frequent local measurements while writing sparse full particle frames. Compare with the original diagnostic traces at equivalent sample times; check whether a sampling mode disables batch overlap.
73. **Give occupied-grid output an independent, optional cadence.** A user may need frequent particle frames but only occasional grid inspection. This would avoid most of the fine-grid output penalty without misrepresenting a stale grid as current.
74. **Investigate a lossless on-disk representation for repeated grid geometry.** The [earlier cache](../benchmark/still_wedge_grid_reuse.md) still wrote the same bytes. Reuse payload only if the VTKHDF layout and real readers support it; otherwise retain the existing representation.
75. **Compute occupied cells on the GPU for large output frames.** Build the unique occupied set from the actual frame positions and transfer the compact set. Include binning and compaction costs, keep the cubic `H / GPUCellSubdivision` edge, and check against both existing writer modes.
76. **Avoid downloading internal cell IDs when no consumer needs them.** An exported grid derives from current positions, not the internal bin snapshot. Audit measurement, progress, and final-state consumers before removing a transfer from a particular output path.
77. **Coalesce small particle-frame copies.** A packed staging transfer might cut CPU submission overhead on very small frames. Compare against the current masked field copies and charge the packing kernel and host unpacking; do not add an extra synchronization.
78. **Tune writer buffering and HDF5 chunk shape for the real workload.** Large frames and a slow disk need different budgets from small frames and fast storage. Measure queue stalls, writer time, resident host memory, and file close time instead of assuming larger batches help.
79. **Try optional lossless HDF5 compression on I/O-bound runs.** Compression can lower bytes written when disk bandwidth is the limit. Benchmark CPU time, output size, and ParaView reads as well as write time; do not switch it on for all runs.
80. **Build the frame conversion plan once.** Cache field indices and conversion choices instead of rebuilding lookup tables in the output collector on every frame. Keep slots tied to their own ghost-owner and particle-state snapshots.

## Numerical methods and problem size

These are not equivalent-workload speedups. Use them when the application
allows the numerical change, and validate against the physical quantity of
interest rather than only comparing elapsed time.

81. **Try the existing single-neighbor integrator.** It avoids one full interaction evaluation on most steps. A [historical StillWedge run](../benchmark/still_wedge_performance.md#historical-scheme-comparison-2026-10-11) took about 34% less time, but had different positions and pressure; validate each intended case before switching.
82. **Inspect the timestep limiter before tuning CFL.** If measured steps are much smaller than necessary for a validated case, a higher documented CFL may reduce the number of steps. Check stability, force and pressure signals, and convergence rather than using elapsed time as a criterion.
83. **Calibrate the artificial sound speed for the intended Mach range.** A less conservative sound speed can relax the acoustic step limit, but changes compressibility and pressure response. Require the physical density and accuracy limits for the case.
84. **Avoid clipping to every diagnostic timestamp when exact integration deadlines are unnecessary.** Use a supported sampling/output schedule or separate clip deadlines. This can avoid short steps, but changes the numerical schedule and may affect diagnostics.
85. **Use variable resolution to spend particles where they matter.** Keep fine resolution near an interface or force measurement and use coarser particles elsewhere. Check mass and momentum through transitions, output cost, and whether the supported backend matches the case.
86. **Sweep smoothing length at fixed particle spacing.** A smaller validated support can reduce neighbor work, but can degrade consistency and stability. Use convergence tests and boundary checks; a faster run with too few neighbors is not an improvement.
87. **Compare validated kernel families at equal physical error.** A kernel with cheaper evaluation or a better accuracy-to-neighbor ratio may reach the desired result with less work. Match the error target rather than assuming the same `h` gives comparable fidelity.
88. **Investigate an alternative pressure integrator for acoustically limited long runs.** An implicit or semi-implicit stage could permit fewer timesteps, but introduces a solve and changes the method. Compare total iterations, memory, stability, and validated observables.
89. **Explore multirate stepping for separated fast and slow regions.** It may avoid advancing an entire domain at the smallest local timestep. Require a conservative interface coupling and synchronized output, then charge the extra bookkeeping and exchanges.
90. **Pick the coarsest globally converged spacing for routine runs.** A modest spacing increase can remove many particles in 3D. Use the resolution study to bound errors in the output that matters, and retain the fine run for validation.

## Startup, hardware, and scaling

See the [dual-GPU design](dual_gpu_simulation.md). Track elapsed time for a
single case separately from aggregate throughput and capacity.

91. **Use profiling counters to select the next bottleneck.** Measure actual memory traffic, cache hit rate, warp stalls, and host gaps in separate profiling runs. This avoids optimizing distance rejects or copies that are not limiting elapsed time.
92. **Select tuning presets by workload and GPU.** A small offline search could rank configurations by particle count, boundary fraction, and device class. Keep a deterministic default, validate accuracy, and report calibration time separately.
93. **Distribute independent runs across GPUs.** One case per device avoids halo communication and can improve sweep throughput immediately. Measure concurrent disk output and resource contention; it does not speed up one case.
94. **Prototype domain splitting for cases that exceed one GPU's capacity.** Keep an owner per particle and exchange stage-specific halos across a shared cell face. Demonstrate complete directed pairs and matching accepted state before measuring speed.
95. **Pack only the state required by a halo and overlap its exchange.** Exchange dense position, velocity, density, and needed boundary data, then run the interior while the halo moves. Measure peer bandwidth and compare pinned host staging when peer access is unavailable.
96. **Balance multiple devices by measured interaction work.** Equal particle counts can still leave one GPU with denser neighborhoods or more walls. Adjust the split only when saved wait time exceeds migration and rebuild costs, and validate ID ownership.
97. **Combine cross-device timestep and stop decisions.** A small reduction of the local force, acoustic, and rebuild criteria can avoid separate host synchronization for each decision. Every device must advance the same accepted step and handle errors without a partial commit.
98. **Reuse an equilibrated checkpoint for parameter sweeps.** Avoid replaying an identical spin-up if the model permits branching from a common state. Save enough accepted state, motion history, and integrator information to verify an uninterrupted restart.
99. **Reuse validated geometry and ghost preprocessing.** Cache expensive sampling or parsing by geometry inputs, spacing, and precision so repeated jobs do not regenerate equivalent data. Invalidate on configuration changes and compare the loaded geometry and normals exactly.
100. **Keep a long-lived Julia process for repeated short jobs.** Package loading, CUDA initialization, and first compilation can dominate small jobs. Amortize them across independent runs, reset per-run state, and report both fresh-process and warmed times.
