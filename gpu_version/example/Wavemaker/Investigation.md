# Matched DBC investigation

The comparison here uses the distributed DualSPHysics 5.4
`main/06_Wavemaker` DBC case. No mDBC correction, added layers, collision
projection or changed fluid volume is used to improve its result.

## What was different originally

The original Julia example was not the same simulation. It had 525,907
particles (63,832 fixed, 30,300 moving, 431,775 fluid), a 2.02 m periodic
interval and the CPU-reference complex diffusion model. The distributed case
has 463,200 original particles, a 2 m interval and Fourtakas inner DDT2.

The original sampling placed fluid as close as 0.000482 m to the beach at
0.02 m spacing. Its wall rasterization also differed from GenCase. At 10 s,
3,568 fluid particles were more than half a spacing below the bed and the
minimum fluid z was approximately -210.7 m. Escaped particles remained in
the calculation and expanded the grid.

There were further numerical differences despite similar option names:

- The package's kernel regularization was `(0.01h)^2`; DualSPHysics uses
  `(0.1h)^2`, a factor of 100 larger. It enters the viscosity, diffusion and
  relative-velocity timestep denominators.
- The existing symplectic scheme moves the accepted position using predictor
  velocity. The reference uses the average of old and accepted velocities.
- The existing timestep uses a particle's absolute position/velocity term.
  The reference uses neighbour-relative motion, maximum particle speed and
  predictor/corrector limits. Its force limit uses interaction acceleration.
- Prescribed wall velocities were analytical stage derivatives; the reference
  advances prescribed walls after the fluid step and uses displacement/dt.
- The reference keeps internal positions and its clock in double precision.
  `SavePosDouble = false` controls output precision, not internal storage.
- The periodic force path bypassed the package's PosCell representation and
  subtracted Float64 absolute coordinates for every candidate pair.
- The reference build projects enable CUDA fast math. Its pressure power and
  fractional hydrostatic power therefore use different arithmetic from the
  package's existing CPU-compatible path.

## Matched setup

`BoundarySection.csv` contains only the 316 integer x/z lattice cells of the
reference's fixed skin. The geometry code generates fluid and piston particles
and repeats the sections over 100 y rows. All 463,200 generated positions were
checked against the exported reference initial frame, with identical position
sets at its Float32 output precision. Double positions are reconstructed from
the GenCase lattice; this is not a claim of a byte-for-byte binary restart.

| Setting | Matched case |
| --- | --- |
| Fixed / moving / fluid | 31,600 / 30,000 / 401,600 |
| Spacing and periodic interval | 0.02 m; y in [0,2) |
| Kernel | Wendland C2; h = 1.5√3 dx |
| Kernel regularization | (0.1h)², explicitly overridden |
| State and force precision | Float32 |
| Position and clock precision | Float64 |
| Sound speed | 52.4099227246139 m/s |
| Density diffusion | Target-oriented Fourtakas inner DDT2, coefficient 0.1 |
| Viscosity | Artificial, coefficient 0.02; boundary factor 1 |
| Boundary condition | DBC, with the reference rasterization |
| Stepping | Reference predictor/corrector and adaptive limits |
| Motion | Reference sine schedule and increment-averaged wall velocities |
| Output | Every 0.1 s, without shortening steps to output times |
| Exclusions | Reference position domain and density bounds 700–1300 kg/m³ |

`DualSPHysicsSymplectic()` selects this separate reference path. The existing
`SymplecticTimeStepping()`, `SingleNeighborTimeStepping()` and diffusion models
retain their CPU-compatible arithmetic. The new path currently requires
DBC, double positions, Wendland C2, artificial viscosity, Fourtakas inner
DDT2, direct execution and one gather lane. Unsupported combinations fail.

The reference rebuilds/reorders at different points. This solver can reuse
cell assignments because it measures conservative offsets for both full and
predictor states. Ordered accepted-pair tests protect completeness and order.
Predictor positions remain continuous across periodic seams while using old
cell assignments; only the position used for exclusion checks is wrapped.
Wrapping the predictor itself made global search bounds span an entire period
when a particle crossed a seam. Accepted positions are wrapped before rebuilding.
Sorting, accumulation, compiler math and overshooting output times still
prevent a claim of bitwise-identical trajectories.

## Leakage must include exclusions

The completed reference run excluded 262 particles, all for position. At its
two-second frame it had already excluded 234; among the 401,366 retained fluid
particles, 31 were more than half a spacing below the bed and 28 were more
than 0.06 m below it. DualSPHysics was not perfectly watertight in this case.

A diagnostic run using the matched initial particles and inner diffusion but
legacy integration/regularization kept every particle. At two seconds, 222
were outside the reference position domain. Of those still inside it, 63
were more than half a spacing below the bed. Thus counting only the reference's
retained particles would hide a substantial part of its leakage.

The reference path applies the same domain and density policy. Excluded
particles are checked after both prediction and correction. Excluded
particles remain stored with `Type = Frozen` for inspection and are omitted
from interactions and active-grid bounds. The containment report separately
counts exclusions and penetration among active particles. This exclusion
policy is part of matching the case, not evidence that wall physics improved.

The completed ten-second comparison is:

| Quantity | DualSPHysics | This solver |
| --- | ---: | ---: |
| Excluded fluid particles | 262 | 279 |
| Active fluid particles | 401,338 | 401,321 |
| Active particles more than 0.01 m below the bed | 0 | 26 |
| Active particles more than 0.06 m below the bed | 0 | 0 |
| Minimum active bed clearance | -0.00806 m | -0.05475 m |
| Steps | 59,411 | 58,862 |
| Reported run time | 860.958 s | 2,149.775 s |

At two seconds this solver had 255 exclusions and 24 active particles more
than 0.06 m below the bed, compared with 234 and 28 in the reference. At five
seconds both had no active particles more than half a spacing below the bed.
The remaining exclusions and ten-second penetration are a real limitation;
matching this DBC case did not make its walls watertight.

The initial density comparison differed by at most one Float32 ulp
(6.104e-5 kg/m³). ID-matched fluid position RMS error, using the minimum
periodic y displacement, was 0.148 μm at 0.1 s, 1.16 mm at 1 s, 20.9 mm at
2 s and 0.908 m at 10 s. Saved-frame times also differ slightly. These are
matching physical inputs and numerical rules, not identical particle paths
or an accuracy validation of the later breaking-wave flow.

## Performance evidence

The runs use an RTX 5080 under Windows WDDM, Julia 1.12.7, CUDA runtime 12.9
for Julia and 12.8 for DualSPHysics, with driver API 13.4. Both cases are 3D
with Float32 force/state fields and Float64 internal positions. DualSPHysics
initially processes 509,520 particles including periodic copies; this solver
stores the 463,200 originals and queries translated images.

Isolated timings are
unprofiled, warmed whole-solver runs in ABBA order, with no concurrent GPU
work. Every variant starts from the same geometry and state. Records include
physical duration, steps, rebuilds, readbacks and ID-sorted state differences.

Initial diagnostic ablations on the matched particle layout identified:

| Isolated change | Baseline median | Variant median | Scope |
| --- | ---: | ---: | --- |
| Periodic absolute to cell-relative force coordinates | 37.1817 s | 18.2985 s | 0.1 physical s; strict math |
| CUDA fast math | 18.1232 s | 16.6037 s | 0.1 physical s; cell-relative coordinates |
| Cache image bounds within neighbour iteration | 16.4993 s | 16.0818 s | 0.1 physical s; fast math |

These diagnostic timings preceded the final pressure/regularization audit;
final matched-case timings are recorded separately below. They identify
implementation costs and must not be substituted for a completed reference
simulation time. Do not multiply their speedups into an overall claim.

All diagnostic runs above took 506 steps, two rebuilds and 1,518 host
criterion readbacks. Cell-relative arithmetic changed maximum position by
about 1.23 μm, velocity by 5.86e-5 m/s and density by 0.00129 kg/m³ after
0.1 s. Fast math changed them by about 1.69 μm, 7.86e-5 m/s and 0.00141 kg/m³.
These are measured rounding changes, not bitwise-preserving speed knobs.
Caching image bounds preserved the entire final state exactly.

Repeating the ablations with the final pressure and regularization gives:

| Isolated change | Baseline median | Variant median | Work in each run |
| --- | ---: | ---: | --- |
| Absolute to cell-relative periodic coordinates, strict math | 36.5053 s | 23.9048 s | 0.1 physical s; 506 steps |
| Strict to reference fast math, cell-relative coordinates | 23.8555 s | 16.4813 s | 0.1 physical s; 506 steps |

Each final timing also used two rebuilds and 1,518 criterion readbacks.
Coordinate conversion changed maximum position by 1.446 μm, velocity by
6.121e-5 m/s and density by 0.001282 kg/m³. Fast math changed them by
1.562 μm, 6.969e-5 m/s and 0.001465 kg/m³. These short-time differences are
small but nonzero; the ten-second trajectory comparison above is the relevant
limit on claiming identical results. A separate warmed unprofiled run took
16.5286 s for the same 506 steps.

Three fresh executions of the installed DualSPHysics binary, each stopped at
0.1 s using the same generated case, recorded simulation times of 7.191975,
7.187347 and 7.181075 s (median 7.187347 s). Their median total executable
runtime, including initialization, was 7.435179 s. Each reported 506 steps,
matching Julia's early count. Comparing the warmed Julia median of 16.4813 s
with the reference's simulation median gives a 2.29× early gap; the full-run
gap below also includes the later flow and much more frequent Julia rebuilds.
Initialization/output scopes are recorded separately rather than folded into
an asserted kernel speedup.

A 64-register cap was rejected: its median was 20.7974 s versus 16.5161 s
uncapped, with identical states and work counts. No bandwidth-saturation
claim follows from candidate counts, GPU utilization or kernel timings.

The original completed logs recorded 2,324.34 s for Julia versus 860.96 s
for DualSPHysics, but with the mismatches described above. DualSPHysics also
recorded approximately 100.61 s of separate initialization. Its first output
interval took 7.3071 s. The original 75 versus 238 s/s observations cannot
establish the cause by themselves; the matched measurements above and the
profile below address the remaining gap.

The final ten-second Julia run took 214.98 wall seconds per physical second,
versus 86.10 for the completed reference: a remaining 2.50× gap. Julia's
reported wall time includes startup and finalization; its initial frame was
logged at 9.274 s and final frame at 2,146.313 s. DualSPHysics separately
reported 100.610 s of initialization. Removing Julia's initial-frame time
does not materially close the gap. The new run used 25,903 rebuilds and
176,586 criterion/displacement readbacks. Its step count is 0.92% lower than
the reference, so a much smaller timestep cannot explain the remaining gap.

### Where the remaining time goes

A separate activity capture of the final 0.1 s setup recorded 15.94 s of GPU
activity, 92.28% of the trace. The interaction kernel accounted for 15.73 s:
91.09% of the trace and approximately 98.7% of GPU busy time. Its 1,012 calls
averaged 15.55 ms each. An adjacent unprofiled run took 16.3848 s; use the
unprofiled runs for elapsed-speed comparisons, since tracing adds overhead.

The CPU spent 15.66 s in stream synchronization while those kernels ran.
That wait overlaps GPU execution; adding it to GPU time would double-count
the same work. CUDA kernel launches took 72.83 ms in total and asynchronous
device-to-host API calls 40.37 ms. Host batching or output changes therefore
cannot account for the large early gap shown here.

The compiled Julia interaction kernel uses 122 registers per thread and
reports 32 bytes of local storage. The reference reports 60 registers for
its fluid kernel and 48 for its boundary kernel. Its separate fluid/boundary
loops use packed PosCell and velocity/density data plus stored periodic
copies. This solver uses a unified gather loop, particle-type checks and
per-target translated-image query bounds. The interaction implementation is
the measured bottleneck; those source and resource differences identify
credible optimization targets. Their individual contributions have not been
isolated, and register counts alone do not prove occupancy or bandwidth
saturation. The rejected register-cap experiment also shows why forcing a
lower count is insufficient.

## 2026-10-28 settings and execution audit

The physical WaveMaker setup was checked against the local reference XML and
source: particle counts, dimension, spacing, smoothing length, sound speed,
CFL, viscosity, inner DDT2, gravity, timestep bounds, density exclusions and
piston schedule agree at the documented precision. The implementations still
differ in rounding, ordering and their Wendland constants: DualSPHysics uses
rounded decimal kernel coefficients, whereas this package uses analytic ones.
That small numerical difference is not evidence for a factor-of-two time gap.

The checked current StillWedge example is a different workload from the
distributed DualSPHysics StillWedge DBC case. Julia selects CFL 0.5,
`SingleNeighborTimeStepping()` and linear diffusion; the reference selects
CFL 0.2, symplectic stepping and Fourtakas inner diffusion. Larger allowed
timesteps and usually one interaction pass per step can account for a raw-time
advantage without demonstrating a faster matched force implementation.
WaveMaker selects CFL 0.2 and two evaluations per step in both solvers.

A fresh reference run at 0.1 s recorded 506 steps, 7.209492 s simulation time
and 105.999 s initialization. The initialization scope is separate from all
solver-time comparisons below. Warmed ABBA runs of the current Julia setup
recorded 506 steps, two rebuilds and 1,518 criterion readbacks in each run:

| Isolated experiment | Baseline median | Variant median | State comparison at 0.1 s |
| --- | ---: | ---: | --- |
| Subdivision 1 to current subdivision 2 | 15.7062 s | 13.7837 s | Max position difference 1.30 μm |
| Stage-correct pressure cache, subdivision 2 | 13.8620 s | 13.9278 s | Max position difference 1.23 μm |
| Precomputed periodic image limits, subdivision 2 | 13.9261 s | 10.2462 s | Identical positions, velocities and densities |
| Forward current compiler flags, subdivision 2, no query cache | 13.7034 s | 11.0151 s | Max position difference 1.47 μm |

These are isolated, unprofiled whole-solver measurements on the RTX 5080.
The final compiler experiment uses `Investigate.jl timing_compiler`, with
false/true/true/false order and a warmup before each measurement. Honoring the
configured compiler flags reduces elapsed time by about 19.6% (1.244× speedup).
Against the fresh reference simulation time, the early gap becomes about 1.53×.
Maximum velocity and density changes were 5.62e-5 m/s and 0.001404 kg/m³;
enabling relaxed arithmetic is not bitwise preserving.
The periodic-query experiment removes about 26.4% of elapsed time, a 1.36×
speedup, with matching work counts and accepted state. It adds 72 bytes per
particle (31.81 MiB here), and does not establish whole-solver memory savings.
It identifies image-bound arithmetic and its kernel context as a substantial
cost; it does not separately apportion the gain to instructions, registers or
memory traffic. Pressure caching did not improve elapsed time.

Two configuration defects were also found: the special driver ignored the
interaction compiler flags, and an unconditional call in `WavemakerCase.jl`
made the wrapper run the simulation twice. The driver now forwards the flags;
the setup file runs automatically only when invoked directly.

The rebuild travel reduction previously measured accepted positions after
periodic wrapping. Crossing a seam by 0.002 m could report about 1.998 m,
forcing a rebuild. Displacement is now measured before wrapping and wrapping
is reported through a separate invalidation flag. The current global halo
representation still requires a rebuild on wrapping: simply skipping it leaves
an old cell anchor a full period away and makes the subsequent search much
broader. A longer diagnostic exposed this cost, so the driver retains explicit
wrap invalidation. No seam-related speedup is claimed. The wrapped position and
exclusion checks retain their existing meaning.
Positive and negative seam regressions check this accounting explicitly.
The old ten-second trajectory comparison above describes the earlier
implementation and compiler settings.

`DualSPHysicsSymplectic(cache_periodic_queries = true)` enables the measured
query preparation, with buffers allocated by the support constructors and
refreshed for every evaluated state. It defaults off, including in WaveMaker.
The cache remains separate from pressure caching. Its isolated gain above was
measured before forwarding the compiler options. With the current compiler
options actually enabled, a subsequent paired run measured 11.0475 s without
the query cache and 11.2389 s with it. Both had identical accepted states and
work counts. These effects overlap; their speedups must not be multiplied.

Final validation used the unmodified setup file with only duration, headless
launch and a new output path supplied as run arguments. The one-second run
completed at 1.0000083 s with 5,217 steps, 43 rebuilds, 15,651 criterion readbacks,
finite particle positions/densities and no exclusions. It wrote normal 0.1 s
frames. Its cold-process wall time is not a paired speed measurement, and the
final ten-second case was not rerun in this audit.

The final assertion groups passed: wave geometry/motion/integration, explicit
seam invalidation, 14,080 periodic pair/vector assertions including cached
queries, effective configuration, general periodicity and 2,412 compact-buffer
assertions. The combined shell session remained open after printing its
completion marker and was stopped after all assertion groups had completed.
The separate final one-second run and final paired compiler benchmark exited
successfully. The full CPU/GPU test suite was not run.

## 2026-10-29 full-run and periodic-representation audit

The user's completed current run recorded 1,525.97 s, 59,068 steps and 26,170
rebuilds with subdivision 2 and both interaction compiler flags enabled. The
saved full reference log records 860.958 s and 59,411 steps, a 1.77× elapsed
gap with nearly equal step counts. A remembered reference time of 780 s would
give a 1.96× gap, but is not the saved measurement. Output writing took 0.75 s
overlapped with solver work. The full run confirms that the earlier 0.1 s ratio
cannot stand in for a complete simulation.

Further experiments used the initial frame and the saved five-second frame,
with freshly rebuilt cell lists. These are force-only diagnostics: their timed
region excludes copy creation, sorting, integration and output. They do not
establish whole-solver speedups.

At the five-second frame, changing 128-thread blocks to 64 or 256 changed the
force timing by less than about 2%. Cooperative gathers with two or four lanes
were slower. A simple two-pass prototype specializing fluid and nonfluid
targets was also slower, about 11.64 ms versus 10.33 ms; this does not rule out
a more complete data-layout/partitioning change.

The substantial result came from explicitly storing periodic copies and using
the ordinary cell-relative neighbor loop instead of translated-image queries:

| Saved state | Translated queries | Stored copies | Force-kernel ratio |
| --- | ---: | ---: | ---: |
| Initial | 10.383 ms | 4.018 ms | 2.58× |
| Five seconds | 10.915 ms | 4.527 ms | 2.41× |

The stored-copy case processes 509,520 particles initially and 510,370 at five
seconds versus 463,200 originals in the query case. This is a particle-count
comparison, not whole-solver VRAM accounting. Repeated measurements alternated
query/copy/copy/query order after warming both variants.

Forces were compared over every original particle by ID. Relative L2
acceleration differences were 4.04e-7 initially and 9.58e-7 at five seconds;
the maximum five-second acceleration difference was 0.000414 m/s². A sampled
neighbor audit included 475 targets in each state, including the largest
force-difference targets. Matched pair vectors differed by at most 7.45e-9 m.
There were 179 and 4 unmatched neighbor identities, respectively, all at the
support-radius rounding boundary. Their maximum Wendland gradient magnitude
was 9.71e-16 initially and zero in the five-second sample. This is not
bitwise-identical arithmetic or an exhaustive pair-set proof, but it is strong
evidence that the periodic-query representation is an expensive implementation
choice. No corresponding factor was found in the physical input settings.

The reproducible force-only experiment is
`benchmark/wavemaker_periodic_images.jl`. It reads a saved case output and the
current case setup, suppresses the setup's automatic launch in an isolated
module, and writes diagnostics to a new output directory. It does not change
the production solver. Before enabling a stored-copy backend, it still needs
device-side copy refresh at both evaluated states, owner mapping across sorts,
original-only integration/reductions/output, explicit memory accounting and
paired whole-solver plus long-run numerical validation. The measured kernel
ratio must not be applied directly to the full-run time.

## Reproduction and validation

From `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Wavemaker/test_wavemaker.jl
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl containment 10
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl timing_coordinates
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl timing_fastmath
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl profile
```

To independently check the initial layout, export the reference initial
frame using its PartVTK tool and run:

```sh
PartVTK -dirdata CaseWavemaker_out/data -first:0 -last:0 \
  -savecsv particles.csv -csvsep:1 -vars:-all,+idp,+vel,+rhop,+type,+mk
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl check particles_0000.csv
```

The focused checks cover geometry, sine motion, gravity sign, the averaged
corrector, timestep limits, regularization, whole-support diffusion suppression
and 10,240 neighbour-pair/vector assertions, including shifted seams, stale
assignments, 2D/3D, subdivisions 1/2, mixed precision and 100 km coordinate
offsets. Existing allocation, predictor, sorting, configuration and prescribed
motion regressions also passed. This is not a run of the full CPU/GPU suite.
The existing `test/periodicity.jl` suite also passed, including its all-pairs
force oracle, shifted-seam interpolation, graph/output crossings and floating
integration checks. Normal package precompilation and a fresh package-load
check with `CUDA.functional()` succeeded.

Primary reference files checked in the local DualSPHysics 5.4 installation:
`CaseWavemaker_Def.xml`, completed `Run.out`/`Run.csv`/`RunPARTs.csv`,
`JSph.cpp` (constants and regularization), `JSphGpuSingle.cpp` (step sequencing),
`JSphGpuSimple_ker.cu` (integration), `JSphGpu_ker.cu` (DBC interactions),
`FunSphEos_iker.h` (pressure), `JSphGpu.h` (position storage), and the CUDA
build options in the supplied Visual Studio projects.

## 2026-10-29 implemented periodic-copy backend

`DualSPHysicsSymplectic(periodic_images=true)` now selects an actual solver
backend. Each force evaluation wraps its evaluated positions into the primary
period, appends near-face copies, sorts dense source fields on the GPU and
uses the ordinary cell-relative interaction loop. A single map selects original
targets; neighbor reads remain dense. Predictor copies use predictor density
and velocity. Originals alone are integrated, reduced and exported. Original
particle sorting retains its existing policy; the separate source list is
rebuilt twice per step. No host source-count download is needed inside a step.
The final diagnostics separately report source rebuilds and source count.

The source grid is initialized from the reference scheme's finite domain bounds,
with periodic and stencil margins. Invalid/nonfinite active sources set a device
flag that is consumed with the existing timestep reduction and raises an error.
The period must exceed twice the support radius, ensuring at most one copy per
original. Query caching and the image backend cannot be enabled together.

The runnable `WavemakerCase.jl` final call and `Wavemaker.jl` CLI explicitly
enable this backend. `WavemakerSetup.jl` loads definitions without running the
editable final call, preventing accidental duplicate full simulations.
The function and scheme defaults stay off for reference comparisons.

`benchmark/wavemaker_solver_backends.jl` checks the real backend against saved
initial and five-second states by original particle ID. The production builder
generated 509,520 and 510,370 sources. Acceleration relative L2 differences were
4.05e-7 and 9.58e-7; both timestep limits matched exactly. Maximum density-rate
differences were 3.89e-7 and 0.00449 against rates reaching roughly 6493.
Neighbor summation order and cutoff rounding differ, so these are accuracy
checks rather than a claim of bitwise equality.

The first warmed, unprofiled 0.1 s ABBA whole-solver comparison was:

| Backend | Elapsed solver times | Steps | Original rebuilds | Step readbacks |
| --- | --- | --- | --- | --- |
| Periodic queries | 11.479 s, 11.235 s | 506 | 2 | 1518 |
| Dense periodic copies | 4.408 s, 4.418 s | 506 | 2 | 1518 |

This includes device copy construction and source sorting, integration,
reductions, final downloads and output finalization. The source backend performs
1012 additional source sorts in these 506 steps. It is about 2.57 times as fast
in this short comparison; it is not a measured ten-second case speedup.
Artifacts: `output/periodic_images_solver_audit.log` and
`output/wavemaker_solver_backends_DGZ090/results.txt`.

Live-buffer accounting at startup was 65.28 MiB for queries and 169.79 MiB for
copies: approximately 104.50 MiB additional device payload for this layout and
configured source grid. These totals include particle/reorder, support, cell
grid, motion and reduction arrays, with shared arrays counted once. They exclude
host/output storage, CUDA context/JIT, allocator reservations and allocation
overhead; they are not process VRAM figures.

A subsequent one-second paired run used the final implementation, full particle
count, moving piston and normal 0.1 s output cadence, with no concurrent GPU
test job:

| Backend | Elapsed solver time | Steps | Original rebuilds | Step readbacks | Source sorts |
| --- | --- | --- | --- | --- | --- |
| Periodic queries | 144.825 s | 5217 | 43 | 15651 | 0 |
| Dense periodic copies | 47.628 s | 5215 | 42 | 15645 | 10430 |

Both completed with 463,200 exported originals, 11 frames, finite states and
zero exclusions. The final image-source count was 509,528. This longer pair is
approximately 3.04 times as fast, but its different steps/rebuilds mean it is
not an exactly matched-work comparison. The 0.1 s ABBA result above supplies
the matched original-work comparison. Neither is a ten-second timing.
Artifacts: `output/periodic_images_one_second.log` and
`output/wavemaker_solver_backends_AuTcI1/results.txt`.

Trajectory differences require attention when evaluating the full case. At
0.1 s the accepted times matched, maximum position difference was 1.16e-6 m,
and relative L2 velocity/density differences were 3.67e-5 and 1.29e-7.
At the one-second endpoints (1.0000083 s versus 1.0001218 s), periodic
minimum-image position error was 1.09 mm RMS, 0.806 mm at the 99th percentile,
and 73.8 mm maximum. Relative L2 velocity/density differences were 0.0135 and
0.000195. The initial maximum absolute position difference of 1.98 m in the
benchmark log counted seam crossings; the minimum-image analysis is in
`output/periodic_images_state_check.log`. The reusable benchmark now reports
minimum-image distances too. Small snapshot-force errors and finite states
do not establish full-run trajectory equivalence. Keep this backend experimental
while checking the ten-second flow and outliers.

Selected regression execution completed with 21,152 passing assertions:
`test/periodic_images.jl`, `test/effective_configuration.jl`,
`test/gpu_memory.jl`, `test/stationary_predictors.jl`, `test/sorting_shadows.jl`,
and `example/Wavemaker/test_wavemaker.jl` (including its periodic-bounds tests).
The new source tests cover 2D/3D, both float precisions, subdivisions 1/2,
every periodic axis, transverse shifts, seam-crossing predictor states,
moving/fixed/frozen sources and explicit invalid-grid failures.
`output/periodic_images_regressions.log` ends with the completion marker and
exit code zero. The full CPU/GPU suite was not run for this change.
