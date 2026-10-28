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
