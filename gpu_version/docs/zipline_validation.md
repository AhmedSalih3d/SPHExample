# Native ZipLine validation and performance

Measured on 5 November 2026 with an RTX 5080, Ryzen 9 9950X3D, Julia 1.12.7
(`-t 2,0`), CUDA.jl 6.4.2 and CUDA runtime 12.9. The reference is the installed
DualSPHysics 5.4.355 GPU executable and `examples/chrono/06_ZipLine` definition.
Our simulation uses Julia/CUDA throughout; it does not load Chrono.

## Warmed complete runs

Both implementations simulated 3.5 seconds and wrote 176 frames. Native physics
uses Float32 with Float64 positions, 32 projection iterations and a 0.002 s
rigid timestep ceiling with motion bounds. The reference uses its supplied
symplectic/acoustic timestep settings and CAD mesh contacts. Native uses
particle contacts. Neither case contains fluid or periodic boundaries.

| Implementation | Seconds, run 1 / run 2 | Steps | Particles |
|---|---:|---:|---:|
| First native implementation, `bec9630` | 41.147 / 40.658 | 3,350 | 32,142 |
| Updated native implementation | 27.769 / 28.144 | 3,315 | 31,727 |
| DualSPHysics supplied case | 32.003 / 31.618 | 14,724 | 32,142 |

The user's independent warmed first-implementation run took **41.10 s**,
consistent with the first row. The updated scenario takes about **28.0 s**:
approximately 1.46 times faster than the earlier native version and 1.14 times
faster than this reference run. Reference numbers are `TSimul`; its total
times, including initialization, were 32.195 and 31.810 s. Native wall time
includes output and final downloads. CAD visualization assembly happens
after the native timer. Compilation and kernel warm-up are excluded.

The first comparison alternated native/reference solver order. Updated native
runs were repeated separately. Both updated repetitions had the same 3,315
steps, 482 grid rebuilds, 1,038 step-state readbacks, eight captured graphs,
176 output frames and identical diagnostic residuals. No particles were
lost or frozen. The remaining particle-count difference is exactly the
reference's 415 **visual cable particles**, replaced by an analytical
point-line constraint and a ParaView line; all 678 moving particles and
31,049 physical boundary particles retain their reference positions.

These are scenario timings, not an isolated, bitwise-equivalent kernel speedup.
The inverse-inertia cache changes Float32 arithmetic order; collision paths,
step counts and rebuild counts differ from the original implementation.
Different timestep controllers and contact surfaces also prevent a claim of
equal trajectories or superior physical accuracy over DualSPHysics.

## Why the first version was slower

The native constraint/contact solver runs in one CUDA block. With ten rigid
bodies, each graph colour has very few independent constraints, leaving little
parallel work for a large GPU. It originally revisited all 24 manifold slots
for every possible body pair on every iteration and repeatedly transformed
inverse inertia between body and world coordinates.

The update compacts occupied manifold slots once per stage, preserving their
order, and caches world inverse inertia for the velocity phase. Orientations
are fixed throughout that phase, so the cache remains valid. Position and
velocity loops are specialized separately. The solver remains entirely on
the GPU. Removing cable points mainly simplifies the model and display; their
1.3% share of particles does not explain the large runtime reduction.

The native step is still more expensive: about 8.43 ms versus 2.16 ms for
the reference. The rigid controller needs far fewer steps without an acoustic
restriction. These results do not imply the same advantage in fluid cases.

The log's large **Read Back State** time is a synchronization boundary that
includes waiting for queued GPU work. It is not evidence that transferring
the small step state consumes that time. Output writing took about 0.3 s
in the first implementation and was not the dominant cost.

## Mechanics checks

Across every saved frame of the updated default run:

| Check | Maximum |
|---|---:|
| Joint attachment displacement | 2.582 mm |
| Hinge-axis alignment residual | 0.00530 rad |
| Centre-of-mass velocity error along the cable before container impact | 0.0000249 m/s |
| Centre-of-mass position error along the cable before impact | 0.00000737 m |

Before container contact, internal impulses cancel and the line reaction is
perpendicular to its axis, so the mass-weighted acceleration along the line
must equal gravity's component along it. This gives an independent analytical
trajectory check.

At 64 iterations the maximum saved attachment residual fell to 1.441 mm;
runtime rose to 55.967 s. Halving the timestep ceiling at 32 iterations took
38.595 s and produced a 2.613 mm maximum residual. Iteration error and the
particle contact approximation remain relevant. Display meshes do not change
the geometry used by the contact solver.

Focused regressions cover analytical sliding, swinging hinges, internal linear
and angular momentum, support, inclined sticking, restitution, two-body impact,
fluid coupling, sorting, graph/direct equality and force-frame ownership.
The final revision test run passed native mechanics, floating output, floating
forces, sorting shadows, effective configuration, timestep diagnostics, automatic
ParaView states, temporal geometry blocks and Poiseuille tests. Compact-buffer
tests also passed during initial validation.

`stationary_predictors.jl` has two separate Float32 cached-field tolerance
failures (2 failed / 2,991 passed). They were reproduced on the isolated
published `bec9630` baseline. Their tolerances were not changed by this update.

## Geometry and force display

The renderer now uses the shared points and indexed cells copied directly
from GenCase's `CaseZipLine__Actual.vtk`. Its 137 points and 220 cells contain
the cable line, five container faces and 214 rigid-body triangles. The first
STL-based export already agreed within 0.00000006 m; copying the indexed mesh
also preserves reference topology and shared-vertex shading. Body poses drive
the surfaces. A Z-up camera avoids the previous plan-view appearance.

The generated default state contains an overview and a tracked body detail
view. Blue arrows are gravity, red arrows are combined joint/contact reactions,
and initially hidden green arrows are net force. Length is normalized for
visibility. The numeric arrays `GravityForce`, `ConstraintForce`, `TotalForce`
and their magnitudes are in newtons under **Body forces [N]**. `ConstraintTorque`
is in N m. The original `Force` field contains only SPH/spring loads and is zero
in this case; it could not show gravity or native constraint reactions.

## Reproduction

From `gpu_version/`:

```powershell
julia --project=. -t 2,0 example/ZipLine3d.jl
julia --project=. -t 2,0 test/rigid_body_dynamics.jl
julia --project=. -t 2,0 benchmark/zipline.jl output/zipline_benchmark
```

For the reference, run GenCase from a directory containing its `ragdoll/`
assets and material file, then:

```text
DualSPHysics5.4_win64 -gpu <generated-CaseZipLine-prefix> <output-directory> -svres
```

`benchmark/zipline.jl` can alternate reference runs when
`DUALSPHYSICS_ROOT` and `DUALSPHYSICS_ZIPLINE_CASE` name the installation and
generated case prefix. See the benchmark header and
[geometry provenance](../example/data/zipline/README.md).

With ParaView installed, check the generated default state and camera tracking:

```text
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_zipline.py <ZipLine3D_SingleVTKHDFStateFile.py>
```
