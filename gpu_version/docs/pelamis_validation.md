# Native CUDA Pelamis validation

`example/Pelamis2d.jl` implements DualSPHysics 5.4's
`examples/chrono/04_Pelamis/CasePelamis_Def.xml`. Four buoyant segments are
connected by pitch hinges. The first attachment slides on an analytical
vertical line; it is not a particle-based axis. Fluid forces and joint
reactions are solved on the GPU using the existing 2D floating
predictor/corrector and `SymplecticTimeStepping()`. No Chrono dependency or
new time integration scheme is introduced.

## Geometry and forcing

The complete GenCase lattice is preserved at dp = 0.025 m: 26,839 fluid
particles, 1,037 fixed beach particles, 300 moving piston particles, and
655, 655, 655 and 660 particles in the four segments, totaling **30,801**.
The fixture also preserves the initial Float32 density field. SHA-256 of
sorted `(group,i,k)` rows, with origin (2.8,0,-0.1), is
`97eb50325d57d7f13706ba41ed6454477cab615fb29481c26013a321abcd91c2`.

The reference is genuinely 2D, with unit width. World X,Z map to native X,Y;
native positive rotation is minus the reference's world-Y pitch. The
display maps world `(x,y,z)` to `(x,z,-y)`, preserving handedness. Body
masses and scalar inertias come from the sampled particles at relative
weight 0.5. The original segment CAD is used for display, not as a 3D
physical model.

The bed is flat until x = 10 m, followed by a 0.15 slope. The piston uses
the reference second-order regular wave: height 0.2 m, period 1.5 s, depth
1.5 m, zero phase and no ramp. Closed-form displacement agrees with the
reference paddle CSV to 0.487 micrometres, within its printed precision.
There is no periodicity, rigid collision model, joint spring or damping.
The three hinges are at world x = 7.15, 8.45 and 9.75 m, z = 1.5 m; the
vertical guide is at x = 5.8 m.

Both configurations use Wendland, h = 1.1 sqrt(2) dp, artificial viscosity
0.01 with zero boundary multiplier, Fourtakas density diffusion 0.1, DBC,
no shifting and the SPH acoustic/force timestep controller. The native
fluid-coupled case has no custom rigid-body timestep ceiling.

## Full simulation and comparison

All runs complete 10 s and write 201 frames at 0.05 s intervals. No native
particles are frozen; the reference reports no excluded particles. Native
runs take 97,000 steps; the reference takes 97,805. Output clipping and
different reductions/controllers mean step counts are close, not identical.

Across the two timed native runs, comparing all four body centres and
pitch angles at every output time with interpolated reference CSV values:

| Measurement | Range across native runs |
|---|---:|
| Body-centre RMS difference | 0.725–0.821 mm |
| Largest body-centre difference | 1.774–2.988 mm |
| Pitch RMS difference | 0.0442–0.0717 degrees |
| Largest pitch difference | 0.193–0.424 degrees |
| Largest joint-position residual | 0.101–0.104 micrometres |
| Maximum heave from initial position | 0.0796–0.0813 m |
| Maximum absolute pitch | 13.44–13.47 degrees |
| Net vertical joint force / total weight, maximum | 0.000073–0.000092 |

The vertical-force check tests the free-heave guide and cancellation of
internal hinge forces. It does not require each body's hinge force to
have zero vertical component. Float32 atomic fluid-load accumulation
changes addition order between runs, so trajectories are not bitwise
reproducible. Graph/direct tests bound physical differences rather than
requiring identical bits. Reference XML mass/inertia printing also rounds
values. This is agreement with another discretized solver, not experimental
validation or proof of greater physical accuracy.

Measured on 5 November 2026: RTX 5080, Ryzen 9 9950X3D, Julia 1.12.7
(`-t 2,0`), CUDA.jl 6.4.2, runtime 12.9. Reference: DualSPHysics 5.4.355
GPU, runtime 12.8, supplied mixed-precision and Chrono configuration.
Native uses Float32 physics and Float64 particle positions.

| Implementation | Warmed solver seconds, run 1 / run 2 |
|---|---:|
| Native CUDA | 61.1831 / 61.6813 |
| DualSPHysics supplied case | 162.6765 / 166.0674 |

The native mean is **2.68 times faster** for this case. These are unprofiled
whole-solver runs with output, in native/reference, reference/native order,
after a full native warmup. Native wall time includes output and final
downloads; reference uses `TSimul`. Initialization, compilation, CAD
assembly and rendering are excluded. A cold reference run took 238.0 s
of solver time plus 114.9 s initialization; it is excluded from the ratio.
Native timed runs record 379–380 cell-list rebuilds, 1,853–1,868 step-state
readbacks, 12 captured graphs and no frozen particles. This is a case-level
speed comparison, not an isolated kernel optimization or memory-saving claim.

## Reproduction and output

From `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Pelamis2d.jl
julia --project=. -t 2,0 test/rigid_body_planar.jl
julia --project=. -t 2,0 test/pelamis.jl
julia --project=. -t 2,0 benchmark/pelamis.jl
```

For optional reference timings set `DUALSPHYSICS_ROOT` and
`DUALSPHYSICS_PELAMIS_CASE` to the installation and a GenCase-generated
case prefix without extension. Ordinary example runs need neither program.
`pelamis_validation(case,directory;reference_directory)` also compares
the extracted FloatingInfo CSVs and the reference wave-paddle CSV.

The focused planar-joint tests passed 86 assertions, including independent
RK4 pendulum and analytical inclined-slide references, both existing
timestepping schemes, Float32/Float64, graphs, initial velocities and
frozen poses. The Pelamis geometry/coupling/export tests passed 28 assertions.
Existing native 3D joint/contact and floating-output regressions also passed,
including 3,232 queued/synchronous snapshot assertions. The complete CUDA
suite was not run.

The default state contains an overview and tracked detail view, copied
indexed CAD, triangulated fluid speed, moving piston, time and analytical
guide. Only sampled wet beach faces are displayed. Blue arrows show gravity,
orange fluid load and red joint reactions, offset 0.55 m above each body
for visibility. Arrow lengths are equal to show direction; magnitudes in
newtons for a one-metre width remain in CSV and `FloatingBodies` arrays.
The optional ParaView regression checks geometry, colour range, force
visibility, first/middle/last frames and camera tracking.

```sh
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_pelamis.py output/pelamis_2d/Pelamis2D_SingleVTKHDFStateFile.py output/pelamis_2d/renders
pvpython --no-mpi --disable-registry --force-offscreen-rendering benchmark/render_pelamis.py output/pelamis_2d/Pelamis2D_SingleVTKHDFStateFile.py output/pelamis_2d/video_frames
ffmpeg -framerate 20 -i output/pelamis_2d/video_frames/frame_%04d.png -c:v libx264 -crf 20 -pix_fmt yuv420p output/pelamis_2d/pelamis.mp4
```

Simulation results and rendered videos remain Git-ignored.
