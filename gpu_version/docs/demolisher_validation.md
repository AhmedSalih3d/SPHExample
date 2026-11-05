# Native CUDA demolisher validation

`example/Demolisher3d.jl` implements DualSPHysics 5.4
`examples/chrono/01_Pendulum/CaseDemolisher_Def.xml`: the four-arm pendulum
with a spherical wrecking ball and sixteen independently moving blocks.
It uses the existing `SymplecticTimeStepping()` and native CUDA joints and
contacts. No Chrono library is imported. `CasePendulum` remains a separate,
collision-free example with its existing layout and settings.

## Reference geometry and materials

The complete quantized lattice matches GenCase: **91,319 particles** at
dp = 0.05 m. The fixed open container has 67,541 samples, the arms have
161, 161, 161 and 160, the ball has 1,839, and each block has 1,331. SHA-256 of
the sorted `(group,i,j,k)` rows, using group = Mk−9 and lattice origin
(−0.5,−0.5,−0.5), is
`4b3effa637621aa835872ded5408e69e38f6cf6a163bddcc23462debad7d415c`.

The container spans x = 0–11 m, y = 2–8 m, z = 0–3 m and has no top face.
Arms start along +X at y = 5 m, z = 5.2 m. The analytical world pivot is
(5,5,5.2), and spherical joints at x = 6, 7, 8 and 9 connect the arms and
ball. The ball CAD radius is 0.35 m and its initial centre is (9.35,5,5.2).
Blocks are 0.5 m cubes in three staggered rows of five, six and five.

Body masses/inertias are calculated from the actual sampled particles. The
reference's relative weights are 1.2 for arms/blocks and 2.5 for the ball:
arm masses 24.15, 24.15, 24.15 and 24 kg; ball mass 574.6875 kg; block mass
199.65 kg each, before printed-XML rounding. A material named steel sets
contact coefficients; it does not replace those specified masses.

Only the ball and blocks have collision meshes in the reference XML. Arms
have no `modelfile` and do not collide, as confirmed by the reference log.
Native `FloatingGroups` therefore enables only the ball and blocks; the
arms' joints and rigid motion remain active. Container contacts use
`FixedGroups=(11,)`. Steel input coefficients are friction 0.45 and
restitution 0.8; soft wood uses 0.75 and 0.5. `RigidMaterial` assigns steel
to the ball while blocks retain wood defaults. Pair coefficients use the
minimum; the fixed container retains the steel coefficients.

There is no fluid, periodicity, spring or joint damping. The reference uses
Chrono NSC CAD contacts and a half-dp collision-overlap allowance. Native
contacts use the sampled particles at a full-dp separation and a half-dp
search margin, fitting the reference kernel support. These are different
contact models: identical collision trajectories and superior contact
accuracy are not claimed.

## Full impact and stability

Native defaults use Float32 physics, Float64 positions, 32 iterations,
a motion-limited 0.001 s timestep ceiling and CUDA graphs. Both solvers run
three seconds and write 301 frames. The native case completes 3,003 steps;
the supplied reference uses 17,320 acoustic steps. All 91,319 particles
remain active; none are frozen or excluded.

| Check | Native | Reference |
|---|---:|---:|
| First block horizontal speed above 0.3 m/s | 1.16 s | 1.160013 s |
| Blocks displaced horizontally by more than 0.1 m at 3 s | 12 / 16 | 13 / 16 |
| Maximum joint-position residual | 0.0732 mm | — |
| Maximum rigid-body centre speed | 9.205 m/s | — |
| Minimum block CAD corner height above floor | 0.0353 m | — |
| Maximum sphere/box CAD overlap | 0 m | — |
| Block horizontal drift before 0.5 s | 0.199 mm | — |

Native total mechanical energy has no significant growth: its largest
increase is 6.74e−8 of initial energy. Impact and friction dissipate energy;
this is not an undamped energy-conservation test. The sphere/box overlap
check uses the 0.35 m sphere and each oriented 0.5 m cube, independently of
the particle contact search. The floor check transforms all eight corners
of each block at every output frame. These checks detect gross penetration
and instability; they do not establish contact-surface convergence.

At a 0.0005 s ceiling, a 1.8 s impact run retains the 1.16 s onset, reduces
the maximum joint error to 0.0140 mm and early horizontal drift to 0.129 mm,
and still has no sphere/box overlap or floor penetration. Post-impact block
paths are sensitive to discretization and timestep; this check establishes
stability, not identical debris trajectories.

## Measurements

Measured on 5 November 2026 using RTX 5080, Ryzen 9 9950X3D, Julia 1.12.7
(`-t 2,0`), CUDA.jl 6.4.2 and CUDA runtime 12.9. Reference executable:
DualSPHysics 5.4.355 GPU, CUDA runtime 12.8, with its supplied mixed precision
and Chrono configuration. Native uses Float32 velocities/forces and Float64
positions; the reference saves particle positions in Float32.

`benchmark/demolisher.jl` warms the complete native path and runs unprofiled
native/reference, reference/native pairs. Native wall time includes output
and final downloads; reference comparison uses `TSimul`. Initialization,
compilation and CAD visualization assembly are excluded. The first reference
launch spent about 103 seconds preparing its collision geometry; subsequent
launches reused its preparation cache. A cold start must not be mistaken
for solver time.

Native timing records include 363 cell-list rebuilds, 766 step-state
readbacks, eight captured graphs and no frozen particles. Particle counts,
dimensionality, frames and input geometry match; step counts and contact
representations differ. No whole-solver memory-saving claim is made.

| Implementation | Warmed solver seconds, run 1 / run 2 |
|---|---:|
| Native CUDA | 51.7408 / 51.4169 |
| DualSPHysics supplied case | 52.0233 / 52.1418 |

These are comparable times, with no substantial performance advantage
established. A prototype that skipped empty contact-colour groups preserved
exact final body/particle states in A/B/B/A runs. It reduced time relative
to its own disabled branch, but its bookkeeping offset the gain against
the original solver. That prototype was not retained. No new integration
scheme or weaker iteration/timestep setting was introduced to improve timings.

## Reproduction and video

From `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Demolisher3d.jl
julia --project=. -t 2,0 test/rigid_body_dynamics.jl
julia --project=. -t 2,0 test/demolisher.jl
julia --project=. -t 2,0 test/pendulum.jl
julia --project=. -t 2,0 benchmark/demolisher.jl
```

For optional reference timings set `DUALSPHYSICS_ROOT` and
`DUALSPHYSICS_DEMOLISHER_CASE` to the installation and a GenCase-generated
case prefix without extension. Keep its generated geometry files and
`Floating_Materials.xml` beside the case. Ordinary example runs need neither
GenCase nor DualSPHysics.

The default ParaView state uses the original indexed GenCase surfaces,
coloured blocks, a grey ball, simulation time and a tracked Z-up detail
view. Particle points are hidden. The detail view displays the ball's blue
gravity and red combined joint/contact reaction arrows. Their lengths are
normalized; all 21 bodies' force magnitudes in newtons remain in the CSV and
`FloatingBodies` arrays. The original SPH/spring `Force` is zero in this dry
case. At time zero a joint/contact reaction has not yet been evaluated.

```sh
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_demolisher.py output/demolisher_3d/Demolisher3D_SingleVTKHDFStateFile.py output/demolisher_3d/renders
pvpython --no-mpi --disable-registry --force-offscreen-rendering benchmark/render_demolisher.py output/demolisher_3d/Demolisher3D_SingleVTKHDFStateFile.py output/demolisher_3d/video_frames
ffmpeg -framerate 100/3 -i output/demolisher_3d/video_frames/frame_%04d.png -c:v libx264 -crf 20 -pix_fmt yuv420p output/demolisher_3d/demolisher.mp4
```

The optional ParaView test checks first/middle/final poses, original mesh
counts, force visibility and camera tracking. The focused Julia regressions
cover group filtering, unequal-material rebounds, reference lattice/masses,
graph/direct equivalence and a full linked-ball impact. Existing pendulum,
ZipLine and contact tests are included. The complete CUDA suite is not part
of this focused run.

The focused Julia files and optional ParaView regression passed on the measured
system. The video was rendered from the simulated poses, visually checked,
and verified as a 1600×900 H.264 MP4 with 101 frames over 3.03 seconds. It is
saved locally at `output/demolisher_3d/demolisher.mp4`; simulation results and
video frames remain Git-ignored.
