# Native PointAbsorber variants

`example/PointAbsorber2d.jl` implements all four `chrono/10_PointAbsorber`
configurations. Select `variant = :direct`, `:point_line`, `:spring` or
`:coulomb` in its direct `let` setup. All use the same 19,897-particle
reference fixture, original initial densities and indexed CAD. Provenance,
geometry hash, masses, forcing and PTO constants are in
`example/data/point_absorber/README.md`.

Direct heave uses the existing component restrictions. The native point-line
variant uses `PointLineJoint(...; LockRotation = true)` to retain pure heave:
the reference permits world-Z rotation, which lies outside this 2D physical
X-Z plane. Rotation-locked point-line joints currently require 2D. They do
not simulate a particle axis. The default unrestricted point-line joint
remains available for mechanisms such as Pelamis.

Spring PTO reuses the existing CUDA linear spring/damper. New
`CoulombDamper{D,T}` applies constant opposing axial force on CUDA, capped by
the finite-stage stopping impulse. It is sliding damping, not a static
friction/sticking constraint. A zero-velocity predictor uses the accepted
velocity direction when completing a stopping impulse, avoiding residual
creep. The default Symplectic scheme remains unchanged; both existing
schemes and graph/direct execution are tested. Free initial body velocities
are now also applied to particle velocities before the first interaction.

All variants retain dp = 0.01 m, Symplectic, Wendland h = 1.2 sqrt(2) dp,
artificial viscosity 0.01 with boundary factor zero, Fourtakas diffusion 0.1,
DBC and no shifting. The default duration is 10 s with 0.05 s output.

## Validation

```sh
julia --project=. -t 2,0 example/PointAbsorber2d.jl
julia --project=. -t 2,0 test/point_absorber.jl
julia --project=. -t 2,0 test/floating_springs.jl
julia --project=. -t 2,0 test/floating_initial_velocity.jl
```

Focused PointAbsorber tests pass 100 assertions, including 2D/3D and
Float32/Float64 damper laws, GPU impulse limits, stopping without accepted
velocity reversal, both schemes, graphs and all reference variants. Existing
spring and initial-velocity regressions pass another 99 assertions.
The complete CUDA suite was not run.

All four full ten-second runs complete with 201 frames, 19,897 particles
retained and none frozen. Horizontal drift stays below 9e-16 m and planar
angle is zero at every frame. Direct/native-guide heave differs by 0.622 mm
RMS and 1.70 mm maximum. Float32 atomic load reduction and the different
constraint paths do not give bitwise-identical fluid trajectories.

| Variant | Heave range from initial centre | Steps | Solver seconds |
|---|---:|---:|---:|
| Direct | −54.4 to +51.6 mm | 114,173 | 11.63 |
| Native point-line | −54.7 to +51.8 mm | 114,173 | 33.53 |
| Spring PTO | −28.6 to +20.0 mm | 114,177 | 12.24 |
| Coulomb PTO | −49.2 to +43.7 mm | 114,176 | 12.27 |

Times include output/final downloads on the RTX 5080 system described in the
FlexibleGate report, after focused warmup runs. These are different physical
variants, not an isolated optimization or matched DualSPHysics benchmark.
The simple direct restriction is cheaper than iterative native projection.
No reference speedup or experimental accuracy claim is made.

Output-sampled PTO dissipation is nonnegative: trapezoidal integration gives
27.84 J for the spring's viscous damper and 11.51 J for Coulomb damping.
The Coulomb force never exceeds 10 N; maximum spring-plus-damper force is
68.95 N. These energy estimates use accepted-state constitutive forces and
0.05 s samples; they are not exact per-step work integrals.

## Default state and forces

The default state contains a flume overview and close-up of the body, forces
and PTO. The coil/rod is drawn 0.04 m in front of the fluid for visibility;
its endpoints follow simulated heave. This depth offset is display-only.
Blue is gravity, red guide reaction and orange the combined SPH/PTO force.
Equal arrow lengths show direction. Separate offset origins prevent parallel
arrows from hiding each other.

Native-guide reactions are exported by the solver. For the direct heave
variants, ParaView calculators reconstruct the exact horizontal reaction
`(-Force_X,0)` and constant weight; raw CSV/HDF body loads are unchanged.
The guide has zero vertical reaction and balances the pitch torque. PTO
CSV/geometry fields describe constitutive forces at accepted poses; body
`Force`/`Torque` retain the last evaluated half-state combined loads.
All force units are N for a one-metre model width.

```sh
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_chrono.py output/point_absorber_spring/PointAbsorberSpring2D_SingleVTKHDFStateFile.py output/point_absorber_spring/renders 1
```
