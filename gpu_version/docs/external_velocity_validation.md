# Native ExternalVelocity family

`example/ExternalVelocity.jl` implements the five configurations in
`chrono/12_ExternalVelocity`, in their reference order: two boxes with inline
and CSV commands, a ball and 27 blocks, a ball and 112 blocks, hauling, and
ship/propeller. Select `variant` at the top of the direct `let` setup.
`example/data/external_velocity/README.md` records the exact upstream
particle counts, lattice hashes, initial densities, masses, materials,
forcing and original indexed CAD. The example is portable and imports no Chrono.

## Native mechanics

`RigidBodyDynamics` now accepts two optional tuples:

```julia
command = RigidVelocity(20;
    Linear = VelocitySeries([0.,0.5,0.5,1.],
        [(0.,0.,1.),(0.,0.,1.),(0.,0.,0.),nothing]))
link = BodySpring(61,60; Attachment=(0.6,0.4), Anchor=(1.2,0.4),
    Stiffness=4000, Damping=0, RestLength=1)
# Use the appropriate markers and dimension for your own geometry.
# RigidBodyDynamics(Velocities=(command,), Springs=(link,))
```

Velocity keys interpolate linearly. Duplicate times are right-continuous
jumps. `nothing` releases a complete vector to dynamics until the next
finite key; the last key is held. Linear vectors use native coordinates;
angular values are native planar scalars or world XYZ vectors in 3D.
Per-component `none` combinations are not supported. In these references,
all three components are always prescribed or released together.
The second box reads the bundled upstream semicolon-separated CSV files;
its program is checked equal to the inline box program.

Programs flatten into device keys once. The existing predictor/corrector
samples them at half/full stage times. Prescribed degrees of freedom have
zero impulse mobility while the other degrees retain physical contact and
joint response. Motor reactions are included with native constraint/contact
reactions. A commanded linear body can still rotate freely, and a commanded
angular body can still translate. Initial particle velocities match the
initial motor state. Pause and permanent freezing take precedence.
The dry native travel controller also bounds future command speeds, avoiding
a large unanticipated first step at a velocity jump.

Moving-body springs apply equal/opposite axial loads and attachment torques
at the evaluated body poses. Their coupled stiffness/damping bound limits the
existing timestep. The hauling coil follows both accepted moving endpoints;
its exported CSV gives length, force magnitude and endpoint coordinates.
It is a display of the actual link, not a spring anchored to the world.

Planar frictional contacts reuse the native neighbour search, body-disjoint
pair colouring, speculative normal impulses and Coulomb tangential projection.
Eight directional/tangent bins use the common 24-slot pair storage. Candidates
and impulses stay on CUDA, including graph replay. Both existing stepping
schemes are retained. No additional global simulation integrator is added.
No motor/spring buffers are allocated when those options are absent.

## Numerical scope

These contacts act on the sampled particle geometry, whereas the original
Chrono cases use CAD mesh contacts. Displayed CAD is copied from GenCase and
does not drive collision mechanics. Frictionless sampled floors can redirect
horizontal momentum through their scalloped geometric normals; this is
distinct from Coulomb friction. The aligned-contact regression independently
checks the actual tangential impulse bound. Sample-spacing, timestep and
iteration convergence remain necessary for quantitative CAD comparisons.
No claim of reference trajectory accuracy or universal speed superiority is made.

The dry cases use the existing native travel controller with a 0.001 s
ceiling. The fluid-coupled ship uses the standard SPH adaptive timestep.
All examples retain the existing Symplectic scheme by default. Original
collision settings remain enabled for the first four variants and disabled
for ship. The ship's hinge and propeller motor remain active; fluid loads
move the hull. No periodic conditions are used.

## Validation commands

```sh
julia --project=. -t 2,0 test/rigid_velocities.jl
julia --project=. -t 2,0 test/external_velocity.jl
julia --project=. -t 2,0 test/rigid_body_planar.jl
julia --project=. -t 2,0 test/rigid_body_dynamics.jl
julia --project=. -t 2,0 test/rz_flow.jl
```

The complete CUDA suite was not run. Focused checks cover both precisions,
both existing stepping schemes, graph/direct execution, motor acceleration,
release under gravity, angular midpoint pose and reaction torque, elastic
impact momentum, floor support, Coulomb friction, moving-spring force balance
and the five exact reference configurations. The fetched RZ-flow changes are
retained and checked separately.

## Default state and forces

Each example writes its original moving CAD into the default ParaView state.
Blue arrows are gravity, red motor/joint/contact reaction and orange SPH/spring
load. Equal arrow lengths show direction; the HDF/CSV magnitude arrays retain
newtons. Arrow lengths and display origins are offset for visibility, without
changing physical application points. Pure motor torque is in the exported
`ConstraintTorque` fields; a linear force arrow does not represent a couple.
Dry cases hide the empty fluid layer and show translucent fixed CAD. The
hauling state also shows a force-coloured moving-endpoint coil in front of the floor.

## Completed runs and focused results

All five full-duration runs complete with the exact input particle counts,
no frozen particles and finite exported body states:

| Variant | Simulated seconds | Steps | Saved frames |
|---|---:|---:|---:|
| Boxes | 2.5 | 3,100 | 51 |
| Ball / 27 blocks | 4 | 6,257 | 41 |
| Ball / 112 blocks | 4 | 5,800 | 41 |
| Hauling | 4 | 4,231 | 81 |
| Ship | 10 | 343,200 | 201 |

Validation output uses 0.05 s intervals, or 0.1 s for the million-particle
wall cases; example defaults retain 0.01 s. Physics is Float32 with Float64
positions. Accepted command errors away from instantaneous jump times stay
below 3.4e-8 m/s or rad/s. Jump and release semantics are checked separately
by the focused tests. All 27 and all 112 blocks move more than one centimetre.
The hauled cube moves 4.43 m and the ship's hull moves 0.818 m in total CoM
translation. Maximum ship hinge-point error over 201 frames is 3.60e-8 m.

The hauling spring spans 0.600-2.349 m and reaches 5,397.16 N. Its default
state uses the actual full-series force range, replacing the automatic
placeholder legend. Its moving endpoints and coloured coil are verified at
first/middle/final output times. All five states pass the same checks for
original CAD, visible forces, fixed fluid speed scale and current time.

Float32 accumulation of the unwrapped planar angle gives final command-phase
errors 0.000217 rad for hauling and 0.002380 rad for ship (analytical integrals
-19.75 / 36 rad). Angular velocities match their commanded values. No
integration scheme was changed to hide these finite-precision differences.

Focused validation passes 147 motor/contact/moving-spring assertions, 41
reference-configuration assertions, 87 existing planar-mechanics assertions,
134 existing native 3D-mechanics assertions and 49 fetched RZ-flow assertions
(458 total). An obsolete rejection test was updated to verify that planar
contacts retain SPH coupling now that they are supported. Off-centre body
springs independently check both equal/opposite forces and attachment torques.

Full-case runs and focused regressions overlapped in part, so their elapsed
times are not presented as solver performance evidence. These checks establish
completion, command enforcement and mechanism behavior, not matched accuracy
or speed against the CAD-contact reference solver.
