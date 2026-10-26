# Prescribed motion examples

These examples use Julia trajectory objects and generated particle blocks to
demonstrate the nine DualSPHysics motion feature families. They are kinematic
demonstrations without fluid or gravity, with independent geometry and APIs.
They do not import upstream XML or reproduce its example geometry.

Run from the repository root:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/example/Motion/Motion01.jl
```

Each script accepts optional `[output_dir] [duration_seconds] [dimensions]`.
Defaults are four seconds, 2D, and `gpu_version/output/Motion/MotionXX`.
Motion05 defaults to 3D to demonstrate arbitrary-axis rotation. Every example
supports both dimensions and writes VTKHDF particle positions and velocities.

| Script | Demonstration |
| --- | --- |
| [Motion01.jl](Motion01.jl) | Constant translation, a pause, return translation, another pause |
| [Motion02.jl](Motion02.jl) | Four ordered translation segments tracing a square |
| [Motion03.jl](Motion03.jl) | Three nested bodies with translation, rotation and local oscillation |
| [Motion04.jl](Motion04.jl) | Acceleration followed by deceleration with continuous velocity |
| [Motion05.jl](Motion05.jl) | Rigid rotation; arbitrary axis in 3D |
| [Motion06.jl](Motion06.jl) | Accelerated rotation and circular translation of separate bodies |
| [Motion07.jl](Motion07.jl) | Sinusoidal translation, rotation and circular translation |
| [Motion08.jl](Motion08.jl) | Translation loaded from [data/translation.csv](data/translation.csv) |
| [Motion09.jl](Motion09.jl) | Matching degree/radian rotations loaded from two files |

Use the shared builder or runner from another Julia script:

```julia
include("gpu_version/example/Motion/MotionExamples.jl")
using .MotionExamples
using SPHExampleGPU

case = motion_example(3; dimensions = 3)
position = motion_position(case.motion, 3, (0.5, 0.0, 0.1), 0.75)
result = run_motion_example(3; dimensions = 3, visualize = false,
    open_log_file = false, time_stepping = SingleNeighborTimeStepping())
```

The scripts use symplectic stepping and CUDA graphs by default. Set
`use_graph = false` for direct kernel execution. Their shared implementation
is in [MotionExamples.jl](MotionExamples.jl); the small numbered scripts only
select a case and command-line arguments. No dependency changes are required.

## Trajectory and frame conventions

Times are seconds, translation is metres, angular quantities and phases are
radians, and sinusoidal frequencies are Hz. 2D coordinates are the solver's
two components, conventionally `(x,z)`; positive 2D angles rotate the first
coordinate toward the second. 3D rotations follow the right-hand rule.

`KinematicSegment` sets its own initial velocity and constant acceleration.
`MotionSequence` accumulates segment displacements, with the next segment
owning a shared endpoint. A zero-velocity segment is a pause. Each sinusoidal
segment subtracts its initial sine value, including nonzero phase, so its
initial displacement is zero. A circular trajectory translates all body
points by the same orbit displacement and preserves body orientation.

`SampledTrajectory(times, values)` uses linear interpolation. Provide
`velocities` to use cubic Hermite interpolation with continuous interior
velocity. Both APIs return the exact velocity and acceleration of their
interpolant. Before the first time and at/after the final time, values are
held and rates are zero. Velocity discontinuities have no represented impulse.

Reference positions describe the zero-pose geometry. A node transforms a point
as `center + translation(t) + R(t)*(reference-center)`. A child transform acts
first and the parent transform acts next. Centers, axes and translation
directions belong to the parent's untransformed reference frame; they move
with the parent. Composition includes angular, tangential, centripetal and
cross terms in the first two derivatives. All reference geometry shares one
zero-pose frame. Use `group_marker = nothing` for virtual parents. Duplicate
IDs/markers, missing parents and cycles are rejected.

## File format

Files contain numeric data separated by whitespace, commas or semicolons.
Blank lines and `#` comments are accepted. Every row must have the same width,
with finite values, strictly increasing nonnegative times and at least two
rows. Headers require explicit `skip_rows`; columns are selected explicitly
with **one-based** Julia indices. There is no automatic axis inference.

```julia
translation = load_translation_trajectory("motion.csv";
    time_column = 1, value_columns = (2,4))  # x,z from time,x,y,z
rotation = load_rotation_trajectory("angles.csv"; angle_units = :degrees)
```

Time is rebased to `start_time` (default zero). The first position or angle is
subtracted by default, matching upstream file-displacement semantics. Set
`relative = false` to retain the file's displacement/angle origin. Angles are
converted to radians and remain unwrapped. Files use linear interpolation,
hold endpoints with zero rates, and are read once before device execution.

## Supported scope and comparison

`SimMotion` supports nonperiodic 2D/3D DBC (`NoMDBC`), both time integrators,
sorting, compact predictors, double-position storage and CUDA graph replay.
Periodic and mDBC cases fail explicitly. Existing `MotionDetails` and
`SimRotation` can drive other groups; overlapping ownership fails explicitly.
The CPU package is unchanged.

This is feature coverage, with a defined trajectory API. It is **not a claim
of complete numerical parity** with DualSPHysics. Wall velocities here are
analytical derivatives; upstream uses finite motion increments. General
simultaneous operations, incremental axis updates and hierarchy events can
also have different semantics. XML event graphs, teleportation, automatic
previous-velocity/phase inheritance, Euler/path-file motions and upstream
file-column numbering are not imported. Explicitly set segment rates/phases
and compose nodes as described above.

The [independent reference check](../../test/reference/README.md) compares
representative inputs for all nine patterns against the actual upstream
motion engine. It verifies positions and finite displacements, separately
from analytical wall velocities. It does not certify fluid-solver equivalence.
