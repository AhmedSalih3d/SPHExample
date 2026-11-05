# Native pendulum validation

The example reproduces DualSPHysics 5.4
`examples/chrono/01_Pendulum/CasePendulum_Def.xml`. It uses the existing native
CUDA joint solver and `SymplecticTimeStepping()`. There is no Chrono dependency
and no additional stepping method or integration keyword.

The reference XML selects Symplectic for SPH. Its Chrono log identifies
**Euler Implicit Linearized** for the rigid system. Selecting SPH Verlet would
not replace that rigid-body joint integrator. Our existing
`SingleNeighborTimeStepping()` also runs this case and produces the same
trajectory as the ordinary scheme: this dry example has constant external
gravity and no fluid forces to resample.

## Geometry and physical model

Both cases contain 7,852 particles: 7,200 fixed floor samples and four floating
groups of 161, 161, 161 and 169 samples, at dp = 0.05 m. The particle bounds
clip the complete container mesh to the floor patch. The four Y-oriented
cylinders start at x = z = 5, spanning y = 5 to 9. Four spherical joints have
anchors (5,5,5), (5,6,5), (5,7,5) and (5,8,5). The first connects to the fixed
world; the others connect adjacent arms. The pivot is analytical.

Body masses are 20.125, 20.125, 20.125 and 21.125 kg; full inertia tensors
come from the same discrete lattice. GenCase's initial output has the same
quantized lattice checksum as the procedural example. The supplied reference
uses collision-free `RigidAlgorithm=0`. Neither run has fluid, periodicity,
contact, springs or joint damping. `CaseDemolisher` is a separate example.

Display surfaces retain GenCase's shared vertices and polygon connectivity.
They follow the simulated quaternions and centres. The generated default
ParaView state includes a Z-up overview, a tracked close view, blue gravity
and red reaction arrows. Lengths are normalized; force arrays and CSV values
are in newtons. The original SPH/spring `Force` is zero in this dry example.

## Correction within the existing scheme

The predictor's bilateral constraint impulse represents a half-step change
in velocity. The midpoint corrector had been adding it only once to its
full-step base velocity. Scaling it by two supplies the full-step correction
and removes the leading artificial damping. The reported joint force and
torque use the same scaling. Contact-enabled systems retain their existing
impact handling. No particle integration equations or fluid-force operators
were changed.

The original collision-free update lost about 5.5% of the available
gravitational potential energy over six seconds at a 0.001 s ceiling. Its
maximum centre error against independent planar dynamics was about 1.65 m.
The corrected scheme uses Float64 physics and positions with 48 joint
iterations by default. Float32 remains selectable but accumulated projection
noise was visible in this undamped, sensitive four-link chain; its timing
advantage alone was not used to select the default.

## Independent accuracy checks

`benchmark/pendulum_reference.jl` derives the four-link planar mass matrix,
gravity and centrifugal terms from the discrete masses, centres and inertia
tensors. It integrates these equations independently with RK4 at 0.00005 s;
it does not call the CUDA constraint solver. Halving the reference step
checks convergence. Native output times and the reference executable's
actual output times are evaluated separately, avoiding an output-time
overshoot being counted as trajectory error.
The generated reference XML rounds centres and inertia components; its error
against the common ideal lattice also includes that parameter rounding.

| Configuration | Maximum centre error | Maximum energy error* |
|---|---:|---:|
| Native default, dt ceiling 0.001 s, 48 iterations | 0.8435 mm | 0.000789% |
| Native, dt ceiling 0.0005 s, 48 iterations | 0.5288 mm | 0.000157% |
| Native, dt ceiling 0.001 s, 64 iterations | 1.4144 mm | 0.000683% |
| DualSPHysics supplied configuration | 25.5185 mm | — |

\* Energy error is normalized by the gravitational potential released by
moving all arms from their initial horizontal positions to vertical, rather
than by an arbitrary zero of potential energy. Maximum native joint-position
residual is 1.74e-12 m at the default settings; motion stays in the initial
Y–Z plane. Changing timestep and iteration count changes numerical error;
in this sensitive chain, endpoint cancellation need not vary monotonically.
Joint residuals alone are not evidence of trajectory accuracy.

## Performance measurement

Measured on 5 November 2026 with RTX 5080, Ryzen 9 9950X3D, Julia 1.12.7
(`-t 2,0`), CUDA.jl 6.4.2 and CUDA runtime 12.9. The installed reference is
DualSPHysics 5.4.355 GPU with CUDA runtime 12.8, using its supplied acoustic
timestep and Chrono.
Native defaults use Float64 physics/positions, CUDA graphs and a 0.001 s
rigid timestep ceiling with motion bounds. DualSPHysics uses its supplied
mixed precision; Chrono's internal rigid solve uses doubles, while floating
velocities and saved particle positions use single precision.

Warmed timings are recorded in ignored `output/pendulum_final_benchmark`
and the final example's `output/pendulum_3d/metrics.toml`. A whole six-second
native run warms all moving-particle rebuild and graph variants before the
timed pairs. Solver order is native/reference, then reference/native.
Both write 601 frames over six physical seconds. Native wall time includes
output and final downloads; reference `TSimul` is used for comparison.
Compilation and CAD display assembly are excluded. Native avoids SPH
interaction loops in a case with no fluid and uses fewer, motion-limited
rigid steps. These are case-level timings with different timestep policies,
not an equal-step kernel comparison or a general speed guarantee.

| Implementation | Solver seconds, run 1 / run 2 | Steps | Frames |
|---|---:|---:|---:|
| Native CUDA, default settings | 20.7615 / 20.7183 | 6,816 | 601 |
| DualSPHysics supplied case | 32.5088 / 32.3909 | 36,967 | 601 |

The native run is **1.56 times faster** on this case while giving the smaller
error against the independent dynamics above. The repeated native CSVs are
byte-identical. Each native run has 925 cell-list rebuilds, 2,181 device-state
readbacks and ten captured graphs, with no frozen particles. Reference total
times including initialization are 32.6943 and 32.5951 seconds. No whole-solver
memory-savings claim is made from these timing results.

## Reproduction

From `gpu_version/`:

```sh
julia --project=. -t 2,0 test/pendulum.jl
julia --project=. -t 2,0 test/rigid_body_dynamics.jl
julia --project=. -t 2,0 test/floating_springs.jl
julia --project=. -t 2,0 test/floating_constraints.jl
julia --project=. -t 2,0 benchmark/pendulum.jl
```

For the optional reference benchmark set `DUALSPHYSICS_ROOT` to the installed
solver and `DUALSPHYSICS_PENDULUM_CASE` to a GenCase-generated case prefix,
without `.xml`. Keep `Floating_Materials.xml` in its working directory.

After running the example, the optional headless ParaView regression checks
the actual generated state at the first, middle and final frames:

```sh
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_pendulum.py output/pendulum_3d/Pendulum3D_SingleVTKHDFStateFile.py
```

An optional second argument saves three rendered PNGs. The focused checks
cover reference geometry, independent dynamics, long-run energy and joint
errors, both existing schemes, Float32 graph/direct equivalence, force
accounting, CAD export and the default state's force visibility and tracking.

All four focused Julia test files and the optional ParaView regression passed
on the measured system. The initial, middle and final rendered frames were
also checked visually. The complete CUDA suite was not run for this change.
