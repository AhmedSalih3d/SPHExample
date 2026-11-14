# SPHERIC Test 06: moving square

**Status: the tested DBC setup does not pass validation.** All nine baseline
simulations completed; see the measured drag errors, resolution sensitivity,
state checks and sound-speed investigation in [REPORT.md](REPORT.md).
The source case is [SPHERIC Test 06](https://www.spheric-sph.org/tests/test-06).

The follow-up [DualSPHysics comparison and proposed solution](COMPARISON.md)
quantifies noise, accuracy and warm solver timings across all three Reynolds
numbers. It includes current field images and isolated motion, diffusion,
regularization, sound-speed and precision controls. The integrated-velocity
motion control removes acceleration introduced by rounded position records;
the remaining large acoustic ringing still requires pressure/boundary work.

## Case definition

The closed tank occupies `[0,10] × [0,5]` m. A unit square initially occupies
`[1,2] × [2,3]` m, centered at `(1.5,2.5)` m. Fluid starts at rest with zero
pressure and no gravity. Tank and square walls are specified as no slip. The
square follows the official `Motion_Body.dat` history in +x through 8 s.
Its last column is **absolute center x**, so the implementation subtracts
1.5 m once. Cubic Hermite interpolation matches both displacement and velocity
at each supplied knot. The table's acceleration column is retained in the
fixture; it is not independently imposed on the fluid or used as an additional
motion constraint. Interpolated acceleration is the derivative of the same
Hermite curve. The tabulated displacement has finite decimal precision.

The nominal scales are Umax = 1 m/s, L = 1 m, transverse depth b = 1 m and
rho0 = 1000 kg/m³. Kinematic viscosity is `Umax*L/Re`: 0.02, 0.01 and
1/150 m²/s for Re = 50, 100 and 150. This is laminar Newtonian flow; no SPS
or artificial-viscosity model is added.

## Numerical setup

All simulation and analysis logic is Julia and uses the existing GPU solver.
`plot.py` is an optional matplotlib figure renderer.

- Uniform cell-centered lattices at dp = 1/20, 1/40 and 1/60 m. The solid/fluid
  interface is halfway between adjacent lattice rows. Four tank-wall layers
  cover the kernel support; the square is filled with solid particles.
- Wendland C2, h = sqrt(2) dp, support radius H = 2h, mass = rho0 dp².
- Existing dynamic boundary conditions (`NoMDBC`), with prescribed solid
  velocity in the laminar interaction. This approximates the specified no-slip
  surface; exact surface no slip is not established by assigning wall velocity.
- Symplectic stepping; CFL = 0.2; c0 = 40 m/s for the baseline study.
  Zero-gravity linear density diffusion, delta = 0.1. No particle shifting.
- Float32 density, velocity and force arithmetic with Float64 positions/clock.
  Both cell subdivisions are 1; radial arithmetic and fast math are disabled.
- Output every 0.02 s, with timesteps clipped to those deadlines and 8 s.
  The CUDA graph path is used. Fixed/prescribed boundary accelerations are not
  evaluated because their trajectories do not depend on fluid forces.

`analyze_history` recomputes the instantaneous pressure and laminar pair
forces from each accepted saved state. Only **square/fluid** pairs enter the
load; tank and internal square forces are excluded. The pair formulas match
the solver and are tested against the GPU kernel in both precisions. The
analysis sums in Float64. Positive drag resists the +x motion:

`Cd_component = -2 * sum(F_square_x_component) / (rho0 * Umax² * L * b)`.

No inherited acceleration, total-force subtraction, body mass/inertia, or
unsteady added-mass correction enters the force measurement. The three-column
ASCII files contain time, pressure Cd and viscous Cd in the requested order.

## Comparison criteria

The source provides finite difference / level set reference solutions and
warns of high-frequency numerical force oscillations. The comparison integrates
both histories over the same 0.1 s bins, from 0.1 to 8.0 s. Raw SPH curves and
reference samples are preserved. The plot and report show both transient and
late-time behavior.

These are **project choices**, not official SPHERIC pass thresholds:

- At the finest spacing, at most 10% normalized RMS error in **each** drag
  component over the full comparison interval.
- At most 5% relative RMS change between the two finest spacings in each
  component, normalized by the reference RMS.
- All frames finite, no lost or frozen particles, prescribed center error
  below 1e-7 m and velocity error below 1e-6 m/s, density variation below 1%,
  peak Mach below 0.1, and no particle-center penetration into nominal solids.

The `drag_pass`, `state_ok` and `resolution_pass` columns are separate checks.
A passing low-resolution component alone cannot establish validation.
`REPORT.md` requires all nine completed runs; a short smoke run is explicitly
marked incomplete by the comparison code.

## Reproduce

Run from `gpu_version/`, using Julia 1.12 and an NVIDIA CUDA GPU. Quote paths
because the requested parent folder contains a space. No dependency changes
are needed.

```sh
# Focused tests, including an end-to-end CUDA smoke run.
julia --project=. -t 2,0 "validation cases/spheric_test_06/test.jl"

# Nine complete 8-second runs: all Reynolds numbers, all three spacings.
julia --project=. -t 2,0 "validation cases/spheric_test_06/run.jl"

# Diagnostic sound-speed study; all other Re=100, dp=1/60 settings are retained.
julia --project=. -t 2,0 "validation cases/spheric_test_06/run.jl" --dx 0.016666666666666666 --re 100 --sound-speed 80 --tag sound_speed_80
julia --project=. -t 2,0 "validation cases/spheric_test_06/run.jl" --dx 0.016666666666666666 --re 100 --sound-speed 160 --tag sound_speed_160

# Recompute all metrics and the report from saved small coefficient histories.
julia --project=. -t 2,0 "validation cases/spheric_test_06/report.jl"

# Optional plots: Python with numpy/matplotlib; --fields also needs h5py.
python "validation cases/spheric_test_06/plot.py" --fields
```

For a short independent run:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/run.jl" --dx 0.05 --re 100 --duration 0.2 --tag smoke
```

Verified reference fixtures and compact numerical results/figures are tracked.
Large particle histories under `runs/`, original downloaded archives and logs
are Git-ignored. About 15 GB of local particle histories are generated by the
baseline study. Completed runs resume from their `.toml` records only when
spacing, duration, output interval and sound speed match. A different setup
requires a new `--tag`; partial particle histories are never overwritten.
The usual `test/runtests.jl` also includes the focused case tests.
