# SPHERIC Test 06: moving square

**Status: the tested DBC setup does not pass validation.** All nine baseline
simulations completed; see the measured drag errors, resolution sensitivity,
state checks and sound-speed investigation in [REPORT.md](REPORT.md).
The source case is [SPHERIC Test 06](https://www.spheric-sph.org/tests/test-06).

The [body-following variable-resolution `SimpleMDBC` study](VARIABLE_RESOLUTION_SIMPLE_MDBC.md)
attaches a moving refinement box (`follow = 3`, $L/dp = 20 \to 40$ and $20 \to 80$)
around the translating square, diagnoses closed-tank interface pressure relaxation
and leading-corner cavitation, and documents three core `VariableResolution`
improvements (kinematic buffer advection, half-step predictor MLS buffer exchange,
reversible flux floor, and Chiron et al. 2018 `PlanarShifting` support).

The [boundary-traction verification, mDBC and spatial-refinement study](MDBC_VALIDATION.md)
establishes analytical hydrostatic/suction pressure and laminar Couette wall-shear
benchmarks, then compares `NoMDBC`, `SimpleMDBC` and `UpdatedMDBC` (with and without
`PlanarShifting`, across Re = 50/100/150, c0 = 40/160 m/s and L/dp = 20/40/60)
under the new default pairwise timestep.

The [force comparison with the new default timestep](PAIRWISE_FORCE_ACCURACY.md)
reruns both legacy and neighbor-based criteria at Re = 50/100/150 and c0 = 40/160.
It isolates the timestep change and reports raw pressure, viscous and total drag.

The follow-up [DualSPHysics comparison and proposed solution](COMPARISON.md)
quantifies noise, accuracy and warm solver timings across all three Reynolds
numbers. It includes current field images and isolated motion, diffusion,
regularization, sound-speed and precision controls. The integrated-velocity
motion control removes acceleration introduced by rounded position records;
the remaining large acoustic ringing still requires pressure/boundary work.

The [four-edge damping experiment](EDGE_DAMPING.md) adds matching 0.5 m box
sponges at strengths of 10 and 100 s⁻¹ in both solvers, across all three
Reynolds numbers. It compares raw drag, noise, reference error and mean-load
changes. Adding these momentum sinks changes the benchmark physics.

The [optional divergence-cleaning study](DIVERGENCE_CLEANING.md) tests the
paper's ψ-based velocity-divergence procedure in Julia GPU, with no edge
sponges and matched-CFL controls. It includes raw force errors, measured
divergence, field images and paired whole-solver costs.

The [160 m/s sound-speed study](SOUND_SPEED_160.md) extends both solvers to
Re = 50, 100 and 150, with smaller-timestep controls in both and optional
Julia cleaning at matched physical CFL. It distinguishes acoustic stiffness
from timestep size and includes current field images and warm solver timings.

The [timestep audit and square-pause diagnostic](TIMESTEP_AUDIT.md) examines
the current timestep estimator and measures pressure/energy relaxation during
a smooth stop, hold and restart at 40 and 160 m/s. The modified motion is
reported separately from the original benchmark.
See [implementation checks and remaining test limitations](IMPLEMENTATION_VALIDATION.md)
for the focused GPU results and the reproduced broad-suite failure.

## Matched coefh=1.2 resolution study

[`coefh_refinement.jl`](coefh_refinement.jl) retains the successful original-tank
DBC configuration: c0=160 m/s, coefh=1.2, integrated motion, CFL=0.2, delta=0.1,
no shifting, cleaning, sponge, hold or additive background pressure. It runs
Re=100 first, then Re=50 and 150, at L/dp=20/40/80 (dp=0.05/0.025/0.0125 m).
The smoothing length scales as `h=1.2*sqrt(2)*dp`; automatic wall support
requires five layers at every spacing. Neither force histories nor reference
samples are acoustically filtered.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/coefh_refinement_tests.jl"
julia --project=. -t 2,0 "validation cases/spheric_test_06/coefh_refinement.jl"
python "validation cases/spheric_test_06/plot_coefh_refinement.py"
```

Pass `100` to the runner for only the initial Re=100 study. Verified archived
n=40 controls and the Re=100/CFL=0.1 control are copied into a separate study
directory, with their original metadata intact; sources and force checksums
are recorded in `results/coefh_refinement/provenance.csv`. Missing coarse/fine
runs use the current solver, and partial particle histories are never
overwritten. This does not modify the user's manual script or supplied runs.
Archived and fresh hardware/timings are not a performance comparison.

The outputs separate raw pressure/viscous/total reference errors, sampled
startup peaks/times, 3-7 s means, central pressure-noise diagnostics, successive
resolution differences and penetration in both millimetres and dp.
Refinement changes the acoustic timestep too; the separate archived CFL
control is retained rather than attributing every difference to space.
The plotting script independently checks the exact piecewise-linear error
integrals and writes three raw all-resolution ParaView entry points under
`runs/coefh_refinement/re<Re>/paraview_force_plot/`.

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

## Optional acoustic controls

### Motion and resolution comparisons

The [fresh matched motion comparison](results/motion_c160/README.md) tests
rounded-position Hermite and integrated-velocity motion at c0=160.0 m/s in
the original tank with DBC and no sponge/cleaning. Run `motion_c160.jl` and
then `plot_motion_c160.py` to reproduce it. Integrated motion reduces the
central pressure-noise proxy by about 69%, while raw pressure error changes
from 24.86% to 20.58%; startup and mean-load discrepancies remain.

For a matched comparison of rounded-position Hermite motion and integrated
velocity at c0=160.0 m/s in the original tank, with DBC and no sponge or cleaning:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/motion_c160.jl"
python "validation cases/spheric_test_06/plot_motion_c160.py"
```

Both are fresh full-duration runs; histories, metrics and plots are stored
separately under `results/motion_c160`.

To double the spatial resolution, halve the spacing from 0.025 m to 0.0125 m:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/motion_c160.jl" 80
python "validation cases/spheric_test_06/plot_motion_c160.py" 80
```

The argument is `L/dp`, not particle spacing. In 2D this gives four times as
many fluid particles, with results saved separately under `results/motion_c160_n080`.
Both motion types retain c0=160 m/s, the original domain, DBC, and no sponge or
cleaning. The plot script also compares each fine run with its L/dp=40 history.

To return to dp=0.025 m and compare both motions at c0=400.0 m/s, use:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/motion_c160.jl" 40 400
python "validation cases/spheric_test_06/plot_motion_c160.py" 40 400
```

The second argument is sound speed; omitting it retains 160 m/s. The legacy
script filenames remain unchanged. Results use the separate `results/motion_c400`
folder, including an overlay against the 160 m/s runs at matching spacing.
The [400 m/s results](results/motion_c400/README.md) show worse full-history
pressure and viscous errors for both motions than the matched 160 m/s cases;
increasing sound speed is not a monotonic accuracy improvement here.

### UpdatedMDBC retry

The [paper audit and corrected retry](results/updated_mdbc_paper_audit_c160/README.md)
fixes a confirmed implementation mismatch: continuity must use physical wall
velocity, while mirrored ghost velocity supplies viscous no-slip treatment.
The corrected unshifted run reduces maximum penetration from 485 mm to 1.69 mm,
with no fluid centres inside the square at the final frame. Force errors remain.
Run `updated_mdbc_retry.jl --paper-audit`, then `plot_updated_mdbc_retry.py --paper-audit`
for new matched controls and images. The flag selects new output paths; the
current solver always uses the correction. The [earlier retry](results/updated_mdbc_retry_c160/README.md)
is retained explicitly as historical pre-fix evidence.

Append `40` to both commands to repeat the corrected comparison at c0=40.0 m/s:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/updated_mdbc_retry.jl" --paper-audit 40
python "validation cases/spheric_test_06/plot_updated_mdbc_retry.py" --paper-audit 40
```

Results are stored separately under `results/updated_mdbc_paper_audit_c40`,
including an overlay against the corrected 160 m/s histories.

To enable divergence cleaning in the same three corrected 40 m/s cases:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/updated_mdbc_retry.jl" --paper-audit --cleaning 40
python "validation cases/spheric_test_06/plot_updated_mdbc_retry.py" --paper-audit --cleaning 40
```

This uses kappa=10, sigma=1, cleaning Courant=0.1 and absolute-pressure decay.
Physical CFL remains 0.2 as in the off controls; cleaning's additional timestep
restriction is active. The separate `results/updated_mdbc_paper_audit_c40_cleaning`
family records the cleaning settings and includes raw on/off overlays.

Replace `40` with `160` in both commands for the same cleaning study at
c0=160.0 m/s. The [160 m/s cleaning results](results/updated_mdbc_paper_audit_c160_cleaning/README.md)
compare against the corrected 160 m/s off controls, with the same kappa=10
and all other physical and numerical settings retained. Kappa is not scaled
with sound speed.

The [startup diagnosis](results/cleaning_startup_c160/README.md) tests smaller
timesteps, kappa=160 decay, and stronger density diffusion at c0=160 m/s.
Run `cleaning_startup_control.jl --full` and `plot_cleaning_startup_control.py`
for eight startup controls and one full-history diffusion check. No tested
setting provides a convincing force improvement; the off-case CFL sweep
also reveals unresolved temporal sensitivity. Defaults remain unchanged.

### Smoothing-length review (current manual case)

The manual case now uses the original closed tank without a sponge and
`coefh=1.2`, meaning `h=1.2*sqrt(2)*dp` in 2D. The [independent coefh review](results/coefh_validation/README.md)
reproduces the supplied history exactly and confirms improved drag using
fixed-wall-layer and nominal-timestep controls. Pressure timestep convergence
and transient penetration remain unresolved. Reproduce the controls with
`coefh_validation.jl` and plot them with `plot_coefh_validation.py`; generic
builder defaults and historical studies retain their original settings.

The [all-Reynolds study](results/coefh_reynolds/README.md) extends this setup
to Re=50,100,150, compares it with stock DualSPHysics 5.4 GPU, and tests
BackgroundPressure=1 Pa against 0 Pa in Julia. Run `coefh_reynolds_study.jl`
and `plot_coefh_reynolds.py` for the combined raw-force figures. Both commands
accept `--native-only` for the Julia subset. The stock Dual comparison uses
zero background pressure; differing solver operators are documented.

The [112000 Pa study](results/background_112000/README.md) distinguishes the
DSPH FAQ's EOS stiffness B=112000 Pa (c0=28 m/s) from an additive background
pressure of 112000 Pa. Re100 controls test both interpretations; neither
improves force agreement. Reproduce with `background_112000.jl` and
`plot_background_112000.py`.

The [coefh=1.2 boundary study](results/coefh_mdbc/README.md) compares DBC,
SimpleMDBC and corrected UpdatedMDBC at Re=50,100,150 with c0=160 and zero
background pressure. Run `coefh_mdbc_study.jl`, then `plot_coefh_mdbc.py`.
It keeps shifting, divergence cleaning and sponges disabled. DBC has the
lowest full-history drag errors in these controls; the report also includes
penetration, density and final-particle checks for both mDBC modes.

### Wider sponge diagnostic (historical)

For an original-sized 10×5 m tank with internal 2 m and 0.5 m sponges at
c0=160 m/s, run `original_tank_sponge.jl`, then `plot_original_tank_sponge.py`.
See [the original-domain results](results/original_tank_sponge_c160/README.md):
both tested widths worsen the reference load comparison. The 2 m layer damps
the physical shear flow alongside the body, and the narrow layer also changes
near-wall return flow. Preserving geometry alone does not preserve the benchmark
when damping is added to its momentum equation.

`manual_case.jl` now uses `c₀ = 160.0` m/s, DBC (`boundary_mode = NoMDBC`), integrated-velocity
motion and no divergence cleaning in an expanded tank. Integrated motion means
that body displacement is obtained by integrating the supplied velocity records,
avoiding acceleration caused by rounding in the displacement records.

The editable settings are `domain_padding = 2.0`, `use_sponge = true`,
`sponge_width = 2.0` m and `sponge_rate = 100.0` s⁻¹. The tank extends from
(-2, -2) to (12, 7) m; the quadratic velocity sponge occupies the outer 2 m.
The original [0, 10] × [0, 5] m box and the entire prescribed square path are
undamped. Results go to `runs/manual_wide_sponge` and
`results/manual_wide_sponge`, separate from earlier manual runs.

This is a changed-domain diagnostic, not the official closed-tank validation.
Moving the no-slip tank walls and damping the surrounding fluid both change
the physical return flow and body loads. For the original benchmark, set
`domain_padding = 0.0`, `use_sponge = false` and choose a different `run_tag`.
Keep `sponge_width <= domain_padding` to keep damping outside the original box.

Run the three matched original-tank, expanded-tank and expanded-sponge controls:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/wide_sponge.jl" 160
python "validation cases/spheric_test_06/plot_wide_sponge.py" 160
```

The coefficient histories and metadata are saved separately under
`results/wide_sponge_c160`; the 40 m/s study remains in `results/wide_sponge`.
Omit the numeric command argument to reproduce the 40 m/s study. The manual
run uses its own `manual_wide_sponge_c160` tag to preserve previous runs.
Reference errors for the expanded cases describe their
difference from the original benchmark, not accuracy for the changed domain.

### Original tank with an internal sponge

Run `original_tank_sponge.jl` to keep the original 10 × 5 m walls and test
0.5 m and 2 m internal sponges at c0=160 m/s, rate 100 s⁻¹, DBC and integrated
motion, with cleaning off. These are separate diagnostics; they do not change
the manual example's expanded-domain preset. The 2 m sponge leaves only the
undamped rectangle (2,2)–(8,3), starting exactly at the body's top/bottom faces
and overlapping its end-of-path neighbourhoods. It therefore damps physical
shear/return flow as well as disturbances. The 0.5 m box keeps the body outside
the damping region, although nearby fluid can still enter it.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/original_tank_sponge.jl"
python "validation cases/spheric_test_06/plot_original_tank_sponge.py"
```

Results and raw plots are in `results/original_tank_sponge_c160`. These runs
reuse the original/expanded no-sponge histories from `results/wide_sponge_c160`
for comparison. Damping adds a momentum sink, so unchanged geometry alone does
not make the damped flow identical to the official benchmark.

The manual case now defaults to `motion_kind = :integrated`, integrating the
official velocity records to avoid artificial acceleration from rounded positions.
Use `:hermite` to reproduce the historical position interpolation. Both modes
support `reference_speed` and `start_delay`.

Set `boundary_mode = SimpleMDBC` in `manual_case.jl` to run ghost-point density
correction on both the tank and moving body; `NoMDBC` remains the default.
Ghosts reflect across the configured rectangular faces, including tied corners.
SimpleMDBC retains the wall density floor and does not resolve negative wall
pressure clipping. These options are diagnostic controls, not a validation fix.
Use a distinct `run_tag` for each comparison to retain separate results.

The reference defaults remain zero background pressure, no initial hold, no
divergence cleaning and no sponge. The editable manual script exposes:

- `BackgroundPressure = 101325.0` in `SimConstants` for a constant atmospheric
  offset. Keep `c₀ = 40.0`: the Tait coefficient remains
  `B = rho0*c0^2/7 = 228571.43 Pa`. Setting `Cb` to atmospheric pressure is
  not equivalent to adding this offset.
- `start_delay = 4.0` for four stationary seconds followed by the complete
  eight-second motion (`duration = 8.0 + start_delay`). The prescribed curve
  is shifted, not rescaled. Force CSVs retain simulation `time` and also store
  `motion_time = time - start_delay`; reference metrics and ParaView overlays
  use that alignment.
- `SimDivergenceCleaning = DivergenceCleaning(kappa = 10, courant = 0.1)`
  for the existing experimental velocity-divergence cleaner. Its decay and
  stability bound both include the same background pressure used in the
  momentum equation, so a large offset can require a smaller timestep.
- `SimDamping = SphericTest06.edge_sponge(layout; width = dx, rate = 100.0)`
  for one-dp fluid damping at all four tank edges. The quadratic profile uses
  the maximum face penetration at corners, not four accumulated reductions.
  It does not damp the tank or prescribed square. One dp is thinner than the
  kernel support and is not claimed to be an effective nonreflecting boundary.

Pressure output and offline square/fluid force reconstruction include the
offset. The offset must not simply be subtracted from the reconstructed SPH
pair forces: a constant pressure can create nonzero discrete residuals.
Saved diagnostics include fluid pressure mean/std/min and kinetic energy to
distinguish pressure positivity, stationary drift and noise reduction.

The reproducible Re = 100 study uses dp = 0.025 m, c0 = 40 m/s, matched physical
CFL = 0.1, cleaning kappa = 10 and a 100 s^-1 sponge where enabled. It runs
baseline, background-only, hold-only, background+hold, cleaning-only,
sponge-only, background+cleaning and the combined twelve-second case:

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/pressure_controls.jl"
```

Pass control names to run a subset, e.g. `baseline background`. Compact force
histories, spectra and `summary.csv` are written under
`results/pressure_controls/<control>/`; particle histories are ignored under
`runs/pressure_controls/<control>/`. Completed records are reused only when
their settings match. Partial histories are never overwritten.

The noise metric is the existing centered 0.5 s high-pass error RMS over motion
time 1.5–6.5 s. Raw pressure/viscous reference error, mean drag over 2–6.5 s,
stationary drift and particle-state checks are reported separately. These
modified equations are diagnostic controls, not an automatic validation pass.

### Measured Re = 100 outcome

All eight controls completed with 84,864 particles, no frozen/lost fluid or
square particles, and 401 frames for eight seconds or 601 frames with the
four-second hold. These are single-resolution controls, not a convergence
study. The measurements below use the same physical CFL = 0.1 baseline.

| Control | Pressure-noise reduction | Raw pressure NRMSE | Raw viscous NRMSE | Mean pressure-drag change |
|:--|--:|--:|--:|--:|
| Baseline | 0.00% | 46.54% | 12.73% | 0.00% |
| Atmospheric offset | 18.66% | 129.72% | 76.78% | +121.40% |
| Initial hold only | 0.00% | 46.54% | 12.73% | 0.00% |
| Offset + hold | 34.04% | 126.54% | 76.65% | +128.32% |
| Cleaning only | 16.93% | 46.56% | 18.48% | +1.14% |
| One-dp sponge only | 1.58% | 46.54% | 12.56% | +0.20% |
| Offset + cleaning | 36.45% | 129.40% | 77.46% | +122.88% |
| All controls | 42.74% | 130.40% | 76.72% | +128.33% |

**Keep the reference defaults; none of these is a demonstrated drag fix.**
The atmospheric offset keeps the minimum fluid pressure above 97.1 kPa and
removes the measured particle-center penetration in these runs, but its
discrete pressure forces more than double the mean drag. Subtracting the
offset from the reported force would hide that effect rather than repair it.
The offset also produces stationary fluid speeds up to 0.0833 m/s during the
hold (0.0881 m/s with all controls); the zero-offset hold remains exactly at
rest and only delays the subsequent force history.

Cleaning alone lowers the central-window noise but increases peak density
departure from 0.814% to 2.268%, peak Mach from 0.0586 to 0.1125, and viscous
reference error. Its unchanged pressure NRMSE is not a validation pass.
The one-dp sponge has only a small effect and leaves particle-center
penetration unresolved. The combined case is quietest by this metric but has
large mean-load/reference errors. Positive pressure, smoother drag and a
completed simulation are distinct checks.

Each control has a generated
`runs/pressure_controls/<control>/paraview_force_plot/ForceComparison_Re100.py`
state. Load it in ParaView to see the particles, raw force curves, correctly
shifted reference, time cursor and automatic screenshot export.
Measurements used Julia 1.12.7 / CUDA.jl 6.4.2 on an RTX A1000 Laptop GPU,
Float32 physics with Float64 positions and Float64 offline force sums.
Saved wall times include cold/warm setup differences and are **not** paired
performance evidence; no speedup is claimed.

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

### Manual editable case

[`manual_case.jl`](manual_case.jl) is a directly runnable setup in the style of
[`Bathymetry2d.jl`](../../example/Bathymetry2d.jl). Its top-level `layout`
controls tank origin/size and body center/size. Like the bathymetry example,
the file exposes `SimConstants`, the `Tank`/`Water`/`Square` geometries,
`SquareMotion`, `SimMetaData`, the kernel/viscosity/diffusion/time-stepping models,
particle allocation and the direct `RunSimulation` call.
Edit physical values in `SimConstants`, output/GPU options and shifting in
`SimMetaData`, and the adjacent resolution, reference-scale and timing settings.
Cell-centered sampling and force analysis remain shared with the batch case.
Tank and body edges must align with `dx`, and the default
layout and physics reproduce Re = 100, L/dp = 40, c0 = 40 m/s. The benchmark's
2D NoMDBC boundary treatment remains fixed. Change `run_tag` to keep repeated
runs in separate folders.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/manual_case.jl"
```

After force analysis finishes, the script writes a Python state and, with
`visualize_in_paraview = true`, opens it directly in ParaView. ParaView's own
Python API builds the particle view above native, interactive pressure and
viscous drag charts, overlaid on the official finite-difference reference,
then saves the combined `.pvsm` state and PNG using `SaveState` and
`SaveScreenshot`. The manual simulation does not launch or require `pvpython`.
Both charts include a dashed vertical time cursor and an orange dot on the SPH
curve, synchronized with ParaView's animation time. When several spacings are
plotted, the dot follows the finest spacing; between saved force samples it
uses linear interpolation without filtering the history.
The force-plot script plots every available spacing in the selected results directory;
the manual run restricts that plot to its own spacing. Open
`validation cases/spheric_test_06/runs/manual_layout/paraview_force_plot/ForceComparison_Re100.py`
in ParaView (**File → Load State**) to create the charts and save the image,
or open the resulting `.pvsm` to inspect the saved layout.
Set `visualize_in_paraview = false` to write only the Python state without
opening the GUI; its `.pvsm` and PNG are generated when you load it in ParaView.
The plotted histories are unfiltered;
the baseline run's 0.1 s binned comparison remains available in
`results/validation/binned.csv`. When you change the benchmark layout or
physical scales, the official force curve is still shown for context, but the
result is no longer an apples-to-apples SPHERIC validation.

The saved Python state can be loaded again without rerunning the solver.
For optional headless regeneration (separate from the manual simulation),
`pvpython` can still plot all three Reynolds numbers in the baseline results:

```sh
pvpython --no-mpi --force-offscreen-rendering "validation cases/spheric_test_06/paraview_force_plot.py" --results "validation cases/spheric_test_06/results/validation" --re all
```

Use `--re 100 --n 40` to limit the chart to a single Reynolds number and
resolution, or `--component pressure` / `--component viscous` for one chart.
Add `--simulation-state "validation cases/spheric_test_06/runs/manual_layout/Re100_n040/MovingSquare_SingleVTKHDFStateFile.py"`
with `--re 100 --n 40` to regenerate the combined particle-and-chart state.
The default output directory is ignored `runs/paraview_force_plot/`.

Test the combined layout, saved-state reload and preservation of raw force
samples without Julia or a CUDA GPU:

```sh
pvpython --no-mpi --force-offscreen-rendering "validation cases/spheric_test_06/test_paraview_force_plot.py"
```

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
