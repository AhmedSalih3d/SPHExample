# Re = 100 at the official benchmark's nominal spacing

The [SPHERIC Test 06 reference](https://www.spheric-sph.org/tests/test-06)
uses a 600 × 300 Cartesian finite-difference grid with a level-set interface.
In the 10 × 5 m tank, this corresponds to approximately 1/60 m spacing.
This run sets the SPH particle spacing to 1/60 m (60 spacings across the
1 m square). It matches nominal spacing, not effective resolution or wall
treatment. The reference drag comes from the finite-difference calculation,
not from an SPH run at this spacing.

## Reproduce

From `gpu_version`, in a Julia session started with `julia --project=. -t 2,0`:

```julia
p = abspath("validation cases", "spheric_test_06")
include(joinpath(p, "case.jl"))
using .SphericTest06, SPHExampleGPU
case = build_case(reynolds = 100, dx = 1/60, duration = 8.0, interval = 0.02,
    sound_speed = 160.0, cfl = 0.2, kernel_h_factor = 2sqrt(2), wall_layers = 7,
    shifting = PlanarShifting, motion_kind = :integrated,
    time_stepping = SingleNeighborTimeStepping(),
    save = joinpath(p, "runs", "official_spacing_re100_replay"))
run_case(case; results_dir = joinpath(p, "results", "official_spacing_re100_replay"))
```

The run used the local compact planar-shifting buffer-check fix. It completed
8 s with 136,000 steps, 401 VTKHDF frames, 176,400 fluid particles and no
frozen particles. See the [run record](Re100_n060.toml), [drag history](Re100_n060.csv)
and [state diagnostics](Re100_n060_state.csv).

## Startup drag

The official reference was linearly interpolated onto the SPH output grid
(0.02 s). Peaks and signed trapezoidal integrals use 0.4–0.8 s inclusive.

| Metric | SPH | Official reference | Difference |
|---|---:|---:|---:|
| Pressure peak | 8.622 at 0.56 s | 8.296 at 0.56 s | +3.9% |
| Viscous peak | 0.903 at 0.68 s | 0.788 at 0.70 s | +14.6% |
| Pressure integral (s) | 2.688 | 2.532 | +6.1% |
| Viscous integral (s) | 0.2894 | 0.2435 | +18.9% |

This run is closer to the official reference on all four startup metrics
than the previously compared 80- and 160-spacing runs. That does not establish
convergence: even at matched nominal spacing, the minimum saved-frame
clearance was −11.5 mm, indicating that some fluid particle centers crossed a
nominal solid boundary.

## DualSPHysics 5.4 comparison

The [overlay](drag_comparison.png) shows the full run and startup detail for
both solvers and the official reference. DualSPHysics used the same nominal
spacing, Re, integrated motion, sound speed, CFL, smoothing-length ratio,
seven wall layers, and 0.02 s output interval. The numerical methods are
not identical: DualSPHysics used symplectic stepping, native DDT1 (0.1), and
classic capped shifting, versus Julia's single-neighbor stepping,
zero-gravity linear diffusion, and uncapped planar shifting.

To reproduce, set `DUALSPHYSICS_ROOT` to a DualSPHysics 5.4 installation and
run the following from the `gpu_version` Julia project:

```julia
p = abspath("validation cases", "spheric_test_06")
include(joinpath(p, "dual_compare.jl"))
C = Test06DualComparison
tag = "official_spacing_re100_replay"
prefix = C.prepare_dual(n = 60, reynolds = 100, sound_speed = 160,
    coefh = 2.0, wall_layers = 7, motion_kind = :integrated,
    trajectory = C.SphericTest06.body_trajectory(motion_kind = :integrated),
    tag = tag, diffusion = 1, shifting = 3)
C.verify_geometry(prefix, 60; coefh = 2.0, wall_layers = 7)
save = joinpath(p, "runs", tag, "dual")
C.dual_run(prefix, save; reynolds = 100)
C.extract_dual(save, joinpath(p, "results", tag, "dual"); square_particles = 3600)
```

DualSPHysics completed 138,071 steps with 401 frames and 192,796 particles.
Its [drag history](dual/dual.csv) was interpolated onto the same nominal
0.02 s startup grid as the other curves.

| Startup metric | Julia SPH | DualSPHysics | Official reference |
|---|---:|---:|---:|
| Pressure peak | 8.622 at 0.56 s | 9.424 at 0.52 s | 8.296 at 0.56 s |
| Viscous peak | 0.903 at 0.68 s | 0.867 at 0.68 s | 0.788 at 0.70 s |
| Pressure integral (s) | 2.688 | 2.666 | 2.532 |
| Viscous integral (s) | 0.2894 | 0.2803 | 0.2435 |

DualSPHysics predicts a higher, earlier pressure peak, but its integrated
pressure is slightly lower than Julia's. Both solvers overpredict both
startup integrals relative to the official reference.
