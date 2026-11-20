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
