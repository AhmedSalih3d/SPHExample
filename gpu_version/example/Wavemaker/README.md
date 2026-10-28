# DualSPHysics 06: sinusoidal wave maker

This case reproduces the distributed DualSPHysics 5.4 `main/06_Wavemaker`
DBC setup. All case files stay together here, with geometry and simulation
settings in separate readable files.

| File | Purpose |
| --- | --- |
| `Geometry.jl` | Tank dimensions, reference particle layout and surface previews |
| `BoundarySection.csv` | GenCase's 316 fixed-wall x/z lattice cells |
| `Wavemaker.jl` | Physical constants, sinusoidal motion and solver setup |
| `Investigate.jl` | Initial-state checks, containment reports and paired timings |
| `test_wavemaker.jl`, `test_periodic_bounds.jl` | Numerical and geometry regressions |
| `Investigation.md` | Findings, measured comparisons and limitations |

The reference has 31,600 fixed, 30,000 moving and 401,600 fluid particles.
The geometry generator repeats its x/z sections over 100 y rows in `[0,2)`.
`BoundarySection.csv` preserves GenCase's rasterized skin and corner cells;
fluid and piston particles are generated from this skin. The matched layout
uses `dx = 0.02 m`. Changing resolution requires regenerating that small
section file, rather than silently changing the reference case.

| Setting | Reference setup |
| --- | --- |
| Tank | 5 m long, 2 m wide, 1.5 m high; open top |
| Beach | Flat to x = 1 m, rising to z = 1.1 m at x = 5 m |
| Initial water polygon | z = 0.75 m; sampled water z = 0.03 to 0.73 m |
| Piston | x = 0 to 0.06 m, driven in x |
| Boundaries | DBC (`NoMDBC`), periodic across y; no ghost correction |
| Physics | Gravity 9.81 m/s², viscosity 0.02, Fourtakas inner DDT2 0.1 |
| Kernel | Wendland C2, h = 1.5√3 dx, CFL = 0.2; no shifting |
| Regularization | η² = (0.1h)², matching the reference |
| Precision | Float64 positions and clock, Float32 fluid state and force coordinates |
| Sound speed | 52.4099227246139 m/s |
| Motion | Amplitude 0.25 m, initially 0.5 Hz; reference frequency schedule |
| Output | 10 s simulation, frames every 0.1 s; full timesteps may overshoot |

From `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Wavemaker/Wavemaker.jl
```

Optional CLI arguments are output directory, spacing (0.02 for the matched
layout) and physical duration. Use a new directory for each run. To avoid
opening ParaView and the log:

```julia
include("example/Wavemaker/Wavemaker.jl")
result = run_wavemaker(
    simulation_time = 2.0,
    visualize = false,
    open_log_file = false,
    save_location = "output/wavemaker_reference",
)
```

Including the source only defines functions. Geometry can be generated on the
CPU without running the CUDA solver. The run saves Fixed/Piston/Water CSVs,
a static geometry preview, an animated prescribed-motion preview and the
solver's particle data. A water-column gauge can be enabled with
`measurements = true`; the reference comparison leaves it disabled.
Surface previews show the source polygons; particle centres follow GenCase's
rasterization.

The first sine segment lasts 5 s at 0.5 Hz, using the XML phase 4.71238898
radians (approximately 3π/2). Subsequent ten-second segments alternate
0.75 Hz and 0.5 Hz with zero phase. Each segment adds its displacement to
the preceding endpoint; motion ends at 100 s. Edit `wavemaker_motion()` to
change this schedule.

The case selects `DualSPHysicsSymplectic()`: the reference predictor/corrector,
neighbour-relative timestep limits, a Float64 integration clock and
increment-averaged prescribed-wall velocities. It also matches the reference
build's fast-math policy. The existing `SymplecticTimeStepping()` and
`SingleNeighborTimeStepping()` schemes keep their CPU-reference behavior.
The new reference path currently uses direct steps, one gather lane, DBC,
Wendland C2, artificial viscosity and Fourtakas inner diffusion. Unsupported
combinations fail explicitly.
For this host-controlled reference path, the readback count records the three
criterion/displacement reductions per step; these are not full particle downloads.

The reference's position domain and density exclusion limits (700–1300 kg/m³)
are configured explicitly and checked at predictor and accepted states.
Excluded particles remain in this solver's output
with `Type = Frozen`, so their IDs and escape positions remain inspectable;
they stop contributing to interactions. Count both exclusions and penetration
among active particles when judging leakage. Exclusion alone is not a
containment improvement. `Investigate.jl` reports both quantities.

Force coordinates use cell-relative arithmetic while positions stay in double
precision. This follows the reference's Pos-Cell approach and changes rounding
relative to the older periodic absolute-coordinate path. Particle ordering,
rebuild cadence, compiler arithmetic and output times can still differ;
matching the physical setup does not imply bitwise-identical trajectories.

```sh
julia --project=. -t 2,0 example/Wavemaker/test_wavemaker.jl
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl containment 10
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl timing_coordinates
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl timing_fastmath
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl profile
```

The timing scripts warm each variant, alternate ABBA order, compare whole-solver
elapsed time and record states plus step/rebuild/readback counts. Run timings
without other GPU work. See [Investigation.md](Investigation.md) for results.
