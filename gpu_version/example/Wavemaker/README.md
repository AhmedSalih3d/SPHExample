# DualSPHysics 06: sinusoidal wave maker

This case reproduces the distributed DualSPHysics 5.4 `main/06_Wavemaker`
DBC setup. All case files stay together here, with geometry and simulation
settings in separate readable files.

| File | Purpose |
| --- | --- |
| `Geometry.jl` | Tank dimensions, reference particle layout and surface previews |
| `BoundarySection.csv` | GenCase's 316 fixed-wall x/z lattice cells |
| `Wavemaker.jl` | Runnable case entry point for the command line or Julia/VS Code |
| `WavemakerCase.jl` | Editable physical setup; final call runs the full case with periodic copies |
| `WavemakerSetup.jl` | Loads the case definitions without starting a simulation |
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
| Search | Current case uses `cell_subdivision = 2`; use 1 for the earlier reference comparisons |
| Regularization | η² = (0.1h)², matching the reference |
| Precision | Float64 positions and clock, Float32 fluid state and force coordinates |
| Sound speed | 52.4099227246139 m/s |
| Motion | Amplitude 0.25 m, initially 0.5 Hz; reference frequency schedule |
| Output | 10 s simulation, frames every 0.1 s; full timesteps may overshoot |

From `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Wavemaker/Wavemaker.jl
```

The default full run now enables the periodic-copy backend and writes to
`output/wavemaker_periodic_copies`. Running `WavemakerCase.jl` in Julia/VS Code
also enables it explicitly in the final call. Restart Julia after updating the
package so an already-loaded module cannot retain the old implementation.
The resolved configuration must show `periodic_images = true` and
`pair_coordinates = periodic_images_cell_relative`.

Optional CLI arguments are output directory, spacing (0.02 for the matched
layout), physical duration and backend (`images` by default, or `queries`).
For example, from `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Wavemaker/Wavemaker.jl output/wavemaker_images_test 0.02 10 images
```

Running or including `Wavemaker.jl` starts the
case; includes use the case defaults and ignore the surrounding Julia process's
arguments. Use a new directory for each run. For a customized run, including
one without opening ParaView and the log, load the definitions-only file:

```julia
include("example/Wavemaker/WavemakerSetup.jl")
result = run_wavemaker(
    periodic_images = true,
    simulation_time = 2.0,
    visualize = false,
    open_log_file = false,
    save_location = "output/wavemaker_reference",
)
```

`WavemakerSetup.jl` loads definitions only; `WavemakerCase.jl` runs its final
editable call. `Wavemaker.jl` remains the entry point for CLI arguments.
Geometry can be generated
on the CPU without running the CUDA solver. The run saves Fixed/Piston/Water CSVs,
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
The experimental `periodic_images = true` backend builds a separate dense source buffer containing
original particles and near-face periodic copies for each predictor/corrector
force evaluation. Its ordinary cell-relative neighbor loop avoids translated
periodic queries. Copy generation and sorting run on the GPU. Originals alone
are integrated, used for timestep reductions and exported. This backend needs
finite domain bounds and a period larger than twice the support radius;
query caching cannot be enabled with it. It uses more device memory and changes
neighbor summation order, so trajectories need not be bitwise identical.
The measured one-second pair took 144.8 s with queries and 47.6 s with copies;
step counts differed by two. Its 99th-percentile periodic position difference
was 0.8 mm, with a 7.4 cm maximum outlier. Full-run trajectory equivalence is
not established; inspect the flow as well as elapsed time. See the investigation
for matched-work short timings, state differences and memory cost.
The package-level scheme and `run_wavemaker` function retain `false` as their
default; the two runnable case entry points explicitly select `true`.
With a custom `time_stepping`, set `periodic_images` on that scheme.
`SimulationResult.diagnostics.periodic_source_rebuilds` reports the additional
source sorts, separately from original-particle rebuilds. `periodic_source_count`
reports originals plus copies in the last evaluated state.
The opt-in `cache_periodic_queries = true` moves periodic search-limit calculations into a
preparation kernel for each evaluated state. The interaction visits the same
rows and pairs in the same order. The extra device buffer costs 72 bytes per
particle in 3D (31.81 MiB for this layout). The default `false` omits it; with the
current compiler options enabled it did not improve the measured solver time.
With an explicit `time_stepping` object, select the option on
`DualSPHysicsSymplectic(cache_periodic_queries = true)` instead.
The case's `interaction_always_inline` and `interaction_fastmath` keywords now
reach the reference interaction launcher. These compiler settings can change
rounding and need not improve elapsed time.
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
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl timing_compiler
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl timing_queries
julia --project=. -t 2,0 example/Wavemaker/Investigate.jl profile
```

The timing scripts warm each variant, alternate ABBA order, compare whole-solver
elapsed time and record states plus step/rebuild/readback counts. Run timings
without other GPU work. See [Investigation.md](Investigation.md) for results.

The performance audit in [Investigation.md](Investigation.md#2026-10-28-settings-and-execution-audit)
also distinguishes the current case from the differently configured StillWedge
example. Periodic seam crossings measure displacement before wrapping and
report a separate cell-anchor invalidation flag. A rebuild is still required
on wrapping to keep the current global search bounds tight.

For the force-only comparison of translated queries and stored periodic
copies, using an existing full-case output:

```sh
julia --project=. -t 2,0 benchmark/wavemaker_periodic_images.jl output/wavemaker/Wavemaker.vtkhdf 1,51
```

Frame numbers are one-based; 1 and 51 are the initial and approximately
five-second frames with 0.1 s output. The benchmark compares particle-ID-matched
forces and sampled neighbor identities. It excludes copy maintenance, sorting
and integration from timings; it does not measure solver speed. See the
[full-run audit](Investigation.md#2026-10-29-full-run-and-periodic-representation-audit)
for results and remaining validation.

To measure the implemented backend, including copy generation, source sorting,
integration, reductions and output, use the whole-solver comparison:

```sh
julia --project=. -t 2,0 benchmark/wavemaker_solver_backends.jl 0.1 4
julia --project=. -t 2,0 test/periodic_images.jl
```

The comparison warms both variants, runs ABBA order and writes each run into a
new output directory. Saved initial/five-second frames, when available, are
also checked against the query backend by original particle ID.
