# SPH Example

A Julia implementation of a basic Smoothed Particle Hydrodynamics (SPH) solver. The code is primarily written for educational purposes and contains several ready to run examples such as 2D and 3D dam breaks or wedge impacts. Custom particle setups can also be loaded from the `input/` folder.

Please consider giving the project a :star: if it is useful in your work. Citation is appreciated for academic use and feel free to reach out with feedback or questions.

The code can produce a 2D dam break ([@DamBreak2D-Video](https://www.youtube.com/watch?v=7kDVjZkc_TI)):

https://github.com/user-attachments/assets/a0070389-e2a5-4bf8-9eda-e40364eea7ce

Or, if you are really patient (1+ day to calculate), a 3D case ([@DamBreak3D-Video](https://www.youtube.com/watch?v=_2e6LopvIe8)):

https://github.com/user-attachments/assets/a38aaf39-3cf3-4041-b983-03f6107de8b9

## Description


The project demonstrates how to assemble a small SPH solver with Julia. It focuses on clarity rather than ultimate performance. Unofficial benchmarks suggest that for 2D cases the CPU runtime is comparable to DualSPHysics. Key features include:

- **Weakly compressible formulation** – density varies ~1 % and pressure is a function of density.
- **Multi-threaded execution** – achieved by spawning the neighbour loop.
- **Per-particle compute loops** – per-particle loops handle threading without chunk
  metadata.
- **Dynamic boundary condition** – inspired by DualSPHysics.
- **Density diffusion** – based on Fourtakas et al. 2019 to reduce pressure noise.
- **Wendland quintic kernel** – simple and stable without tensile corrections.
- **Symplectic time stepping** – choose between symplectic two-loop and single-loop midpoint updates.

Time-stepping behavior is selected via `RunSimulation(..., SimTimeStepping=...)` with either
`SymplecticTimeStepping()` or `SingleNeighborTimeStepping()` depending on the desired update path.
`Laminar()` and the laminar part of `LaminarSPS()` use the viscosity pair
denominator `(ρᵢ + ρⱼ) * (d² + η²)`, following the
[DualSPHysics formulation](https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation).
This corrects an earlier addition in that denominator and changes results for
simulations that select either viscosity model.

## Folder Structure

```
.
├── example/          # Ready to run simulations
├── input/            # Pre-generated particle layouts (.csv)
├── src/              # Package source code
├── images/           # Images used in this README
├── Project.toml      # Package dependencies
└── Manifest.toml     # Exact dependency versions
```

Example scripts live in `example/` and read geometry from `input/`. The solver code itself is in `src/`:

```
src/
├── AuxiliaryFunctions.jl            # Small helper utilities
├── OpenExternalPrograms.jl          # Convenience wrappers for logs and ParaView
├── PreProcess.jl                    # Load inputs and allocate arrays
├── ProduceHDFVTK.jl                 # Write simulation data in HDF5/VTK format
├── SPHCellList.jl                   # Custom neighbour search and time stepping
├── SPHDensityDiffusionModels.jl     # Density diffusion implementations
├── SPHExample.jl                    # Glue module re-exporting all functions
├── SPHKernels.jl                    # SPH kernel definitions
├── SPHViscosityModels.jl            # Viscosity models such as Laminar or SPS
├── SimulationConstantsConfiguration.jl  # User-facing solver parameters
├── SimulationEquations.jl           # Core SPH physics equations
├── SimulationGeometry.jl            # Domain and geometry definitions
├── SimulationLoggerConfiguration.jl # Logging helpers for timer outputs
├── SimulationMetaDataConfiguration.jl  # Metadata such as run time and output path
└── TimeStepping.jl                  # Controls for Δt and CFL condition
```

## Getting Started

### Installation

The easiest way to experiment with the code is to clone the repository and activate it in Julia:

```julia
using Pkg
Pkg.activate("/path/to/SPHExample")
Pkg.instantiate()
```

Alternatively, install it directly:

```julia
using Pkg
Pkg.add(url="https://github.com/AhmedSalih3d/SPHExample")
```

### Running an Example

Open one of the files in `example/`, for instance `example/StillWedgeMDBC.jl`,
and adjust the simulation parameters or the `ComputerInteractions!` function.
Run the script to start the simulation. Results are written in `hdfvtk` format
which can be loaded with ParaView 5.12 or newer. Output is written
asynchronously, so files finish flushing when the simulation completes.
Pressing `Ctrl+C` during `RunSimulation` also flushes and closes the active
VTKHDF output, finalizes the `.log` file, and opens the data written so far in
ParaView.
To color exported cell grids by particle counts, set
`ExportGridCellParticleCounts=true` in `SimulationMetaData`. This also adds a
`ParticleNeighborsPerCell` array that includes each cell's particle count minus
one plus the particles in its neighbor stencil.

### Performance Diagnostics

Neighbor rebuilds use a stable index permutation and reusable `NeighborSortScratch`
buffers, moving each particle record once instead of repeatedly moving all its
fields during sorting. Equal-cell order is preserved, including optional MDBC
and kernel-output fields. Interaction helpers also propagate the neighbor loop's
bounds-check guarantee; direct checked calls retain bounds checks. These changes
preserve the force formulas, support radius, and integration settings.
The Wendland C2 gradient also uses the kernel's precomputed inverse smoothing
length, avoiding a division for each accepted particle pair. Linear density
diffusion skips boundary pairs before evaluating its hydrostatic correction and
distance reciprocal. In the measured solver cases, these changes preserved the
timestep sequence and changed final fields only by floating-point roundoff.

The particle and MDBC loops distribute contiguous batches of 64 particles using
a shared atomic counter, so workers can take more work when they finish a batch.
Single-threaded runs and small inputs use a serial path. Each particle retains
its neighbor summation order and owns its output writes.

Three-dimensional runs with `NoShifting` and `NoKernelOutput` also cache
particle candidates within `1.125H`. Before every force evaluation, including
predictor states, the solver checks displacement from copied reference positions
and rebuilds this cache if any particle has moved more than `H/32`. Two particles
can therefore close by at most `H/16` during reuse, leaving half the `H/8` margin
unused. Grid rebuilds and particle reordering invalidate the cache as well.
The force cutoff remains `H`, and retained candidates keep their original order.
The 2D and optional-output paths keep the original traversal because the measured
2D benefit did not justify the additional storage.

In the 17,446-particle 3D dam-break benchmark, this reduces initial candidates
from 13.28 million to 3.93 million per evaluation. Six alternating warmed runs
on Julia 1.12.7 with 32 default threads reduced median solver time from 0.533 s
to 0.491 s (8.0% less time, 0.02 s simulated). Timesteps matched exactly and
maximum scaled final-field error was `6.4e-13`. The compact cache adds about
16.3 MB of reachable storage (932 bytes per particle in this geometry); capacity
and rebuild allocations can require more. Gains depend on motion and geometry.

This particle cache filters the existing cell candidates. It preserves the
cell-grid coverage limitation described below; its distance margin does not
expand the grid stencil to find previously excluded cells.

For fixed MDBC ghost points, neighboring cell indices are cached and only active
ghosts are scheduled. The cache is refreshed after each neighbor rebuild and
particle reordering. Moving or fluid ghost points retain the direct lookup path.
Density and kernel contributions are still evaluated every time, in the original
neighbor order; the cache does not change the timestep or neighbor-reuse policy.

Neighbor-cell tables use `PackedNeighborCellLists`: contiguous arrays of
unsigned start/end cell IDs plus native-integer offsets. Consecutive neighboring
cells form a single run, prepared during rebuilds and visited as one particle
loop. This preserves the exact interaction order while reducing loop overhead.
The ID width is chosen from
8, 16, 32, or 64 bits using the maximum possible cell count, including the
sentinel, so movement and rebuilds cannot overflow it. The supplied benchmark
cases use 16-bit IDs and require about 88–89% less neighbor-table memory.
Particle coordinates and physics calculations retain their original precision.
The existing vector-of-vectors neighbor API remains supported. Compression's
runtime effect is small and workload dependent. The run encoding measured
about 3–4% less 3D solver time beyond the earlier packed-list implementation.
Including solver arrays, live temporary buffers, and both output snapshots,
compression saves about 2.5–3.7% in the supplied cases. These storage measurements
exclude Julia/compiler memory, allocator slack, input loading, and HDF5 overhead;
they do not establish a maximum safe simulation size.
Larger storage measurements using 32-bit IDs project about 2–3% more particles
from neighbor compression at a fixed RAM budget, or 23–28% when also counting
the earlier sorting-scratch reduction against the original implementation.
These estimates assume comparable geometry and output settings; runtime and
peak memory still limit practical simulation size.

See [the reproducible benchmarks](benchmark/README.md) for measured speedups,
numerical comparisons, and commands for profiling your own machine.

### Neighbor reuse: known coverage limitation

The current rebuild decision is a motion heuristic: it accumulates
`4 * maximum(norm.(Positionₙ⁺ - Position))` and rebuilds when this reaches `h`.
`Positionₙ⁺` is the predictor state, rather than the position at the last rebuild.
The cached cells have width `H` (the interaction radius), and the search visits
only immediately adjacent cells. There is no extra search margin guaranteeing
that a previously excluded pair stays outside the interaction radius.

The regression in `test/neighbor_rebuild_coverage.jl` demonstrates the gap:
particles in cells 0 and 2 can move from separation `1.001953125H` to
`0.998046875H`, while the motion estimate is only `0.00390625H`, below the
default `h = 0.5H` threshold. Their speeds can satisfy `c₀ >= 10 * max(speed)`.
The suite marks this coverage assertion as a known broken test. This limitation
predates packed neighbor storage and remains unresolved; matching older solver
states does not establish that every physically interacting pair was included.

A replacement should add an explicit search margin (a neighbor "skin"), track
movement from the last build, and check validity before every force evaluation,
including predictor states, moving boundaries, and MDBC ghost interactions.
For a pair list built to `H + Skin`, keeping twice the largest displacement
below `Skin` gives the required geometric bound. This is the established
[skin-based neighbor-list approach](https://docs.lammps.org/Developer_par_neigh.html).
Changing only the current displacement threshold does not supply that margin.

### Further performance priorities

In a warmed four-thread 3D dam-break run lasting 0.06 s of simulated time,
interaction evaluation accounts for about 98% of solver time, while neighbor
maintenance including rebuild-triggered derivative refreshes accounts for about
0.2%. Reducing the number of unnecessary candidate-pair evaluations is therefore
a more promising speed target than merely rebuilding less often in this case.

| Candidate | Effort | Main consideration |
|---|---|---|
| Explicit neighbor skin and displacement validation | Moderate to large | Correctness prerequisite for safely tuning reuse; storage and search volume must be benchmarked |
| Tighter bins or pruned cell searches | Moderate to large | May further reduce pair-loop work; the 3D particle candidate cache above already trades memory for fewer checks |
| Thread-count and batch-size tuning | Low | Measure representative sizes; more threads can add overhead for small cases |
| SIMD blocks or evaluating pair geometry once for both particles | Large | Requires careful accumulation, thread ownership, and model-specific physics checks |

`ComplexDensityDiffusion` now skips the inverse hydrostatic equation of state
for boundary pairs whose diffusion is disabled by the existing model. This
measured about 5.8% less solver time in the 3D benchmark using that model, with
identical timesteps and roundoff-scale state differences. The default linear model keeps its existing path:
the analogous shortcut produced small, mixed timing changes.

At shutdown, the solver prints the full hierarchical run sorted by elapsed
time, followed by a flattened global ranking of recorded sections by allocated
bytes. Parent rows are inclusive of their children. A gray `~section~` row is
the time or memory used directly by that section but not yet covered by a named
child timer; `~untimed~` is work outside all timed sections. The GC column
reports time spent in garbage collection. Allocation totals around threaded
sections are process-wide deltas during that wall-time interval, so they
localize an expensive phase but do not by themselves identify a worker task.

The MDBC rows are divided into reusable-buffer acquisition,
`NeighborLoopMDBC!`, and `ApplyMDBCCorrection`. Neighbor-data maintenance
separately reports `UpdateΔx!`, `UpdateNeighbors!`, and
`BuildNeighborCellLists!`, so hot rows map directly to their Julia functions.

`NeighborLoop` is intentionally a fused, threaded particle-pair kernel. Timing
its density-diffusion, pressure, viscosity, and kernel formulas individually
would require a shared timer inside worker tasks and would dominate the very
small operations being measured. Use Julia's standard `Profile.@profile` for
portable function- and line-level CPU sampling, and
`Profile.Allocs.@profile sample_rate=0.1` followed by
`Profile.Allocs.print()` for allocation stacks. Run a representative warm-up
first so compilation is excluded. The `@profview RunSimulation(...)` form in
the MDBC examples provides a graphical CPU view when ProfileView tooling is
available. Use the tables for trustworthy phase-level wall time and allocations.

### Output scheduling and background writes

The solver advances through one continuous timestep loop. `OutputTimes` only
defines sampling deadlines; it does not restart neighbor construction,
derivative initialization, or adaptive timestep state. A timestep is shortened
only at the final `SimulationTime`, so changing visualization cadence does not
change the integration sequence.
`SimulationMetaData.TimeSteps` records every accepted physical timestep, rather
than one sample per output frame.

`SingleNeighborTimeStepping` periodically re-evaluates its carried derivative
from the accepted full state every 20 physical steps. This suppresses the
long-time staggered drift without changing the requested simulation inputs, and
the correction cadence is based only on integration steps, never output events.
It adds one neighbor evaluation on correction steps (about 5% at this interval);
a derivative refresh caused by a neighbor-list rebuild satisfies the same
correction and is not duplicated.

A snapshot is taken from the first completed timestep at or after each deadline.
The file therefore records the state's actual simulation time, which can be
slightly later than the requested deadline. If one timestep crosses several
deadlines, the corresponding frames intentionally contain the same state and
timestamp; exact-time output would require interpolation.

VTKHDF writes run in one persistent `Threads.@spawn` writer task while the solver
continues on the other Julia threads. Launch with at least two threads (for
example, `julia --threads=auto`) to overlap blocking HDF5 work with simulation.
The transient writer keeps its HDF5 group and dataset handles open across frames,
uses larger grid chunks, and obtains each grid frame's cell offset from the
existing extent instead of reading the full history. The output timer now includes
progress logging and the time spent waiting for a free snapshot buffer.
Particle snapshots use a bounded two-buffer pool: if storage cannot sustain the
requested output rate, the solver waits instead of dropping frames or allowing
memory use to grow without bound. Finalization always waits for queued writes to
finish, so reducing output frequency still reduces total data-copying and HDF5
work.

For the four-second StillWedge example with particle and grid output on local
storage, 32 default threads, and three warmed runs per interval, `OutputTimes=0.01`
produced 401 frames and took 2.17 s median; `0.1` produced 41 frames and took
2.02 s. Before the writer changes these were 2.28 s and 2.05 s. Both intervals
produced identical timesteps and final particle fields. The first measured run
included compilation and took about 15 s at `0.01` in both versions. Disk and
console speed can change the size of the remaining output cost. The two VTKHDF
files total about 201 MB at `0.01` versus 21 MB at `0.1`, so storage throughput
still matters when asking for ten times as many frames.

## Help

Questions or issues can be posted on the GitHub issue tracker. Response times may vary but all feedback is welcome.

## Authors

Written by Ahmed Salih ([AhmedSalih3d](https://github.com/AhmedSalih3d)).

## Version History

| Version | Description |
|---------|-------------|
| 0.7.0  | Per-particle compute loops, multiple-dispatch main simulation path, and streamlined output metadata |
| 0.6.12 | Simulation code uses now multiple dispatch procedure in main simulation code instead of `if` coding statements, to increase run time performance |
| 0.6.11 | Removed chunk metadata after per-particle compute loops landed |
| 0.6.10 | Implemented concepts of tests, aim is to understand allocations and run time |
| 0.6.9  | Specify output times via `OutputTimes` (float or vector). |
| 0.6.8  | Select which variables are written to `vtkhdf` files. |
| 0.6.7  | Introduced mDBC boundary conditions and other improvements allowing particles to interact with boundaries. |
| 0.6.6  | Added neighbour grid visualisation in ParaView for debugging. |
| 0.6.5  | Linearised density diffusion, optional single-file output and performance improvements. |
| 0.6.4  | Revised geometry configuration interface and added time-step plot. |
| 0.6.3  | Added automatic log visualisation and `CloseHDFVTKManually` helper. |
| 0.6.2  | Added automatic ParaView visualisation support. |
| 0.6    | Major rewrite with solver setup changes and moving object support. |
| 0.5    | Logging and `hdfvtk` output added. |
| 0.4    | Complete rewrite focusing on custom cell lists. |
| 0.3    | Highly optimised CPU version with minimal allocations. |
| 0.2    | Cleanup of initial release. |
| 0.1    | Initial release. |

## License

This project is licensed under the MIT License – see [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- [DualSPHysics](https://dual.sphysics.org/) for inspiration.
- Many thanks to the Julia community and especially [Leandro Martínez](https://github.com/lmiq) for guidance on neighbour-list algorithms.
- Thanks to [PharmCat](https://github.com/PharmCat) for suggestions and code contributions.

[![Star History](https://api.star-history.com/svg?repos=AhmedSalih3d/SPHExample)](https://star-history.com/#AhmedSalih3d/SPHExample)
