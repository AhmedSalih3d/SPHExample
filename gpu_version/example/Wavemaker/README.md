# Sinusoidal wave maker — DualSPHysics example 06

[Wavemaker.jl](Wavemaker.jl) sets up a CUDA simulation of a piston generating
waves that travel towards a sloping beach. Its setup follows the readable
constants / geometry / metadata / models / allocation / solver sequence of
[DucklingMDBC.jl](../DucklingMDBC.jl).

[Geometry.jl](Geometry.jl) contains the geometry. It draws
the x/z cross section, samples particles and repeats the section across y.
Geometry generation works without an NVIDIA GPU. No external geometry files
or DualSPHysics installation are needed to run this example.

The source case is the distributed DualSPHysics 5.4
`examples/main/06_Wavemaker/CaseWavemaker_Def.xml`, described in the
[official test-case guide](https://github.com/DualSPHysics/DualSPHysics/wiki/7.-Testcases#716-wavemaker).

| Setting | Default |
| --- | --- |
| Coordinates | x along the tank, y across it, z upwards |
| Tank | 5 m long, 2 m wide, 1.5 m high |
| Beach | Flat to x = 1 m, rising to z = 1.1 m at x = 5 m |
| Initial water polygon | z = 0.75 m, clipped by the beach |
| Piston | x = 0 to 0.06 m, z = 0 to 1.5 m |
| Particle spacing | 0.02 m |
| Piston amplitude | 0.25 m; 0.5 m peak-to-peak stroke |
| Simulation | 10 s, output every 0.1 s |
| Boundaries | DBC (`NoMDBC`), periodic across y, open top |
| Physics | Gravity 9.81 m/s², artificial viscosity 0.02, density diffusion 0.1 |
| Numerics | Symplectic, Wendland C2, h = 1.5√3 dx, CFL = 0.2, no shifting |

Run from the repository root:

```sh
julia --project=gpu_version -t 2,0 gpu_version/example/Wavemaker/Wavemaker.jl
```

Or from `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Wavemaker/Wavemaker.jl
# Optional arguments: output directory, particle spacing, duration in seconds.
julia --project=. -t 2,0 example/Wavemaker/Wavemaker.jl output/wavemaker_preview 0.05 2.0
```

The default output directory is `gpu_version/output/wavemaker`. Use a new
directory for each run. The main script opens the solver output in ParaView
and opens its log. For a run without opening these programs:

```julia
include("example/Wavemaker/Wavemaker.jl")
result = run_wavemaker(
    dx = 0.05,
    simulation_time = 2.0,
    visualize = false,
    open_log_file = false,
    save_location = "output/wavemaker_preview",
)
```

Including either source file only defines functions. To inspect the geometry
without running the solver:

```julia
include("example/Wavemaker/Geometry.jl")
using .WavemakerGeometry
case = wavemaker_geometry(; dx = 0.02)
length.(values(case.positions))  # fixed, piston, water particle counts
```

The simulation saves:

- `Wavemaker_Fixed.csv`, `Wavemaker_Piston.csv` and
  `Wavemaker_Water.csv`: particle positions and initial densities, readable by
  `SPHGeometry`. These are inspection/export copies; the solver uses the
  in-memory particles.
- `Wavemaker_Geometry.vtkhdf`: stationary polygon geometry.
- `Wavemaker_Motion.vtkhdf`: animated polygons showing the prescribed piston
  motion. Its water polygon shows the initial volume; simulated water motion
  is in the solver's particle output.
- Solver particle output, logs and the WG0 water-column measurement at
  `(x, y) = (1, 1)` m. Measurement blocks are stored in the solver VTKHDF file.
  Set `measurements = false` to omit the gauge.

The piston follows the reference's piecewise sinusoidal schedule. Each segment
adds a displacement to the previous endpoint:

```text
Δx(τ) = A [sin(2π f τ + φ) - sin(φ)]
vx(τ) = 2π f A cos(2π f τ + φ)
```

Here τ is time since that segment started. The first segment lasts 5 s with
f = 0.5 Hz and φ = 3π/2, starting at rest and moving into the water. Subsequent
segments last 10 s with zero phase, alternating f = 0.75 Hz and f = 0.5 Hz.
Displacement remains continuous; velocity can jump at a segment change, as
commanded by these separate sinusoids. The motion stops at 100 s. Edit
`wavemaker_motion()` to change the amplitude, frequency or schedule, checking
that the complete piston stroke stays over the flat bed.

The geometry uses an outward boundary band, three particle layers thick on
the flat floor at the default spacing. At coarse spacing the piston thickens
to retain at least that many layers. Both sampled endpoint rows across y are
kept, with periodic faces half a spacing outside them; at dx = 0.02 the period
is 2.02 m and the seam spacing is 0.02 m. Fixed walls own shared lattice nodes,
so the moving piston and water do not duplicate them.

This is an adaptation of the reference geometry and commanded motion to this
solver. It uses DBC rather than Duckling's mDBC, so it needs no ghost-node CSV.
Hydrostatic fluid density and sound speed use the sampled water extrema.
The boundary sampling, analytical wall velocities, density diffusion,
time-step clipping and particle escape handling can differ from DualSPHysics;
matching these inputs does not establish matching wave heights or trajectories.

Run the focused geometry, motion and short CUDA regressions from `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Wavemaker/test_wavemaker.jl
```
