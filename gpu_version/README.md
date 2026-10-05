# SPHExampleGPU – CUDA version of SPHExample

`gpu_version/` is a copy of `src/` ported to NVIDIA GPUs with
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). It is a self contained Julia
package (`SPHExampleGPU`) with the same public API as the CPU package: the
example scripts only differ in `using SPHExampleGPU` instead of
`using SPHExample`. All physics (Wendland/cubic kernels, artificial, laminar
and laminar+SPS viscosity, all density diffusion models, dynamic and mDBC
boundary conditions, particle shifting, prescribed motion of bodies) is
supported in 2D and 3D, in `Float64` or `Float32`.

## Requirements

* An NVIDIA GPU with a recent driver (CUDA 12 or newer). The CUDA toolkit is
  downloaded automatically by CUDA.jl, `nvcc` is not needed.
* Julia 1.10 or newer.

## Getting started

```bash
# from the repository root
julia --project=gpu_version -e 'using Pkg; Pkg.instantiate()'
julia --project=gpu_version gpu_version/example/Dambreak2dMDBC.jl
```

The examples in `gpu_version/example/` mirror those in `example/`. Choose the
precision with `FloatType = Float32` or `Float64` at the top of a script.

### Automatic ParaView visualization

With `VisualizeInParaview = true` and `ExportGridCells = true`, automatic ParaView
opening loads both the particles and cell grid in the same session and view.
The grid appears as a wireframe. This works with single-file VTKHDF output and
numbered VTKHDF file series; the generated state script can also be opened later.

Particles are colored by pressure when `Pressure` is exported, otherwise density.
`RunSimulation` fixes the pressure color range to `0` through `ρ₀ g h` Pa, where
`h` is the initial fluid height plus one particle spacing (`y` in 2D, `z` in 3D).
The fixed scale makes colors comparable across timesteps. For impact pressures
or a different scale, pass `ParaviewPressureRange = (0.0, 20000.0)` to
`RunSimulation`; negative lower limits are also supported. Values outside the
range use the endpoint colors. With zero gravity or no fluid, the state uses
the initial pressure data range, with a nonzero fallback for uniform data.
When generating a state directly, use
`AutoOpenParaview(metadata, constants, variables; pressure_range = (0.0, 20000.0))`.
The generated Python state's `pressure_range` can also be edited before opening.

### Using generated particles directly

`SPHGeometry` holds input particles in `geometry.Particles`, a `StructArray`.
Pass sampled positions directly without writing or reading CSV:

```julia
sampled = sample_particles(regions, constants.dx)
geometries = [SPHGeometry{2, Float32}(region.positions;
    Density = region.type == Fluid ?
        hydrostatic_density(region.positions, constants) : constants.ρ₀,
    GroupMarker = marker, Type = region.type)
    for (marker, region) in enumerate(sampled)]
particles = AllocateDataStructures(geometries, metadata)
```

Alternatively, use `Particles = StructArray((Position = positions,
Density = densities, ID = ids))`. IDs are one based and unique across groups;
when omitted they are assigned automatically after explicit IDs. Optional
`Velocity`, `GhostPoints`, and `GhostNormals` fields initialize the solver.
The group's `Type`, `GroupMarker`, `Motion`, and `Floating` settings still apply.
Allocation copies inputs and converts them to the simulation precision, so
geometries can be reused without being changed by a run.

Existing `CSVFile = path` calls remain supported: they load the `StructArray`
once when constructing the geometry, preserving CSV position precision and
converting zero based IDs. Allocation no longer reads files. The StillWedge
and other examples with generators sample particles inside their main run files.
Their previous `CSVFile` settings remain commented out, and standalone generators
remain available for CSV/VTKHDF export.
The first-region ownership rule applies only when all regions are passed to
`sample_particles` together; CSV loading preserves every row, including
coincident positions. `write_particle_csv` writes coordinates and densities as
`Float64`, preserving `Float32` values when promoted for double-position runs.
Boundary normals can still be loaded separately with `ParticleNormalsPath`.
For `SimpleMDBC`, supplying `GhostPoints` and `GhostNormals` on every boundary
group also supports a run entirely from memory; cavity and 2D dam break use this.

### Generating a case from polygons (StillWedge 2D)

`example/GenerateStillWedgeMDBC.jl` builds the 2D still wedge case from
Meshes.jl `PolyArea`s instead of hand made CSV files. It needs no GPU and runs
no simulation. From the repository root:

```bash
julia --project=gpu_version gpu_version/example/GenerateStillWedgeMDBC.jl [output_dir] [dx]
```

The tank is one polygon (floor and walls of 0.04 m carrying a hollow 45°
wedge with its tip at `(1.1, 0.26)` m; the inner V of the wedge shell continues
through the floor, so there is no boundary below the wedge), the water a
second polygon up to 0.5 m. The script writes to `output_dir` (default
`input/still_wedge_generated/`):

* `StillWedge2D_Geometry.vtkhdf` – polygon cells with a `Region` cell array
  (`1` = tank, `2` = water); concave and holed faces are triangulated;
* `StillWedge2D_Dp<dx>_Bound.csv`, `..._Fluid.csv` – particles in the layout
  read by `SPHGeometry` (with `dx = 0.02` they reproduce `input/still_wedge/`
  exactly, densities included);
* `StillWedge2D_Dp<dx>_Particles.vtkhdf` – the same particles with `Density`,
  `Pressure`, `Type` and `GroupMarker` for ParaView.

Boundary particles start at `ρ₀`. Fluid particles start in hydrostatic
equilibrium: `hydrostatic_density` inverts the solver's equation of state
(`EquationOfStateGamma7`) for `P = ρ₀ g (water_level - y)`. By default the water
level is the highest fluid particle (the DualSPHysics convention); pass
`water_level` to override it, and the `SimConstants` of your simulation so that
`ρ₀`, `g` and `c₀` agree (the default matches `StillWedgeMDBC.jl`).

The particles come from `ParticleRegion` and `sample_particles`
(`src/ParticleGenerator.jl`): one lattice of spacing `dx` is laid over all
regions with `RegularSampling` and each lattice point goes to the first region
that contains it. Walls are listed first and own their outline, the fluid only
takes the open interior of its polygon, so boundary and fluid particles never
overlap and the fluid stops one spacing short of the walls and the free
surface. Pass `offset` to a `ParticleRegion` to shrink it by that distance (for
example `ParticleRegion("Fluid", water, Fluid; offset = 0.5dx)` keeps the fluid
at least half a spacing away from every edge, including slanted ones); a
negative `offset` grows the region. A `Float32` spacing such as
`SimConstants.dx` is sampled as the decimal it represents (`0.02f0` → `0.02`):
widening it to `0.019999999552965164` would drift the lattice off the wall
outlines and drop wall particles. `SavePolygonVTKHDF` writes convex polygons
without holes as n-sided cells; concave or holed faces use constrained
Delaunay triangulation to preserve their actual area. The single file
`SaveVTKHDF(path, points, ...)`
lives in `src/ProduceHDFVTK.jl`. The geometry tests run without a GPU:
`julia --project=gpu_version gpu_version/test/still_wedge_geometry.jl`.

### Generating the middle-square still-wedge geometry

`example/GenerateStillWedgeMiddleSquareMDBC.jl` builds the tank, hollow wedge,
water and fixed central block from `Meshes.jl` polygons, then uses the package's
`ParticleRegion`, `sample_particles` and hydrostatic initialization tools. It
does not read input CSVs, run a simulation or need a GPU:

```bash
julia --project=gpu_version gpu_version/example/GenerateStillWedgeMiddleSquareMDBC.jl [output_dir] [dx]
```

The tank and wedge use the StillWedge dimensions above. The fixed block is
0.4 m wide and 0.5 m high, with its lower-left corner at `(0.9, 0.36)` m; the
water excludes it. At `dx = 0.02`, sampling reproduces the reference case's
1,126 boundary and 2,300 fluid positions without copying them.

Output defaults to `gpu_version/input/still_wedge_middle_square_generated/`:

* `StillWedge_MiddleSquare_Geometry.vtkhdf` – polygon cells with `Region`
  labels (`1` = tank/wedge, `2` = water, `3` = fixed block); concave and
  holed faces are triangulated;
* `StillWedge_MiddleSquare_Dp<dx>_Bound.csv`, `..._Fluid.csv` – sampled
  particles for `SPHGeometry`, with hydrostatic fluid densities;
* `StillWedge_MiddleSquare_Dp<dx>_Particles.vtkhdf` – sampled points with
  density, pressure, type and group marker (`1` = all fixed boundaries,
  `2` = fluid).

The default generated files are included with the example. Pass an output
directory and optionally `dx` to regenerate at another resolution. The
original MDBC simulation inputs and ghost-node data remain unchanged.
The geometry tests need no GPU:
`julia --project=gpu_version gpu_version/test/still_wedge_middle_square_geometry.jl`.

### Generating the 2D DamBreak case

`example/GenerateDamBreak2DMDBC.jl` builds the 2D dam-break tank and its
initial water column from `Meshes.jl` polygons. It uses the same lattice
sampling and hydrostatic initialization as the StillWedge generator, writes
CSV files for `SPHGeometry`, and writes VTKHDF files for ParaView. It needs no
GPU and does not run a simulation:

```bash
julia --project=gpu_version gpu_version/example/GenerateDamBreak2DMDBC.jl [output_dir] [dx]
```

The default geometry is a 4 m × 3 m tank with 0.06 m walls and a 1 m × 2 m
water column. Output defaults to `gpu_version/input/dam_break_2d_generated/`;
pass a simulation's `SimConstants` to
`generate_dam_break_2d_example` when a different `c₀`, `ρ₀`, or `g` is used.

### Generating the 2D MovingSquare case

`example/GenerateMovingSquare2D.jl` builds the closed tank, fluid and moving
square from `Meshes.jl` polygons. The tank is 10 m × 5 m with 0.04 m walls;
the 1 m square starts at `(1, 2)` m. The generator needs no GPU and does not
run a simulation:

```bash
julia --project=gpu_version gpu_version/example/GenerateMovingSquare2D.jl [output_dir] [dx]
julia --project=gpu_version gpu_version/example/MovingSquare2d.jl
```

The first command writes to `gpu_version/input/moving_square_2d_generated/`
by default. It creates VTKHDF files for the polygons, particles and prescribed
motion sequence, plus `Fixed`, `Fluid` and `Square` CSV files for optional file
input. It opens the motion preview in ParaView when `paraview` is on `PATH`.
This generator does not run the
simulation or need a GPU. The simulation samples its own particles at its
configured `dx`; the CSV export is optional.

To preview a prescribed motion before running the simulation, write a temporal
VTKHDF `PolyData` file. `motions` maps region names to `MotionDetails`, and
`times` gives the frame times in seconds:

```julia
using SPHExampleGPU
using StaticArrays: SVector

tank = outline(rectangle((0.0, 0.0), 10.0, 5.0);
               thickness = 0.04, side = :outward)
square = rectangle((1.0, 2.0), 1.0, 1.0)
motion = MotionDetails{2, Float64}(
    Velocity = 2.8,
    StartTime = 0.0,
    Duration = 3.0,
    Direction = SVector{2, Float64}(1.0, 0.0),
)
SavePolygonMotionSequence(
    "MovingSquare2D_Motion.vtkhdf",
    (; tank, square);
    motions = (; square = motion),
    times = 0.0:0.1:3.0,
)
```

This writes a single temporal VTKHDF `PolyData` file with the polygon geometry
and time steps. Regions omitted from
`motions` remain fixed, as do regions whose `MotionDetails` has
`MoveParticles = false`. The preview follows the prescribed translation only;
it does not predict fluid interactions.

### 2D lid-driven cavity (Re = 100)

`example/GenerateLidDrivenCavity2D.jl` and
`example/LidDrivenCavity2d.jl` set up the square, fluid-filled cavity from the
Rocky 2025 R1 SPH verification manual. The cavity is 1 m × 1 m; the bottom
and side walls are stationary, while the top boundary moves at 1 m/s in `+x`
without changing its position. The lid spans the complete 1.1 m outside width
of the tank and overlaps both side walls. The sample spacing is 0.01 m, wall
thickness 0.05 m, and simulation duration 60 s.

The configuration specifies an initial density of 10 kg/m³, dynamic viscosity
0.1 Pa·s, and sound speed 10 m/s. The solver takes kinematic viscosity, so the
example sets `ν₀ = μ/ρ₀ = 0.01 m²/s`, which gives `Re = ρ₀ U L/μ = 100`.
It uses zero gravity, laminar (Morris) viscosity, `CFL = 0.2`, and a
Wendland C2 kernel with `h = 1.2√2 dx` (the manual does not specify a kernel
or smoothing length). The thermal model and the 3D periodic thickness are not
represented in this 2D solver case.

The lid uses `MotionDetails(...; MoveParticles = false)`: its boundary
particles retain their positions while carrying the prescribed velocity used
in the no-slip viscous interaction. This differs from a moving body, whose
particles translate with the prescribed velocity.

Three choices stabilize the cavity run:

* **mDBC walls.** The generator writes one ghost node per wall/lid particle in
  `LidDrivenCavity2D_Dp<dx>_GhostNodes.csv`, mirrored across the wetted face
  into the fluid; corner particles mirror across both faces. With plain DBC,
  the sliding lid and fixed side-wall particles have different prescribed
  velocities, so their wall-wall continuity terms grow corner densities
  exponentially, over-pressurize the corners, and push fluid out of the
  cavity. mDBC interpolates boundary density from fluid neighbours and avoids
  this DBC failure.
* **Zero-gravity linear density diffusion** (`δᵩ = 0.1`) suppresses density
  drift and prevents the fluid under the lid from rarefying and seeping into
  it.
* **Planar shifting**, the DualSPHysics setting for internal flows, keeps the
  fluid particles evenly distributed and improves the velocity profiles.

The GPU run disables `GPUBoundaryForces`, since the fixed/prescribed walls do
not respond to their reaction accelerations. Including those unused
accelerations in the global time-step constraint shrinks `dt` dramatically.
It also uses `GPUMaxStepsPerSync = 256` and writes frames every 0.5 s by
default. Set the `mdbc`, `density_diffusion`, or `shifting` keyword to `false`
in `run_lid_driven_cavity_2d` to disable each option for comparison.

At `dx = 0.01`, the full 60 s run takes about 54 s on an RTX A1000 laptop GPU
(176,758 steps). Fluid density stays within about 5% of `ρ₀`. At `t = 60 s`,
the centreline profiles agree with Ghia, Ghia & Shin (1982) to within
`0.016 U` for `u(y)` on `x = 0.5` and `0.017 U` for `v(x)` on `y = 0.5`.

```bash
julia --project=gpu_version gpu_version/example/GenerateLidDrivenCavity2D.jl
julia --project=gpu_version gpu_version/example/LidDrivenCavity2d.jl
```

Both scripts accept an optional output directory and `dx`; the simulation
script additionally accepts a duration:
`[save_dir] [dx] [duration]`. Its former `input_dir` argument remains available
for the commented CSV configuration. From the Julia REPL, `include` the
simulation script and call `run_lid_driven_cavity_2d()`. Particles and ghost nodes
are sampled in that function; running the export script is optional. Exported inputs go to
`gpu_version/input/lid_driven_cavity_2d_generated/` by default. The simulation
writes velocity, density and pressure frames to
`C:\TestSimulations\LidDrivenCavity2D_GPU` by default.

### Floating bodies: the 2D falling cylinder and 3D rigid bodies

A `Floating` particle group is a rigid body moved by gravity and the forces of
the surrounding particles (the DualSPHysics "floating" object, `RigidAlgorithm
= 1`). Give the group `Type = Floating` and its `FloatingDetails`:

```julia
Cylinder = SPHGeometry{2, FloatType}(CSVFile = "...", GroupMarker = 3, Type = Floating,
    Floating = FloatingDetails{FloatType}(RelativeWeight = 1.2, PauseTime = 1.0))
```

`RelativeWeight` is the body density relative to `ρ₀`. The body is held still
until `PauseTime` (DualSPHysics `FtPause`). Its particles join the neighbour
loops like boundary particles, but their momentum terms are always evaluated.
After each neighbour loop, their accelerations are summed into the force
`m₀ Σ aₖ` and the torque about the body centre. The body mass is
`RelativeWeight m₀ N`, and its moment of inertia comes from the particle
positions. The centre, velocity and angular velocity advance with the
symplectic predictor and corrector, and the particles are then moved rigidly.
The same `FloatingDetails` also works with `SPHGeometry{3, FloatType}`:
3D bodies use a full inertia tensor, vector angular velocity and quaternion
orientation. All of this runs on the device, so steps are still batched and
replayed as CUDA graphs (`src/GPUFloating.jl`). Floating bodies need
`SymplecticTimeStepping()`. `<SimulationName>_Floating.csv` records the centre
and velocity at each output time; its 2D angle/omega columns remain unchanged,
while 3D output includes the scalar-first quaternion (`Orientation:0` through
`Orientation:3`) and the three angular-velocity components.

Force reduction and rigid placement use a compact list of floating particles,
refreshed after cell sorting without changing its device address. Each warp
reduces only the bodies it contains. Rotation coefficients are computed once
per body and stage, including the promoted precision used for 3D double
positions. These changes preserve CUDA graph replay and the rigid-body scheme;
force summation order can differ slightly due to floating-point rounding.

On an RTX 5080, a warmed GPU benchmark with 250,000 particles (5,000 floating,
`Float32` physics and `Float64` positions) measured 1.5 times faster floating
stages for eight 2D bodies and 2.5 times faster 3D particle placement. These are
floating-kernel gains, not whole-simulation speedups: a single 2D cylinder's
complete timestep remained approximately unchanged because fluid interactions
dominate. Run `benchmark/floating_kernels.jl` for device timings or
`benchmark/floating_steps.jl` for the complete cylinder timestep, supplying a
saved baseline `GPUFloating.jl`; the latter also accepts its Git revision.
The focused physics checks run with `julia --project=. -t 1,0 test/run_floating.jl`
from `gpu_version/`.

`example/GenerateFloatingCylinder2D.jl` and `example/FloatingCylinder2d.jl`
reproduce DualSPHysics `examples/main/11_Floating/CaseFloatingSphereVal2D`.
A cylinder of radius 1 m and relative weight 1.2 starts half submerged in a
10 m wide, 14 m deep tank and sinks after a 1 s pause. The case uses
`dp = 0.025`, `coefh = 1.2`, laminar + SPS viscosity, density diffusion 0.1,
`CFL = 0.2` and `c₀ = 30 √(g · 0.8)`. Two things differ from DualSPHysics:
* The periodic sides are fixed walls.
* The cylinder is sampled as concentric rings (`sampling = :conforming`)
  rather than on the lattice.

```bash
julia --project=gpu_version gpu_version/example/GenerateFloatingCylinder2D.jl [output_dir] [dx]
julia --project=gpu_version gpu_version/example/FloatingCylinder2d.jl
```

The validation data of the DualSPHysics case (Fekken 2004; Moyo and Greenhow
2000) give the sinking distance and velocity against `t* = (t − 1 s) √(g/R)`
for `t* ≤ 8`. At `dx = 0.025` (about 233 000 particles, 84 000 steps, about
7 minutes on an RTX A1000 laptop GPU) the GPU run follows Fekken closely up to
`t* ≈ 3`. After that it lies between the two experiments, ending at a sink
depth of 3.9 R (Fekken 3.4 R, Moyo and Greenhow below 4 R). Over the window,
its velocity differs from Fekken's by 0.07 √(gR) on average. Past the window
(`t > 3.5 s`) the wake loses its symmetry, the cylinder drifts sideways, and
near the end it touches a side wall; this is where fixed walls differ from
the periodic sides of the original case.

### Drawing shapes for particle generation

`src/PolygonDrawing.jl` provides drawing helpers that return Meshes.jl
`PolyArea`s, which go straight into `ParticleRegion`, `Multi` and
`SavePolygonVTKHDF`. Points may be tuples, vectors, `SVector`s or `Point`s;
angles are in radians (use `deg2rad` for degrees). The outer ring of every result
runs counter-clockwise and its holes clockwise, whatever the input orientation.

| Helper | Draws |
| --- | --- |
| `polygon(vertices; holes)` | any simple polygon, optionally with holes (vertex lists or shapes) |
| `triangle(a, b, c)` | a filled triangle |
| `rectangle(corner, w, h; angle, centered)`, `square(corner, a; ...)` | rectangles, optionally rotated about the corner or centre |
| `regular_polygon(center, r, n; angle)`, `circle(center, r; segments = 128)` | regular polygons; a circle is one with its vertices on the circle |
| `arc(center, r, θ₀, θ₁; segments)` | points along an arc, to build paths or outlines |
| `line(a, b; thickness, side, offset)` | a straight wall |
| `polyline(points; thickness, side, offset, closed, miter_limit)` | a wall along a path |
| `outline(shape; thickness, side, offset, miter_limit)` | a wall around a shape (every ring, holes included) |
| `offset_polygon(shape, distance)` | the shape grown (or shrunk, `distance < 0`) along its normals |
| `translate`, `rotate(shape, θ; origin)`, `mirror(shape; origin, direction)` | moved copies |
| `prism(base, bottom, top)` | a 3D extrusion of a 2D shape along `z` (box, cylinder, wall…) |

Walls are placed relative to the drawn path with `side`. For paths, `:left`,
`:right` or `:center` are taken looking along the drawing direction. For
`outline`, `:outward` (default), `:inward` or `:center` are taken relative to
the shape. `offset` then moves the wall further along the normal: towards the
left for paths, away from the shape for outlines. It leaves a gap between the
drawn surface and the wall. Joints are mitred, so straight walls keep sharp
corners. Convex joints whose miter would exceed `miter_limit` (default 4) times
the offset are bevelled. A wall too thick for its path throws an
`ArgumentError` instead of returning an invalid polygon. This happens when an
inner offset reverses an edge or crosses itself.

```julia
dx = 0.02
domain = rectangle((0, 0), 2.0, 1.0)
tank   = outline(domain; thickness = 3dx)                      # closed tank, walls outside
open_tank = polyline([(0, 1), (0, 0), (2, 0), (2, 1)];         # wetted surface drawn,
                     thickness = 3dx, side = :right)           # walls grow outwards
pipe   = outline(circle((1.0, 0.5), 0.2); thickness = 2dx, side = :inward)
ramp   = line((1.2, 0.0), (2.0, 0.3); thickness = 3dx, side = :right)
water  = rectangle((0, 0), 2.0, 0.5)
regions = [ParticleRegion("Bound", Multi([tank, pipe, ramp]), Fixed),
           ParticleRegion("Fluid", water, Fluid)]
```

Shapes listed earlier claim the lattice points first, so obstacles inside the
water rarely need to be cut out as holes. In 3D, a `ParticleRegion` takes a
prism or a tuple of prisms, which is filled as their union. `sample_particles`
then fills a 3D lattice, and `SavePolygonVTKHDF` writes the prism surfaces:

```julia
interior = rectangle((0, 0), 1.0, 0.6)
tank  = (prism(outline(interior; thickness = 3dx), -3dx, 0.6),     # walls
         prism(offset_polygon(interior, 3dx), -3dx, 0.0))          # floor
water = prism(interior, 0.0, 0.3)
```

#### Sampling a shape along its own outline

On the lattice, a circle or a tilted wall becomes a staircase. For selected 2D
regions, pass `sampling = :conforming` to place the particles along the shape
instead. They lie on layers parallel to the outline, at depths `0, dx, 2dx, …`.
Each layer is split evenly into steps of about `dx`, and every sharp corner
gets a particle. A circle becomes concentric rings, and an arc wall becomes
arcs one spacing apart. All other regions stay on the common lattice:

```julia
regions = [ParticleRegion("Cylinder", circle((1.0, 0.3), 0.15), Fixed;
                          sampling = :conforming),                # rings to the centre
           ParticleRegion("Pipe", circle((0.4, 0.5), 0.2), Fixed;
                          sampling = :conforming, layers = 3),    # hollow: 3 rings
           ParticleRegion("Baffle", polyline(arc((1.6, 0.7), 0.3, π, 2π);
                                             thickness = 2dx), Fixed;
                          sampling = :conforming),                # 3 arcs
           ParticleRegion("Tank", open_tank, Fixed),              # lattice
           ParticleRegion("Fluid", water, Fluid)]                 # lattice
```

The first layer lies on the outline, or one spacing inside it with
`include_surface = false`; `offset` moves it further in. `layers` limits the
number of layers, and the core beyond them stays free for later regions (here
the pipe fills with water). A conforming region claims its particles plus
`dx / 2` on either side of its first and last layers. Later lattice regions
skip that band, so lattice particles stay at least half a spacing away from
conforming ones. Conforming particles in an earlier region, or within `dx / 2`
of an earlier particle, are dropped, so list conforming boundaries first.

`example/GenerateShapesShowcase.jl` draws a 2D tank with a cylinder, wedge,
ramp, arc baffle and tilted square, and a 3D tank of prisms. The cylinder,
baffle and square are sampled along their outlines. It writes both scenes as
VTKHDF and CSV files:

```bash
julia --project=gpu_version gpu_version/example/GenerateShapesShowcase.jl [output_dir] [dx]
```

The 2D water polygon excludes the portions occupied by the cylinder, wedge,
ramp, baffle and square, including cutouts where shapes cross the waterline.
Concave polygons are triangulated with constrained Delaunay so empty regions
such as the open tank interior are not filled in ParaView.

The DamBreak and MovingSquare generators above are drawn with `polyline`,
`outline`, `rectangle` and `polygon`. They reproduce their previous particles
exactly. The drawing tests need no GPU:
`julia --project=gpu_version gpu_version/test/polygon_drawing.jl`.

### Which precision?

* `Float64` reproduces the CPU results to round-off (the test suite checks
  that densities, positions and velocities agree to about `1e-14` relative
  after a few dozen steps).
* `Float32` is where consumer and laptop GPUs are fast. GeForce/RTX-A/Quadro
  cards execute double precision at 1/32 (Ampere, Ada) or 1/64 (Blackwell) of
  their single precision rate, so `Float64` on such a GPU is not faster than a
  modern multi core CPU. Datacenter GPUs (A100, H100) have fast `Float64`.
  `Float32` is the precision DualSPHysics uses by default as well; densities
  stay well within the ~1 % variation of weakly compressible SPH.
* `Float32` with `GPUDoublePosition = true` keeps the positions alone in
  `Float64`, like the `posdouble` mode of DualSPHysics. A `Float32` position
  is quantised to an ulp of its distance from the origin (1e-3 at 10 000 m,
  a tenth of a typical `dx`), and every step adds increments that are far
  smaller than that ulp; a `Float64` position is not. The pair loops never
  read the `Float64` positions, they work on cell relative `Float32`
  coordinates (see "Double positions" below), so the cost is a `Float64`
  add per particle per step and there is no measurable slowdown.

### Startup time and precompilation

The package ships a [PrecompileTools](https://github.com/JuliaLang/PrecompileTools.jl)
workload (`src/PrecompileWorkload.jl`). When the package is precompiled it
writes tiny versions of the example cases (2D/3D, mDBC and DBC, a moving body
with planar shifting) for one short output interval. This exercises the solver
and output paths without simulating extra physical time. The host code (CSV
input, logging, VTKHDF output, the time loop) is then cached in the package
image. If a working GPU is present during precompilation, so are the CUDA
kernels for one lane and the automatic lane choice. On an RTX A1000 laptop GPU
the first 2D dam break run in a new session drops from about 45 s to about 1 s.
Loading the package (`using SPHExampleGPU`, about 5 s, mostly CUDA.jl) stays the
same, and no sysimage is needed.

This has a cost and some limits:

* With dependencies already precompiled, the package takes about 90 s to
  precompile on the RTX A1000 above, and about 30 s without a working GPU. A
  clean setup also needs to precompile dependencies, which adds to this time.
  Package precompilation runs once per package or dependency change, not once
  per session.
* Kernels are cached only for the GPU architecture present during
  precompilation. Without a GPU (for example on an HPC login node) only the host
  code is cached, and kernels compile on the first run. Precompile on a GPU node
  to cache them as well.
* Only the precompiled types and lane counts are cached. Other types and
  explicit lane counts still work but compile on first use. By default that
  means `Float32`, the `WendlandC2` kernel, `SymplecticTimeStepping`, the
  viscosity and density diffusion models of the examples, and one-lane plus
  automatic gather kernels. Keep `FloatType`, the models and (for explicit
  tuning) `GPULanesPerParticle` consistent between runs to benefit.

The following preferences, stored in `LocalPreferences.toml`, change the
workload. Restart Julia after setting them; the package then precompiles again.

```julia
using Preferences, SPHExampleGPU
# also cache Float64 (the test suite and CPU parity runs use it)
set_preferences!(SPHExampleGPU, "precompile_float_types" => ["Float32", "Float64"])
# also cache every supported explicit gather lane count (default [1], plus automatic)
set_preferences!(SPHExampleGPU, "precompile_gpu_lanes" => [1, 2, 4, 8, 16, 32])
# also cache the kernels of GPUDoublePosition = true (cell relative positions)
set_preferences!(SPHExampleGPU, "precompile_double_position" => true)
# turn the workload off, e.g. while editing the package source
set_preferences!(SPHExampleGPU, "precompile_workload" => false; force = true)
```

## What the GPU does differently

The CPU code loops over cells, evaluates each particle pair once and scatters
the two halves into per thread arrays that are reduced afterwards. On the GPU
that would need atomics or large reductions, so the GPU version instead uses
the standard *gather* formulation:

* **One thread (or a few warp lanes) per particle.** The thread of particle
  `i` loops over the particles of the 9 (2D) or 27 (3D) surrounding cells and
  accumulates `dρ/dt`, the acceleration and the optional kernel and shifting
  sums in registers, then writes them once. No atomics, no per thread arrays,
  no reduction step, no zeroing of arrays. For small particle counts the
  neighbour loop is split over several lanes of a warp (each lane takes every
  `K`-th candidate) and the partial sums are merged with warp shuffles, so a
  3 000 particle 2D case still keeps the whole GPU busy.
* **Exact CPU semantics.** The CPU's density diffusion uses `Dⱼ = -Dᵢ` as a
  short cut that is not symmetric in the particle roles. The gather kernel
  reconstructs which particle of a pair was the CPU's `i` (lower index inside
  a cell, higher index across cells) and calls the very same
  `compute_viscosity`/`compute_density_diffusion` functions with the same
  signature, so the GPU reproduces the CPU results instead of a "close"
  variant. As on the CPU, the models receive the densities of the state being
  evaluated (`ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹`) as arguments, so the corrector loop sees the
  predictor density and velocity, and the reciprocals are computed once per
  particle instead of once per pair.
* **Fluid gating of density diffusion by select, not early exit.** The
  `LinearDensityDiffusion` and `ComplexDensityDiffusion` terms act only between
  two fluid particles. The CPU code returns early when either particle is not
  fluid (about 6% faster there). On the GPU that `return` is a divergent branch
  inside the pair loop: it pays off for warps of boundary particles and costs
  in warps of fluid particles with boundary neighbours. Measured on an RTX
  A1000 (Float32, pair kernel time, 9 interleaved rounds, median ratio to the
  old MotionLimiter multiply): early exit 1.00 (DamBreak2D), 1.02
  (StillWedge2D), 1.01 (DamBreak3D), 0.99 (Duckling3D), a wash; a branch
  free select of the finished term 0.98, 0.99, 0.90, 0.96. The models
  therefore compute the term unconditionally and select it. A multiply by one
  is exact, so this is equivalent to the old form; the only differences seen
  in tests are ulp level and come from FMA contraction in `dot`, which can
  change with any recompile.
* **Dense uniform grid + counting sort.** Particles are binned into cells of
  edge `H` on a grid covering their bounding box (plus a one cell margin).
  A histogram (atomics), a prefix scan and a scatter reorder all particle
  arrays by cell in a single fused gather kernel; the three cells of one row
  form one contiguous index range so a 3D particle scans 9 ranges instead of
  27 cells. `GPUCellSubdivision = 2` bins at `H/2` with a 5x5(x5) stencil
  instead (25 row ranges in 3D), which scans 42 % less volume in 3D and 31 %
  in 2D; see the next point for what that buys. The reorder is followed by an in-cell insertion sort, which makes
  runs bitwise reproducible (`GPUDeterministicSort = false` skips it). The
  cell list is only rebuilt when the accumulated displacement exceeds `h`,
  exactly like the CPU version. Gravity and motion factors are derived from
  particle type instead of stored or reordered as separate arrays.
* **Half width cells (`GPUCellSubdivision = 2`).** Candidates per particle
  drop from 30.5 to 21.8 (DamBreak2D), 34.3 to 23.7 (MovingSquare2D), 520 to
  322 (DamBreak3D) and 262 to 159 (Duckling3D); the share of candidates
  inside the support rises from 20-31 % to 32-46 %. The pairs found are the
  same (tested with the density diffusion off, where every term is
  antisymmetric: both grids agree to 1e-14). The price is 25 instead of 9 row
  ranges per particle in 3D (iterated lazily; a 25-tuple would unroll the
  pair body 25 times), a grid with 4-8 times more cells, and an mDBC kernel
  that visits the fluid ranges of 125 instead of 27 cells. Measured on an RTX
  A1000 (Float32, device time per step, 9 interleaved rounds, median ratio to
  `GPUCellSubdivision = 1`): with one lane per particle the pair kernel takes
  0.84 (DamBreak2D mDBC), 0.73 (MovingSquare2D), 0.77 (DamBreak3D), 0.65
  (Duckling3D mDBC) and the whole step 0.94, 0.76, 0.77, 0.73. With the
  automatic lane split of the small cases (32, 4, 8, 4 lanes) the pair kernel
  takes 1.41, 0.85, 0.94, 0.81 and the step 1.33, 0.87, 0.93, 0.94: the lanes
  stride each of the 25 short row ranges separately, and the mDBC kernel
  time roughly doubles. The default therefore stays 1, the grid whose pair
  orientation reproduces the CPU results; use 2 for large cases that run with
  one lane per particle. Splitting the rows rather than the candidates of a
  row over the lanes, and merging the fluid ranges of a row in the mDBC
  kernel, are the obvious follow ups. A full run of the 3D dam break at
  `dp = 0.0085` (1.6 s simulated, `GPUCellSubdivision = 2`,
  `GPULanesPerParticle = 1`) finished 40 s sooner than with the `H` grid.
* **Lanes per particle (`GPULanesPerParticle`).** With one lane, particle
  `i` gets thread `i`, and that thread visits every candidate itself. With
  `K` lanes, particle `i` gets `K` consecutive threads of a warp instead:
  lane `k` starts at candidate `k` of each row range and steps by `K`, so the
  `K` lanes together visit every candidate exactly once, each accumulating
  its own partial sums. At the end the partial sums are merged across the
  lanes with warp shuffles (`lanes_sum`) and lane 0 writes the result; the
  mDBC kernel does the same for its ghost node matrix. The option exists
  because a small case has too few particles to fill the GPU: an RTX A1000
  holds roughly 16 000 resident threads per pass, so a 3 000 particle 2D
  case with one lane leaves most of the chip idle. The automatic choice
  (`0`) picks `K` such that at least four times the resident thread capacity
  is launched, which gives 32 lanes to a 6 700 particle 2D case, 4 to a
  55 000 particle 3D case and 1 above roughly 100 000 particles. Lanes
  interact badly with half width cells: they stride within one row range and
  only finish together when the range holds many candidates. With `H/2`
  cells there are 25 rows per particle, each about five times shorter, so
  lanes sit idle while the loop overhead is paid 25 times instead of 9 (the
  1.41 above). Even on the `H` grid, 32 lanes took 2.6 times the device time
  of 1 lane on the pair kernel of the 6 700 particle 2D dam break; many lanes
  only help when the GPU would otherwise be idle, and they cost real work in
  shuffles and duplicated loop control. The heuristic was tuned on wall time
  on Windows, where launch overhead dominates the small cases, so device
  time alone is not the whole picture, but a sweep with `benchmark/tune.jl`
  on the small cases is worth doing. Rule of thumb: large cases run with 1
  lane and `GPUCellSubdivision = 2`; small cases keep both defaults unless
  measured otherwise.
* **Fused element-wise kernels.** Half step, density limiting, prescribed
  motion and pressure are one kernel; the final step also computes the
  pressure for the next step and the block wise reduction of the time step
  limits and the maximum displacement. The corrector advances the position
  with the half step velocity times `dt`, the same symplectic scheme as the
  CPU `FullTimeStep`. A step therefore consists of 6 kernel launches (8
  with mDBC and moving bodies), two of which are single block kernels that
  do the loop control.
* **Device resident time step.** The time step, the simulated time, the
  displacement bound that triggers a cell list rebuild and the loop flags
  live on the device. A single block kernel turns the block wise reduction
  of the previous step into `dt` and decides whether the step may run; the
  other kernels read `dt` from device memory and exit at once when it may
  not. The host enqueues a batch of steps (its size estimated from the last
  read back, at most `GPUMaxStepsPerSync`), reads the state back once, and
  only then rebuilds the cell list or writes an output. Because nothing in
  the launch sequence of a step changes from step to step, it is captured
  once as a CUDA graph and replayed (`GPUUseGraph`), which removes most of
  the per launch overhead that bounds the small 2D cases on Windows.
* **mDBC on the GPU.** One thread (or `K` lanes) per boundary particle with
  a ghost node gathers the fluid neighbours of the node, assembles the
  `(D+1)×(D+1)` system in registers and solves it with StaticArrays inside
  the kernel. The launch covers only those particles: the reorder of every
  cell list rebuild also relists the ghost node owners in cell order
  (`GhostIndex`, a stream compaction of the non-zero `GhostPoints`), so the
  warps are dense instead of scattered over the fluid particles that are
  interleaved with the boundary in cell order and exit at once. The neighbour
  loop tests the distance before it loads the particle type, so only the
  candidates inside the support (roughly a third) pay for the second load.
* **Asynchronous output.** Output fields are downloaded asynchronously into
  two pinned staging buffers. A collector copies completed downloads into a
  reusable host frame queue and immediately returns the staging buffers. A
  Progress log lines are emitted and flushed at each output interval, independently
  of queued frame writes. A separate task writes queued frames to disk in order,
  so HDF5 writes do not
  prevent the collector from receiving subsequent frames. The collector runs
  in the simulation's thread pool; the writer prefers the other pool. Use
  `julia -t 2,1` to give collection and simulation enough CPU threads while
  disk writes run separately. With only one CPU thread, blocking writes can
  still interrupt simulation launches.
  `GPUOutputQueueBytes` budgets the queued frame arrays separately from the
  file batching budget (at least one frame is allocated). No frames are dropped:
  sustained disk throughput below frame production eventually fills the queue
  and makes the simulation wait. GPU downloads still precede subsequent kernels
  on the simulation stream. File writes
  are batched: appending one frame to the transient `vtkhdf` file costs some
  35 tiny HDF5 operations (extend and write one element of every
  bookkeeping dataset, update the step count), about 20 µs each, which for a
  3 000 particle frame was three times the cost of the particle data. The
  writers (`PolyDataFrameWriter`, `GridFrameWriter`) keep the dataset
  handles open and collect frames in host memory, at most 16 or as many as
  fit into `GPUOutputBufferBytes`, then write them with one hyperslab write
  per dataset into chunks that hold exactly one batch. The cell grid frame is
  built in preallocated buffers instead of one array per cell. The
  measurements of the previous two-staging-buffer pipeline (Float32, RTX A1000,
  cell grid exported, default threads) show
  the run time has become nearly independent of the number of frames: the
  2D dam break (6 678 particles, 4 448 steps) takes 0.79 s with 2 frames,
  0.82 s with 51 and 0.90 s with 201 frames (1.04 s before batching, 1.63 s
  before the asynchronous download); the 2D still wedge (3 027 particles,
  8 499 steps) takes 0.86 s with 41 frames and 0.96 s with 401 frames
  (0.98 s and 1.49 s before batching, when the simulation thread waited
  0.34 s for the writer). What remains of the difference is the host
  synchronization of the step loop at every output time, the enqueueing of
  the copies and the file close, not the writing. `GPUAsyncOutput = false`
  writes the frame on the simulation thread instead (the download stays
  asynchronous). `benchmark/bench_output_interval.jl` measures the run time
  as a function of the output interval; its `--buffer-mib` option sets the
  frame buffer.
  Pressing `Ctrl+C` during `RunSimulation` drains queued downloads, flushes and
  closes VTKHDF and log files, then opens the results written so far in
  ParaView. A Julia exit while the simulation is running triggers the same
  finalization.
* **Double positions (`GPUDoublePosition = true`).** DualSPHysics stores
  `posxy`/`posz` as doubles and everything else as floats, and hands the
  interaction kernels a `float4 poscell`: the position relative to the
  corner of the particle's cell plus the cell index, so that a pair distance
  is `(poscell₁ - poscell₂) + cellsize * (cell₁ - cell₂)` in single
  precision. The GPU version does the same. With the option on, `Position`,
  the half step position and the mDBC ghost node positions are `Float64`
  arrays on the device (and on the host when the particles are allocated
  with `AllocateDataStructures(SimGeometry, SimMetaData)`); velocities,
  densities, pressures, accelerations, the kernel sums and all constants
  stay `FloatType`. The element-wise kernels add the `Float32` increment to
  the `Float64` position (one promoted add per particle) and write the
  `PosCell` of the half step position; a small kernel does the same for the
  start-of-step position before every neighbour loop that evaluates it. The
  interaction and mDBC kernels take a *pair source* argument, either the
  position array (plain mode, unchanged code path and bitwise unchanged
  results) or the `PosCell` array, and form every pair vector with
  `pair_vector`: `(relᵢ - relⱼ) + s * (cellᵢ - cellⱼ)` with `s = H / R` the
  cell edge and the cell difference `(rowcell - cellⱼ, -dy, -dz)` read off
  the stencil row being scanned. A `PosCell{3, Float32}` is 16 bytes, so the
  pair loop moves the same data as before instead of the 24 bytes of a
  `Float64` position and does no `Float64` arithmetic. A ghost node is
  referred to its own cell in the same way. With `FloatType = Float64` the
  option is a no-op. Measured on the 2D still wedge (DBC, `dp = 0.01`)
  translated by 10 000 m in both directions and run for 218 steps: plain
  `Float32` ends 1.1e-3 from the `Float64` reference with a 2e-3 density and
  8 % velocity error, `Float32` with double positions 1.7e-6, 1.2e-5 and
  2e-4, the same as the untranslated `Float32` run; the step time was
  identical.

Data stays on the GPU for the whole run; the host `StructArray` passed to
`RunSimulation` holds the state of the last written output (reordered by
cell, like the CPU version) when the function returns.

### Measurements

Use `MeasurementConfig` to opt into pressure and velocity probes, water-column
heights, free-surface tracking, or any combination. Each probe uses the
simulation's coordinate dimension; in 2D the vertical axis defaults to 2 and
in 3D it defaults to 3.

```julia
measurements = MeasurementConfig(
    pressure_probes = [MeasurementProbe("gauge", (0.25, 0.5))],
    velocity_probes = [MeasurementProbe("outlet", (1.0, 0.1))],
    water_column_probes = [WaterColumnProbe("column", (0.25, 0.0))],
    free_surface = FreeSurfaceDomain((0.0, 0.0), (1.2, 0.0), 0.02),
    sample_every = 1,
)

RunSimulation(;
    SimGeometry, SimMetaData, SimConstants, SimKernel, SimLogger, SimParticles,
    SimViscosity, SimDensityDiffusion, SimTimeStepping,
    SimMeasurements = measurements,
)
```

The default combined output stores all selected series in the same
`<SimulationName>.vtkhdf` file under `/Measurements`; set
`ExportSingleVTKHDF = true` when constructing the metadata. Samples share the
particle output times, and `sample_every` can reduce their frequency without
changing the simulation step. The GPU computes and writes measurements on its
existing output worker, not in the timestep kernels. Pressure and velocity use
the nearest fluid particle. Water-column height is the nonnegative height above
the probe's vertical coordinate within its radius (the kernel support radius by
default; zero means the surface is below the probe and `NaN` means no particle
was found).
Free-surface values are the highest fluid-particle coordinate in each
horizontal bin; empty bins contain `NaN`. `Locations`, `Names`, `Values`,
`Radii`, `GridShape`, `HorizontalAxes`, and the shared `Time` dataset describe
the measurement groups.
Omitting `SimMeasurements` creates no measurement datasets or extra downloads.
Fields omitted from `OutputVariables` are added only to the existing GPU output
staging when the selected measurements need them.

### Modes are type parameters

As in the CPU package, the optional features of a run are type parameters of
`SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}` rather
than boolean flags, so the fused kernels are specialised at compile time:

| Parameter | Types | Meaning |
|-----------|-------|---------|
| `SMode` | `NoShifting`, `PlanarShifting` | particle shifting |
| `KMode` | `NoKernelOutput`, `StoreKernelOutput` | store the kernel and kernel gradient sums for output |
| `BMode` | `NoMDBC`, `SimpleMDBC` | mDBC boundary condition (needs `ParticleNormalsPath`) |
| `LMode` | `NoLog`, `StoreLog` | write the simulation log file |

Trailing parameters can be omitted, `SimulationMetaData{2, Float32}(...)`
selects all defaults. `RunSimulation` takes the time stepping scheme as
`SimTimeStepping`: `SymplecticTimeStepping()` evaluates two neighbour loops per
step and is the scheme validated against the CPU; `SingleNeighborTimeStepping()`
reuses the corrector derivative of the previous step as the next predictor and
evaluates one neighbour loop per step. Like the CPU scheme it applies the mDBC
correction (with `SimpleMDBC`) to the boundary densities before every half
step, and it evaluates the derivative at the accepted full state (mDBC,
pressure, neighbour loop) before the first step and after every cell list
rebuild. Unlike the CPU it does not re-evaluate the carried derivative every
20 steps. Because both packages share these types
and the `AllocateDataStructures(SimGeometry, SimMetaData)` form, the case
definitions in `benchmark/cases.jl` construct against either package.

### GPU specific `SimulationMetaData` options

| Option | Default | Meaning |
|--------|---------|---------|
| `GPUSyncTimers` | `false` | Synchronize after every phase so the `TimerOutputs` table shows real per kernel times (costs a few percent). |
| `GPUDeterministicSort` | `true` | Sort particles inside each cell after the counting sort; results become bitwise reproducible between runs. |
| `GPUMaxCells` | `50_000_000` | Abort with a clear message if the neighbour grid would need more cells (a particle escaped). |
| `GPUInteractionThreads` | `128` | Threads per block of the interaction and mDBC kernels. |
| `GPULanesPerParticle` | `0` (auto) | Warp lanes that share one particle's neighbour loop (1, 2, 4, ... 32). Small cases cannot fill the GPU with one thread per particle, so the automatic choice launches at least four times the resident thread capacity of the device and lets several lanes scan alternating neighbours, combined with warp shuffles. Use `1` for large cases and together with `GPUCellSubdivision = 2` (see "Lanes per particle" above). |
| `GPUBoundaryForces` | `true` | Evaluate the momentum equation for boundary particles too, as the CPU does (their acceleration only enters the force based time step limit). `false` skips it and saves 10-20 % in cases with many boundary particles, at the price of a slightly different adaptive time step. |
| `GPUAsyncOutput` | `true` | Write output files on a task while the GPU continues (see "Asynchronous output"). |
| `GPUMaxStepsPerSync` | `32` | Upper bound on the steps enqueued between two host read backs of the device resident step state. The actual batch is the estimated number of steps until the next cell list rebuild or output. `1` reproduces a synchronization per step. |
| `GPUUseGraph` | `true` | Capture the launch sequence of a step as a CUDA graph and replay it. Disabled automatically with `GPUSyncTimers`. |
| `GPUCellSubdivision` | `1` | Cells per support radius `H` along each axis: `1` bins at `H` with a 3^D stencil (the CPU's cells), `2` at `H/2` with a 5^D stencil. Same neighbour pairs, fewer distance checks, more cell ranges per particle. Only `1` reproduces the CPU's orientation of the asymmetric density diffusion term (the results of the two grids differ by that term only). `2` pays off with one lane per particle (large cases); with many lanes it is slower. |
| `GPUDoublePosition` | `false` | Store and integrate the particle positions (and the mDBC ghost node positions) in `Float64` while everything else stays in `FloatType`; the pair loops use cell relative `FloatType` coordinates (see "Double positions" above). Use it for `Float32` runs of domains far from the origin or with many steps. No effect with `FloatType = Float64`. Allocate the particles with `AllocateDataStructures(SimGeometry, SimMetaData)` so that the input is read in `Float64`. |
| `GPUOutputQueueBytes` | `256 * 2^20` | Separate host memory budget for reusable frames between the download collector and disk writer; at least one frame. Used with `GPUAsyncOutput`. |
| `GPUOutputBufferBytes` | `256 * 2^20` | Host memory for output frames that are held before they are written to the single `vtkhdf` file (see "Asynchronous output"). The number of frames per flush is the budget divided by the size of a frame (particle fields plus, with `ExportGridCells`, an upper bound of the grid frame), at least 1 and at most 16. `0` writes every frame at once. |

`OutputTimes` also accepts `Float64` values or vectors when `FloatType = Float32`.
The positions of `Float32` runs are written as `Float64` to the output files
in either case (that is the point precision of the writer).

## Layout

```
gpu_version/
├── Project.toml          # package SPHExampleGPU (adds CUDA.jl)
├── src/
│   ├── SPHExampleGPU.jl              # module glue, same exports as SPHExample
│   ├── GPUReductions.jl              # fused SVector block reductions
│   ├── GPUCellGrid.jl                # dense grid, counting sort, reorder kernel, cell relative positions
│   ├── GPUKernels.jl                 # interaction, mDBC, half/final step kernels
│   ├── GPUFloating.jl                # floating rigid bodies: force sums, body update, rigid placement
│   ├── SPHCellList.jl                # device containers, time loop, RunSimulation
│   ├── SPHMeasurements.jl            # output-time pressure, velocity and surface samples
│   ├── PrecompileWorkload.jl         # tiny example runs cached at precompile time
│   └── ...                           # unchanged host side files from src/
├── example/              # GPU versions of the example scripts
├── benchmark/            # CPU/GPU benchmark harness (shared case list)
└── test/                 # unit tests and CPU vs GPU comparison
```

Files not listed under `src/` above are identical to the CPU version except
for small changes that make them precision generic (`Float32` literals), the
`Float32` variant of `Estimate7thRoot`, and the removal of the CPU only
`AllocateThreadedArrays`.

## Tests

```bash
julia --project=gpu_version gpu_version/test/runtests.jl
```

The suite checks the grid helpers and reductions, bitwise reproducibility,
`Float32` sanity, the cell relative pair vectors and the double position
mode (including a case translated by 10 000 m), and runs four cases (2D
mDBC wedge, 2D moving square with shifting and SPS turbulence, 3D dam break,
3D duckling with mDBC) with the CPU package in a separate process and
compares the final state. `test/geometry_particles.jl` checks direct and CSV-backed particle holders,
precision conversion and reusable allocations. `test/generated_examples.jl`
checks that all six main examples with generators prepare particles in memory.
`test/floating_bodies.jl` checks that a floating
body falls freely and stays rigid in air, without numerical error. It also
checks that in water a neutrally buoyant body stays put while a heavier one
sinks. `test/lid_driven_cavity.jl` checks the extended lid geometry, generated
mDBC ghost nodes, bounded wall/fluid densities in a short cavity run, and the
wall-wall continuity regression on a fixed wall next to a sliding wall.

## Benchmarks

```bash
# CPU (use -t N,0 on Julia 1.12: `-t auto` adds an interactive thread which the
# CPU code does not account for and crashes)
julia -t 24,0 --project=. gpu_version/benchmark/benchmark_cpu.jl
# GPU
julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl --float32
julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl --float32 --double-position  # Float64 positions
julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl            # Float64
# kernel level profile of one case
julia --project=gpu_version gpu_version/benchmark/profile_steps.jl --float32 DamBreak3D_dp0.0085
# run time as a function of the output interval (frames written)
julia --project=gpu_version gpu_version/benchmark/bench_output_interval.jl --float32 --time 0.5 --intervals 0.0025,0.01,0.05,0.5 --grid DamBreak2D_MDBC
```

Each case runs the same physics as the corresponding example script for a
few hundred steps after a warm up run that compiles everything. Results
measured on an Intel i7-12850HX (16 cores, 24 threads) and an NVIDIA RTX
A1000 Laptop GPU (16 SMs, 4 GB, 1/32 rate `Float64`) are listed in the table
below (time per step of the time loop, output excluded).

| Case | Dim | Particles | CPU 24 threads, Float64 [ms/step] | GPU Float32 [ms/step] | Speedup Float32 | GPU Float64 [ms/step] | Speedup Float64 |
|------|-----|----------:|------------:|------------:|--------:|------------:|--------:|
| StillWedge2D_MDBC_dp0.02 | 2D | 3,027 | 1.27 | 0.217 | **5.8x** | 3.16 | 0.4x |
| DamBreak2D_MDBC_dp0.01 | 2D | 6,678 | 1.21 | 0.245 | **4.9x** | 4.16 | 0.3x |
| StillWedge2D_DBC_dp0.01 | 2D | 11,246 | 1.83 | 0.302 | **6.1x** | 8.55 | 0.2x |
| MovingSquare2D_dp0.04 | 2D | 33,020 | 7.48 | 0.710 | **10.5x** | 14.24 | 0.5x |
| MovingSquare2D_dp0.02 | 2D | 128,775 | 29.20 | 1.598 | **18.3x** | 41.63 | 0.7x |
| DamBreak3D_dp0.02 | 3D | 17,446 | 8.23 | 1.367 | **6.0x** | 33.52 | 0.2x |
| Duckling3D_MDBC_dp0.01 | 3D | 54,817 | 20.45 | 2.747 | **7.4x** | 72.35 | 0.3x |
| DamBreak3D_dp0.0085 | 3D | 171,496 | 99.78 | 15.425 | **6.5x** | 478.41 | 0.2x |
| Duckling3D_MDBC_dp0.005 | 3D | 365,656 | 161.08 | 15.746 | **10.2x** | 476.85 | 0.3x |

Notes on reading the table:

* Times are the best of 2-3 runs of the whole time stepping loop (a few
  hundred adaptive steps each), file output excluded; measured 2026-09-25.
* The RTX A1000 Laptop GPU is a small 16 SM GPU whose `Float64` throughput is
  1/32 of `Float32` — that is why `Float64` loses to 24 CPU cores here. On a
  datacenter GPU (A100/H100, 1/2 rate) `Float64` scales like `Float32` does.
* The `Float32` speedup grows with the particle count (18x at 129k particles
  in 2D, 10x at 366k in 3D) because large launches amortize the fixed
  per-kernel overhead (~16 µs on Windows/WDDM); a desktop GPU with more SMs
  will extend this trend to even larger cases.
