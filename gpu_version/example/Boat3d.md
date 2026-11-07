# Fixed boat in a 3D current

Run from `gpu_version/`:

```sh
julia --project=. -t 2,0 example/Boat3d.jl
```

The default is the principal hull from DualSPHysics 5.4
`examples/inletoutlet/07_CurrentHull/Hull.vtk`, converted to a closed STL.
It is a fixed-hull current test. Separate propellers, fittings and decorative
components are omitted. This is therefore a reduced geometry case, not an
exact reproduction or validated resistance prediction for the full reference.
See [geometry provenance](data/current_hull/README.md) for the repair details.

For your own closed STL, pass its path on the command line, or configure units,
placement and resolution explicitly:

```julia
include("example/Boat3d.jl")
setup = boat_setup("my_hull.stl"; scale=0.001, translation=(0.75,0,0.13))
println(setup.packing) # CPU-only preprocessing and diagnostics
result = run_boat("my_hull.stl"; scale=0.001, translation=(0.75,0,0.13))
```

STL has no units. The reference fixture is already in metres; its translation
is `(0.75,0,0.13)`, matching the reference XML. Coordinates are `(x,y,z)` with
vertical `z`. Keep the hull clear of the floor, side walls and open planes by
at least one kernel support radius. The initial waterline must intersect it.

## Geometry discretization

The loader reads ASCII and binary STL, rejects nonfinite/degenerate triangles
and checks that every welded edge has two oppositely oriented incident faces.
STL vertices must already agree at shared edges. It does not silently close
holes or certify that a mesh has no self-intersections.

A bounding-volume hierarchy accelerates exact point-to-triangle distance and
inside/outside ray tests. A Cartesian volume seed is constrained so that solid
particle centers remain at least half a spacing inside the surface. Crowded
seeds from projection around edges are removed; local spatial-hash repulsion
then improves spacing while respecting the distance constraint. Fluid centers
start at least half a spacing outside the solid. The full solid volume is
sampled for DBC, rather than using isolated triangle vertices or a single thin
surface layer. All particles use the solver's standard mass `rho0 * dx^3`.

This follows the signed-distance/constrained-relaxation preprocessing approach
used by [TrixiParticles](https://trixi-framework.org/TrixiParticles.jl/dev/preprocessing/preprocessing/)
and [SPHinXsys](https://www.sphinxsys.org/html/examples/example4_2D_multiResolutionParticleDistribution.html).
It is a geometric repulsion implementation, not their SPH packing equations,
hierarchical winding-number segmentation, or level-set solver. Narrow features
below the chosen spacing need refinement. Inspect the initial particle output
and compare successively smaller `dx` before interpreting forces quantitatively.
`setup.packing` reports particle count, represented volume, minimum neighbor
spacing and minimum surface clearance. The boat setup rejects spacing below
`0.65dx`. This check alone does not establish geometric convergence.

## Inlet and outlet

The channel implementation now supports 2D and 3D. It starts from the settings in
DualSPHysics `CaseCurrentHull_Def.xml`, with the hydrostatic-background
correction below explicitly enabled for this case:

- Domain planes at `x=-0.2` and `x=2`, side walls at `y=+/-0.5`, bed at `z=0`.
- Initial waterline `z=0.2`, uniform velocity `(1.5,0,0)` m/s.
- Five inlet layers and eight outlet layers at the default `dx=0.01` m.
  Buffer thickness must span the kernel support; setup expands it if needed.
- Prescribed positive streamwise velocity at **both** ends. The reference's
  outlet `v=-1.5` is relative to its inward normal pointing in negative x;
  its physical velocity is positive x, not an incoming current.
- Zeroth-order, volume-weighted Shepard extrapolation at mirror ghost
  locations using interior fluid only. The boat opts into
  `HydrostaticCorrection=true`: extrapolate the departure from hydrostatic
  density and restore the background at the ghost height. Unsupported ghosts
  retain that hydrostatic background. Initial and emitted fluid use the same
  hydrostatic density. Ordinary `OpenChannel` keeps its original direct-density
  extrapolation and reference-density fallback by default.
- Inlet emission, fluid-to-outlet conversion and retirement run on CUDA.
  Fluid going upstream through the inlet is removed. Outlet particles that
  return to the interior are reclassified as interior sources.

The floor and side walls extend beneath the buffers; there are no end caps or
periodic connections. This prescribed-velocity one-way treatment is the
reference's configuration. It is not a characteristic/nonreflecting outlet
and is not intended for strong reverse flow. The density fallback also does
not implement DualSPHysics's optional first-order extrapolation. The driver
retains the existing open-channel restrictions: DBC, no shifting, expanding
grid, standard stepping, and no prescribed particle motion or damping.

Other reference settings are `c0=60` m/s, gamma 7 and `h=2dx`,
artificial viscosity `alpha=0.02` with zero boundary multiplier, Fourtakas
inner diffusion, CFL 0.2 and duration 2 seconds. The native symplectic solver
is used. The example defaults to `WendlandC2()` rather than the reference's
cubic spline. Both uncorrected kernels showed a sharp timestep decrease
after about 1.5 s in the coarse run; the fastest particles in a completed
one-second snapshot were near the inlet/sidewall corners. The hydrostatic
correction addresses the truncated support there rather than changing the
kernel alone. To investigate the reference kernel explicitly, use
`run_boat(; sph_kernel=CubicSpline{Float32}())`. Numerical equivalence to
DualSPHysics is not claimed.

## Output and checks

Output goes to `output/boat3d` as VTKHDF and the normal simulation logs/ParaView
files. Marker 20 is the hull, marker 10 the tank, and markers 1/2/3/4 distinguish
interior, inlet, outlet and reserve particles. Inactive reserve and retired
entries remain allocated; they are not active physical fluid.

```sh
julia --project=. -t 2,0 test/boat_geometry.jl
julia --project=. -t 2,0 test/open_boundaries.jl
julia --project=. -t 2,0 test/open_channel_3d.jl
julia --project=. -t 2,0 test/boat_simulation.jl
```

Geometry checks can run without a functional GPU. The 3D channel tests cover
independent all-pairs ghost-density sums in both precisions and subdivisions
1/2, plus uniform-flow preservation and particle accounting with direct and
CUDA graph execution. The separate boat smoke test uses `dx=0.025` and 0.03 s;
it writes to `output/boat_validation`. It verifies finite state and particle
balance, not a converged wake, drag, or the complete two-second production run.
`benchmark/boat_discretization.jl` reports volume error and sampled surface coverage at three resolutions; see [measured validation results](../docs/boat_validation.md).
