# Impinging jet (2D, mDBC)

This example follows DualSPHysics 5.4
`examples/inletoutlet/08_ImpingingJet/CaseJet2D_Def.xml`. The original XML and
upstream LGPL licence are in `data/impinging_jet/`.

All simulation settings and geometry construction are in
`ImpingingJet2d.jl`: plate particles, mDBC ghosts and normals, initial jet,
inlet and outlet layers, and the inactive inlet reserve. Running or including
the file starts the simulation. No geometry fixture is loaded at runtime.

Run from `gpu_version`:

```powershell
julia --project=. example/ImpingingJet2d.jl
```

For a smaller run through first impact, suppress the automatic run before
including the file and call the simulation with a coarser spacing:

```julia
const RUN_IMPINGING_JET = false
include("example/ImpingingJet2d.jl")
run_impinging_jet(spacing = 0.0025, duration = 0.008, output_interval = 0.008)
```

The default jet is 0.03 m wide, directed downward at 20 m/s from z = 0.125 m
onto a 0.4 m plate. Initial fluid occupies z = 0.12 through 0.1245 m.
Particle spacing is 0.0005 m; inlet and side outlets use four layers.
The run lasts 0.1 s, with output every 0.0002 s. It uses Wendland C2,
h = sqrt(2) dp, sound speed 230 m/s, CFL 0.1, gravity 9.81 m/s^2,
LaminarSPS with kinematic viscosity 1e-6 m^2/s, full Fourtakas diffusion,
symplectic stepping and no shifting.

The default wall treatment is **SimpleMDBC**, corresponding to the original
mDBC formulation used by the upstream case. The physical wall plane is
z = dp/2. Each plate particle at (x,z) owns a reflected ghost at (x,dp-z),
and `GhostNormals` stores the boundary-to-ghost displacement (0,dp-2z).
The mDBC correction reconstructs wall density from neighboring fluid at
these ghosts, using the first-order fit when supported and a Shepard fallback
otherwise. A dry wall resets to reference density so a formerly wet wall
does not retain its previous impact pressure. The original mDBC zero wall
velocity convention is retained.
`UpdatedMDBC` is also available through the `boundary_mode` keyword for
integration checks; it is a different wall formulation and is not the reference
default.

The inlet fixes density to the reference density and velocity to (0,-20).
At x = +/-0.2 m, fluid converts to outlet particles only within the outlet
height, 0 <= z <= 0.03 m. Fluid returning from an outlet into the interior
converts back to interior fluid. Backflow entering the top nozzle is retired
only within its width; spray above the inlet outside that width remains fluid
while it stays within the configured vertical domain. Outlet velocity is
extrapolated from mirrored interior ghost locations with a zeroth-order
Shepard sum. An outlet particle retains its existing velocity when its ghost
has no donor neighbors; coincident ghost/donor pairs are excluded, following
the reference neighborhood rule. Outlet density stays at the reference density: the upstream outlet zones omit
`imposerhop`, whose default is fixed density. Particles beyond the buffer
are retired. The native implementation supports the outward jet flow;
DualSPHysics' general reverse-flow and advanced refilling machinery is not
implemented. Quantitative agreement with the upstream pressure and velocity
fields still requires a paired reference comparison.

Geometry was checked with GenCase v5.4.354.01. Its seed case contains 3,248
plate particles, 610 initial jet particles, 61 inlet seeds and 60 seeds per
outlet. The rightmost plate column is x = 0.2025 m (the GenCase domain is
half-open). The example constructs that lattice and expands the seeds to
four-layer buffers. Inlet reserves are allocated for the whole duration:
at the default settings there are 244,061 inactive reserve particles.
These count toward GPU memory and are not lost fluid. Output is written to
`output/impinging_jet` as VTKHDF.

Focused checks:

```powershell
julia --project=. test/impinging_jet.jl
julia --project=. test/open_boundaries.jl
julia --project=. test/updated_mdbc.jl
```

The jet regressions check the wall plane and reflected ghosts, an independent
constant-density mDBC correction after sorting, pressure and reciprocal-density
caches, dry-wall pressure reset, fixed inlet/outlet density, outlet velocity extrapolation, finite-zone conversion and backflow handling,
and a coarse
jet through first impact. The impact runs exercise CUDA graphs and interaction
caches with mDBC and require a positive wall-pressure response, finite particle
states and balanced emission/retirement. They do not establish convergence or
agreement with upstream results.

Validation on 2026-11-06 used Julia 1.12.7, CUDA.jl 6.4.2 and an RTX 5080.
The jet, open-channel and mDBC regression files passed, including driver-level
wet-to-dry wall correction and finite inlet/outlet zone checks in both precisions.
A complete SimpleMDBC run at dp = 0.0005 m through 0.1 s, with graphs enabled,
interaction caches disabled and output every 0.01 s, passed finite-state and
particle-balance checks. It emitted 244,000 particles, retired 205,093 and ended
with 40,241 active fluid particles from 1,334 initially; 61 reserves remained.
Final active density was 998.7855 to 1004.64386 kg/m^3, wall density was
1000 to 1004.4624 kg/m^3, and maximum fluid speed was 20.806 m/s. The lowest
active fluid particle at the final frame was z = 0.000293006 m, above the
physical wall interface at z = 0.00025 m. These checks validate the native
run and boundary integration; they are not a paired upstream comparison.
