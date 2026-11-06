# Impinging jet (2D)

This example is based on DualSPHysics 5.4
`examples/inletoutlet/08_ImpingingJet/CaseJet2D_Def.xml`. The original XML and
upstream LGPL licence are in `data/impinging_jet/`.

Run from `gpu_version`:

```julia
julia --project=. example/ImpingingJet2d.jl
```

For a smaller smoke run:

```julia
julia --project=. -e 'include("example/ImpingingJet2d.jl"); run_impinging_jet(spacing=0.0025, duration=0.008, output_interval=0.008)'
```

The default jet is 0.03 m wide, directed downward at 20 m/s from z = 0.125 m
onto a 0.4 m plate. Initial fluid occupies z = 0.12 through 0.1245 m.
Particle spacing is 0.0005 m; inlet and side outlets use four layers.
The run lasts 0.1 s, with output every 0.0002 s. It uses Wendland C2,
h = sqrt(2) dp, sound speed 230 m/s, CFL 0.1, gravity 9.81 m/s²,
LaminarSPS with kinematic viscosity 1e-6 m²/s, full Fourtakas diffusion,
symplectic stepping and no shifting.

The native inlet fixes density to the reference density and velocity to
(0,-20). Particles convert to interior fluid below the inlet plane.
At x = ±0.2 m, fluid converts to outlet particles; velocity and density
are extrapolated from mirrored interior ghost locations with a zeroth-order
Shepard sum. An empty ghost neighbourhood falls back to zero velocity and
reference density. Particles beyond the outlet buffer are retired. The native implementation also
retires interior particles that pass above the inlet plane or below the plate
buffer; it does not reproduce every upstream domain-exclusion rule.

This initial implementation uses **DBC instead of the reference mDBC**.
Outlet buffers are initially filled, but do not implement DualSPHysics'
full refilling/reverse-flow machinery. It is intended for outward jet flow;
reversing outlets are not supported. No quantitative agreement with the
DualSPHysics pressure or velocity fields is claimed. DBC impact pressure and
outlet transients require validation before using this as a benchmark.

Geometry was checked with the installed GenCase v5.4.354.01. Its seed case
contains 3,248 plate particles, 610 initial jet particles, 61 inlet seeds
and 60 seeds per outlet. The rightmost plate column is x = 0.2025 m
(the GenCase domain is half-open). Native construction retains that lattice
and expands the seeds to four-layer buffers. Inlet reserves are allocated
for the whole duration: at the default settings there are 244,061 inactive
reserve particles. These count toward GPU memory and are not lost fluid.
Output is written to `output/impinging_jet` as VTKHDF.

Focused checks:

```julia
julia --project=. test/impinging_jet.jl
julia --project=. test/open_boundaries.jl
julia --project=. test/fixed_grid.jl
```

The CUDA checks exercise a coarse jet through first impact with graphs both
enabled and disabled, finite particle states and emission/retirement balance.
They do not establish full-resolution convergence or agreement with upstream.

Validation on 2026-11-06 (Julia 1.12.7, CUDA.jl 6.4.2, NVIDIA RTX 5080):
a complete default-resolution 0.1 s run with output every 0.01 s completed
with finite active densities and velocities. It emitted 244,000 particles,
retired 205,128 and ended with 40,206 active fluid particles (1,334 initially).
The remaining 61 reserve particles were inactive. Final active density ranged
from 999.598 to 1005.1282 kg/m^3 and maximum speed was 20.9455 m/s.
These are native-run sanity checks, not a comparison against upstream results.
The jet, existing open-channel and fixed-grid regression files all passed.
