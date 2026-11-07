# Boat validation — 7 November 2026

Environment: Julia 1.12.7, CUDA.jl 6.4.2, NVIDIA GeForce RTX 5080, Float32
physics and Float64 positions. Cell subdivision remained 1 for boat runs.
No timing comparison or speedup is claimed.

## Geometry

The original DualSPHysics hull is reduced and repaired as described in
[the fixture provenance](../example/data/current_hull/README.md). Its closed
principal-hull mesh volume is 0.01981824304 m³. Run
`julia --project=. -t 2,0 benchmark/boat_discretization.jl` from `gpu_version/`
to reproduce these CPU preprocessing measurements (30 relaxation iterations):

| Spacing (m) | Hull particles | Represented volume error | Minimum spacing / dx | Minimum clearance / dx | Maximum sampled surface gap / dx |
|---|---:|---:|---:|---:|---:|
| 0.025 | 1,066 | −15.96% | 0.9205 | 0.5000 | 2.5467 |
| 0.020 | 2,070 | −16.44% | 0.9251 | 0.5000 | 2.6138 |
| 0.010 | 18,550 | −6.40% | 0.8539 | 0.5000 | 1.8434 |

Represented volume is particle count times `dx^3`, using the solver's uniform
particle mass. Surface gap samples every mesh vertex and triangle centroid;
it is not a rigorous continuous Hausdorff bound. The coarse grids underresolve
the thin hull tips. Neither the nonmonotonic coarse errors nor these three
measurements establish a convergence rate. The 1 cm default improves geometric
coverage, but further refinement is needed for quantitative resistance work.

## CUDA smoke case

`test/boat_simulation.jl`: 0.025 m spacing, 0.03 s, 73,271 allocated particles,
30,769 initially active fluid particles and 1,066 hull particles. CUDA graph
execution completed 186 steps and wrote four VTKHDF frames. It emitted 312
particles and retired 312, preserving the active-fluid count. Final density
range over stored particles was 994.2352–1007.8613 kg/m³. All seven checks passed,
including finite positive density, finite positions and fluid accounting.
These checks do not validate the developed wake or resistance.

The test output is `output/boat_validation/Boat3D.vtkhdf`; the generated
`Boat3D_SingleVTKHDFStateFile.py` opens it in ParaView.

## Focused regressions

Commands from `gpu_version/`:

```sh
julia --project=. -t 2,0 test/boat_geometry.jl
julia --project=. -t 2,0 test/open_boundaries.jl
julia --project=. -t 2,0 test/open_channel_3d.jl
julia --project=. -t 2,0 test/impinging_jet.jl
```

The geometry tests cover binary/ASCII STL, invalid topology, independently
known distances including rotation, packing clearance and initial fluid
exclusion. 3D ghost-density values are compared with independent all-pairs
Shepard sums in Float32/Float64 with cell subdivisions 1 and 2. Uniform channel
runs verify particle accounting, density and velocity with graphs on and off.

The final focused run passed 277 assertions: 20 geometry, 71 existing 2D channel, 33 new 3D channel (including six hydrostatic-background checks), 146 impinging-jet checks and seven actual-hull smoke checks. The full package suite and CPU package suite were not run; changes are confined to the GPU package.

## Two-second corrected boat run

The native solver completed the full 2 s at `dx=0.02` m with output every
0.1 s: 193,809 allocated entries, 2,070 hull particles, 15,370 timesteps and
21 VTKHDF frames. The maximum stored velocity magnitude was 2.1127 m/s, and
the final stored density range was 978.1476–1040.6350 kg/m³. All state values
were finite, all densities positive, and the active-fluid accounting matched:
58,579 initially active + 73,500 emitted − 70,919 retired = 61,160 active.
The final output interval used 768 steps, with no late timestep collapse.
This is a successful coarse solver run, not a resistance or wake-convergence
validation. The default 1 cm case has not been run for the full 2 s.

Reproduce from `gpu_version/`:

```julia
include("example/Boat3d.jl")
result = run_boat(; dx=0.02, duration=2.0, output_interval=0.1,
    save_location="output/boat3d_hydrostatic_validation")
```

The saved file is `output/boat3d_hydrostatic_validation/Boat3D.vtkhdf`;
`Boat3D_SingleVTKHDFStateFile.py` is its generated ParaView loader. A preview
is saved beside it as `boat_preview.png`. Readback independently checked all
21 output times and finite positions/velocities in the final frame.

The boat enables `HydrostaticCorrection=true`. Direct zeroth-order density
extrapolation biases a hydrostatic profile when mirror neighborhoods are
truncated by the bed or side wall. The correction extrapolates density minus
the hydrostatic background and restores the background at the ghost's height.
The focused tests check this operation against an independently computed
profile with a constant density perturbation, including unsupported ghosts,
in both precisions.

The uncorrected coarse cubic and Wendland trials were stopped after their
step counts rose sharply beyond about 1.5 s. At 1.7 s the uncorrected cubic
case used 5,395 steps per 0.1 s; at 1.8 s the uncorrected Wendland case used
14,796. A completed uncorrected one-second snapshot located its fastest
particles near the inlet-sidewall corners, upstream of the hull. The corrected
Wendland run retained approximately 768 steps per 0.1 s through 2 s. These
observations motivate the opt-in correction; they do not establish a general
stability guarantee or an elapsed-time speedup. Generic `OpenChannel` keeps
its original direct-density default for existing cases.
