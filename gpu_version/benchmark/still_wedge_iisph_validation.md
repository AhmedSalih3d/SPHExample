# StillWedge with IISPH

The existing `example/StillWedgeMDBC.jl` now forwards its solver selection to
`RunSimulation(SimSolver=...)`. `WCSPH()` remains the default. Select IISPH by
changing `SimSolver` near the top of that example or passing `--iisph`.
The helper also accepts `run_still_wedge(;solver=IISPH(...),...)`.

The original particle layout is retained: 3,027 particles, comprising 2,447
fluid and 580 fixed particles, spacing 0.02 m, 2.2 m wide tank, 0.04 m floor
and walls, and the hollow 45-degree wedge with apex `(1.1,0.26)` m and 0.06 m
shell. No extra wall layers or replacement tank geometry were introduced.
The initial particle coordinates, IDs and input densities are identical between
the two choices; IISPH recomputes summation density before advancing.

## Selection and numerical settings

```julia
result = RunSimulation(;
    SimSolver = IISPH(max_timestep=0.002),
    SimGeometry, SimMetaData, SimConstants, SimKernel, SimLogger, SimParticles,
    SimViscosity = Laminar(),
    SimDensityDiffusion = ZeroDensityDiffusion(),
    SimTimeStepping = nothing,
    ParticleNormalsPath = nothing,
)
```

For direct calls, use NoMDBC/NoShifting/NoKernelOutput metadata with WCSPH
radial, scalar-SPS and interaction-cache options disabled. The StillWedge
helper selects these compatible settings automatically. Its WCSPH settings
remain artificial viscosity, linear density diffusion, and the selected WCSPH
integrator (symplectic by default). These are different numerical models;
solver selection does not imply identical trajectories or pressures.

IISPH uses Float32 particle arithmetic, Float64 positions, Wendland C2 with
`h=0.04 m`, `H=0.08 m`, subdivision one, a 0.002 s timestep cap, relaxation
0.1, tolerance 0.001, and warm start 0.5. Kinematic viscosity is `1e-6 m^2/s`.
The IISPH CFL factor is 0.25; the retained constants' `CFL=0.5` belongs to
the WCSPH setup. CUDA graphs and asynchronous output remain deferred for IISPH.

## Actual four-second run

Validated on 2026-11-07 using Julia 1.12.7, CUDA.jl 6.4.2, and an NVIDIA
GeForce RTX 5080 (16 GB), with four compute threads and no interactive thread.

| Measure | Result |
| --- | --- |
| Simulated duration | 4.0 s |
| Accepted steps | 2,000 |
| Saved particle frames | 401, including t=0 |
| Fluid identities | All 2,447 preserved in every frame |
| Fixed-particle displacement | Exactly zero |
| Escaped particles / floor or wedge intrusion | None detected in the saved frames |
| Final fluid density | 995.36-1001.00 kg/m^3 |
| Largest saved-frame fluid density | 1001.21 kg/m^3 |
| Final maximum fluid speed | 0.12763 m/s |
| Maximum saved-frame fluid speed | 0.76190 m/s |
| Pressure updates | 64,885 total; maximum 130 in one step |
| Maximum accepted relative linearized density residual | 0.000999939 |

All saved positions, velocities, densities and pressures were finite. The
containment check inspects the original exterior wall faces and hollow wedge
cavity with half-spacing allowance; the reported maximum intrusion relative
to the floor and outer wedge surface was zero. This case result does not
remove the soft-wall limitation observed in the separate dam-break screen.

Normalized SPH fluid-pressure probes at x=0.3 m were compared with
`rho0*g*(0.48-y)`, using the initial highest fluid particle as the reference
water level. The case starts with zero IISPH pressure and settles from that
initial state. The final flow is not perfectly stationary.

| y (m) | Probe pressure (Pa) | Initial hydrostatic reference (Pa) | Difference |
| --- | --- | --- | --- |
| 0.1 | 3397.83 | 3727.80 | -8.85% |
| 0.2 | 2449.63 | 2746.80 | -10.82% |
| 0.3 | 1567.86 | 1765.80 | -11.21% |
| 0.4 | 749.75 | 784.80 | -4.47% |

Individual pressures remain noisy (final maximum approximately 15.82 kPa).
This establishes that the actual example runs and retains its layout; it
does not validate impact loads, pointwise pressure accuracy, or a speedup
over WCSPH. No paired, warmed performance comparison was performed.

## Reproduce and inspect

From the repository root:

```bash
julia --project=gpu_version -t 4,0 gpu_version/example/StillWedgeMDBC.jl --iisph
# Choose an output directory explicitly, retaining the same example path:
julia --project=gpu_version -t 4,0 gpu_version/benchmark/still_wedge_iisph.jl gpu_version/output/StillWedgeIISPH 4.0 0.002
python gpu_version/benchmark/analyse_still_wedge_iisph.py gpu_version/output/StillWedgeIISPH
```

The optional Python analysis uses numpy, h5py and matplotlib. It checks every
saved frame, writes a pressure-profile CSV and JSON checks, and renders four
snapshots. The pressure colour scale matches the initial hydrostatic range;
larger nodal pressures saturate at the top colour.

The particle/grid VTKHDF files, generated ParaView state and optional Typst/PDF
report keep their existing names and formats. The report now reads stored
pressure when available and labels its source correctly. IISPH's per-step
history and a case-summary text file are additional sidecars.

## Regression checks

The solver-choice, shared-output, report, original geometry, middle-square
geometry and generated-example regressions passed (335 assertions). The source fixtures compare the two
solver choices' particle arrays and output metadata directly. The optional
report is also compiled and visually inspected with the existing template.

```bash
julia --project=gpu_version -t 4,0 gpu_version/test/still_wedge_solvers.jl
julia --project=gpu_version -t 4,0 gpu_version/test/typst_report.jl
julia --project=gpu_version -t 4,0 gpu_version/test/generated_examples.jl
```
