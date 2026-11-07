# IISPH fixed-wall examples

This is the initial GPU IISPH milestone. It provides an independent pressure
projection solver for uniform volume samples of fluid and stationary walls.
It does not establish engineering validation or a speedup over WCSPH.

From the repository root:

```bash
julia --project=gpu_version -t 4,0 gpu_version/example/IISPH/run_tank.jl path/to/output
```

The default case has a 0.6 m wide tank, a 0.4 m water column, 0.025 m particle
spacing, five layers of fixed wall particles, gravity, and laminar viscosity.
The kernel has `h=2dx` and support radius `H=4dx`. The wall depth covers this
support plus one spacing, including intrusion into the first sampled layer.
The run starts with zero velocity and pressure, allowing the density constraint
to establish the supporting pressure. This startup is not a hydrostatic pressure
initialization. `IISPHTank_IISPH.csv` (or the chosen simulation name) records the
per-step pressure convergence. ParaView is disabled by default.

`cases.jl` supplies `tank_case(T, save; ...)`. For a dam break, use a wider tank
and a fluid column occupying only its left side:

```julia
using SPHExampleGPU
include("gpu_version/example/IISPH/cases.jl")
using .IISPHCases

case = tank_case(Float32, "IISPHDamBreak";
    name="DamBreak", nx=48, ny=24, fluid_width=16, fluid_height=20,
    duration=0.2, output_interval=0.01,
    solver=IISPH(max_timestep=0.002))
try
    result = RunSimulation(;case...)
finally
    isopen(case.SimLogger.LoggerIo) && close(case.SimLogger.LoggerIo)
end
```

The example uses `NoMDBC`, `NoShifting`, `NoKernelOutput`, zero density
diffusion, and direct synchronous execution. `benchmark/iisph.jl` runs tank and
dam-break screens and reports elapsed whole-run time, accepted timesteps,
pressure iterations, density range and maximum speed. Startup compilation and
output are included; these numbers are not paired, warmed solver speedups.

For accuracy studies, vary resolution, timestep cap and pressure tolerance,
and compare physical observables at equal simulated times. Surface density
deficiency from kernel truncation is distinct from positive compression and
from the linearized residual in the history file. Fixed-wall exported pressure
is zero; use fluid probes to evaluate hydrostatic pressure. Individual fluid
particle pressures remain noisy even with tighter pressure-iteration tolerance;
this milestone does not validate pointwise pressures or impact loads. The
hydrostatic regression lets a more viscous column settle for one second and
compares normalized pressure probes at three depths with analytical values.
The fixed-volume wall is a density constraint rather than hard contact.
Under-supported particles at the dam-break front intrude approximately
half a spacing into the sampled floor at 0.2 s, including with smaller
timesteps. Use this example as a solver development screen, not a validated
wall-impact model.
