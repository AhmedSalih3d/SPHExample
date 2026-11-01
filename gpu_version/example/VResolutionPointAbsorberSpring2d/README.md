# 06 Special: point absorber on a linear spring

This case implements DualSPHysics 5.4
`vresolution/06_Special/CasePointAbsorberSpringVRes` in the native Julia/CUDA
solver. The box has the source case's dimensions, 0.30 m wide by 0.16 m high,
and starts at `(2.0,0.4)` m. Its density is 500 kg/m³. Only vertical
translation is free; horizontal translation and rotation are locked.

The bottom anchor is `(2.0,0.0)` m and the initial body attachment is
`(2.0,0.325)` m. The spring has stiffness 100 N/m, damping 500 Ns/m and rest
length 0.325 m. It acts on the fine body once, including its actual attachment
velocity. All link forces, body stages, prescribed piston positions, patch
transforms and interface transfers run on the GPU. Fixed anchors do not
require Chrono. The spring also limits the adaptive timestep when necessary.

The tank is flat up to x=2.5 m and slopes to `(9.2,0.6)` m. Initial still water
level is 0.4 m. A piston at x=0 generates second-order regular waves, height
0.1 m and period 1.2 s, with a smooth one-period ramp. The fine rectangle
starts at `(1.7,0.2)`–`(2.3,0.6)` m and follows body marker 50. The piston is
independent moving marker 10 in the coarse domain.

| Setting | Native example |
| --- | --- |
| Coarse / fine spacing | 0.01 / 0.005 m |
| Kernel | Wendland C2, h=1.5dp, η²=(0.1h)² |
| Equation of state | gamma 7, reference density 1000 kg/m³ |
| Sound speed | 20√(9.81×0.4) m/s |
| Stepping | `SymplecticTimeStepping()`, CFL 0.2 |
| Viscosity | `ArtificialViscosity()`, coefficient 0.05 |
| Density diffusion | `LinearDensityDiffusion()`, coefficient 0.1 |
| Boundaries / shifting | DBC (`NoMDBC`), no shifting |
| Precision | Float64 positions and clock, Float32 fluid/body arithmetic |
| Output | 10 s duration, 0.05 s intervals, CUDA graphs |

These are our ordinary solver parameters. The reference instead used
h=1.2√2dp, viscosity 0.01 with zero boundary factor, advanced shifting and
Chrono coupling. The native wave driver uses the prescribed second-order
theory; it does not reproduce the reference's AWAS feedback after 2 s.
Consequently the bundled body history is a comparison, not a numerical
equivalence target. The moving MLS/flux method is experimental; assess the
represented-fluid-mass history and convergence for longer runs.

Run from the repository root:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/example/VResolutionPointAbsorberSpring2d.jl
```

For a shorter run or changed settings, load definitions only:

```julia
include("gpu_version/example/VResolutionPointAbsorberSpring2d/Setup.jl")
result = run_point_absorber_spring(duration=1.3, output_interval=0.05,
    save_location="C:/TestSimulations/PointAbsorberSpring2D")
```

Use a new empty output directory. Editable physical and solver settings are
in the entry point and `point_absorber_spring_case`. `metadata_options` can
override GPU/output controls. The inclusive fine body lattice has 2,013
particles and mass 25.1625 kg, matching the generated reference. The original
2,832-point coarse boundary skin is retained. Fluid and fine-floor particles
are generated natively; their skin clearance and buffer extent can differ
from GenCase. The full fine floor supports subsequent patch motion.

Outputs include the ordinary particle VTKHDF, `_Floating.csv`, `_Springs.csv`,
the represented-mass `_History.csv`, and `_Comparison.csv` evaluated at the
existing reference times. Open `PointAbsorberSpring2D_Link.vtkhdf.series`
beside the particles to animate the body outline, tracked rectangle and
spring link. The link is drawn as a thin strip; its forces use the axial
spring law, independent of that display thickness.

The bundled reference data was extracted from the user's already completed
run. No DualSPHysics solver or GenCase is invoked during generation, tests,
simulation or comparison. Source hashes and extraction details are in
[the data notes](../data/vresolution_special/README.md).

Focused tests, from the repository root:

```powershell
julia --project=gpu_version gpu_version/test/floating_springs.jl
julia --project=gpu_version gpu_version/test/point_absorber_spring.jl
julia --project=gpu_version gpu_version/test/resolution_attachment.jl
```

A native Float32 check on the RTX 5080 / Julia 1.12.7 reached 2.5 s in 66,050
steps with CUDA graphs. There were no density exclusions or non-finite states.
The body centre ranged from 0.38204 to 0.40394 m, with horizontal position and
rotation locked. Represented owned-fluid mass changed from 1776.500 to
1773.800 kg (−0.152%). This is a short-run regression, not a full 10 s
convergence or DualSPHysics-equivalence result. The existing reference was
only read.
