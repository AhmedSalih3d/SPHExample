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
| Kernel | Wendland C2, h=1.2√2dp on both levels, η²=(0.1h)² |
| Equation of state | gamma 7, reference density 1000 kg/m³ |
| Sound speed | 20√(9.81×0.4) m/s |
| Stepping | `SymplecticTimeStepping()`, CFL 0.2 |
| Viscosity | `ArtificialViscosity()`, coefficient 0.05 |
| Density diffusion | `LinearDensityDiffusion()`, coefficient 0.1 |
| Boundaries / shifting | DBC (`NoMDBC`), no shifting |
| Precision | Float64 positions and clock, Float32 fluid/body arithmetic |
| Output | 10 s duration, 0.05 s intervals, CUDA graphs |

The default `h_coefficient=1.2sqrt(2)` matches the reference's 2D smoothing-length
ratio on both resolution levels: h≈0.0169706 m for dp=0.01 m and h≈0.00848528 m
for dp=0.005 m. The other numerical settings remain this package's ordinary
choices. The reference used viscosity 0.01 with zero boundary factor, advanced
shifting with corrected non-conservative pressure, and Chrono coupling.
The native wave driver uses the prescribed second-order
theory; it does not reproduce the reference's AWAS feedback after 2 s.
Consequently the bundled body history is a comparison, not a numerical
equivalence target. The moving MLS/flux method is experimental; assess the
represented-fluid-mass history and convergence for longer runs.

Run from the repository root:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/example/VResolutionPointAbsorberSpring2d.jl
```

Running or including the main example starts the simulation immediately; there
is no `abspath(PROGRAM_FILE)` guard. For a shorter run or changed settings, edit
the keyword defaults in the main example or its final call, for example:

```julia
run_point_absorber_spring(duration=1.3, output_interval=0.05,
    save_location="C:/TestSimulations/PointAbsorberSpring2D")
```

Use a new empty output directory. Editable physical and solver settings are
in the main example: `SimulationConstants`, kernel, `Constraints.LinearSpring`,
`Constraints.FloatingConstraints`, `FloatingDetails`, refinement, metadata and
solver models are visible there. `Setup.jl` contains geometry/wave mathematics
and reference postprocessing. `metadata_options` can override GPU/output controls
while retaining single-file output (`ExportSingleVTKHDF=true`, `ExportGridCells=false`).
The inclusive fine body lattice has 2,013
particles and mass 25.1625 kg, matching the generated reference. The original
2,832-point coarse boundary skin is retained. Fluid and fine-floor particles
are generated natively; their skin clearance and buffer extent can differ
from GenCase. The full fine floor supports subsequent patch motion.

Fluid rows follow the bed with the area-preserving shear
`(x,y) -> (x,y + bed(x))`, clipped at the horizontal still-water level. On the
slope this keeps the first row one vertical spacing above the bed. Cutting
horizontal rows at `bed(x) + dp` left a repeating one-to-two-spacing clearance,
which produced the stepped fluid edge and large periodic DBC wall pressures.
The flat part of the tank retains its original lattice, and the wall skin,
particle mass, DBC model, viscosity and density-diffusion operators are unchanged.
Filling the previously omitted near-bed strips changes the initial fluid count
and represented mass relative to the earlier Cartesian fill; compare mass drift
within each run rather than treating the old initial mass as a target.

All visualization data is saved in **`PointAbsorberSpring2D.vtkhdf`**. Its
`Particles` and `Geometry` blocks share the same frame times. Geometry includes
the tank, body outline, tracked rectangle and a spring strip whose anchor stays
fixed and attachment follows the accepted body position. No separate geometry
file or numbered VTKHDF series is produced. Open the single file in ParaView,
press **Apply**, and use its time controls. The spring's display thickness does
not affect the force law.

`_Floating.csv`, `_Springs.csv`, represented-mass `_History.csv`, and
`_Comparison.csv` retain their numerical histories. Render a saved frame with:

```powershell
pvpython --dr gpu_version/example/RenderRefinement2d.py path/to/PointAbsorberSpring2D.vtkhdf preview.png
```

Add `--all-frames` and give an output directory instead of `preview.png` to
render an animation as numbered PNGs. A reduced-resolution smoke run can use
`dx=0.02, duration=0.002, output_interval=0.001` with
`region=VariableResolutionBox((1.6,0.1),(2.4,0.7))` to retain the required body
clearance at that coarser spacing. This checks output and execution without
running the production resolution or assessing long-run physics.

The bundled reference data was extracted from the user's already completed
run. No DualSPHysics solver or GenCase is invoked during generation, tests,
simulation or comparison. Source hashes and extraction details are in
[the data notes](../data/vresolution_special/README.md).

Focused tests, from the repository root:

```powershell
julia --project=gpu_version gpu_version/test/floating_springs.jl
julia --project=gpu_version gpu_version/test/point_absorber_spring.jl
julia --project=gpu_version gpu_version/test/point_absorber_slope.jl
julia --project=gpu_version gpu_version/test/resolution_attachment.jl
```

An earlier native Float32 check with h=1.5dp on the RTX 5080 / Julia 1.12.7
reached 2.5 s in 66,050 steps with CUDA graphs. There were no density exclusions
or non-finite states.
The body centre ranged from 0.38204 to 0.40394 m, with horizontal position and
rotation locked. Represented owned-fluid mass changed from 1776.500 to
1773.800 kg (−0.152%). This is a short-run regression, not a full 10 s
convergence or DualSPHysics-equivalence result. It predates the matched
`h=1.2√2dp` default and does not validate its long-run behavior. The existing
reference was only read.

The slope regression isolates a static 3 m DBC tank at dp=0.02 m with the same
kernel and solver settings for both particle layouts. At 0.5 s, following the
bed reduced the innermost wall-pressure RMS error from about 5.85 kPa to
0.37 kPa, and maximum error from 16.73 kPa to 0.71 kPa. It checks particle
preservation and finite densities as well as the pressure improvement. These
are reduced-resolution checks of the sampling artifact; they do not establish
long-run convergence or equivalence with DualSPHysics.

A separate 2.5 s check at the same reduced spacing retained the improvement:
wall-pressure RMS error was 5.25 kPa before and 0.30 kPa after the placement
change. The coupled spring/piston/refinement case also completed 1.3 s at
dp=0.02 m in 15,184 steps with no density or non-finite exclusions. No
production-resolution simulation was run to validate this change.
