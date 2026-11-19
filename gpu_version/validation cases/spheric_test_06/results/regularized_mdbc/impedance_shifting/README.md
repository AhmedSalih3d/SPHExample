# DBCPlusImpedance with PlanarShifting

Both runs use Re=100, L/dp=80, CFL=0.10, c0=160 m/s, integrated translation
and fluid-only density diffusion. Each completed 67,905 steps and saved
all 46 frames through 0.90 s. The only changed simulation setting is
`PlanarShifting`. Errors are unfiltered relative L2 norms against the
finite-difference reference over 0.10–0.90 s.

| Measure | NoShifting | PlanarShifting |
| --- | ---: | ---: |
| Pressure drag error | 33.27% | 24.64% |
| Viscous drag error | 13.65% | 15.42% |
| Pressure Cd at 0.82 s | +3.29 | −0.71 |
| Pressure Cd at 0.88 s | +3.14 | +3.01 |
| Minimum saved clearance | −1.99 mm | +1.95 mm |
| Mean fluid pressure rise at 0.90 s | 2.57 kPa | 3.81 kPa |
| Peak fluid pressure inside the central rear wall stencil | 8.89 kPa | 5.07 kPa |

All 20 central rear-wall particles have fluid within kernel range at every
saved frame in both runs. The shifted reversal at 0.82 s thus occurs with
central rear support present. The shifted fluid pressure in the plotted
area peaks at 35.45 kPa near the lower rear corner at that time. The
positive clearance establishes particle
separation in the 20 ms output frames. Crossings between outputs remain
unmeasured.

[Velocity fields at 0.82 and 0.88 s](speed_comparison.png) ·
[Fluid pressure fields](pressure_comparison.png) ·
[Drag and clearance histories](comparison.png)

Run these commands from `gpu_version`. Keep both particle histories for
the field comparisons.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/regularized_mdbc_experiment.jl" impedance half-step
julia --project=. -t 2,0 "validation cases/spheric_test_06/regularized_mdbc_experiment.jl" impedance half-step-planar
python "validation cases/spheric_test_06/plot_impedance_shifting.py"
```
