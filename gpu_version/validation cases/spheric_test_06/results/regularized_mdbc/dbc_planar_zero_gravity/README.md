# DBC with zero-gravity linear diffusion and PlanarShifting

Both n80 runs use DBC and PlanarShifting at CFL 0.10. Both completed
67,905 steps and saved 46 frames through 0.90 s. The comparison changes
only the density-diffusion model. Force errors are relative L2 norms
against the finite-difference reference over 0.10–0.90 s.

[Metrics](metrics.csv) · [Force histories](comparison.png) ·
[Velocity fields](speed_comparison.png) · [Pressure fields](pressure_comparison.png)

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/regularized_mdbc_experiment.jl" dbc half-step-planar-zero
julia --project=. -t 2,0 "validation cases/spheric_test_06/regularized_mdbc_experiment.jl" dbc half-step-planar-zero-gravity
python "validation cases/spheric_test_06/plot_dbc_planar_zero.py" --zero-gravity
```
