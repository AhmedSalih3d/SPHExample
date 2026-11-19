# SPHERIC Test 06 refinement at matched nominal timestep

At Re = 100, vary `L/dp` from 40 to 80 to 160 on the original tank and
square. Use DBC, PlanarShifting, zero-gravity linear density diffusion,
Wendland C2, integrated prescribed motion, `coefh = 2.0`, and a sound speed
of 160 m/s. Run through 8 s with 20 ms output. All drag histories use the
same saved-frame force reconstruction.

Use CFL 0.05, 0.10, and 0.20 at the three spacings. Since
`h = 2 sqrt(2) dp`, this holds the nominal acoustic timestep
`CFL * h / c0` at about 22.1 microseconds. The solver still applies its
state-dependent timestep limits and output clipping. The analysis reads
the realized step counts from the logs. The archived n80/CFL 0.20 case
gives an additional temporal comparison at fixed spacing.
The `fine-time` run halves CFL at n160 and samples the startup through
0.90 s. It checks timestep sensitivity where the n80 control showed the
largest differences.

The difference metrics integrate squared differences between the saved
pressure and viscous histories using piecewise-linear time interpolation.
Each comparison in a given window uses the same normalization. The
apparent order is the base-two logarithm of the coarse-to-medium
difference divided by the medium-to-fine difference. It is a diagnostic
for contraction of the histories at these spacings.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/matched_timestep_refinement.jl"
julia --project=. -t 2,0 "validation cases/spheric_test_06/matched_timestep_refinement.jl" fine-time
python "validation cases/spheric_test_06/plot_matched_timestep_refinement.py"
```

[Drag histories](drag_histories.png) · [Relative differences](differences.csv)

## Result

All three full runs finished with 401 saved frames. The mean timesteps
were 22.072, 22.075, and 22.075 microseconds at n40, n80, and n160.
The two finest runs each accepted 362,400 steps. Over 0.10–8.00 s, the
pressure differences were 8.71% and 8.76% for the successive spatial
refinements, while the viscous differences were 14.47% and 13.20%.
After 1.00 s the pressure differences decreased from 6.00% to 5.07%,
while the viscous differences increased from 10.98% to 19.06%.
Minimum saved square/tank clearance across the three runs was
−34.21, −8.15, and −5.03 mm, respectively.

The n160 half-timestep startup run accepted 81,498 steps through 0.90 s.
Its pressure drag differed by 81.9% over 0.10–0.90 s from n160 at the
matched nominal timestep. At 0.82 s, pressure drag Cd changed from +3.03
to −27.58. These data leave spatial convergence of the full drag histories
unverified and identify substantial timestep sensitivity in startup pressure.
