# Manual-case coefh 2.0 matrix

The matrix uses the original closed 10 m by 5 m tank, the 1 m translating
square, and the integrated benchmark motion from `manual_case.jl`. The
simulation uses Wendland C2, `h = 2.0 * sqrt(2) * dp`, seven tank-wall
layers, `c0 = 160 m/s`, `CFL = 0.20`, laminar viscosity, and
PlanarShifting. Each simulation runs to 8 s with output every 0.02 s.

- Reynolds numbers 50, 100, and 150
- Geometry resolutions `L/dp = 20, 40, 80`, matching the coefh refinement study
- DBC (`NoMDBC`) and `SimpleMDBC` using the manual geometry builder
- Zero diffusion, zero-gravity linear, fluid-only linear, complex,
  Fourtakas inner (DDT2), and Fourtakas full (DDT3)

The 108 GPU runs retain their particle histories under
`runs/manual_coefh2_matrix/`. The runner verifies and reuses completed
members. A partial particle history requires a separate directory before a
retry. Each force history uses the manual-case saved-density convention,
including the stored wall density for SimpleMDBC.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/manual_coefh2_matrix.jl"
python "validation cases/spheric_test_06/plot_manual_coefh2_matrix.py"
```

Pass `20`, `40`, or `80` to run a subset of the resolutions. Force
histories, per-frame state summaries, aggregate metrics, and reference
comparison plots are in `results/manual_coefh2_matrix/`. The aggregate
compares pressure, viscous, and total drag against the finite-difference
reference over 0.10–8.00 s and reports startup pressure error separately
over 0.10–0.90 s. Negative minimum clearance indicates that a saved fluid
particle center crossed a nominal solid face.
