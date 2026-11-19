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
# Matched UpdatedMDBC check

The archived `updated-mdbc/zero-gravity-linear/Re100_n040` run uses the
same manual geometry and settings as the SimpleMDBC control, changing only
the boundary mode. It completed all 401 frames through 8 s.

| Boundary | Pressure relative L2 | Viscous relative L2 | Total relative L2 | Viscous peak Cd | Peak time (s) |
|---|---:|---:|---:|---:|---:|
| Finite-difference reference | - | - | - | 0.7885 | 0.702 |
| DBC | 0.0460 | 0.0258 | 0.0418 | 0.7976 | 0.700 |
| SimpleMDBC | 0.0521 | 0.0328 | 0.0466 | 0.8012 | 0.700 |
| UpdatedMDBC | 0.0649 | 0.4544 | 0.0479 | 0.5686 | 0.660 |

UpdatedMDBC underpredicts the reference viscous peak by about 28%.
Total drag error alone masks the deterioration in the viscous component.
This check is specific to Re = 100 and L/dp = 40.

## Fine-resolution DBC comparison with DualSPHysics

[Drag overlay](results/manual_coefh2_matrix/dual_native_dbc_n080.png) compares
the archived native trace with DualSPHysics 5.4 and the finite-difference
reference. Peak viscous drag Cd was 0.955 (native), 0.928 (DualSPHysics),
and 0.788 (reference).

Both timing runs used the RTX A1000 Laptop GPU, Re = 100, L/dp = 80,
coefh = 2.0, c0 = 160 m/s, CFL = 0.2, and 401 frames through 8 s.
The native run took 1661.31 s (181,200 steps); DualSPHysics took 2297.95 s
elapsed (2296.71 s solver-reported; 184,163 steps), excluding GenCase and
force postprocessing. Native elapsed time was about 28% lower in this
single comparison. Shifting (PlanarShifting vs. NoFixed), output paths,
and timing scopes differ, so this is not an isolated kernel speedup.
