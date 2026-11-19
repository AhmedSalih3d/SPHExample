# Moving-square wall penetration controls

The native study uses Re = 100, Wendland C2, `coefh = 2.0`, integrated
square motion, zero-gravity linear density diffusion, and the original
closed tank. The control resolution is `L/dp = 40`, the baseline uses
CFL 0.20 and PlanarShifting, and the saved frame interval is 20 ms.

Signed square clearance is the maximum coordinate distance from the moving
square center minus its half-side. A negative value indicates a fluid
particle center inside the nominal square. The diagnostics separately
count fluid centers outside the fixed tank. They use a 0.1 nm tolerance
for native Float64 positions and 1 micrometre for the DualSPHysics
PartVTK export. Saved-frame clearance cannot rule out transient crossings
between frames.

Removing shifting, halving CFL, adding a 1 kPa background pressure, and
aligning the outer square-particle row to the nominal face all left
penetration in both DBC and SimpleMDBC. Aligning the outer row was the
worst intervention. The first baseline crossings occurred at the leading
corners of the moving square. For this lattice, the initial volume-weighted
Wendland sum from square particles is 0.500 at the center of the leading
face and 0.250 at its corner.

The two impedance variants add pressure for approaching ghost fluid.
`DBCPlusImpedance` uses the physical viscous wall velocity.
`UpdatedMDBCWithImpedance` retains UpdatedMDBC's mirrored tangential
velocity for viscosity. Both use their boundary-aware postprocessor.

| Native treatment | L/dp | Duration [s] | Minimum saved clearance [mm] |
|---|---:|---:|---:|
| DBC | 40 | 8 | −16.55 |
| SimpleMDBC | 40 | 8 | −15.99 |
| DBCPlusImpedance | 40 | 8 | +3.87 |
| UpdatedMDBCWithImpedance | 40 | 8 | +3.63 |
| DBCPlusImpedance | 80 | 2 | +1.47 |
| UpdatedMDBCWithImpedance | 80 | 2 | +2.33 |

The full n40 pressure/viscous relative L2 errors over 0.10–8.00 s were
13.1%/17.5% with DBCPlusImpedance and 16.3%/21.8% with
UpdatedMDBCWithImpedance. Plain DBC gave 4.6%/2.6%. The positive clearance
comes with worse force agreement at the tested impedance strength.

DualSPHysics is evaluated from its PartVTK particle positions and original
frame times. The archived n40 run used `coefh = 1.2`. The fine rerun at n80
uses `coefh = 2.0` and DualSPHysics NoFixed shifting, which differs from
native PlanarShifting. The n40 no-shifting rerun provides a closer control
at the same smoothing length and CFL, with each solver's own DBC
implementation and timestepping.
Over the first 2 s of that matched case, DualSPHysics reached −8.48 mm
and native DBC reached −23.26 mm. The archived n40 run reached −1.05 mm.
Across all 401 saved frames in the fine DualSPHysics rerun, the first
crossing was at 0.380 s, the deepest was −5.78 mm at 2.200 s, and at most
six fluid centers were inside the square at once. No saved fluid center
crossed the outer tank walls in the reported runs.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/wall_penetration_controls.jl"
julia --project=. -t 2,0 "validation cases/spheric_test_06/wall_penetration_controls.jl" impedance-full
julia --project=. -t 2,0 "validation cases/spheric_test_06/wall_penetration_controls.jl" updated-impedance-full
julia --project=. -t 2,0 "validation cases/spheric_test_06/wall_penetration_controls.jl" impedance-fine
julia --project=. -t 2,0 "validation cases/spheric_test_06/dual_wall_penetration.jl"
julia --project=. -t 2,0 "validation cases/spheric_test_06/dual_wall_penetration.jl" matched
python "validation cases/spheric_test_06/plot_wall_penetration_controls.py"
```

[Native metrics](metrics.csv) · [Full-run native drag and clearance](full_comparison.png) ·
[DualSPHysics and native clearance](dual_comparison.png) · [DualSPHysics metrics](dual_metrics.csv)
