# SPHERIC Test 14 · Free heave of rounded floats

The GPU case simulates three unconstrained releases from rest and compares
vertical center-of-mass displacement with the measured mean. The datum is
the specified still-water float position. Both histories retain the original
release time and vertical offset.

| Release | Mass (kg) | Draft (m) | Center of gravity above base (m) | Initial heave (mm) |
| --- | ---: | ---: | ---: | ---: |
| Float 1, down | 9.75 | 0.187 | 0.090 | −74 |
| Float 1, up | 9.75 | 0.187 | 0.090 | +76 |
| Float 2, down | 23.42 | 0.230 | 0.100 | −88 |

The solid particles use axisymmetric profiles from the specification and
top heights checked against the supplied STL meshes. Float 1 has a
hemispherical base. Float 2 has a rounded base with a 50 mm flat radius.
The GPU estimates inertia from the sampled particles about the specified
center of gravity. The benchmark supplies the masses and centers of gravity.

## Reproduce

Download the [official Test 14 archive](https://www.spheric-sph.org/tests/test-14)
and extract it into `validation cases/spheric_test_14/reference`.
The tested ZIP has SHA-256
`03d264db42797da73011e20418b5ffc1a2e763aa7a338c68abe4dbd997b0f8cf`.
The archive names the positive Float 1 history `Fl1+0p76.txt`.

From `gpu_version`, run:

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_14/run.jl"
python "validation cases/spheric_test_14/plot.py"
```

The Julia arguments are particle spacing, tank half-width, duration, and an
optional single release name (`float1_down`, `float1_up`, or `float2_down`).
The defaults are 0.025 m, 1.2 m, and 4.5 s. The plotter accepts the resulting
directory tag, for example `dp0.025_half1.2_t4.5`. It compares raw heave and
reports displacement RMSE, mean period across rising zero crossings, mean
heave in the last second, and maximum horizontal drift. The reported
period is relative to the prescribed still-water datum.

## ParaView

For an editable single run, use [manual_case.jl](manual_case.jl). It puts
the release, tank dimensions, spacing, damping, output settings, and solver
choices in one script, following the Bathymetry example and the Test 06
manual case. Run it from `gpu_version`:

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case.jl"
```

The manual case writes `FreeHeave_Comparison.py`. ParaView runs this
launcher and saves `FreeHeave_Comparison.pvsm` and a PNG preview.
The ParaView layout stacks the fluid cutaway above the synchronized heave
chart, as in the Test 06 particle-and-force view. Edit
`visualize_in_paraview` to control whether ParaView opens automatically.

The batch runner writes a `FreeHeave_Comparison.py` launcher in every
completed release directory under
`runs/<tag>/<release>`. It opens the first release after the invocation.
Pass a release name to run and open that release only. The view places
the animated fluid and float next to the measured and simulated heave curves.
The blue water is cut away in front of the orange float so the submerged
particles remain visible. Press **Play** to animate the particles and move
the chart playhead with them.

Use **File → Load State** to open the `.py` launcher or a previously
generated `.pvsm`. The saved state reads the VTKHDF and two chart CSVs
from the release directory. You can also open the launcher directly:

```powershell
paraview --state="validation cases/spheric_test_14/runs/manual_dp0.025_half1.2_t4.5/float1_down/FreeHeave_Comparison.py"
```

Use `--no-paraview` with `run.jl` for batch runs that only save simulation
output. ParaView state generation and GUI launch are both skipped.

## Pilot settings and limitations

The water depth is 0.8 m. The compact square tank and a cylindrical damping
zone spanning radii 0.7–1.2 m are numerical choices. The published CFD used
an 8 m radial domain with a 1 m outer damping region. The pilot uses hydrostatic
initial density, three layers of DBC walls, Wendland C2 with `h = 1.3 dp`,
`c₀ = 25 m/s`, CFL 0.2, artificial viscosity coefficient 0.01, Fourtakas
inner diffusion coefficient 0.1, and standard symplectic stepping. Particle
shifting is disabled for this free-surface baseline. Julia saves output every
10 ms. The experimental histories are sampled every 5 ms.

The setup establishes a reproducible starting point for particle-spacing,
boundary formulation, and tank-size studies. Such studies are needed to
attribute differences in the period or decay envelope.

## Sensitivity checks

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_14/run.jl" 0.02 1.2 4.5
python "validation cases/spheric_test_14/plot.py" dp0.02_half1.2_t4.5
julia --project=. -t 2,0 "validation cases/spheric_test_14/run.jl" 0.025 1.6 4.5 float1_down
python "validation cases/spheric_test_14/plot.py" --compare dp0.02_half1.2_t4.5 --compare dp0.025_half1.6_t4.5
```

The spacing check holds the tank and release setup fixed. The larger-tank
check keeps the 25 mm spacing and moves the half-metre-wide damping zone
outward with the wall. It tests the combined effect of tank extent and
damping-zone placement for the downward Float 1 release.

## Measured pilot results

All seven GPU runs reached 4.5 s with 451 saved frames each and zero
frozen particles. The heave RMSE used the experimental sample times and
linear interpolation of the simulated trace.

| Release | 25 mm RMSE (mm) | 20 mm RMSE (mm) | 20 mm period (s) | Measured period (s) |
| --- | ---: | ---: | ---: | ---: |
| Float 1, down | 22.16 | 20.07 | 0.807 | 0.871 |
| Float 1, up | 22.03 | 19.34 | 0.818 | 0.865 |
| Float 2, down | 27.16 | 23.94 | 0.942 | 1.008 |

At 20 mm spacing, the late mean heave bias fell in every release. The
oscillations still decayed faster than the measured traces. Widening the tank
from 2.4 to 3.2 m changed the downward Float 1 RMSE from 22.16 to 21.99 mm.
That control leaves the main decay discrepancy open.

[Pilot heave histories](results/dp0.025_half1.2_t4.5/heave_comparison.png)
· [Spacing and tank-width comparison](results/dp0.025_half1.2_t4.5/heave_sensitivity.png)

[SPHERIC benchmark specification](https://www.spheric-sph.org/tests/test-14)
· [Gu et al., 2018](https://doi.org/10.1016/j.jfluidstructs.2018.06.012)
