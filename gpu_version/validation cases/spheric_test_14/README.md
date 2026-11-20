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

Every completed release saves a standalone `<release>_heave_comparison.png`
next to its CSV in `results/<tag>`, before ParaView starts. This requires
Python 3.11 or newer with NumPy and Matplotlib. To regenerate the plot for
one existing release, run `plot.py <tag> --release float1_down`.

For an editable single run, use [manual_case.jl](manual_case.jl). Running
the file or including it starts the simulation immediately. Edit the release,
tank, damping, and solver settings at the top of `run_manual_case()`.
Set `kernel_family` to `WendlandC2()` or `CubicSpline{FloatType}()` and
`kernel_h_factor` to the desired ratio `h / dx`. The run record saves the
kernel and actual smoothing length. Run it from `gpu_version`:

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case.jl"
```

The manual case writes `FreeHeave_Comparison.py`. ParaView runs this
launcher and saves `FreeHeave_Comparison.pvsm` and a PNG preview.
The ParaView layout stacks the fluid cutaway above the synchronized heave
chart, as in the Test 06 particle-and-force view. The layout follows the
available space in the ParaView window, and the PNG captures that layout
at its rendered size. Edit
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

## Held-float buoyancy check

Hold Float 1 at its prescribed draft of 0.187 m for 3 s and compare the
vertical fluid load to its 95.65 N weight. This control uses two spacings
with the same `h/dx = 1.3`, laminar viscosity, linear density diffusion,
and the 2.4 m tank with its original damping zone. The float position is
checked at every saved time. The initial frame precedes the first force
evaluation and is omitted from the plot.

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_14/held_float.jl"
python "validation cases/spheric_test_14/plot_held_float.py"
```

Over 2–3 s, the mean vertical fluid forces were 93.12 N at 25 mm and
93.79 N at 12.5 mm, respectively 2.52 N and 1.86 N below the weight. The
static load deficit has the opposite sign to the upward drift observed
in the free-heave runs. [Held-float force history](results/held_float_t3.0/buoyancy_comparison.png)

[SPHERIC benchmark specification](https://www.spheric-sph.org/tests/test-14)
· [Gu et al., 2018](https://doi.org/10.1016/j.jfluidstructs.2018.06.012)

## Cartesian 2D section

[Gu et al., §3](https://doi.org/10.1016/j.jfluidstructs.2018.06.012) reduce
the heave CFD to an axisymmetric wedge with azimuthal periodicity. The
Cartesian SPH solver uses a constant-width section. The two models have
different displaced-volume, added-mass and wave-spreading behavior.

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case_2d.jl"
```

The [2D manual case](manual_case_2d.jl) uses the same float profile, release
offset and 0.8 m water depth, with 8 m of water to either side and a 1 m
damping zone at each end. Translation and rotation are locked except for
heave. The rigid-body mass per metre of span is calculated from the
submerged section area at the specified draft. The case compares heave
with the experiment as a Cartesian sensitivity check. It saves the heave
PNG before opening the synchronized ParaView view.

The Float 1 downward release at 12.5 mm spacing completed through 4.5 s.
The simulated period was 0.866 s versus 0.871 s measured, while the full
heave RMSE was 20.89 mm. [Cartesian heave comparison](results/planar_dp0.0125_half8.0_t4.5/float1_down_heave_comparison.png)

### 200 Hz benchmark-format output

By default the 2D manual case runs all three releases. Set `selection` to a
release name for one run. The solver saves a frame every 0.005 s and writes
headerless, tab-separated time (s) and simulated vertical displacement (mm)
files in `results/planar_dp0.0125_half8.0_t4.5_200Hz`.

| Release | Simulated file |
| --- | --- |
| Float 1, −74 mm | [Fl1-0p074.txt](results/planar_dp0.0125_half8.0_t4.5_200Hz/Fl1-0p074.txt) |
| Float 1, +76 mm | [Fl2+0p076.txt](results/planar_dp0.0125_half8.0_t4.5_200Hz/Fl2+0p076.txt) |
| Float 2, −88 mm | [Fl2-0p088.txt](results/planar_dp0.0125_half8.0_t4.5_200Hz/Fl2-0p088.txt) |

The upward release follows the filename on the benchmark page. The downloaded
experiment for that release is `Fl1+0p76.txt`. The plots use the experimental
means from the archive. The text files do not include the ranges shown on the
benchmark page.

[All three heave comparisons](results/planar_dp0.0125_half8.0_t4.5_200Hz/heave_comparison.png)
