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

The Julia arguments are particle spacing, tank half-width, and duration.
The defaults are 0.025 m, 1.2 m, and 4.5 s. The plotter accepts the resulting
directory tag, for example `dp0.025_half1.2_t4.5`. It compares raw heave and
reports displacement RMSE, mean period across rising zero crossings, and
maximum horizontal drift. The reported period is relative to the prescribed
still-water datum.

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

[SPHERIC benchmark specification](https://www.spheric-sph.org/tests/test-14)
· [Gu et al., 2018](https://doi.org/10.1016/j.jfluidstructs.2018.06.012)
