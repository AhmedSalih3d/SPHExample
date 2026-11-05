# Current wheel and pulley reference fixtures

Exported from DualSPHysics 5.4 `examples/chrono/11_CurrentWheelPulley` using
its GenCase and `benchmark/export_chrono_fixture.py`. The supplied XML,
integer particle lattice, initial density and original indexed Actual CAD
surfaces are bundled under the upstream LGPL licence (`LICENSE.DualSPHysics`).
No DualSPHysics or Chrono installation is required to run the Julia example.

| Variant | GenCase points | Initial active points after buffers | Lattice SHA-256 |
|---|---:|---:|---|
| `:wheel` | 13,050 | 13,397 | `b503382308144466f3b1864ae9c3e74afdeb9541bad2fbe1cde1ae2c4b3ea693` |
| `:pulley` | 13,227 | 13,574 | `29abfaf7837ac291d716977e83c481b099cf1ce3b51e3f480ccb83b4e280757b` |

Both use dp = 0.0075 m, a 1 m/s current, 0.3 m water depth, a snapped
inlet seed at x = -0.0025 m, an outlet at x = 2 m, five inlet and eight
outlet layers. Only the 39 seed rows below the fixed free surface are
active; 82 unused seed points above it remain inactive. Physical 2D uses
world X-Z and unit out-of-plane depth. There is no periodic wrapping.

The 2,271-point wheel has relative density 1.5 (`:wheel`) or 0.5
(`:pulley`), with shaft at (1,0,0.6). The pulley variant adds a 177-point
body with relative density 0.5 and shaft (1.0025,0,1.2025). Its angular
ratio is 0.3/0.05 = 6, in the same direction. Shafts and belt are exact
constraints, without axis or cable particles. The uniform gear's black
rotating pointer is display-only, drawn ahead of the original CAD surface.

Defaults retain four seconds, 0.01 s output, Symplectic, Wendland
h = sqrt(2) dp, c0 = 62.641839 m/s, artificial viscosity 0.05 with zero
boundary factor, Fourtakas diffusion 0.1, DBC and no shifting.

The native open boundary extrapolates zeroth-order Shepard ghost density
and prescribes streamwise buffer velocity. It extends ghost density to
fixed floor particles outside the open planes, avoiding a truncated-wall
continuity artefact. This is a native implementation of the reference
configuration, not a claim of identical DualSPHysics inlet/outlet operators.
See `docs/current_wheel_validation.md` for validation and resource accounting.
