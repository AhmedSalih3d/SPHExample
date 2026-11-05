# External velocity reference fixtures

Portable exports of DualSPHysics 5.4 `examples/chrono/12_ExternalVelocity`,
using its GenCase and `benchmark/export_chrono_fixture.py`. The upstream XML,
velocity CSVs, integer lattice, original initial densities and indexed Actual
CAD are retained under `LICENSE.DualSPHysics` (LGPL). Ordinary Julia runs need
neither DualSPHysics nor Chrono installed.

| `variant` | Dimensions | dp (m) | Particles | Floating bodies | Duration (s) |
|---|---:|---:|---:|---:|---:|
| `:boxes` | 3 | 0.05 | 17,004 | 2 | 2.5 |
| `:breaking_wall` | 3 | 0.02 | 1,040,455 | 28 | 4 |
| `:many_blocks` | 3 | 0.02 | 1,002,272 | 113 | 4 |
| `:hauling` | 2 | 0.02 | 4,541 | 2 | 4 |
| `:ship` | 2 | 0.005 | 12,791 | 2 | 10 |

Lattice SHA-256:

- boxes: `d6242a071fc245df4959599fc4ea259d82299bd1742f58bb9d7185d37494bb4b`
- breaking_wall: `25e6c9eac39da645f9989b7344cf999e191aa549f864f72ca78b28527170e318`
- many_blocks: `022d281753c72f4cd822b0bdd2377ef5b33dac71aafca12b879cb389e1c0b465`
- hauling: `7262041d2ee71f0ae72b43bf1ba5b32fec9397272d7992404a0fcf79b384fa07`
- ship: `b7140926e2933f613bf48792173566aeea65426f263b7ee9330a510f15616a2d`

Origins are (-0.5,-0.5,-0.5) for boxes, (-1,-4,-1) for wall variants and
(-1,0,-1) for both planar variants. The planar coordinate mapping is world
(X,Z) to native (X,Y); native angular velocity is minus world-Y omega.
Two-dimensional masses assume one metre out-of-plane depth.

The example transcribes inline velocity keys and reads the second box's
original semicolon-separated linear/angular CSVs. Duplicate times are jumps;
`none` releases a whole vector. The two box programs are verified identical.
Bodies retain density ratio 1.2. Wall cases retain a density-ratio-30 steel
ball and 27/112 density-ratio-1 wooden blocks. Hauling retains ratios 2/0.5,
the imposed world-Y 5 rad/s ramp, and the moving-body spring with endpoints
(0.6,0,0.4)/(1.2,0,0.4), stiffness 4000 N/m, damping zero and rest length 1 m.
The disabled `_linearvel` element is deliberately inactive. Ship retains
ratios 0.5/0.5, hinge (1.755,0,0.145), the propeller ramp to world-Y -4 rad/s
at 2 s, and the reference's disabled rigid collision switch. Its actual
exported floating markers are 23/24, because its mkconfig has two fluid slots.

The bundled material file gives soft wood friction/restitution 0.75/0.5 and
steel 0.45/0.8. Native pairs use the smaller coefficient. Native contacts
resolve dp-spaced particle samples rather than CAD triangles, so upstream
Chrono's mesh overlap/compliance settings are not interchangeable with the
native contact diameter. The original CAD is for display, not collision.

All variants retain Symplectic, Wendland h = sqrt(D) dp, no shifting, DBC,
CFL 0.2 and 0.01 s default output. Sound speeds are 60/100/100/160/48.522 m/s.
Artificial viscosity is 0.01 except ship (0.05), with boundary factor one.
Wall/box cases have no density diffusion; hauling/ship use Fourtakas 0.1.
No periodic boundaries are enabled. Dry cases use the existing native travel
controller with ceiling 0.001 s, retaining the existing predictor/corrector.
Ship retains the ordinary coupled SPH timestep.
