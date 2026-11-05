Portable fixtures from DualSPHysics 5.4 `examples/chrono/07_DamBreakCubes`.
The upstream LGPL 2.1 license is in `LICENSE.DualSPHysics`. Each directory
preserves the full GenCase lattice, initial Float32 densities and indexed
Actual CAD surfaces. Coordinates use the origin and spacing in the example.

| Variant | Reference | Total / fixed / fluid / floating particles |
|---|---|---|
| `one_cube` | `DamBreak1Cube_Def.xml` | 209630 / 66633 / 140800 / 2197 |
| `three_cubes` | `DamBreak3Cubes_Def.xml` | 214024 / 66633 / 140800 / 3×2197 |
| `blocks` | `CaseSolidsCHRONO_Def.xml` | 115436 / 29108 / 79200 / 18×396 |

Sorted `(group,type,i,j,k)` lattice SHA-256 values:

- one cube: `f35595a156d24d1a3b7daba4385a3f5b961bd8ac4d543da7ce557a4d309db70`
- three cubes: `23122312cab66f3bcb6316fd51d1356d8dc616e984a418192ac5a8130def533d`
- blocks: `85c4d97c3ba9a1772d9917c22919beae746da9d337280a8874c13c35f34c657b`

Cubes use dp = 0.0125 m, origin (-2.05,-0.05,-0.05), mass 2.7 kg each,
PVC friction 0.15 and restitution 0.6. Tank steel uses 0.35 and 0.8.
Blocks use dp = 0.01 m, origin (-0.05,-0.05,-0.05), relative weight 1.25
(0.495 kg each), hard-wood friction 0.7 and restitution 0.6. The tank's
restitution override is 0.65. Pair coefficients use the smaller value.

The block XML explicitly selects `masspart="distributed_bodymass"`, giving
0.00125 kg per floating SPH sample instead of the fluid's 0.001 kg. Native
`InteractionMass` preserves this distinction while rigid mass and inertia
still come from relative weight. Other variants retain uniform SPH masses.
Reference CAD contacts and native sampled-particle contacts are different
discretizations; identical collision trajectories are not claimed.
