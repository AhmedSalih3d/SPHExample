# ZipLine reference geometry

Source: DualSPHysics 5.4, `examples/chrono/06_ZipLine/CaseZipLine_Def.xml`.
The original CAD meshes are retained in `ragdoll/`, with the distribution's
LGPL 2.1 license in `LICENSE.DualSPHysics`. Attribution: DualSPHysics developers,
https://dual.sphysics.org/ . No Chrono code or library is included or required.

`lattice.csv` contains the cable and ten rigid parts after the reference's
GenCase operations at dp = 0.04 m. Integer indices reconstruct coordinates as
`(-0.5, -1.5, -10) + 0.04*(i,j,k)` metres. Group = mkbound + 1. The container
is generated directly in Julia, with lattice nodes owned by the cable/parts
removed. The complete layout matches all 32,142 reference nodes and markers:

| Group | Object | Particles |
|---|---|---:|
| 1 | Cable (visual guide) | 415 |
| 2 | Container | 31,049 |
| 3 | Head | 111 |
| 4 | Body | 204 |
| 5 / 6 | Left / right arm | 35 / 47 |
| 7 / 8 | Left / right forearm | 32 / 26 |
| 9 / 10 | Left / right leg | 69 / 71 |
| 11 / 12 | Left / right calf | 40 / 43 |

The fixture records GenCase's surface sampling and ownership, including the
reference's small CAD translations. It contains no dynamics or animation.
The body particle density is initialized to 1000 kg/m³; `RelativeWeight = 0.001`
reproduces the specified rigid density of 1 kg/m³ and mass of each part.
The example uses this fixed reference resolution; changing it requires
regenerating the fixture. Primitive mechanics regressions vary resolution
independently of these CAD data.

Provenance commands (run GenCase where `ragdoll/` and the reference material
file are accessible):

```text
GenCase_win64 CaseZipLine_Def CaseZipLine -save:all
PartVTK_win64 -filedata CaseZipLine.bi4 -filexml CaseZipLine.xml -savecsv initial -vars:-all,+idp,+rhop,+mk,+type -csvsep:1
```

All non-container points are rounded back to their integer lattice indices;
the reconstructed container/fixture union was checked against every exported
reference point and marker. The run itself uses the bundled data and Julia/CUDA.
