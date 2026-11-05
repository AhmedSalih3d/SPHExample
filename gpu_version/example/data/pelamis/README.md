The data comes from DualSPHysics 5.4 GenCase using
`examples/chrono/04_Pelamis/CasePelamis_Def.xml` and its `Segment.stl`.
See `LICENSE.DualSPHysics` for the upstream LGPL 2.1 license. The native
example needs neither the upstream installation nor an STL reader.

`lattice.csv` stores the complete 30,801-point reference layout and initial
Float32 densities read from `CasePelamis_All.vtk`. Coordinates are integer
indices from world origin (2.8,0,−0.1), at spacing 0.025 m; columns `i,k`
represent world X,Z. Groups retain the actual Mk values: 1 fluid, 10–13
segments, 20 piston and 21 beach. Counts are 26,839 fluid, 1,037 beach,
300 piston and 655,655,655,660 segment nodes. Later STL fills own shared
terminal samples, matching GenCase's overwrite order.

The surface CSVs preserve the indexed cells of the first segment and the
piston from `CasePelamis__Actual.vtk`. The other three segment surfaces are
translations of the original template by 1.3 m, as in the geometry commands.
Each segment has 10,911 shared vertices and 17,542 triangles. The four moving
surfaces therefore have 43,644 vertices and 70,168 faces. The piston has eight
vertices and six faces. Only the two wet beach faces selected by the
reference's boundary mask are displayed; the prism's unsampled end/top faces
are excluded. Full CAD is used for display; the physical solve remains 2D.

World coordinates `(x,y,z)` map to render coordinates `(x,z,−y)`, preserving
handedness. The SPH plane and force vectors use render X,Y with zero Z.
Bodies rotate about render Z, so native `Angle` is minus the reference's
world-Y pitch. Surface motion comes from the simulated centres and angles.
The piston display follows the same prescribed second-order displacement.

The fixed vertical guide is an analytical point-line constraint, not a
particle axis. Native hinge axes are world Y. The solver keeps the existing
2D floating integrator and scalar inertia, with GPU joint projection at
its predictor/corrector stages. No rigid contact model is active here.

In the default state, fluid points are given display vertex cells before
selecting the fluid and triangulating it. The finite Delaunay alpha avoids
bridging body-sized voids. Force arrows are offset 0.55 m above their bodies
for visibility; this changes only the display positions. Blue is gravity,
orange fluid load and red joint reaction. Magnitudes remain in the CSV and
`FloatingBodies` arrays; the 2D model assumes one metre of width.
