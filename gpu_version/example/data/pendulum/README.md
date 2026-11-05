These indexed surfaces were extracted from DualSPHysics 5.4 GenCase's
`CasePendulum_Actual.vtk`, generated from
`examples/chrono/01_Pendulum/CasePendulum_Def.xml`. They retain the original
204 vertices and 197 polygon cells: five container faces and four cylinders
with 49 vertices and 48 faces each. See `LICENSE.DualSPHysics` for the upstream
LGPL 2.1 license. No Chrono library is used by this example.

`surface_points.csv` stores one-based vertex IDs and coordinates in metres.
`surface_cells.csv` stores the original shared vertex connectivity. Its group
is the reference Mk minus nine: arms 1–4 and fixed container 6. The container
mesh shows the complete box; GenCase's particle generation bounds clip its
particles to the floor patch at x in [4,6), y in [0.5,9.5), z = 0.

The simulation lattice is generated in `example/Pendulum3d.jl`, independently
of the display mesh. At dp = 0.05 m it matches GenCase's 7,852 particles:
7,200 fixed and 161, 161, 161, 169 floating particles. The cylinders use the
reference eight-point cross-section and shared-end ownership, including the
cap centre retained by the preceding arm. All masses and full inertia tensors
are computed from those particles, with density 1,000 kg/m³.

Four ideal spherical joints connect the arms to each other and to the fixed
pivot at (5,5,5). The pivot is an analytical constraint; its display sphere
does not participate in the particle discretization. The reference selects
`RigidAlgorithm=0`: no collisions, fluid, springs, joint damping or periodic
boundaries. `CaseDemolisher` in the same upstream folder is a separate case.

The example writes moving CAD surfaces into the simulation VTKHDF file and
adds them to the generated default ParaView state. Its Z-up overview and
tracked detail view show gravity in blue and joint reaction in red. Arrow
lengths are normalized to remain visible; inspect the floating-body force
arrays or CSV for magnitudes in newtons. `Force` contains SPH/spring loads and
is zero in this dry case; `TotalForce` includes gravity and joint reactions.
