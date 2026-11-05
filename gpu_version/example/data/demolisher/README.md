The surfaces in this folder were extracted from DualSPHysics 5.4 GenCase's
`CaseDemolisher_Domain_Actual.vtk`, `CaseDemolisher_Pendulum_Actual.vtk`,
`CaseDemolisher_WreckingBall_Actual.vtk` and `CaseDemolisher_Boxes_Actual.vtk`.
They were generated from `examples/chrono/01_Pendulum/CaseDemolisher_Def.xml`.
See `LICENSE.DualSPHysics` for the upstream LGPL 2.1 license. The Julia example
does not load Chrono or require a DualSPHysics installation.

The combined mesh retains 886 shared vertices and 869 polygons: five
container faces, 192 arm faces, 576 sphere faces and 96 block faces. Moving
surfaces have 878 vertices and 864 cells. CSV vertex IDs are one-based;
`surface_cells.csv` stores the original connectivity with at most four
vertices per cell. Its groups equal the original Mk minus nine: arms 1–4,
ball 5, container 11 and blocks 52–67.

The simulation's particle geometry is procedural in `example/Demolisher3d.jl`.
At dp = 0.05 m it matches GenCase's complete integer lattice, including shape
overwrite ownership: 67,541 container points; 161, 161, 161, 160 arm points;
1,839 sphere points; and 1,331 points in each of sixteen solid boxes. In sphere
coordinates centred on (9.35,5,5.2), the exact GenCase sampling is the integer
set i²+j²+k² ≤ 57 in [-7,7]³. The radius of the display/reference CAD sphere
is 0.35 m. The cylinders have CAD radius 0.025 m; their mask=1 particle rings
are wider at this coarse resolution, as in GenCase.

The pivot at (5,5,5.2) is an analytical constraint, with a display sphere as a
visual guide. Four additional spherical joints connect adjacent arms and the
ball. Reference arms have no collision model file. Native `FloatingGroups`
therefore includes only the ball and blocks. Steel and wood contact
coefficients match the material inputs; native pairs use their minimum.
Particle contacts differ from the reference's Chrono mesh contacts and overlap
allowance. CAD is display geometry, transformed from the actual simulated
centres and quaternions.

The generated default state colours each body, hides the particle points,
shows simulation time and follows all moving surfaces in a detail view. Blue
and red arrows show the ball's gravity and combined joint/contact reaction.
Other body forces remain available in the `FloatingBodies` block and CSV.
