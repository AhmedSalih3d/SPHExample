# CurrentHull geometry provenance

`principal_hull.stl` is derived from DualSPHysics 5.4's
`examples/inletoutlet/07_CurrentHull/Hull.vtk`. The bundled reference is
provided by the DualSPHysics project under its distribution's LGPL 2.1
license, reproduced in `LICENSE`. Project: https://dual.sphysics.org/.
`principal_hull.json` records the original file hash and conversion counts.

The source model has 8,829 triangles in many separate components. A strict
solid fill of the complete source is inappropriate: it includes open edges,
nonmanifold propeller components, duplicate triangles and separate fittings.
The retained principal hull component has 2,440 triangles. Its small seams
are welded at a 1 micrometre quantization tolerance. Its one cabin opening is
closed with 38 fan triangles; the lowest cap vertex is at local z=0.1675452 m,
or world z=0.2975452 m after the reference's 0.13 m vertical translation.
The cap is entirely above the initial 0.2 m waterline. The submerged principal
hull surface is preserved to the weld tolerance. Other components, including
propellers, cabin and decorative fittings, are omitted from this reduced
simulation geometry. Breaking waves or a changed waterline can wet the cap;
it is then an intentional geometry approximation.

Reproduce from an installed reference using Julia and the package environment:

```powershell
julia --project=. -t 2,0 example/data/current_hull/convert_reference.jl W:\DualSPHysics_v5.4\examples\inletoutlet\07_CurrentHull\Hull.vtk
```

The converter asserts the expected component size, requires a single directed
boundary loop, rejects wet caps and verifies closed oriented edge pairing.
It is a case-specific conversion, not a general mesh-repair library. Original
coordinates and SI scale are preserved; the Julia setup applies translation.
