# Wave run-up data

These files are copied without changes from DualSPHysics 5.4,
`examples/main/17_WaveRunup` (`CaseWaveRunup`).
[WaveRunup3d.jl](../../WaveRunup3d.jl) uses the geometry and piston signal.

- `Slope.stl`: binary STL with 8 triangles forming a convex wedge.
- `Blocks_3D_scaled.stl`: binary STL with 1,176 triangles forming 98 separate
  convex armour blocks, 12 consecutive triangles per block. Use the scaled
  file specified in the reference XML, rather than `Blocks_3D.stl`.
- `Mov_piston.dat`: time in seconds and horizontal piston displacement in
  metres, sampled every 0.025 s. The first `(0, 0)` row is repeated; the example
  removes this identical duplicate before linear interpolation. It stops the
  signal at 15 s, as the XML's `mvpredef` does, and then holds zero displacement.
- `EXP_CaseWaveRunup_CIEMito.txt`: experimental time, four wave-gauge elevations
  and run-up height from the CIEMito flume, in seconds and metres. The time
  column starts before zero; preserve it when selecting a comparison interval.
  The wave gauges report deviations from still water, whereas the run-up
  column reports absolute height (its still value is 0.25 m).

The XML rotates both STLs by −90° about z and then moves them by
`(5.95, 0.37, 0)` m: `(x, y, z) → (5.95 + y, 0.37 − x, z)`.
The example applies exactly that transformation and retains all block poses.
It samples the convex interiors on its shared particle lattice; it does not
load a precomputed particle cloud or require GenCase at run time.

Source distribution: [DualSPHysics 5.4](https://dual.sphysics.org/).
The distribution's GNU Lesser General Public License 2.1 is retained verbatim
in `LICENSE.txt`. The Julia example is newly implemented in this repository;
this notice concerns the bundled third-party data.

Original SHA-256 values before Git line-ending normalization:

| File | SHA-256 |
| --- | --- |
| `Slope.stl` | `BB6ED4811E1D24A05D72F2212B887FDC46AB354D93DF13481618557333F95C90` |
| `Blocks_3D_scaled.stl` | `528052692D4802601A7DBD92BFA9BCC00605559E1920628EADAE3B76A39CF118` |
| `Mov_piston.dat` | `C8319AF36CA85DD33E56B722D888D1646912B0F602577D73814FAC267DF26A2E` |
| `EXP_CaseWaveRunup_CIEMito.txt` | `6195706559C8565E22DFEF1AFE7E5E72F4DE3E5C7B4BB97133B71FE8EF88E7D6` |
