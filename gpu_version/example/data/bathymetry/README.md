# Bathymetry data

`Bathymetry.xyz` comes from DualSPHysics 5.4, `examples/main/18_Bathymetry`,
copied without changes. [Bathymetry2d.jl](../../Bathymetry2d.jl) uses it.

Its 23,446 rows hold tab-separated x, y and z coordinates in metres: easting
and northing from 439,998.3 to 440,366.7 m and 1,148,505 to 1,148,647.53 m,
and heights from −1 to 16.4 m. The points lie on a 0.3 m lattice, with gaps
where the survey has no data. As the reference's `drawbathymetry` does, the
example moves them by (−439,998.30, −1,148,505.00, 0) m, scales the heights by
1.8 and keeps the 5,200 points with 170 ≤ x ≤ 280 m and 20 ≤ y ≤ 90 m.

Source distribution: DualSPHysics 5.4 (<https://dual.sphysics.org/>).
The distribution's GNU Lesser General Public License 2.1 is retained verbatim
in `LICENSE.txt` alongside this third-party data file. The Julia example is
newly implemented in this repository; this notice concerns the bundled data.

Original SHA-256 before Git line-ending normalization:

| File | SHA-256 |
| --- | --- |
| `Bathymetry.xyz` | `C2122235323B952BBC59EBB48563CD39FAAFF192B5A757C47C2CC5728F86C05E` |
