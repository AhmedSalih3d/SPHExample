Portable GenCase fixture from DualSPHysics 5.4
`examples/chrono/05_OWSC/CaseOWSC3D_Def.xml` and its tank/flap STL files.
The upstream LGPL 2.1 license is in `LICENSE.DualSPHysics`.

The exact 173,388-particle lattice uses dp = 0.05 m and origin (-10,-1,-0.2).
There are 25,926 tank particles (Mk10), 2,294 piston particles (Mk20),
2,436 flap particles (Mk15) and 142,732 fluid particles (Mk1).
Sorted `(group,type,i,j,k)` rows have SHA-256
`b1f9457a178af97043bcc28c567e5974754212fd6db0fdddcda9169fb7b9829e`.
Initial Float32 densities and indexed Actual CAD surfaces are preserved.

The reference prescribes flap centre (0,0,0.33) and diagonal body inertia
(27.6,14.8,13.1) kg m²; these override sampled defaults. Relative weight is
0.461. The world-Y hinge is at the origin, with stiffness 15 Nm/rad and
damping 100 Nms/rad. The piston makes second-order 0.15 m, 2 s waves in
1.3 m depth, with the reference one-period polynomial ramp.
