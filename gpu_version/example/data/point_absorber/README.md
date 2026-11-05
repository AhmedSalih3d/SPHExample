GenCase fixtures from DualSPHysics 5.4 `chrono/10_PointAbsorber`.
All four supplied variants have identical particle/surface geometry and
initial densities; only their constraints/PTO differ. The bundled data was
generated from `CasePointAbsorber_Def.xml`. See `LICENSE.DualSPHysics`.

dp = 0.01 m, origin (-0.2,0,-0.05), total 19,897 particles: 1,081 bed,
360 piston, 527 floater and 17,929 fluid. Floater relative weight 0.5 gives
26.35 kg for the two-dimensional model's one-metre width. Its initial centre
is world (2,0,0.4). The indexed CAD and initial Float32 densities are preserved.
Sorted `(group,type,i,j,k)` SHA-256:
`c9f778b9154e45750e04fa88d58c66ce85dfe4b77168d3224be6c2142933913d`.

The piston generates second-order 0.1 m, 1.2 s waves in 0.4 m depth with a
one-period startup ramp. Direct restrictions allow only world-Z translation.
The point-line variant's rotation vector is world Z, outside the physical
X-Z plane, so it likewise suppresses planar pitch. The spring PTO has fixed
anchor (2,0,0), body attachment (2,0,0.325), rest length 0.325 m, stiffness
100 N/m and damping 500 Ns/m. The Coulomb PTO uses the same endpoints and
10 N sliding resistance. No periodicity or rigid contacts are enabled.
