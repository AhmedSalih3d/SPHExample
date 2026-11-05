Original DualSPHysics 5.4 GenCase fixtures from
`examples/chrono/08_WaterMill/CaseWaterMill_Def.xml`.
See `LICENSE.DualSPHysics` for the upstream LGPL 2.1 license.

The full 917,323-point lattice uses dp = 0.025 m and origin (0,0,-0.3).
Counts are 178,050 tank, 40,481 wheel, four buckets of 41,105 each,
126,735 upper-fluid and 407,637 lower-fluid particles. Both fluid groups'
initial density fields are preserved, including the reference's separate
density-depth initialization. Body density is 40 kg/m³ (relative weight 0.04).
The wheel and buckets have masses 25.300625 and 25.690625 kg, respectively.

Sorted `(group,type,i,j,k)` rows have SHA-256
`ec69d0650539237ebae2c5b909b25f5918eb39403176767d4383fd4899df2c0e`.
The surface CSVs preserve all indexed GenCase Actual geometry. No STL reader,
GenCase executable or Chrono library is required for a native run.

The fixed world-Y wheel hinge is at (5.5,0,3), damping 500 Nms/rad.
Bucket hinge anchors are (5.5,0,1.1), (5.5,0,4.8), (7.35,0,3) and
(3.65,0,3), all world Y with damping 100 Nms/rad. Stiffness is zero and
no rigid collision model is enabled in the reference.
