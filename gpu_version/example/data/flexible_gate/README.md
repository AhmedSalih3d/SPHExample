Fixtures generated with DualSPHysics 5.4 GenCase from
`examples/chrono/03_FlexibleGate/CaseFlexibleGate_Def.xml`.
The upstream LGPL 2.1 license is in `LICENSE.DualSPHysics`.

The integer lattice uses dp = 0.01 m and world origin (-0.1,0,-0.1).
`type` is GenCase's encoding: 0 fixed, 1 moving, 2 floating, 3 fluid.
The 4,489 particles retain their Mk markers and initial Float32 densities.
There are 419 tank particles, five gate blocks with 60/80/80/80/84 samples,
and 3,686 fluid particles. The gate blocks have relative weight 0.8.
Sorted `(group,type,i,j,k)` rows have SHA-256
`bee49ac1ac587c81cab7b3bf8ea6a4ab9ffd6eb89b9937a5fbb972438aa501bb`.

Surface CSVs preserve the original indexed GenCase Domain/Gate CAD.
World (x,y,z) maps to display (x,z,-y), while the physical calculation is
2D in world X,Z with one metre of width. `benchmark/export_chrono_fixture.py`
reproduces these files from GenCase `-save:all` output using ParaView's Python.
No upstream executable, Chrono library, XML parser or STL reader is needed
to run the native example.
