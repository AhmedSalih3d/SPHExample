# Native WaterMill

`example/WaterMill3d.jl` implements DualSPHysics 5.4's
`chrono/08_WaterMill` using the existing GPU floating predictor/corrector
and native damped hinges. The full GenCase geometry, initial densities and
indexed CAD are bundled. Counts, masses, links and provenance are recorded
in `example/data/water_mill/README.md`.

The wheel is hinged to the world; four independent buckets are hinged to
the wheel. Damping is 500 Nms/rad at the world hinge and 100 Nms/rad at each
bucket hinge. No elastic stiffness, contacts, prescribed motion or periodicity
is enabled. Fluid from the upper channel falls onto the mechanism and drains
to the lower flume. The two fluid groups retain their distinct initial density
fields; neither is replaced by an artificial driving force or imposed wheel
rotation.

Defaults preserve dp = 0.025 m, 5 s duration, 0.01 s output, Symplectic,
Wendland h = 1.5 sqrt(3) dp, artificial viscosity 0.01 with boundary
multiplier 1, Fourtakas diffusion 0.1, DBC and no shifting. Native body density
is the reference's 40 kg/m³. The sound speed comes from the reference B.

```sh
julia --project=. -t 2,0 example/WaterMill3d.jl
julia --project=. -t 2,0 test/water_mill.jl
```

The focused tests pass 12 assertions, covering the complete 917,323-particle
layout, all five damping coefficients, fluid coupling, hinge residuals,
finite velocities and original CAD export. The full CUDA suite was not run.
The default state displays the wheel/bucket CAD, tank wireframe, fluid speed
on a fixed 0–8 m/s scale, time and offset force directions. Blue is gravity,
red joint reaction and orange SPH/spring force. Equal lengths show direction;
physical magnitudes and moments remain in the floating CSV/HDF arrays.

A full-resolution 1.3 s validation run covers the pictured 0.70–1.22 s motion.
It completes 15,378 steps, writes 27 frames at 0.05 s intervals and retains
all 917,323 particles with none frozen. The largest joint-position residual
over all frames is 1.25e-7 m. The wheel and independent buckets move under
the falling stream. This run deliberately thins output; the example retains
the original 5 s duration and 0.01 s output defaults, which were not fully run.
It records 270 cell-list rebuilds, 554 step-state readbacks and 11 CUDA graphs.
Solver wall time is 348.7 s on the RTX 5080 system described in the FlexibleGate
report. No matched reference timing comparison or speedup is claimed.

```sh
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_chrono.py output/water_mill_3d/WaterMill3D_SingleVTKHDFStateFile.py output/water_mill_3d/renders 8
```
