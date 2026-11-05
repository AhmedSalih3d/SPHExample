# Native OWSC

`example/OWSC3d.jl` implements DualSPHysics 5.4
`examples/chrono/05_OWSC/CaseOWSC3D_Def.xml` with native CUDA mechanics.
It preserves all 173,388 GenCase particles and initial densities at
dp = 0.05 m: 25,926 tank, 2,294 piston, 2,436 flap and 142,732 fluid.
Fixture provenance, digest and license are in `example/data/owsc/README.md`.

The flap's reference relative weight is 0.461, with mass 140.3745 kg
before XML printing rounds it. Its explicitly specified centre is
(0,0,0.33) m and diagonal body-frame inertia is (27.6,14.8,13.1) kg m².
These override sampled defaults via `FloatingDetails.Center` and `Inertia`.
The world-Y hinge at (0,0,0) has stiffness 15 Nm/rad and damping 100 Nms/rad.
No contacts or periodic conditions are enabled in this reference.

The piston produces second-order regular waves of height 0.15 m and period
2 s in 1.3 m depth. Its one-period ramp is
`s²(6−8s+3s²)`, with s = t / 2 s. The native displacement and analytical
velocity include this ramp and its derivative. Displacement agrees with
the installed reference's paddle CSV to its six-decimal printed precision.
Motion is sampled with Hermite interpolation using exact velocities at
400 samples per period.

The fluid uses the original Symplectic predictor/corrector, Wendland
h = 1.2 sqrt(3) dp, artificial viscosity 0.01 with zero boundary multiplier,
Fourtakas diffusion 0.1, DBC and no shifting. Native joint/torque projection
stays on CUDA. The default is 10 s with output every 0.02 s.

## Validation

```sh
julia --project=. -t 2,0 test/owsc.jl
julia --project=. -t 2,0 test/floating_constraints.jl
julia --project=. -t 2,0 example/OWSC3d.jl
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_chrono.py output/owsc_3d/OWSC3D_SingleVTKHDFStateFile.py output/owsc_3d/renders 0.5
```

The focused OWSC tests pass 31 assertions: geometry, prescribed mass centre
and inertia, invalid properties, wave displacement/velocity checks, and
fluid/piston/native-hinge coupling with CAD export. Existing floating
constraint regressions pass 934 assertions, including old positional
`FloatingDetails` construction. The full CUDA suite was not run.

The complete ten-second native run also passed: 25,000 steps, 501 frames,
173,388 retained particles and no frozen particles. The largest hinge-position
residual across all frames is 2.81e-8 m. It recorded 136 cell-list rebuilds,
688 step-state readbacks and 12 captured CUDA graphs. Solver wall time was
100.4 s on the RTX 5080 system described in the FlexibleGate report, after
a short 0.05 s warmup. This does not constitute a paired, fully warmed
performance comparison with DualSPHysics; no OWSC speedup is claimed.

The default state uses original indexed GenCase CAD and fluid speed points,
with Z-up camera, time and offset force arrows. Blue denotes gravity, red
joint reaction and orange SPH/spring force. Equal arrow lengths show direction;
CSV/HDF arrays retain physical magnitudes. Torsional moment is included in
the body torque array, not represented by a linear-force arrow.
