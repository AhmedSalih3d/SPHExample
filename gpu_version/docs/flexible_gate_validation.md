# Native FlexibleGate

`example/FlexibleGate2d.jl` reproduces the geometry and configuration of
DualSPHysics 5.4 `examples/chrono/03_FlexibleGate/CaseFlexibleGate_Def.xml`.
The tank is 2 m long; fluid starts behind the gate up to 0.6 m and covers
the downstream bed to 0.05 m. The five gate blocks span z = 0.05–1 m at
x = 0.5–0.53 m. Hinges at z = 0.2, 0.4, 0.6 and 0.8 m connect neighbours;
the upper block is hinged to the world at z = 1 m. All axes are world Y.

Every hinge has stiffness 750 Nm/rad, damping 50 Nms/rad and zero initial
rest angle. Bodies have relative weight 0.8. Initial densities and all
4,489 particles match GenCase: 419 fixed, 384 floating, 3,686 fluid.
Fixture provenance and the complete lattice digest are in
`example/data/flexible_gate/README.md`. There is no periodicity or rigid
contact model in the reference.

Native elastic/damping torques are evaluated on CUDA at the existing
floating predictor/corrector states. Body-pair loads are equal and
opposite. A conservative coupled stiffness/damping rate bound limits the
existing timestep, just as fixed-anchor linear springs do. This does not
add an integration scheme or call an external multibody engine. The 3D
torsion law uses principal relative twist; nonzero-stiffness 3D hinges
must remain within pi radians. Pure damping also supports continuous
rotation. In 2D the scalar angle remains unwrapped.

The original fluid settings are retained: Symplectic, Wendland with
h = sqrt(2) dp, artificial viscosity 0.02 with zero boundary multiplier,
Fourtakas diffusion 0.1, DBC and no shifting. dp = 0.01 m, rho0 = 1000 kg/m³,
g = 9.81 m/s² and sound speed is calculated from the reference B.

## Checks

```sh
julia --project=. -t 2,0 test/rigid_torsions.jl
julia --project=. -t 2,0 test/flexible_gate.jl
julia --project=. -t 2,0 test/rigid_body_planar.jl
julia --project=. -t 2,0 test/rigid_body_dynamics.jl
julia --project=. -t 2,0 benchmark/chrono.jl FlexibleGate2d.jl
```

The focused torsion tests pass 26 assertions, covering input validation,
an independent analytical damped oscillator in both 2D and 3D with both
existing timestepping schemes, and equal/opposite body-pair torques.
FlexibleGate geometry, coupled deflection, joint residual and moving CAD
checks pass 12 assertions. Existing planar and 3D joint/contact regressions
also pass. The full CUDA suite was not run.

The complete two-second native run writes 201 frames, retains all particles
and reaches a maximum pitch of 37.4 degrees. Across all 1,005 body samples,
the largest hinge-position residual is 1.58e-8 m. Comparison with the
reference CSVs gives a 2.96 mm RMS centre difference, a 20.1 mm maximum
centre difference, and 0.375 / 2.14 degree RMS / maximum pitch differences.
Agreement between discretized solvers does not establish experimental
accuracy. Splashes and later body motion are sensitive to timestep and
Float32 fluid-load reduction order.

An initial, more conservative damping limit took 108,600 steps. Separating
the elastic-frequency bound (dt*rate <= 0.2) from the damping-decay bound
(dt*rate <= 0.5, four times below the midpoint decay stability endpoint)
reduces the default to 42,000 steps. Its centre RMS / maximum differences
from the conservative run are 2.36 / 12.3 mm; maximum pitch difference is
1.66 degrees. This is a timestep-sensitivity check, not a convergence-order
claim. The analytical oscillator regressions also pass with the final bound.

On RTX 5080, Ryzen 9 9950X3D, Julia 1.12.7 (`-t 2,0`), CUDA.jl 6.4.2 and
runtime 12.9, a full warmup followed by an unprofiled native run takes
23.585 s including output and final downloads. It records 829 cell-list
rebuilds, 1,758 step-state readbacks and 12 CUDA graphs. DualSPHysics 5.4.355
GPU, runtime 12.8, takes 35.241 / 35.041 s (`TSimul`) for 35,828 steps and
201 frames, with no exclusions. Its 119–125 s initialization is excluded.
The measured native run is approximately 1.49 times faster; only one warmed
native timing was collected, so this is indicative case-level evidence.
CAD assembly and rendering are excluded from both solver timings. Both are
2D with Float32 physics; native particle positions are Float64. These timings
do not establish a whole-solver memory advantage.

The default ParaView state displays the original GenCase CAD, a fluid-speed
surface and simulation time. Blue arrows indicate gravity, orange SPH/spring
force and red constraint reaction. Arrow lengths are normalized to show
direction, with their display origins shifted right of the gate to avoid
occlusion. Physical force magnitudes and elastic torque remain in the
floating-body CSV/HDF arrays. The 2D calculation assumes one metre of width.

```sh
pvpython --no-mpi --disable-registry --force-offscreen-rendering test/paraview_chrono.py output/flexible_gate_2d/FlexibleGate2D_SingleVTKHDFStateFile.py output/flexible_gate_2d/renders
```
