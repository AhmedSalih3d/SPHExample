# Reference pressure and artificial-viscosity fusion

Ordinary reference interactions now combine pressure and artificial-viscosity
coefficients before their shared gradient multiplication when interaction fast
math is enabled. The density-diffusion formula, pair orientation, kernel gradient,
boundary viscosity factor and particle-mass scaling are retained. This changes
floating-point association and can change trajectories.

The selected diffusion remains reported independently, for example
`density_diffusion_operator = :reference` with `force_fusion = true`. Set
`GPUInteractionFastMath = false` for the original separate reference forces.
Standalone `launch_interactions!` additionally accepts `fuse_reference_forces`,
defaulting to `fastmath`, for an independent comparison. Existing radial force
fusion remains enabled. Other viscosity models, the DualSPHysics-style force
path and conservative variable-resolution execution retain their existing
arithmetic. No device buffers or separate launches are added.

## Reproduce

From `gpu_version`:

```powershell
julia --project=. -t 4,0 test/reference_force_fusion.jl
julia --project=. -t 4,0 benchmark/operator_fusion.jl --reference --rounds=7 --duration=0.03 --output=benchmark/reference_force_fusion_results.csv StillWedge2D_MDBC_dp0.02 DamBreak3D_dp0.02 DamBreak3D_dp0.0085
julia --project=. -t 4,0 benchmark/operator_fusion.jl --reference --rounds=9 --duration=0.03 --output=benchmark/reference_force_fusion_confirmation_results.csv DamBreak3D_dp0.0085
julia --project=. -t 4,0 benchmark/operator_fusion.jl --reference --strict --rounds=3 --duration=0.003 --max-position-dx=0 --output=benchmark/reference_force_fusion_strict_results.csv StillWedge2D_MDBC_dp0.02 DamBreak3D_dp0.02
julia --project=. -t 4,0 benchmark/operator_fusion.jl --reference --rounds=5 --duration=4 --output=benchmark/reference_force_fusion_long_results.csv StillWedge2D_MDBC_dp0.02
```

The benchmark's baseline restores separate vector forces in a cloned current
kernel. The initial measurements held a complete-batch graph experiment constant.
The publication confirmation uses the original power-of-two scheduler in a
force-only checkout, excluding that unproven graph optimization. A same-source candidate clone must match the installed
solver exactly before comparison with old force arithmetic. Warmups precede
alternating unprofiled complete `RunSimulation` calls, including graph capture,
stepping and particle output. Package loading, geometry/host-particle construction,
pre-run GC and output verification are excluded. Requested physics, output
settings and work counts match between variants.

Both use Float32 physics, Float64 positions, cached interaction fields, reference
linear diffusion, type preloading, `(R,Rx)=(1,4)`, symplectic stepping and CUDA
graphs. Reference diffusion retains its existing orientation on that grid;
this is not a CPU-parity claim. The StillWedge case here uses **Simple mDBC**
and its input files; the generated `StillWedgeMDBC.jl` example uses NoMDBC and
already enables radial force fusion.

## Measurements

Measured 2026-11-14 on an NVIDIA RTX 5080 / driver 616.92, Julia 1.12.7,
CUDA.jl 6.4.2, Windows, `-t 4,0`. Other test/benchmark processes were not run
concurrently; desktop applications and idle Julia sessions remained open.

| Case / duration | Baseline median | Fused median | Median paired ratio | Paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| StillWedge mDBC, 3,027 particles / 0.03 s | 0.023396 s | 0.026741 s | 1.05251 | 5.25% slower | 3/7 |
| DamBreak 3D, 17,446 particles / 0.03 s | 0.037580 s | 0.036637 s | 0.98567 | 1.43% less time | 4/7 |
| DamBreak 3D, 171,496 particles / 0.03 s | 0.379235 s | 0.372919 s | 0.97990 | 2.01% less time | 7/7 |
| StillWedge mDBC, 3,027 particles / 4 s | 0.330667 s | 0.342831 s | 1.01409 | 1.41% slower | 1/5 |

The large 3D paired ratios span 0.96294–0.98877; this short workload establishes
a modest case-specific elapsed reduction. Small-case ranges are much wider:
0.90981–1.25174 for short StillWedge, 0.94474–1.07222 for small DamBreak, and
0.94973–1.08225 for four-second StillWedge. No StillWedge speedup is established.
These results do not establish a speedup for a full-duration 3D production run.

### Force-only confirmation with the original graph scheduler

Nine further alternating warmed pairs on the 171,496-particle 3D case used
only the force changes, with the original power-of-two graph scheduler. The
median paired ratio is **0.97770**, or **2.23% less complete runtime**, faster
in **9/9 pairs**. Ratios span 0.96920–0.98596; median totals are 0.352485 s and
0.344586 s. All pairs retain 339 steps, 2 rebuilds and 14 state readbacks.
Numerical differences match the short large-case comparison reported below.
This confirms the benefit without publishing the graph-batching experiment.
The physical-duration target remains 0.03 s, so this is case-specific evidence.

All short-run pairs match accepted step/rebuild/readback counts: 64/1/4,
144/1/7 and 339/2/14 respectively. Maximum final-position differences are
5.89e-6, 3.88e-6 and 1.16e-5 particle spacings. Final density differences are
0.001160, 0.000427 and 0.000793 kg/m^3. Particle identities, output shapes and
frame counts match; values and timestamps can differ.

The four-second mDBC comparison matches 8,498 steps, 2 rebuilds and 402 readbacks.
Maximum final-position difference is 0.0009674 spacings, approximately **0.01935 mm**.
Maximum velocity, density and acceleration differences are 3.094e-5 m/s,
0.000916 kg/m^3 and 0.02710 m/s^2. Maximum output-time difference is 9.55e-9 s.
These pass the explicitly reported 0.1-spacing screening threshold; that threshold
and this case are not a universal accuracy or convergence guarantee.

Strict reference controls at 0.003 s match accepted state, time and checked
particle output exactly on both cases, over three pairs each. They verify the
opt-out, not a speed benefit. Focused tests compare fused/unfused interaction
outputs for 2D/3D, both precisions, cached/uncached fields, multiple diffusion
models, wall factors, lane counts, kernel types and mass scaling.

Raw data: [short timings and numerical differences](reference_force_fusion_results.csv),
[force-only confirmation](reference_force_fusion_confirmation_results.csv),
[four-second comparison](reference_force_fusion_long_results.csv), and
[strict controls](reference_force_fusion_strict_results.csv).

## Publication regression result

The force-only checkout, retaining the original graph scheduler, passed 3,240
assertions across 15 testsets from `reference_force_fusion.jl`,
`batch_commit.jl`, `effective_configuration.jl` and `radial_benchmark.jl`.
These ran sequentially in one Julia 1.12.7 process with `--project=. -t 4,0`
and exited successfully. The focused force entry point contains 2,160 assertions.
`git diff --check` passed. Earlier combined graph/force checks also covered
sorting, Updated mDBC, grid output, periodicity and interaction masses, with
8,652 passing assertions; the graph experiment is not part of this change.
The full package/CPU comparison suite was not run.
