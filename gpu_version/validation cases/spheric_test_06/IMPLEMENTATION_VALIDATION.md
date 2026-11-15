# Implementation checks

The optional GPU divergence-cleaning regression passed **205 assertions**,
including independent dense operators at positive and negative gauge pressure
in 2D/3D and Float32/Float64, sorting, density-diffusion separation, scalar
decay, cleaning CFL, fused prediction, a damped discrete Fourier wave,
graph/direct agreement, stationary mDBC and translated periodic crossings.

```sh
julia --project=. -t 2,0 test/divergence_cleaning.jl
julia --project=. -t 2,0 test/runtests.jl
```

The canonical CUDA suite was attempted and resumed in sections after repairing
stale test assumptions about composite VTKHDF particle roots, equivalent PTX
load spellings and multiple configuration fallbacks. The generated-example
capture now rejects logger/cleanup paths outside its temporary directory and
disables report generation for the mocked solver; its 68 assertions passed.
Large groups covering memory, predictor fusion, graphs, reductions, sorting,
outputs, periodicity, motion, damping and geometry passed.

The broad suite is **not fully green**. It stops at the exact-equality checks
in `test/variable_resolution_solver.jl`, testset
`normal timestep schemes reuse ordinary solver numerics`. An isolated copy
using the published `31e16b5` versions of the three changed integration modules
reproduces the same differing arrays (including Float32 densities
`[1000.0391,1004.9523]` versus `[1000.03937,1004.952]`). Those assertions have
not been relaxed. Later suite sections were not reached in that run.

The canonical core testset was then run separately: **18,485 assertions passed
and two failed**, both the Float32 floating-body angle check `abs(angle)<0.05`.
Current angles were about 0.06312 and 0.06946 radians for the two ordinary
schemes. The isolated published-source runs also fail the same bound, with
angles about 0.06340 and 0.06957. No angle tolerance was relaxed. These are
reproduced baseline failures, not a fully passing broad suite.

The core runner reached its CPU-reference selection, but skipped those
comparisons because the parent CPU checkout lacks the required mode-type API.
No CPU/GPU reference comparison pass is claimed.

The new optional kernel argument initially broke direct variable-resolution
helper calls. Backward-compatible defaults on the predictor/corrector kernels
repair that introduced error; the scripted-helper/model/timestep test then
passed its six assertions before the independently reproduced equality failure.
Cleaning itself explicitly rejects variable resolution.

No Julia dependency changed. Development runs used Julia 1.12.7 with explicit
`-t 2,0` and `--compiled-modules=existing` to avoid repeatedly rebuilding package
caches. Scientific run data, pressure-extension failures and benchmark limitations
are documented separately; passing implementation tests is not a claim that the
moving-square DBC setup passes physical validation.

The timestep regressions additionally passed **19 assertions** in
`test/timestep_floor.jl`: synthetic acceleration limits, the low-level device
floor guard, preservation of nonzero laminar forces while collecting neighbor
rates, the physical viscosity cap and explicit rejection of unsupported strict
refinement. Four complete moving-square controls also assert that every maximum
accepted current-stage timestep ratio is at most 1.0. These bounds are the
computed model criteria, not a mathematical proof of stability or convergence.
