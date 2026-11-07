# Initial IISPH validation — 2026-11-07

Environment: NVIDIA GeForce RTX 5080 (16 GB), Julia 1.12.7, CUDA.jl 6.4.2,
Windows, four Julia compute threads and no interactive thread. The examples
use `GPUCellSubdivision=1`, Wendland C2 with `h=2dx`, and direct synchronous
IISPH execution. No dependencies were changed.

## Focused regressions

From the repository root:

```bash
julia --compiled-modules=existing --project=gpu_version -t 4,0 gpu_version/test/iisph.jl
```

**1,675 assertions passed.** Coverage includes:

- Independent all-pairs CPU density and dense acceleration/continuity matrices
  in 2D/3D and Float32/Float64, including a direct dense pressure solve.
- Pressure preservation through sorting, compact/full particle storage and
  mixed position precision.
- Unsupported combinations, sound-speed-independent timestep selection,
  viscosity and timestep caps, and explicit nonconvergence before integration.
- A settled 2D viscous tank with pressure probes within 5% of analytical
  hydrostatic values at three depths, stationary wall particles, and less than
  1% positive summation-density deviation.
- Stored pressure in numbered and single-file VTKHDF output; probes use solved
  pressure even when pressure is omitted from particle output.
- Fluid-only 2D/3D free fall, a short 3D fixed-wall projection, external-force
  gravity override, and a 2D dam break preserving fluid identities/mass.
- Exact agreement between the default WCSPH selection and its explicitly
  selected symplectic integration path.

An additional three-check IISPH grid smoke run verified that exported grid
files exist and host `Cells` match the current positions and occupied-cell set.

The raw particle-pressure field is noisy. The hydrostatic check uses normalized
SPH probes after a one-second settling run with `ν₀=0.01 m²/s`, `dx=0.025 m`,
and a 0.002 s timestep cap. Probe locations are y=0.1, 0.2 and 0.3 m.
Tightening the linear solve alone did not remove large variations between
individual particle pressures in an additional settling screen.
This is not validation of pointwise pressure, impact loads or production cases.

## Dam-break timestep screen

Uniform 2D particles at `dx=0.025 m`, fluid column 16 by 20 particles, fixed-wall
tank 48 by 24 spacings with five wall layers, gravity, `ν₀=1e-6 m²/s`, density tolerance 0.001,
relaxation 0.1, and 0.2 s simulated. All accepted linearized residuals met the
requested tolerance. The density constraint is not a hard contact constraint:

| Timestep cap (s) | Leading x (m) | Maximum floor intrusion (m) | Intrusion / dx |
| --- | --- | --- | --- |
| 0.002 | 0.79416 | 0.01289 | 0.516 |
| 0.001 | 0.78939 | 0.01305 | 0.522 |
| 0.0005 | 0.78903 | 0.01254 | 0.502 |

The front position is reasonably consistent under timestep refinement, while
the under-supported leading particle retains approximately half a spacing of
floor intrusion. Increasing the wall depth from three to five layers reduced
the initial screen's intrusion from about 0.8–0.9 spacings.
The regression bounds this known intrusion below one spacing; it does not
claim nonpenetration. Improved near-wall treatment is required before using
this solver for wall-impact or coupled-body studies.

## Performance scope

`benchmark/iisph.jl` reports whole-run time, steps, pressure iterations, density
range and maximum speed. Startup compilation and output are included. It is
an accuracy/development screen, not an alternating, paired, warmed comparison
against WCSPH. **No solver speedup or whole-solver VRAM saving is claimed.**

CUDA graph execution, reducing pressure-loop readbacks, broader boundary/body
support, and paired equal-duration performance studies remain later milestones.
The full CUDA/CPU-reference suite is separate from these focused checks.

## Existing WCSPH checks

The same Julia flags were used for these focused test files:

| Test file | Result |
| --- | --- |
| `effective_configuration.jl` | 365 assertions passed |
| `simulation_result.jl` | 410 assertions passed |
| `gpu_memory.jl` | 2,860 assertions passed |
| `sorting_shadows.jl` | 1,008 assertions passed |
| `stationary_predictors.jl` | 2,991 passed; two Float32 cached-force tolerance failures also reproduce on the original committed solver |
| `updated_mdbc.jl` | 1,303 assertions passed |
| `grid_export.jl` | 1,719 assertions passed |

For example:

```bash
julia --compiled-modules=existing --project=gpu_version -t 4,0 gpu_version/test/gpu_memory.jl
```

The predictor comparison was repeated against an isolated copy of the original
source at `aa91e13`, using the same Project/Manifest and GPU. Both runs produced
the identical compared Float32 force vectors at `stationary_predictors.jl:134`:
`[-5.068672, 47.450024]` versus `[-5.0689507, 47.45018]`. The existing
`50eps(Float32)` comparison fails for those vectors. This change does not alter
the WCSPH interaction or predictor kernels, and does not relax that tolerance.

A stale helper in `simulation_result.jl` was corrected to read step values from
the existing `VTKHDF/Particles` block for floating-body composite output. The
output writer itself was unchanged.
