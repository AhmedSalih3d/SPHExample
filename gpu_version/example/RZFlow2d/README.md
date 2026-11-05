# CaseRZ_Flow2D

Native Julia/CUDA adaptation of the recirculating flow shown in DualSPHysics'
`CaseRZ_Flow2D` video. Water is driven leftward beneath an L-shaped barrier,
rises on the left, flows over its upright, then returns around its right end.

The reference is [`CaseRZ_Flow2D_Def.xml`](https://github.com/Nicolasg2c/DualSPHysics/blob/5b670dd3b05cd5d0fa27e6fe140a4fb96d9bbb6f/examples/wavecoupling/04_RZ_Flow2D/CaseRZ_Flow2D_Def.xml).
Reference x/z coordinates map to solver x/y; dimensions are metres.

| Setting | Value |
| --- | --- |
| Tank | 1.3 × 0.8, closed single-particle outer perimeter |
| Upright | x = 0.20–0.24, y = 0.20–0.50 |
| Shelf | x = 0.20–1.05, y = 0.20–0.24 |
| Fill upper limit | 0.45 (highest fluid row at 0.445) |
| Particle spacing | 0.005 |
| Viscosity regularization | `η² = (0.1h)²`, approximately `5e-7 m²` at the default spacing |
| Particles | 23,990 (2,847 fixed, 21,143 fluid) |
| Uniform-flow box | x = 0.60–0.90, y = 0–0.20 |
| Direction | negative x |
| Time/speed samples | (0, 0), (1, 0.3), (3, 0.4), (6, 0.7), (8, 0.2) |
| Relaxation | coefdt = 1000, psi = 0.7, beta = 3 |
| Duration / output interval | 9 s / 0.025 s |

Run from the repository root on a computer with Julia 1.12 and an NVIDIA GPU:

```sh
julia --project=gpu_version -e 'using Pkg; Pkg.instantiate()'
julia --project=gpu_version -t 2,0 gpu_version/example/RZFlow2d.jl
```

The example writes to `gpu_version/output/rz_flow_2d` and opens ParaView using
the existing integration. Colour fluid by the magnitude of `Velocity`, use
a side view and play the time series to compare with the video. The exported
`DampingZones` and `DampingProfiles` blocks show the forcing box and its
normalized blending profile. Hide those blocks for a clear fluid view.

To change duration, spacing, output path or automatic visualization, edit
`rz_flow_case()` in the runner, for example
`rz_flow_case(; duration = 0.05, visualize = false, save_location = "D:/SPH/RZFlow")`.
Spacing must align all reference faces; 0.01 is a coarser supported choice.
`FlowSetup.jl` can be included separately to inspect geometry without running CUDA.

This is an adaptation, not a claim of frame-by-frame agreement. It uses the
package's symplectic integrator and dynamic boundaries, hydrostatic initial
density, and double-precision positions with Float32 physics. It preserves
the reference's Wendland smoothing length, artificial viscosity 0.05, disabled
shifting and density diffusion, and uniform-flow history. The blend is capped
at one to prevent extrapolation for large timesteps. Rotated relaxation boxes,
wave-generating relaxation zones and external SWASH coupling are not implemented.

Focused validation (host checks run without an NVIDIA GPU):

```sh
julia --project=gpu_version gpu_version/test/rz_flow.jl
```
