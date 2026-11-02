# Falling wedge with three resolution levels

[The main example](../VResolutionFallingWedge2d.jl) drops a force-responsive
2D wedge into 1.5 m of water with initial velocity `(0,-6.15)` m/s and body
mass 241 kg per unit out-of-plane width. Gravity and fluid forces change its
velocity after initialization; its descent is not prescribed.

The geometry follows DualSPHysics 5.4 `vresolution/04_FallingWedge2D`:
the apex starts at `(0,1.5)` m and the upper corners are `(±0.25,1.64433)` m.
Four conforming boundary layers represent the wedge. Its relative weight is
computed from the actual finest-level particle count to retain the requested
mass at different spacings. Mirrored mDBC ghost points move and rotate with
the body; corner ghosts use the corner bisector rather than choosing a side.

Two nested rectangles follow the accepted rigid-body pose:

| Level | Default spacing | Reference rectangle |
| --- | --- | --- |
| 0 | 0.002 m | Whole tank |
| 1 | 0.001 m | (-0.49,1.31) to (0.49,1.90) m |
| 2 | 0.0005 m | (-0.39,1.40) to (0.40,1.798) m |

All three domains advance together with a common stable timestep on CUDA.
Each adjacent pair exchanges reconstructed buffer states and interface flux.
The middle domain has both an outer buffer and an inner exclusion. Ownership
is unique, newly created particle IDs use separate level namespaces, and
floating forces are evaluated only on the finest-level wedge. CUDA graphs
and device-side rebuild/creation phases are supported.

The main file exposes the constants, kernels, `FloatingDetails`, nested
`VariableResolutionConfig` structs, metadata, viscosity, diffusion and damping.
The native settings are Wendland C2 with **h=2dp**, sound speed 80 m/s,
artificial viscosity 0.01, CFL 0.2, Fourtakas full density diffusion 0.1,
Simple mDBC, no shifting and ordinary symplectic stepping. Positions and the
clock use Float64 while physics defaults to Float32.

This is a native version of the setup, rather than a validated reproduction
of the reference trajectory. The ordinary native solver uses the gamma-7
equation of state; the reference uses gamma 1. The tank uses closed end walls
with side/bottom damping instead of the reference's inlet/outlet buffers.
Particle generation and mDBC corner sampling can also differ from GenCase.
No DualSPHysics solver or GenCase is invoked.

Run from a **fresh Julia session** after updating the package, since this
change adds fields to configuration and body structs:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/example/VResolutionFallingWedge2d.jl
```

Running or including the main file starts the default 0.025 s case. Change
the keyword defaults or final call for a different run. By default the spare
particle capacity scales with refinement spacing and the area swept by the
descending boxes, including a margin. This lets the fine level fill initially
dry space as it enters the water; a fixed 20,000-slot reserve is insufficient
at production spacing. Pass `particle_reserve=...` to choose an explicit count.
This changes allocation capacity rather than the numerical inputs. The production
spacing is expensive; validation uses a coarser case with wider boxes to
retain the required support clearance:

```julia
run_falling_wedge(
    dx = 0.02,
    duration = 0.025,
    output_interval = 0.0025,
    regions = (
        VariableResolutionBox((-0.65,1.15),(0.65,2.0)),
        VariableResolutionBox((-0.45,1.3),(0.45,1.9)),
    ),
    particle_reserve = 10000,
    visualize = false,
    save_location = "C:/TestSimulations/FallingWedge2D_Coarse",
)
```

Use a new output directory. **`FallingWedge2D.vtkhdf`** contains all particle
levels, both moving box outlines, the wedge, tank and damping geometry.
Open it directly in ParaView and press Apply. Point data `Resolution` is
0, 1 or 2; `Spacing`, `Mass` and `Owned` describe the active frame. CSV files
save accepted body states and per-level counts, represented fluid mass and
creation/retirement/exclusion events.

Render a saved frame or a numbered animation with ParaView's bundled Python:

```powershell
pvpython --dr gpu_version/example/RenderFallingWedge2d.py path/to/FallingWedge2D.vtkhdf impact.png
pvpython --dr gpu_version/example/RenderFallingWedge2d.py path/to/FallingWedge2D.vtkhdf levels.png --color resolution
pvpython --dr gpu_version/example/RenderFallingWedge2d.py path/to/FallingWedge2D.vtkhdf frames --all-frames
```

`--time`, `--overview` and `--pressure-max` control the frame and view.

Focused checks, from the repository root:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/test/floating_initial_velocity.jl
julia --project=gpu_version -t 2,0 gpu_version/test/nested_resolution.jl
julia --project=gpu_version -t 2,0 gpu_version/test/falling_wedge.jl
```

These check force-responsive initial velocities in 2D/3D and both precisions,
uniform flow across nested interfaces with creation on every level, unique
IDs, both graph modes, single-file temporal output and the moving wedge.

A reduced run at dp=0.02/0.01/0.005 m completed the 0.025 s impact in 1,010
steps with CUDA graphs and no density or non-finite exclusions. The wedge's
vertical velocity changed from -6.15 to -4.60 m/s. Represented owned-fluid mass
changed from 4425.90 to 4402.68 kg (-0.525%); assess mass drift and convergence
before using this experimental nested method for quantitative impact loads.
That reduced check did not run the production resolution.

A subsequent full production run at dp=0.002/0.001/0.0005 m completed the same
0.025 s interval in 10,050 steps on the RTX 5080 with CUDA graphs (356 graph
batches and no capture failures). It saved 51 frames in one VTKHDF. The final
level counts, including walls/body, were 1,060,681 / 136,663 / 685,080. There
were no density or non-finite exclusions. The wedge slowed from -6.15 to
-4.56516 m/s; represented owned-fluid mass changed from 4490.76421 to
4492.41471 kg (+0.03675%). This validates execution and output at the production
spacing; reference-model differences above still apply. The default reserve
now scales with the falling boxes' swept area after the original fixed
20,000-slot count was exhausted at 0.001 s.
