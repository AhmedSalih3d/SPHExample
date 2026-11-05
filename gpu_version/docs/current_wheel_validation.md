# Native current wheel and pulley

`example/CurrentWheelPulley2d.jl` implements `chrono/11_CurrentWheelPulley`.
Select `variant = :wheel` or `:pulley` in the direct `let` setup. Original
particle geometry, initial densities, CAD, shaft locations, masses and wave-free
1 m/s inlet are retained. Provenance is in `example/data/current_wheel/README.md`.

`OpenChannel{T}` provides genuine planar inlet/outlet conditions on CUDA.
This case is **not periodic**. Five/eight layers use prescribed streamwise
velocity and zeroth-order Shepard ghost density. Fixed buffer floors use the
same extension, avoiding unphysical density accumulation at the outer edge.
Emission, conversion, retirement and all native joint solves stay on CUDA.
The existing floating predictor/corrector is used without another integrator.
`PulleyJoint(60,61; Ratio=6)` couples angles and velocities; separate analytical
hinges locate the shafts. No Chrono dependency or particle axis is introduced.

## Capacity and supported scope

The initial active particle counts match the installed reference's buffer
initialization: 13,397 and 13,574. Of these, 10,818 are fluid. The native array
also reserves all inlet emissions for the requested duration. Four-second
capacity is 34,344/34,521 entries, including 20,865 inactive inlet slots and
82 unused upper seed points. This differs from DualSPHysics' dynamically
resized storage. Retired/reserved entries are `Frozen`; they are not escaped
physical particles. Report active and allocated counts separately.

Emission activates stable IDs and forces a rebuild before the next interaction.
Retirement only removes sources, so conservative existing ranges remain valid;
all readers skip Frozen entries. This avoids rebuilding for individual outlet
particles. Both density/pressure cache modes refresh the evaluated buffer state.

The first implementation supports fresh, nonperiodic, expanding-grid 2D DBC,
one-way positive-X flow, no shifting, no variable resolution and the two
standard stepping schemes. Unsupported combinations fail explicitly. Reserve
duration must cover simulation time. It does not implement arbitrary open
boundary shapes, time-varying discharge or restart replenishment.

## Verification

```sh
julia --project=. -t 2,0 test/open_boundaries.jl
julia --project=. -t 2,0 test/current_wheel.jl
julia --project=. -t 2,0 example/CurrentWheelPulley2d.jl
```

Open-boundary tests pass 71 assertions, including an independent nonuniform
Shepard sum, constant-field preservation, emission/retirement balance, both
precisions, both schemes and graphs/direct execution. Wheel tests pass 36
assertions, including the analytical pulley impulse, both reference fixtures,
0.1 s coupled runs, the cached-pressure path, bounded floor density and a
step-count guard against the outlet-pressure timestep collapse. Existing
planar joint tests passed 86 assertions before the optional motor extensions.
The complete CUDA suite was not run.

Both full four-second runs save 401 frames with finite body state. The default
ParaView states pass first/middle/final frame checks for original CAD, hidden raw
particles, visible labelled forces, fixed speed range 0-1.5 m/s and correct time.
Blue arrows are gravity, red joint reaction and orange SPH force. Equal lengths
show direction; exported magnitude arrays retain newtons. The pulley gear has
a display-only pointer because its uniform spherical CAD hides rotation.

## Reference comparison

Installed DualSPHysics 5.4.355 GPU runs complete four seconds with 119,648 /
119,567 steps, no excluded particles and solver times 116.49 / 120.30 s for
wheel / pulley. Same RTX 5080; Float32 physics and Float64 positions in both
solvers. Original GenCase input counts and initial active counts match.
Native storage includes inactive capacity as described above. These are single
whole-solver runs, not an alternating paired optimization benchmark.

Final native whole-solver times are 75.07 / 82.33 s for 119,588 / 119,579
steps (approximately 1.55 / 1.46 times faster than the reference single runs).
Both runs emit 20,787 particles, retire 20,436 / 20,504 and end with
11,169 / 11,101 active fluid particles, exactly satisfying initial + emitted
- retired. Maximum shaft-point residual is 9.4e-11 m; maximum pulley angle
residual is 5.1e-6 rad. Both runs retain 78 unused reserve slots.

Removal-only rebuild elimination reduces full-run grid rebuilds from roughly
17,400 to 1,277 / 1,265. The earlier complete native runs took 346.59 / 352.90 s.
Changing rebuild ordering can change Fourtakas pair orientation and later
trajectories; these are not bitwise-identical runs or a paired optimization
experiment. The final implementation is faster in these single comparisons,
with larger allocated particle storage than the reference.

Interpolated reference angular-velocity differences are 0.088 rad/s RMS
(single wheel), 0.082 rad/s RMS (pulley wheel) and 0.489 rad/s RMS (small gear).
Maximum centroid difference is 0.055 mm. Native and reference flow/contact
operators are not identical; the gear amplifies wheel-speed differences by six.
No reference trajectory-accuracy or universal speedup claim is made.
