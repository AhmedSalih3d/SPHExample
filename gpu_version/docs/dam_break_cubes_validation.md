# Native dam-break solids

`example/DamBreakCubes3d.jl` implements all three configurations pictured
under DualSPHysics 5.4 `chrono/07_DamBreakCubes`. Select `variant` at the
top of its direct `let` setup: `:one_cube`, `:three_cubes`, or `:blocks`.
Each selects its original geometry, spacing, fluid density field, sound
speed, body masses, material coefficients, duration and output names.
Defaults remain 2 s / 2 s / 1.5 s, respectively, with output every 0.01 s.

Full particle counts are 209,630, 214,024 and 115,436. Cubes have mass
2.7 kg each; blocks have relative weight 1.25 and mass 0.495 kg each.
Fixture provenance, material values and lattice hashes are in
`example/data/dam_break_cubes/README.md`.

The block reference's `distributed_bodymass` setting is preserved through
`FloatingDetails.InteractionMass`. Neighbour continuity, pressure and
viscosity use the neighbour's interaction mass, and floating force/torque
reduction uses the target body's interaction mass. This preserves equal
and opposite fluid/body momentum exchange. Rigid mass/inertia are independent
of the SPH interaction mass. The default `nothing` allocates no extra mass
array and specializes to the existing uniform-mass path. Custom mass mode
currently requires nonperiodic, single-resolution flow and zero/Fourtakas
density diffusion; unsupported combinations fail explicitly.

All variants use the existing Symplectic scheme, Wendland h = sqrt(3) dp,
Fourtakas diffusion 0.1, artificial viscosity 0.05, DBC and no shifting.
The reference cube cases disable boundary viscosity; the block case retains
it. There are no joints or periodic conditions. Native CUDA contacts use
the exact sampled bodies/tank, Coulomb friction and restitution, with a
half-dp search margin. Reference Chrono contacts instead use CAD meshes and
a half-dp allowed overlap. Similar macroscopic motion is expected; neither
identical impacts nor greater physical accuracy is claimed.

```sh
julia --project=. -t 2,0 example/DamBreakCubes3d.jl
julia --project=. -t 2,0 test/floating_interaction_mass.jl
julia --project=. -t 2,0 test/dam_break_cubes.jl
julia --project=. -t 2,0 benchmark/chrono.jl DamBreakCubes3d.jl
```

The focused mass tests cover Float32/Float64 in 2D/3D, pair momentum,
continuity/viscosity scaling, force reduction and invalid inputs. Geometry
and fluid/contact/export smoke tests cover all three reference variants.
Default states display original moving CAD, tank wireframe, fluid speed,
time and offset force directions. Blue means gravity, red contact reaction
and orange SPH/spring force; physical magnitudes remain in CSV/HDF arrays.

The mass tests passed 44 assertions, the three reference setups and coupled
smoke runs passed 33, and existing body-load/output regressions passed 584.
All three complete native runs also passed, with no frozen particles:

| Variant | Duration / frames | Steps | Minimum transformed CAD height above floor |
|---|---:|---:|---:|
| One cube | 2 s / 201 | 18,338 | 11.16 mm |
| Three cubes | 2 s / 201 | 20,277 | 8.90 mm |
| Eighteen blocks | 1.5 s / 151 | 27,163 | 7.11 mm |

The floor check transforms the original CAD vertices at every output frame,
independently of the contact search. Every body moves horizontally by more
than 1 cm. Maximum horizontal displacements are 1.790 m, 0.937 m and 0.880 m,
respectively. The native three-cube run retains a largely stacked arrangement
while sliding; detailed separation seen in the reference video is not
reproduced closely. Sampled contacts can interlock differently from smooth
CAD contacts, so these runs establish stable coupled examples, not contact
convergence or trajectory equivalence.

Default ParaView states passed first/middle/final geometry, force visibility,
time and colour-range checks and were visually inspected. Native wall times
after short setup warmups were 58.8, 133.4 and 419.0 s on the RTX 5080 system
described in the FlexibleGate report. No matched reference timing runs were
made for these variants; no performance advantage is claimed. The complete
CUDA suite was not run. Simulation output and renders remain Git-ignored.
