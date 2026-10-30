# Piston-wave reference tables

These tables were exported by DualSPHysics 5.4.355 (08-04-2025) from
`examples/main/09_WavesPiston/CasePistonBeach_{REG,IRREG}_Def.xml`.
The damping cases use the same wave definitions. All tables cover 0–30 s
at 0.01 s intervals, retaining the reference export's six decimal places.

- `RegularTheory.csv`: regular second-order piston displacement and theoretical
  first/second-order elevation and velocity at x = 6 m, z = −0.26 m below the
  still surface. Wave height 0.15 m, period 2 s, depth 0.66 m, phase zero,
  one-period startup ramp.
- `IrregularMotion.csv`: time and second-order piston displacement in metres.
  JONSWAP, significant height 0.15 m, peak period 2 s, peak coefficient 3.3,
  1024 energy-stretched components, seed 1234, no ramp. `serieini = 560` with
  `autofit = true` gives the reference's actual start of 560.075 s. Its small
  initial displacement (0.000025 m) is retained. The Julia case interpolates
  linearly and rejects durations beyond 30 s.
- `IrregularTheory.csv`: the same motion plus theoretical first/second-order
  elevation and velocity at the same measurement location. In the reference
  export, the second-order vertical velocity column equals the first-order
  column; it is retained unchanged.

The theoretical elevations are relative to still water. The velocity columns
use simulation x/y (reference x/z). These are wave-theory predictions, not
experimental measurements or simulated SPH results. Startup predictions at a
distant gauge do not model propagation of the startup transient.

To regenerate, run `example/WavesPiston/export_reference.jl` with the path to
a local DualSPHysics 5.4 distribution. Only `savemotion` output spacing/length
is changed and the unrelated 1000 s series export is disabled. The physical
duration remains 30 s; `-nsteps:1` stops after initialization and one step.
No DualSPHysics installation is needed to run the bundled Julia cases.

Source distribution: <https://dual.sphysics.org/>. Its GNU Lesser General
Public License 2.1 is retained verbatim in `LICENSE.txt`. The Julia geometry,
case setup and damping implementation are newly implemented in this repository.
