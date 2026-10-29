# Floating waves data

These files come from DualSPHysics 5.4, `examples/main/12_FloatingWaves`.
[FloatingWavesFlap2d.jl](../../FloatingWavesFlap2d.jl) and
[FloatingWavesPiston2d.jl](../../FloatingWavesPiston2d.jl) use them.

`CaseFloatingWavesVal_Flap.dat` is the original flap signal of
`CaseFloatingWavesVal`, copied without resampling. Its 10,901 rows run from 0
to 10.9 seconds at 0.001-second intervals. Columns are time (s) and flap angle
(degrees), from −4.67° to 4.28°. A positive angle turns the flap about its
hinge at the bed away from the water, as DualSPHysics turns the reference's
flap. `FloatingWavesFlap2d.jl` interpolates the angle linearly between samples,
relative to its first value, as the reference's `mvrotfile` does. The
distribution's `.ods` copy of the experiment lists this wave-maker signal with
the opposite sign.

`EXP_FloatingWavesVal_Hadzic_et_al_2005.txt` is the validation data for that
case, from Hadžić I, Hennig J, Perić M, Xing-Kaeding Y (2005), Computation of
flow-induced motion of floating bodies, *Applied Mathematical Modelling* 29,
1196–1210. After a citation line and a header row, it has five pairs of
tab-separated time (s) and value columns: the free-surface elevations (m) at
wave probes 1 (3–10 s) and 2 (5–9.8 s), and the box's heave (m), roll
(degrees) and sway (m) from 6.3 to 8.1 s. The reference case measures the
elevations at x = 1.16 and 2.66 m. The roll is positive counterclockwise, with
x along the tank and z up, like the solver's 2D angle. DualSPHysics' pitch, and
the roll in the distribution's `.ods` copy, have the opposite sign.

`EXP_FloatingWavesVal2_Ren_et_al_2005.csv` holds the measured box motion of
`CaseFloatingWavesVal2`, from Ren B, He M, Dong P, Wen H (2015), Nonlinear
simulations of wave-induced motions of a freely floating body using WCSPH
method, *Applied Ocean Research* 50, 1–12. It was converted from the first
sheet (`FloatingMotion_mk61`) of the distribution's
`EXP_FloatingWavesVal2_Ren_et_al_2005.ods`, written with 10 significant digits.
The other sheet holds DualSPHysics results and was not converted. Its 107 rows
give time (s), surge (m), heave (m) and roll (degrees) from 1.81 to 8.88
seconds.

Source distribution: DualSPHysics 5.4 (<https://dual.sphysics.org/>).
The distribution's GNU Lesser General Public License 2.1 is retained verbatim
in `LICENSE.txt` alongside these third-party data files. The Julia examples are
newly implemented in this repository; this notice concerns the bundled data.

Original SHA-256 before Git line-ending normalization:

| File | SHA-256 |
| --- | --- |
| `CaseFloatingWavesVal_Flap.dat` | `2C2E532C85B19B8F463544C53ED4AEFC2D6D2891927E4A0E3AD11DD20EF7FBE0` |
| `EXP_FloatingWavesVal_Hadzic_et_al_2005.txt` | `479C9B800EFDEA45469D33F7CF6A4894E9B77DD18AC4D331208C6C746120DE8D` |
| `EXP_FloatingWavesVal2_Ren_et_al_2005.ods` (source of the CSV) | `682C36EFE00D51FDCED3F6B7BEDAB9C86358293A81975B4107C6270911D8E937` |
