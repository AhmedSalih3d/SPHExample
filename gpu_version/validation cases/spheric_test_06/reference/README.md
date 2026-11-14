# Official reference data

Source: [SPHERIC Test 06](https://www.spheric-sph.org/tests/test-06), Andrea
Colagrossi and Giuseppina Colicchio, INSEAN, Italy. Retrieved 2026-11-14.
These numerical fixtures are unmodified extracts from the official
[SPHERIC_TestCase6.zip](https://9449af45-2363-44f6-8b03-03b6b7c2aee5.usrfiles.com/archives/9449af_f76080c1d94f480fb29f1ee0834d7b57.zip).
The archive SHA-256 is
`f2734405bee666e63c45698ac09c9fe3af85a820a2323c210556a19d06dc8e40`.
Individual SHA-256 values are in `sha256.toml`; the runner verifies them.

- `Motion_Body.dat`: time, acceleration, velocity, **absolute center x**.
  The initial center x is 1.5; subtract it once to obtain translation.
  The table continues past 8 s, avoiding endpoint velocity clamping during
  the force comparison. The supplied maximum speed is approximately 0.999899;
  drag normalization uses the specified nominal Umax = 1, without rescaling data.
- `Force_Re050.dat`, `Force_Re100.dat`, `Force_Re150.dat`: time, pressure Cd,
  viscous Cd. The reference is a finite difference / level set computation,
  not experimental measurements. Its finest mesh is 600 × 300.

The source warns of small high-frequency numerical oscillations in the force
records. Comparisons therefore use the same 0.1 s time bins for the reference
and SPH, retaining both raw histories. No phase shift, fitted multiplier,
or reference extrapolation is used. Reference authorship remains with the
contributors; this repository does not claim ownership of their data.
