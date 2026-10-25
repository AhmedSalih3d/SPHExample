# Sloshing excitation data

`CaseSloshingMotionData.dat` is the original 167,001-sample time/angle table
from the same DualSPHysics 5.4 example, copied without resampling. Time is in
seconds and angles are in degrees about the reference +y axis. The Julia
example maps these to negative radians in its 2D x/y plane and rotates the
tank physically about the bottom midpoint `(0, 0)`.

Original motion-file SHA-256 before Git line-ending normalization:
`7233D65922138113DA3454DAD884C2F77012BB94FD0BD31BDA4384C2FF3B2C86`

`CaseSloshingAccData.csv` is copied without resampling from DualSPHysics 5.4,
`examples/main/05_SloshingTank/CaseSloshingAccData.csv`. It contains 167,001
samples from 0 to 8.35 seconds at 0.00005-second intervals. Columns are time,
three linear accelerations (m/s²), and three angular accelerations (rad/s²).
The linear components already include gravity in the tank frame.

The accompanying DualSPHysics `FromMotionToAcc.txt` identifies the angular
excitation as the experimental SPHERIC benchmark data. This is a measured
history, not an analytical sinusoid. Retaining all samples also preserves the
reference's interval-end integration of frame velocity.

Source distribution: DualSPHysics 5.4 (<https://dual.sphysics.org/>).
The distribution's GNU Lesser General Public License 2.1 is retained verbatim
in `LICENSE.txt` alongside these third-party data files. The Julia example is newly
implemented in this repository; this notice concerns the bundled data.

Reference SHA-256 before Git line-ending normalization:
`6F42660ACB261A5E451436BB3087AF86D09310085EA1EE48D0B02EAE8B3205D3`
