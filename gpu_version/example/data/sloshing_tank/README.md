# Sloshing excitation data

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
in `LICENSE.txt` alongside this third-party data. The Julia example is newly
implemented in this repository; this notice concerns the bundled data.

Reference SHA-256 before Git line-ending normalization:
`6F42660ACB261A5E451436BB3087AF86D09310085EA1EE48D0B02EAE8B3205D3`
