# Wave-maker file data

`CaseWavemaker2D_Piston_Movement.dat` is the original piston signal from
DualSPHysics 5.4, `examples/main/07_WavemakerFile`, copied without resampling.
Its 1,826 rows run from 0 to 45.625 seconds at 0.025-second intervals. Columns
are time (s), piston x displacement (m) and an unused constant. The stroke is
±0.026 m; motion begins at 0.925 s. `WavemakerFile2d.jl` interpolates the
displacement linearly between samples, as the reference's `mvfile` does.

`EXP_CaseWavemaker2D_CIEMito.txt` is the CIEMito flume validation data from the
same example: time (s), the measured piston displacement (m) and the free-surface
elevations (m) at wave gauges WG1–WG3, located at x = 1.38, 2.83 and 4.12 m. Its
401 rows cover 0–10 seconds at 0.025-second intervals. Its piston column equals
the motion file over that interval.

Source distribution: DualSPHysics 5.4 (<https://dual.sphysics.org/>).
The distribution's GNU Lesser General Public License 2.1 is retained verbatim
in `LICENSE.txt` alongside these third-party data files. The Julia example is
newly implemented in this repository; this notice concerns the bundled data.

Original SHA-256 before Git line-ending normalization:

| File | SHA-256 |
| --- | --- |
| `CaseWavemaker2D_Piston_Movement.dat` | `B003B483C1EC7D528486C41A246218FEFAB2DDF8B622CB30E43417C007205F84` |
| `EXP_CaseWavemaker2D_CIEMito.txt` | `C598C0F773ACB985FE33AF74D20E9547263AFFCD75DB41273B58177B29A42DA1` |
