# Existing 06 Special output

These small fixtures were read from the completed local DualSPHysics 5.4.355
run at `examples/vresolution/06_Special/CasePointAbsorberSpringVRes_out`.
No solver or GenCase was launched to create them.

- `BoundarySection.csv`: the first 2,832 fixed x/z points in the generated
  coarse `_All.vtk`, read as big-endian Float32 and written as x/y coordinates.
- `PistonTheory.csv`: the time and prescribed displacement columns of
  `WavePaddle_mkb0010.csv`; this is second-order wave theory, not AWAS motion.
- `FloatingReference.csv`: 200 existing Chrono exchange/link records, with
  DualSPHysics x/z mapped to native x/y. Columns retain the actual output
  times, body centre, vertical velocity and force on the floater.
  The link CSV reports the opposite-side reaction: its fz sign is reversed
  here so both solvers report force on the floater. It agrees with
  `-100*(center_y-0.4)-500*velocity_y` to the precision of the saved histories.
- `SourceHashes.json`: SHA-256 hashes of the original source output files.

The source geometry uses a 0.30×0.16 m floating box at `(2,0.4)` m, water
depth 0.4 m, a 0.01/0.005 m refinement pair, and a fixed-anchor spring with
stiffness 100 N/m, damping 500 Ns/m and rest length 0.325 m. The native
example intentionally selects this project's ordinary SPH settings, so these
histories document the existing comparison rather than a bitwise target.
