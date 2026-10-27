# Distributed Motion01-09 reference fixtures

These fixtures were produced from `DualSPHysics_v5.4.3.zip`, whose
internal root directory is `DualSPHysics_v5.4`. Source XML, data and tool binary
SHA-256 hashes are recorded in [source_hashes.json](source_hashes.json).
The package is distributed at the [official downloads page](https://dual.sphysics.org/downloads/).

The original case data used by Motion08/09 is preserved under
`example/Motion/data`. The Julia implementation independently constructs the
geometry and expresses the case trajectories; upstream implementation code
is not included.

For each unmodified case, the reference was generated with the distribution's
tools, in a scratch directory:

```text
GenCase_win64.exe MotionXX reference/MotionXX/Case -save:-all
BoundaryVTK_win64.exe -filexml reference/MotionXX/Case.xml -motiontime:<TimeMax>:<TimeOut> -savemotion reference/MotionXX/transforms
```

The three motion data files must also be accessible beside the generated XML.
`TimeMax` and `TimeOut` are taken from the original case. Generated global MK
values use `mkboundfirst=10`; fixture `id` values subtract that offset and refer
to the original `mkbound`/motion reference IDs.

Collect the generated observations with ParaView's Python (substitute your
distribution and scratch paths):

```powershell
pvpython --no-mpi --disable-registry gpu_version/test/reference/motion_cases/collect_fixtures.py --reference-root WORK/reference --case-source DUAL/examples/motion --tools DUAL/bin/windows
pvpython --no-mpi --disable-registry --force-offscreen-rendering gpu_version/test/reference/motion_cases/render_comparison.py --reference-root WORK/reference
```

The renderer reads the Julia animations from `gpu_version/output/Motion` by
default and saves paired images there under `comparison`. Its contact sheet
uses Pillow. Source hashes describe the original distribution bytes; Git may
normalize line endings in the copied example input files.

- [transforms.csv](transforms.csv) stores the tools' cumulative affine matrices
  at 0.1-second intervals over the complete cases (3,324 samples).
- [geometry_vertices.csv](geometry_vertices.csv) stores the complete initial
  vertex sets from `Case__Actual.vtk`, grouped by case and motion ID.
- [geometry_bounds.csv](geometry_bounds.csv) stores bounds and per-body point
  and polygon-cell counts from those same VTK surfaces.

During validation, all original output times were compared with the Julia
trajectories, including the denser 0.01/0.02/0.05-second frames. The largest
matrix coefficient error was below `9e-13`; the CSV output itself is rounded
to 13 significant digits. Tests use `5e-11` for these matrices. GenCase emits
Float32 geometry with accumulated construction rounding; complete vertex sets
are compared within `2e-5` m, with identical vertex and cell counts. Julia uses
Float64 geometry and does not reproduce those rounding artifacts.

Run the automated checks from the repository root:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/test/motion_examples.jl
```

The older `../dual_motion_reference.cpp` fixtures test representative motion
families. These additional fixtures instead cover the **actual distributed
cases**, their scenery, trajectories, simultaneous tracks, resets and file
transitions. They establish kinematic agreement, not SPH fluid-solver equality.
