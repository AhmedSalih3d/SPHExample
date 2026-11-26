# Reuse cell-grid geometry between StillWedge output frames

The generated [StillWedge example](../example/StillWedgeMDBC.jl) uses
`NoMDBC`, the symplectic integrator, and single-file particle and grid output.
It emits particle frames every 0.01 simulated seconds. The grid is derived
from the coordinates downloaded for each frame, so it can stay identical
even when the particles move and the integrator advances.

The output worker calls
[`occupied_cells_host`](../src/GPUCellGrid.jl#L1513) to compute the
sorted occupied-cell coordinates, then
[`append_grid_frame!`](../src/ProduceHDFVTK.jl#L1358) constructs the grid
geometry anew. In 2D, the latter writes four three-component points and four
connectivity entries per cell, and also computes offsets, types and cell
IDs. The geometry is determined by the ordered cell coordinates and the
cell edge. Particle coordinates within those cells have no effect on it.

## Proposed change

Give the single-file grid writer an owned copy of the previous frame's cell
coordinates and edge. After deriving the current frame's occupied cells as
usual, compare the dimension, edge and complete ordered cell vector. On an
exact hit, append the writer's existing geometry buffers into the pending
frame. On a miss, call `fill_grid_geometry!` and refresh the owned key.
An owned copy matters because the output worker reuses its cell vector.

This trades a linear comparison for a linear geometry construction. A miss
adds a comparison and snapshot copy, so time-varying geometries should also
be included in the benchmark. The pending frame still gets its own time,
metadata and dataset entries. This preserves VTKHDF layout and leaves file
size and HDF5 write volume unchanged. It also preserves the rule that only
cells occupied by the current frame can be exported, even between neighbor
rebuilds. Empty cell sets keep their existing handling.

## Measuring reuse

A four-second run on an NVIDIA GeForce RTX 5080 with Julia 1.12.7 completed
8,502 steps and three cell-list rebuilds. It wrote 400 nonempty grid frames
of 240 cells and 960 points each. All 399 consecutive frame pairs had
exactly equal points, connectivity, offsets, types and cell IDs. In that
run, the cache could have reused the first frame's geometry for each
subsequent frame.

Start `julia --project=. -t 4,0` from `gpu_version/` and run the following
to count identical consecutive grid frames in a four-second run. The wrapper
uses a fresh output directory and skips the external viewers and PDF report.
The example's last line is removed before inclusion so it runs only once.

```julia
using HDF5

example = abspath("example/StillWedgeMDBC.jl")
source = replace(read(example, String), r"\nrun_still_wedge\([^\r\n]*\)\s*\z" => "\n")
include_string(Main, source, example)

output = mktempdir()
run_still_wedge(; save_location = output, visualize = false,
    open_log_file = false, generate_pdf_report = false)

h5open(joinpath(output, "StillWedge_GridCells.vtkhdf")) do file
    grid = file["VTKHDF"]
    points = read(grid["Points"])
    counts = vec(read(grid["NumberOfPoints"]))
    ends = cumsum(counts)
    starts = ends - counts .+ 1
    hits = count(2:length(counts)) do i
        view(points, :, starts[i]:ends[i]) == view(points, :, starts[i - 1]:ends[i - 1])
    end
    @show length(counts) hits extrema(counts)
end
```

For a prototype, warm both variants on the same device and alternate their
order. Use different output folders for paired four-second runs with the
default physics, grid settings and output cadence. Compare elapsed solver
and writer times, step and rebuild counts, and the exact particle and grid
datasets, including grid connectivity, offsets, types and cell IDs. Include
both a changing-cell case and an edge-length change in focused grid tests.
The output worker overlaps the GPU loop. Shorter geometry preparation can
reduce output queue backpressure or shorten the final drain. A paired timing
is needed to quantify the effect on elapsed simulation time.

## Paired prototype results

On an RTX 5080 with Julia 1.12.7, a temporary writer cache was tested on the
four-second example. Both paths were warmed with a full run, then tested in
12 pairs with alternating order and fresh output folders. The baseline used
the same instrumented writer with caching disabled. Physics settings and
output cadence were fixed. Both paths skipped viewers and reporting, and
validation ran outside the timed interval.

| Measurement | Baseline median | Cache median | Median paired cache/baseline | Cache faster |
| --- | ---: | ---: | ---: | ---: |
| Full example | 255.55 ms | 251.02 ms | 0.984 | 9/12 |
| Solver wall time | 252.35 ms | 247.52 ms | 0.983 | 9/12 |
| Output writer time | 47.50 ms | 47.00 ms | 1.039 | 5/12 |

All runs completed 8,502 steps with three rebuilds and 404 timestep readbacks.
Final particle state matched exactly by ID, and dataset hashes matched in
both VTKHDF files across all 401 particle frames and 400 grid frames.

The median paired wall-time improvement was about 1.7%, with individual
pairs ranging from about 30% faster to 6% slower. Output writer timings also
varied and favored the cache in five pairs. The experiment was inconclusive
on runtime gain, so the temporary cache was removed.
