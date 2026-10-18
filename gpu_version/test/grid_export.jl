module GridExportTests

using Test
using SPHExampleGPU
using CUDA
using StaticArrays
using HDF5

const GRID = SPHExampleGPU.GPUCellGrid

@testset "occupied cells use current positions and cell-list rounding" begin
    for D in (2, 3), R in (1, 2, 8), T in (Float32, Float64), TP in (Float32, Float64)
        inverse_cell = T(R) * inv(T(0.08))
        expected = [CartesianIndex(ntuple(d -> sign * d, D)) for sign in (-1, 0, 1)]
        positions = [SVector{D, TP}(Tuple(cell)) / TP(inverse_cell)
                     for cell in expected[[3, 1, 3, 2, 1]]]
        @test GRID.occupied_cells_host(positions, inverse_cell) == expected
        @test isempty(GRID.occupied_cells_host(SVector{D, TP}[], inverse_cell))
        shifted = positions .+ Ref(SVector{D, TP}(ntuple(_ -> TP(4) / TP(inverse_cell), D)))
        @test GRID.occupied_cells_host(shifted, inverse_cell) ==
              [cell + CartesianIndex(ntuple(_ -> 4, D)) for cell in expected]

        ties = [SVector{D, TP}(ntuple(_ -> sign * TP(0.25), D)) for sign in (-1, 1)]
        @test GRID.occupied_cells_host(ties, T(2)) ==
              [CartesianIndex(ntuple(_ -> sign, D)) for sign in (-1, 1)]
    end
end

@testset "grid geometry has no gaps between adjacent occupied cells" begin
    for D in (2, 3), R in (1, 2, 8), T in (Float32, Float64)
        ranges = ntuple(d -> (-3 - d):(3 + d), D)
        origin = Int32.(first.(ranges))
        dims = Int32.(length.(ranges))
        grid = CellGrid{D, R}(origin, dims, Int32(prod(dims)))
        cells = filter(cell -> cell[1] != 0 || cell[2] == 0, vec(collect(CartesianIndices(ranges))))
        geometry = SPHExampleGPU.ProduceHDFVTK
        points, connectivity, offsets, types, data, dims =
            geometry.compute_grid_geometry(T(0.08) / R, cells)
        buffers = GridGeometryBuffers()
        fill_grid_geometry!(buffers, T(0.08) / R, cells)
        @test stack(points) == buffers.points
        @test connectivity == buffers.connectivity
        @test offsets == buffers.offsets
        @test types == buffers.types
        @test data == buffers.cell_data
        @test dims == D
        expected_ids = [LinearIndices(Int.(grid.dims))[ntuple(d -> cell[d] - grid.origin[d] + 1, D)...]
                        for cell in cells]
        @test data == expected_ids

        ncorners = 2^D
        lookup = Dict(cell => i for (i, cell) in enumerate(cells))
        for d in 1:D
            step = CartesianIndex(ntuple(axis -> axis == d ? 1 : 0, D))
            @test all(maximum(buffers.points[d, (i - 1) * ncorners + 1:i * ncorners]) ==
                      minimum(buffers.points[d, (lookup[cell + step] - 1) * ncorners + 1:lookup[cell + step] * ncorners])
                      for (cell, i) in lookup if haskey(lookup, cell + step))
        end
        @test fill_grid_geometry!(buffers, T(0.08) / R, CartesianIndex{D}[]) == 0
    end
end

function cycle_grid_geometry!(buffers, edge, frames)
    for cells in frames
        fill_grid_geometry!(buffers, edge, cells)
    end
    return nothing
end

@testset "grid coordinates reuse capacity across changing and empty frames" begin
    geometry = SPHExampleGPU.ProduceHDFVTK
    for D in (2, 3), T in (Float32, Float64)
        edge = T(0.08) / T(2)
        cells = [CartesianIndex(ntuple(d -> d == 1 ? i - 5 : -d, D)) for i in 1:4096]
        counts = (256, 96, 0, 224, 64, 32, 256, 0, 128, 256)
        frames = [cells[1:n] for n in counts]
        buffers = GridGeometryBuffers()
        fill_grid_geometry!(buffers, edge, first(frames))
        original_points = buffers.points
        for frame in frames
            npts = length(frame) * 2^D
            @test fill_grid_geometry!(buffers, edge, frame) == npts
            @test buffers.points isa Matrix{Float64}
            @test size(buffers.points) == (3, npts)
            @test pointer(buffers.points) == pointer(original_points)
            if isempty(frame)
                @test isempty(buffers.connectivity)
                @test buffers.offsets == [0]
                @test isempty(buffers.types) && isempty(buffers.cell_data)
            else
                points, connectivity, offsets, types, data, _ = geometry.compute_grid_geometry(edge, frame)
                @test buffers.points == stack(points)
                @test buffers.connectivity == connectivity
                @test buffers.offsets == offsets
                @test buffers.types == types
                @test buffers.cell_data == data
            end
        end

        cycle_grid_geometry!(buffers, edge, frames) # Warm the changing-size path.
        allocated_bytes = @allocated cycle_grid_geometry!(buffers, edge, frames)
        @test allocated_bytes < sizeof(original_points) ÷ 2

        # Growth uses new storage; retained matrices must keep owning their old data.
        retained_points = buffers.points
        expected_points = copy(retained_points)
        @test fill_grid_geometry!(buffers, edge, cells) == length(cells) * 2^D
        @test pointer(buffers.points) != pointer(retained_points)
        @test retained_points == expected_points
        large_points = buffers.points
        @test fill_grid_geometry!(buffers, edge, CartesianIndex{D}[]) == 0
        @test pointer(buffers.points) == pointer(large_points)
        @test size(buffers.points) == (3, 0)
        @test fill_grid_geometry!(buffers, edge, cells) == length(cells) * 2^D
        @test pointer(buffers.points) == pointer(large_points)
    end
end

@testset "dense grid frames respect the pending output budget" begin
    geometry = SPHExampleGPU.ProduceHDFVTK
    function write_frames(path, budget, D; capacity = 4)
        h5open(path, "w") do file
            root = HDF5.create_group(file, "VTKHDF")
            GenerateGeometryStructure(root; vtk_file_type = "UnstructuredGrid", chunk_size = 64)
            GenerateStepStructure(root; vtk_file_type = "UnstructuredGrid")
            writer = if budget === nothing
                nothing
            else
                @test_throws ArgumentError GridFrameWriter(root; budget_bytes = -1)
                GridFrameWriter(root; capacity, budget_bytes = budget)
            end
            for (k, n) in enumerate((1, 2, 20, 3, 2))
                cells = vec(CartesianIndices(ntuple(d -> d == 1 ? (1:n) : (-1:1), D)))
                if writer === nothing
                    AppendVTKHDFGridData(root, Float64(k), 0.1, cells, nothing)
                    continue
                end
                append_grid_frame!(writer, Float64(k), 0.1, cells)
                @test writer.pending_bytes <= budget
                @test frames_written(writer) + frames_pending(writer) == k
                if geometry.grid_frame_bytes(length(cells), D) >= budget
                    @test frames_pending(writer) == 0
                end
            end
            if writer !== nothing
                flush_frames!(writer)
                @test writer.pending_bytes == 0 && frames_written(writer) == 5
            end
        end
        return h5open(path, "r") do file
            root = file["VTKHDF"]
            names = ("Points", "Connectivity", "Offsets", "Types", "NumberOfPoints",
                     "NumberOfCells", "NumberOfConnectivityIds", "CellData/CellData",
                     "Steps/Values", "Steps/PointOffsets", "Steps/NumberOfParts",
                     "Steps/PartOffsets", "Steps/CellOffsets", "Steps/ConnectivityIdOffsets")
            data = Dict{String, Any}(name => read(root[name]) for name in names)
            data["Steps/NSteps"] = HDF5.read_attribute(root["Steps"], "NSteps")
            data
        end
    end
    mktempdir() do directory
        for D in (2, 3)
            reference = write_frames(joinpath(directory, "reference_$(D).vtkhdf"), nothing, D)
            for capacity in (1, 2, 5, MAX_BUFFERED_FRAMES),
                budget in (0, geometry.grid_frame_bytes(3^(D - 1), D), geometry.grid_frame_bytes(4 * 3^(D - 1), D), typemax(Int))
                path = joinpath(directory, "budget_$(D)_$(capacity)_$(budget).vtkhdf")
                @test write_frames(path, budget, D; capacity) == reference
            end
        end
    end
end

function check_grid_file(path, positions, expected_time, edge, inverse_cell, D; frame = 1)
    return h5open(path, "r") do file
        root = file["VTKHDF"]
        if haskey(root, "Steps")
            @test read(root["Steps/Values"])[frame] == expected_time
            start = read(root["Steps/PointOffsets"])[frame]
        else
            start = 0
        end
        npoints = read(root["NumberOfPoints"])[frame]
        points = root["Points"][:, start + 1:start + npoints]
        ncorners = 2^D
        opposite = D == 2 ? 3 : 7
        centers = (points[:, 1:ncorners:end] + points[:, opposite:ncorners:end]) / 2
        occupied = unique([CartesianIndex(ntuple(d ->
            round(Int, positions[d, i] * inverse_cell, RoundNearestTiesAway), D))
            for i in axes(positions, 2)])
        actual = [CartesianIndex(ntuple(d -> round(Int, centers[d, i] / edge), D))
                  for i in axes(centers, 2)]
        expected = sort(occupied)
        @test actual == expected
        @test read(root["NumberOfCells"])[frame] == length(expected)
        for d in 1:D
            widths = [maximum(points[d, i:i + ncorners - 1]) -
                      minimum(points[d, i:i + ncorners - 1]) for i in 1:ncorners:npoints]
            @test all(w -> isapprox(w, edge; rtol = 1e-12), widths)
        end
        D == 2 && @test all(iszero, points[3, :])
        return (; path, step = frame - 1, npoints, ncells = length(expected),
                dimension = D, edge, cells = Tuple.(expected))
    end
end

function simulate_grid(save, D, R; async, single_file, double_position, export_grid = true)
    T = Float32
    constants = SimulationConstants{T}(dx = 0.02, c₀ = 20, g = 0, CFL = 0.2)
    kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
    edge = kernel.H / R
    positions = [SVector{D, Float64}(Tuple(c) .* (2.1 * Float64(edge)))
                 for c in vec(CartesianIndices(ntuple(_ -> -1:2:1, D)))]
    geometry = [SPHGeometry{D, T}(positions; Density = 1000, Type = Moving, GroupMarker = 1,
        Motion = MotionDetails{D, T}(Velocity = 1, StartTime = -1, Duration = 2,
            Direction = SVector{D, T}(ntuple(d -> d == 1 ? 1 : 0, D))))]
    meta = SimulationMetaData{D, T, NoShifting, NoKernelOutput, NoMDBC, NoLog}(
        SimulationName = "grid", SaveLocation = save, SimulationTime = 0.012,
        OutputTimes = 0.004, OutputVariables = ["ID"],
        ExportGridCells = export_grid, ExportSingleVTKHDF = single_file,
        GPUCellSubdivision = R, GPUDoublePosition = double_position,
        GPUAsyncOutput = async, GPUUseGraph = false,
        GPUOutputQueueBytes = 4096, GPUOutputBufferBytes = 4096,
        VisualizeInParaview = false, OpenLogFile = false)
    particles = AllocateDataStructures(geometry, meta)
    logger = SimulationLogger(save)
    try
        redirect_stdout(devnull) do
            RunSimulation(; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
                SimKernel = kernel, SimParticles = particles, SimLogger = logger,
                SimViscosity = ArtificialViscosity(), SimDensityDiffusion = ZeroDensityDiffusion(),
                SimTimeStepping = SymplecticTimeStepping())
        end
    finally
        close(logger.LoggerIo)
    end
    return (; particles, meta, kernel)
end

@testset "GPU grid export includes only cells occupied in each output frame" begin
    if !CUDA.functional()
        @test_skip false
    else
        mktempdir() do directory
            reader_frames = NamedTuple[]
            for D in (2, 3), R in (1, 2, 8), async in (false, true), single_file in (false, true)
                save = mkpath(joinpath(directory, "$(D)_$(R)_$(async)_$(single_file)"))
                result = simulate_grid(save, D, R; async, single_file, double_position = async)
                (; particles, meta, kernel) = result
                edge = Float64(kernel.H / R)
                inverse_cell = Float32(R) * kernel.H⁻¹
                if single_file
                    h5open(joinpath(save, "grid.vtkhdf"), "r") do file
                        root = file["VTKHDF"]
                        times = read(root["Steps/Values"])
                        offsets = read(root["Steps/PointOffsets"])
                        # The initial particle frame has no grid frame.
                        for frame in 2:length(times)
                            positions = root["Points"][:, offsets[frame] + 1:offsets[frame] + length(particles)]
                            push!(reader_frames, check_grid_file(joinpath(save, "grid_GridCells.vtkhdf"),
                                positions, times[frame], edge, inverse_cell, D; frame = frame - 1))
                        end
                    end
                else
                    for frame in 2:meta.OutputIterationCounter
                        suffix = lpad(frame, 6, "0")
                        positions = h5open(joinpath(save, "grid_$(suffix).vtkhdf"), "r") do file
                            read(file["VTKHDF/Points"])
                        end
                        push!(reader_frames, check_grid_file(joinpath(save, "CellGrid_grid_$(suffix).vtkhdf"),
                            positions, nothing, edge, inverse_cell, D))
                    end
                end
                # Output must not change the integration or the actual cell-list bins.
                control = mkpath(joinpath(save, "without_grid"))
                reference = simulate_grid(control, D, R; async, single_file,
                    double_position = async, export_grid = false)
                @test particles.Position == reference.particles.Position
                @test particles.Velocity == reference.particles.Velocity
                @test particles.Density == reference.particles.Density
                @test particles.Cells == reference.particles.Cells
                @test meta.Iteration == reference.meta.Iteration
            end

            pvpython = Sys.which("pvpython")
            if pvpython === nothing
                @test_skip false
            else
                script = joinpath(directory, "check_grid_reader.py")
                open(script, "w") do io
                    write(io, """
                    from vtkmodules.vtkIOHDF import vtkHDFReader

                    def check_grid(path, step, npoints, ncells, dimension, edge, expected_cells):
                        reader = vtkHDFReader()
                        reader.SetFileName(path)
                        reader.UpdateInformation()
                        reader.SetStep(step)
                        reader.Update()
                        grid = reader.GetOutput()
                        assert grid.GetNumberOfPoints() == npoints
                        assert grid.GetNumberOfCells() == ncells
                        bounds = {}
                        for c in range(ncells):
                            cell = grid.GetCell(c)
                            corners = cell.GetNumberOfPoints()
                            assert [cell.GetPointId(i) for i in range(corners)] == list(range(c * corners, (c + 1) * corners))
                            box = cell.GetBounds()
                            key = tuple(round((box[2*d] + box[2*d+1]) / (2*edge)) for d in range(dimension))
                            bounds[key] = box
                        assert len(bounds) == ncells
                        assert set(bounds) == set(expected_cells)
                        for key, box in bounds.items():
                            for d in range(dimension):
                                neighbor = tuple(k + (axis == d) for axis, k in enumerate(key))
                                if neighbor in bounds:
                                    assert box[2*d+1] == bounds[neighbor][2*d], (key, neighbor, box, bounds[neighbor])
                    """)
                    for (; path, step, npoints, ncells, dimension, edge, cells) in reader_frames
                        println(io, "check_grid(", repr(path), ", $step, $npoints, $ncells, $dimension, $edge, ", repr(cells), ")")
                    end
                end
                process = run(pipeline(ignorestatus(`$pvpython $script`); stdout, stderr))
                @test process.exitcode == 0
            end
        end
    end
end

end
