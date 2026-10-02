using Test
using SPHExampleGPU
using CUDA
using StaticArrays
using StructArrays
using LinearAlgebra
using HDF5

include(joinpath(@__DIR__, "still_wedge_geometry.jl"))
include(joinpath(@__DIR__, "still_wedge_middle_square_geometry.jl"))
include(joinpath(@__DIR__, "moving_square_geometry.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

# Project of the CPU package used as the reference. Defaults to the repository
# containing `gpu_version`; set `SPHEXAMPLE_CPU_REF` to compare against another
# checkout (e.g. a branch of the CPU code). The shared cases construct the meta
# data with mode type parameters, so the CPU checkout must provide that API.
const REPO      = normpath(get(ENV, "SPHEXAMPLE_CPU_REF", joinpath(@__DIR__, "..", "..")))
const CPU_REF   = joinpath(@__DIR__, "cpu_reference.jl")
const CPU_META  = joinpath(REPO, "src", "SimulationMetaDataConfiguration.jl")
const HAVE_CPU  = isfile(joinpath(REPO, "src", "SPHExample.jl")) && isfile(CPU_META) &&
                  occursin("TimeSteppingMode", read(CPU_META, String))

"""
Run `case` on the GPU for `simtime` seconds of physical time and return the
particles sorted by ID together with the meta data.
"""
function run_gpu(case::BenchCase, ::Type{T}, simtime; time_stepping = nothing, kwargs...) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    if time_stepping !== nothing
        kw = merge(kw, (; SimTimeStepping = time_stepping))
    end
    kw.SimMetaData.SimulationTime = T(simtime)
    kw.SimMetaData.OutputTimes    = T(simtime)
    for (k, v) in kwargs
        setproperty!(kw.SimMetaData, k, v)
    end
    particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
    logger    = SimulationLogger(save; to_console = false)
    RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
    order = sortperm(particles.ID)
    return particles[order], kw.SimMetaData
end

"""
Run the CPU reference in a separate Julia process (separate environment) and
return the stored state.
"""
function run_cpu_reference(case::BenchCase, simtime; time_stepping = nothing)
    out = tempname() * ".h5"
    scheme = time_stepping === nothing ? String[] : [string(nameof(typeof(time_stepping)))]
    cmd = `$(Base.julia_cmd()) -t 8,0 --project=$(REPO) $(CPU_REF) $(case.name) $(simtime) $(out) $(scheme)`
    run(pipeline(cmd; stdout = devnull, stderr = devnull))
    return h5open(out, "r") do fid
        (ID = read(fid["ID"]), Density = read(fid["Density"]), Pressure = read(fid["Pressure"]),
         Position = read(fid["Position"]), Velocity = read(fid["Velocity"]),
         Iteration = read(fid["Iteration"]), TotalTime = read(fid["TotalTime"]))
    end
end

relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), eps(eltype(b))))

@testset "SPHExampleGPU" begin
    @test CUDA.functional()

    @testset "mode types mirror the CPU API" begin
        save = mktempdir()
        meta = SimulationMetaData{2, Float32}(SimulationName = "m", SaveLocation = save)
        @test meta isa SimulationMetaData{2, Float32, NoShifting, NoKernelOutput, NoMDBC, NoLog}
        @test meta.TimeSteppingMode isa SingleNeighborTimeStepping
        meta = SimulationMetaData{2, Float32, PlanarShifting}(SimulationName = "m", SaveLocation = save)
        @test meta isa SimulationMetaData{2, Float32, PlanarShifting, NoKernelOutput, NoMDBC, NoLog}
        meta = SimulationMetaData{3, Float64, NoShifting, StoreKernelOutput, SimpleMDBC, StoreLog}(
            SimulationName = "m", SaveLocation = save, OutputTimes = 0.01)
        @test meta isa SimulationMetaData{3, Float64, NoShifting, StoreKernelOutput, SimpleMDBC, StoreLog}
        @test meta.OutputTimes === 0.01
        # the shared case file builds against this API: typed meta data plus
        # the time stepping scheme for `RunSimulation`
        for c in BENCH_CASES
            kw = c.build(Float64, save)
            @test kw.SimTimeStepping isa TimeSteppingMode
            @test kw.SimMetaData isa SimulationMetaData{c.dims, Float64, S, K, B, StoreLog} where {S, K, B}
            mdbc = kw.SimMetaData isa SimulationMetaData{c.dims, Float64, S, K, SimpleMDBC, L} where {S, K, L}
            @test mdbc == (kw.ParticleNormalsPath !== nothing)
        end
        kw = BENCH_CASES[1].build(Float64, save)
        particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
        @test length(particles) == length(AllocateDataStructures(kw.SimGeometry))
        @test hasproperty(particles, :GhostPoints)
    end

    @testset "output schedule is clamped to the simulation end" begin
        save = mktempdir()
        meta = SimulationMetaData{2, Float64}(SimulationName = "m", SaveLocation = save,
                                              SimulationTime = 0.25, OutputTimes = 0.1,
                                              OutputIterationCounter = 1)
        # one based frame counter: frame 1 is the initial state at t = 0
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.1
        meta.OutputIterationCounter = 2
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.2
        # the last interval ends at the simulation end, not at 0.3
        meta.OutputIterationCounter = 3
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
        meta.OutputIterationCounter = 4
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
        meta.OutputTimes = [0.1, 0.2]
        meta.OutputIterationCounter = 1
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.1
        meta.OutputIterationCounter = 2
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.2
        meta.OutputIterationCounter = 3
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
        meta.OutputTimes = [0.5]
        meta.OutputIterationCounter = 1
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
    end

    @testset "type-derived factors" begin
        for T in (Float32, Float64)
            for (type, gravity, limiter) in ((Fluid, -1, 1), (Fixed, 0, 0),
                                              (Moving, 1, 0))
                @test GravityFactorValue(T, type) === T(gravity)
                @test MotionLimiterValue(T, type) === T(limiter)
            end
        end
    end

    @testset "output variables are validated against the modes" begin
        save = mktempdir()
        meta = SimulationMetaData{2, Float32}(SimulationName = "m", SaveLocation = save,
                                              OutputVariables = ["Kernel", "Density", "GhostPoints", "Density"])
        kept = @test_logs (:warn, r"Kernel") (:warn, r"GhostPoints") resolve_output_variables!(meta)
        @test kept == ["Density"]
        @test meta.OutputVariables == ["Density"]
        meta = SimulationMetaData{2, Float32}(SimulationName = "m", SaveLocation = save, OutputVariables = ["ChunkID"])
        @test_throws ErrorException resolve_output_variables!(meta)
        meta = SimulationMetaData{2, Float32, NoShifting, StoreKernelOutput, SimpleMDBC}(
            SimulationName = "m", SaveLocation = save, OutputVariables = ["Kernel", "GhostNormals", "Acceleration"])
        @test resolve_output_variables!(meta) == ["Kernel", "GhostNormals", "Acceleration"]
        @test DEFAULT_OUTPUT_VARIABLES == ["Velocity", "Density", "Pressure", "ID", "Type", "GroupMarker"]
    end

    @testset "only the requested variables are written" begin
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        save = mktempdir()
        kw   = case.build(Float32, save)
        meta = kw.SimMetaData
        meta.SimulationTime  = 0.004f0
        meta.OutputTimes     = 0.002f0
        meta.ExportGridCells = true
        meta.OutputVariables = ["Density", "Velocity", "ID", "Acceleration", "GhostPoints", "Kernel"]
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger    = SimulationLogger(save; to_console = false)
        RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
        kernel_mode = meta isa SimulationMetaData{2, Float32, S, StoreKernelOutput} where {S}
        expected = kernel_mode ? ["Density", "Velocity", "ID", "Acceleration", "GhostPoints", "Kernel"] :
                                 ["Density", "Velocity", "ID", "Acceleration", "GhostPoints"]
        @test meta.OutputVariables == expected
        n = length(particles)
        h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do fid
            pd = fid["VTKHDF"]["PointData"]
            @test sort(keys(pd)) == sort(expected)
            @test size(pd["Velocity"]) == (3, 3n)        # 2D widened to 3 components, 3 frames
            @test size(pd["GhostPoints"]) == (3, 3n)
            @test length(pd["Density"]) == 3n
            @test any(!iszero, read(pd["GhostPoints"]))  # ghost data of the boundary particles
        end
        h5open(joinpath(save, meta.SimulationName * "_GridCells.vtkhdf"), "r") do fid
            @test keys(fid["VTKHDF"]["CellData"]) == ["CellData"]
        end
        # the complete final state is on the host, including fields not written
        @test all(isfinite, particles.Pressure)
        @test length(unique(particles.ID)) == n
    end

    @testset "buffered grid frame writer matches the reference append" begin
        # The buffered writer (`append_grid_frame!`) must produce the same file
        # content as the allocating reference (`AppendVTKHDFGridData`), for
        # every number of frames held before a flush (including a final
        # partial flush by `flush_frames!`).
        function write_grid(dir, frames, append!, finish = _ -> nothing)
            h5open(joinpath(dir, "grid.vtkhdf"), "w") do fid
                root = HDF5.create_group(fid, "VTKHDF")
                GenerateGeometryStructure(root; vtk_file_type = "UnstructuredGrid", chunk_size = 64)
                GenerateStepStructure(root; vtk_file_type = "UnstructuredGrid")
                for (k, cells) in enumerate(frames)
                    append!(root, 0.01 * k, cells)
                end
                finish(root)
            end
            return h5open(joinpath(dir, "grid.vtkhdf"), "r") do fid
                root = fid["VTKHDF"]
                names = ["Points", "Connectivity", "Offsets", "Types", "NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds"]
                d = Dict{String, Any}(n => read(root[n]) for n in names)
                d["CellData"] = read(root["CellData"]["CellData"])
                for n in ("Values", "PointOffsets", "NumberOfParts", "PartOffsets", "CellOffsets", "ConnectivityIdOffsets")
                    d["Steps/" * n] = read(root["Steps"][n])
                end
                d["NSteps"] = HDF5.read_attribute(root["Steps"], "NSteps")
                d
            end
        end
        for D in (2, 3)
            frames = [unique([CartesianIndex(ntuple(_ -> rand(-5:12), D)) for _ in 1:n]) for n in (7, 40, 1, 25, 12)]
            ref = write_grid(mktempdir(), frames, (root, t, cells) -> AppendVTKHDFGridData(root, t, 0.05, cells, nothing))
            for capacity in (1, 2, 5, MAX_BUFFERED_FRAMES)
                w   = nothing
                buf = write_grid(mktempdir(), frames, (root, t, cells) -> begin
                    w === nothing && (w = GridFrameWriter(root; capacity))
                    append_grid_frame!(w, t, 0.05, cells)
                end, _ -> begin
                    @test frames_pending(w) == length(frames) % capacity
                    flush_frames!(w)
                    @test frames_pending(w) == 0 && frames_written(w) == length(frames)
                end)
                @test sort(collect(keys(ref))) == sort(collect(keys(buf)))
                for k in keys(ref)
                    @test ref[k] == buf[k]
                end
                @test buf["NSteps"] == length(frames)
                @test buf["Steps/CellOffsets"] == cumsum([0; length.(frames[1:end-1])])
            end
        end
        # the buffers only grow and keep the exact column count of the frame
        g = GridGeometryBuffers()
        @test fill_grid_geometry!(g, 0.1, [CartesianIndex(1, 2), CartesianIndex(3, 4)]) == 8
        @test size(g.points) == (3, 8) && g.offsets == [0, 4, 8] && g.connectivity == 0:7
        @test fill_grid_geometry!(g, 0.1, CartesianIndex{2}[]) == 0
        @test size(g.points, 2) == 0 && g.offsets == [0]
    end

    @testset "buffered particle frame writer matches the reference append" begin
        # `PolyDataFrameWriter` must produce the same transient PolyData file
        # as `AppendVTKHDFData` called once per frame, for every capacity.
        n     = 17
        names = ["Velocity", "Density", "ID", "Type", "GhostPoints"]
        types = [Fixed, Fluid, Moving]
        frames = [(pos = [SVector{2, Float32}(rand(2)...) for _ in 1:n],
                   vel = [SVector{2, Float32}(rand(2)...) for _ in 1:n],
                   rho = rand(Float32, n), id = rand(Int, n), typ = rand(types, n),
                   gp  = [SVector{2, Float64}(rand(2)...) for _ in 1:n]) for _ in 1:7]
        function read_all(path)
            h5open(path, "r") do fid
                root = fid["VTKHDF"]
                d = Dict{String, Any}("Points" => read(root["Points"]), "NumberOfPoints" => read(root["NumberOfPoints"]),
                                      "NSteps" => HDF5.read_attribute(root["Steps"], "NSteps"))
                for v in names
                    d["PointData/" * v] = read(root["PointData"][v])
                    d["PointDataOffsets/" * v] = read(root["Steps"]["PointDataOffsets"][v])
                end
                for s in ("Values", "PointOffsets", "NumberOfParts", "PartOffsets", "CellOffsets", "ConnectivityIdOffsets")
                    d["Steps/" * s] = read(root["Steps"][s])
                end
                for c in ("Vertices", "Lines", "Polygons", "Strips"), s in ("NumberOfCells", "NumberOfConnectivityIds", "Offsets", "Connectivity")
                    d[c * "/" * s] = read(root[c][s])
                end
                d
            end
        end
        function write_particles(capacity)
            path = joinpath(mktempdir(), "p.vtkhdf")
            h5open(path, "w") do fid
                root = HDF5.create_group(fid, "VTKHDF")
                f1   = frames[1]
                # arrays with the element types and component counts as written (`Type` as `Int8`)
                descriptors = (Matrix{Float32}(undef, 3, 0), f1.rho, f1.id, Int8[], Matrix{Float64}(undef, 3, 0))
                GenerateGeometryStructure(root, names, descriptors...; chunk_size = n)
                GenerateStepStructure(root, names, descriptors...)
                if capacity === nothing
                    for (k, f) in enumerate(frames)
                        # the reference takes the 3 × N component form for the positions and vector fields and Int8 types
                        AppendVTKHDFData(root, 0.01 * k, stack(to_3d(f.pos)), names, stack(to_3d(f.vel)), f.rho, f.id, Int8.(f.typ), stack(to_3d(f.gp)))
                    end
                else
                    w = PolyDataFrameWriter(root, f1.pos, names, descriptors...; capacity)
                    for (k, f) in enumerate(frames)
                        append_frame!(w, 0.01 * k, f.pos, f.vel, f.rho, f.id, f.typ, f.gp)
                    end
                    @test frames_pending(w) == length(frames) % capacity
                    flush_frames!(w)
                    @test frames_written(w) == length(frames)
                    flush_frames!(w)   # nothing pending: no-op
                    @test frames_written(w) == length(frames)
                end
            end
            return read_all(path)
        end
        ref = write_particles(nothing)
        @test ref["NSteps"] == length(frames)
        @test size(ref["Points"]) == (3, n * length(frames))
        @test length(ref["Steps/NumberOfParts"]) == length(frames)
        @test eltype(ref["PointData/Type"]) == Int8
        @test eltype(ref["PointData/GhostPoints"]) == Float64
        for capacity in (1, 3, 7, MAX_BUFFERED_FRAMES)
            buf = write_particles(capacity)
            @test sort(collect(keys(ref))) == sort(collect(keys(buf)))
            for k in keys(ref)
                @test ref[k] == buf[k]
            end
        end
        @test_throws DimensionMismatch begin
            h5open(joinpath(mktempdir(), "q.vtkhdf"), "w") do fid
                root = HDF5.create_group(fid, "VTKHDF")
                GenerateGeometryStructure(root, ["Density"], frames[1].rho; chunk_size = n)
                GenerateStepStructure(root, ["Density"], frames[1].rho)
                w = PolyDataFrameWriter(root, frames[1].pos, ["Density"], frames[1].rho)
                append_frame!(w, 0.0, frames[1].pos[1:n-1], frames[1].rho[1:n-1])
            end
        end
        # frames per flush from the memory budget: a power of two within the cap
        @test buffered_frames(100, 1000) == 8
        @test buffered_frames(100, 700) == 4
        @test buffered_frames(1000, 100) == 1
        @test buffered_frames(1, 10^9) == MAX_BUFFERED_FRAMES
        @test buffered_frames(0, 0) == 1
        @test ispow2(MAX_BUFFERED_FRAMES)
    end

    @testset "asynchronous output frames match a synchronous write" begin
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        function frames(async; double = false, host32 = false)
            save = mktempdir()
            kw   = case.build(Float32, save)
            meta = kw.SimMetaData
            meta.SimulationTime    = 0.006f0
            meta.OutputTimes       = 0.002f0
            meta.ExportGridCells   = true
            meta.GPUAsyncOutput    = async
            meta.GPUDoublePosition = double
            meta.OutputVariables   = ["Density", "Velocity", "ID", "Type", "GhostPoints"]
            particles = host32 ? AllocateDataStructures(kw.SimGeometry) : AllocateDataStructures(kw.SimGeometry, meta)
            logger    = SimulationLogger(save; to_console = false)
            RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
            data = h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do fid
                root = fid["VTKHDF"]
                (Points = read(root["Points"]), Steps = read(root["Steps"]["Values"]),
                 PointData = Dict(k => read(root["PointData"][k]) for k in keys(root["PointData"])))
            end
            grid = h5open(joinpath(save, meta.SimulationName * "_GridCells.vtkhdf"), "r") do fid
                (NumberOfCells = read(fid["VTKHDF"]["NumberOfCells"]), CellData = read(fid["VTKHDF"]["CellData"]["CellData"]))
            end
            log = read(joinpath(save, "SimulationOutput.log"), String)
            return (; data, grid, particles, log)
        end
        sync  = frames(false)
        async = frames(true)
        n = length(sync.particles)
        nframes = length(sync.data.Steps)
        @test nframes >= 3
        # deterministic runs: the asynchronously written frames are bitwise identical
        @test async.data.Steps == sync.data.Steps
        @test size(sync.data.Points) == (3, nframes * n)
        @test async.data.Points == sync.data.Points
        @test sort(collect(keys(sync.data.PointData))) == ["Density", "GhostPoints", "ID", "Type", "Velocity"]
        for k in keys(sync.data.PointData)
            @test async.data.PointData[k] == sync.data.PointData[k]
        end
        @test eltype(sync.data.PointData["Type"]) == Int8
        @test size(sync.data.PointData["Velocity"]) == (3, nframes * n)
        # the initial frame has no cell list yet, so the grid file holds one frame less
        @test length(sync.grid.NumberOfCells) == nframes - 1
        @test async.grid.NumberOfCells == sync.grid.NumberOfCells
        @test async.grid.CellData == sync.grid.CellData
        # the log line of every frame is printed by the writer as well
        @test count("Part_", async.log) == count("Part_", sync.log) == nframes
        # the host arrays hold the final state in both modes
        order(p) = sortperm(p.ID)
        @test async.particles[order(async.particles)].Density == sync.particles[order(sync.particles)].Density
        @test async.particles[order(async.particles)].Cells == sync.particles[order(sync.particles)].Cells

        # Float32 host positions of a double position run are converted on the writer task
        d64 = frames(true; double = true)
        d32 = frames(true; double = true, host32 = true)
        @test eltype(d32.particles.Position) == SVector{2, Float32}
        @test size(d32.data.Points) == size(d64.data.Points)
        # the rounded Float32 start can move a particle across a cell boundary and
        # change the cell order of a frame: compare every frame sorted by ID
        function by_id(r)
            ids = r.data.PointData["ID"]
            perm = reduce(vcat, [(k - 1) * n .+ sortperm(ids[(k - 1) * n + 1:k * n]) for k in 1:nframes])
            return r.data.Points[:, perm], r.data.PointData["Density"][perm]
        end
        p64, ρ64 = by_id(d64)
        p32, ρ32 = by_id(d32)
        @test maximum(abs.(p64 .- p32)) < 1e-4
        @test ρ32 ≈ ρ64 rtol = 1e-3
        @test all(p -> all(isfinite, p), d32.particles.Position)
    end

    @testset "cell grid helpers" begin
        invH = 1 / 0.04
        @test map_floor(0.0, invH)   == 0
        @test map_floor(0.019, invH) == 0
        @test map_floor(0.021, invH) == 1
        @test map_floor(-0.021, invH) == -1
        @test map_floor(-0.019, invH) == 0
        grid = CellGrid{2}((Int32(-3), Int32(-2)), (Int32(10), Int32(8)), Int32(80))
        @test grid isa CellGrid{2, 1}
        for c in ((-2, -1), (0, 0), (5, 4))
            lin = SPHExampleGPU.GPUCellGrid.linear_cell(grid, Int32.(c))
            l   = SPHExampleGPU.GPUCellGrid.local_coords(grid, lin)
            @test l .+ grid.origin == Int32.(c)
        end
        g1 = CellGrid{3}((Int32(0), Int32(0), Int32(0)), (Int32(10), Int32(8), Int32(6)), Int32(480))
        @test SPHExampleGPU.GPUCellGrid.row_offsets(g1) == (-90, -80, -70, -10, 0, 10, 70, 80, 90)
        rows = cell_rows(g1)
        @test rows isa Tuple && length(rows) == 9
        @test [r.off for r in rows] == [-90, -80, -70, -10, 0, 10, 70, 80, 90]
        @test [r.dy for r in rows] == [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        @test [r.dz for r in rows] == [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        @test all(r.off == r.dy * 10 + r.dz * 80 for r in rows)
        @test all(r.dz == 0 for r in cell_rows(grid))
        CellStart = Int32.(0:10:1000)
        @test SPHExampleGPU.GPUCellGrid.row_range(g1, CellStart, Int32(50), Int32(0)) == (CellStart[49] + 1, CellStart[52])
        @test SPHExampleGPU.GPUCellGrid.row_range(g1, CellStart, Int32(50), rows[5]) == (CellStart[49] + 1, CellStart[52])
        @test cell_size(g1, 0.04f0) === 0.04f0
        @test SPHExampleGPU.GPUCellGrid.global_coords(g1, SPHExampleGPU.GPUCellGrid.linear_cell(g1, (Int32(3), Int32(4), Int32(3)))) == (3, 4, 3)
    end

    @testset "cell grid helpers, half width cells (reach 2)" begin
        G = SPHExampleGPU.GPUCellGrid
        grid = CellGrid{2, 2}((Int32(-3), Int32(-2)), (Int32(10), Int32(8)), Int32(80))
        @test G.reach(grid) == 2
        @test G.bin_scale(grid, 0.5f0) === 1.0f0
        for c in ((-1, 0), (0, 0), (4, 3))
            lin = G.linear_cell(grid, Int32.(c))
            @test G.local_coords(grid, lin) .+ grid.origin == Int32.(c)
        end
        # clamped into the margin: never closer than R cells to the edge
        @test G.local_coords(grid, G.linear_cell(grid, (Int32(-3), Int32(-2)))) == (2, 2)
        @test G.local_coords(grid, G.linear_cell(grid, (Int32(100), Int32(100)))) == (7, 5)
        @test collect(G.row_offsets(grid)) == [-20, -10, 0, 10, 20]
        @test [r.dy for r in G.cell_rows(grid)] == [-2, -1, 0, 1, 2]
        @test G.cell_size(grid, 0.04f0) === 0.02f0
        g3 = CellGrid{3, 2}((Int32(0), Int32(0), Int32(0)), (Int32(10), Int32(8), Int32(6)), Int32(480))
        offs = collect(G.row_offsets(g3))
        @test length(offs) == 25
        @test offs[13] == 0 && offs[1] == -2 * 80 - 2 * 10 && offs[25] == 2 * 80 + 2 * 10
        rows3 = collect(G.cell_rows(g3))
        @test length(rows3) == 25 && eltype(rows3) == CellRow
        @test all(r.off == r.dy * 10 + r.dz * 80 for r in rows3)
        @test rows3[1].dy == -2 && rows3[1].dz == -2 && rows3[25].dy == 2 && rows3[25].dz == 2
        @test length(unique((r.dy, r.dz) for r in rows3)) == 25
        CellStart = Int32.(0:10:1000)
        @test G.row_range(g3, CellStart, Int32(50), Int32(0)) == (CellStart[48] + 1, CellStart[53])
        @test_throws ArgumentError SimulationMetaData{2, Float32}(SimulationName = "x", SaveLocation = mktempdir(),
                                                                  GPUCellSubdivision = 0)
    end

    @testset "half width cells find the same pairs" begin
        # With the density diffusion off every pair term is antisymmetric, so
        # the H and H/2 grids must agree to rounding (same pairs, different
        # summation order). With it on, only the orientation of the asymmetric
        # term differs (`GPUCellSubdivision = 1` follows the CPU's choice).
        with_constants(c::SimulationConstants{T}; kwargs...) where {T} = begin
            names = fieldnames(typeof(c))
            nt = NamedTuple{names}(ntuple(i -> getfield(c, i), length(names)))
            SimulationConstants{T}(; merge(nt, values(kwargs))...)
        end
        for (name, simtime) in (("StillWedge2D_MDBC_dp0.02", 0.004), ("DamBreak3D_dp0.02", 0.002))
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            nodiff = BenchCase(case.name, case.dims, (T, s) -> begin
                kw = case.build(T, s)
                merge(kw, (; SimConstants = with_constants(kw.SimConstants; δᵩ = zero(T))))
            end)
            p1, m1 = run_gpu(nodiff, Float64, simtime; GPUCellSubdivision = 1)
            p2, m2 = run_gpu(nodiff, Float64, simtime; GPUCellSubdivision = 2)
            @test m1.Iteration == m2.Iteration > 1
            @test p1.ID == p2.ID
            @test relerr(p2.Density, p1.Density) < 1e-9
            @test maximum(norm.(p2.Position .- p1.Position)) < 1e-12
            @test maximum(norm.(p2.Velocity .- p1.Velocity)) < 1e-9
            q1, _ = run_gpu(case, Float64, simtime; GPUCellSubdivision = 1)
            q2, _ = run_gpu(case, Float64, simtime; GPUCellSubdivision = 2)
            @test relerr(q2.Density, q1.Density) < 1e-3
        end
    end

    @testset "fused reduction" begin
        n = 100_003
        x = CuArray(rand(Float64, n))
        ws = ReductionWorkspace{SVector{2, Float64}}(n)
        f(i, x) = (@inbounds v = x[i]; SVector(v, -v))
        op(a, b) = SVector(max(a[1], b[1]), min(a[2], b[2]))
        r = reduce_svector(ws, f, op, SVector(-Inf, Inf), n, x)
        xh = Array(x)
        @test r[1] == maximum(xh)
        @test r[2] == -maximum(xh)
    end

    @testset "counting sort orders particles by cell" begin
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        initial = AllocateDataStructures(case.build(Float64, mktempdir()).SimGeometry)
        device = upload_particles(initial)
        @test !hasproperty(device, :GravityFactor)
        @test !hasproperty(device, :MotionLimiter)
        @test !hasproperty(device.scratch, :GravityFactor)
        @test !hasproperty(device.scratch, :MotionLimiter)
        particles, meta = run_gpu(case, Float64, 1e-4)
        # after the run the host arrays are the (cell sorted) device state; the
        # sort above restored ID order, so check on a fresh download instead
        @test length(unique(particles.ID)) == length(particles)
        @test all(isfinite, particles.Density)
        @test !hasproperty(particles, :GravityFactor)
        @test !hasproperty(particles, :MotionLimiter)
        # derived or diagnostic-only fields are neither stored nor written
        @test !hasproperty(device, :BoundaryBool)
        @test !hasproperty(device, :ChunkID)
        @test !hasproperty(particles, :BoundaryBool)
        @test !hasproperty(particles, :ChunkID)
    end

    @testset "ghost node owners are relisted in cell order after a rebuild" begin
        # The mDBC kernel is launched over `GhostIndex` alone, so after every
        # reorder the list must name exactly the particles whose (reordered)
        # ghost point is non-zero, ascending, and the length must not change.
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        kw   = case.build(Float64, mktempdir())
        host = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
        _, gp, _ = LoadBoundaryNormals(Val(2), Float64, kw.ParticleNormalsPath)
        for gi in eachindex(gp)
            host.GhostPoints[gi] = gp[gi]
        end
        device = upload_particles(host)
        expected = Int32.(findall(!iszero, host.GhostPoints))
        @test Array(device.GhostIndex) == expected
        @test 0 < length(expected) < length(host)
        # scramble the host order before uploading so the rebuild has to move things
        perm    = sortperm(rand(length(host)))
        device2 = upload_particles(host[perm])
        cl      = CellListWorkspace{2, Float64}(length(host))
        SPHExampleGPU.SPHCellList.rebuild_cell_list!(device2, cl, kw.SimKernel.H⁻¹)
        idx = Array(device2.GhostIndex)
        @test length(idx) == length(expected)
        @test idx == Int32.(findall(!iszero, Array(device2.GhostPoints)))
        @test issorted(idx)
        @test all(==(Fixed), Array(device2.Type)[idx])
        # generic helper: indices of the non-zero entries in order
        x   = CuArray(Float64[0, 3, 0, 0, 1, 2, 0])
        ws  = CellListWorkspace{2, Float64}(length(x))
        out = CuVector{Int32}(undef, 3)
        @test Array(compact_nonzero!(ws, x, out)) == Int32[2, 5, 6]
        @test_throws ArgumentError compact_nonzero!(ws, CUDA.zeros(Float64, 2), out)
    end

    @testset "deterministic repeat" begin
        case = BENCH_CASES[findfirst(c -> c.name == "DamBreak2D_MDBC_dp0.01", BENCH_CASES)]
        p1, _ = run_gpu(case, Float64, 0.005)
        p2, _ = run_gpu(case, Float64, 0.005)
        @test p1.Density == p2.Density
        @test p1.Position == p2.Position
    end

    @testset "Float32 runs and stays physical" begin
        for name in ("StillWedge2D_MDBC_dp0.02", "DamBreak3D_dp0.02")
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            p, meta = run_gpu(case, Float32, 0.01)
            @test eltype(p.Density) == Float32
            @test all(isfinite, p.Density)
            @test all(x -> all(isfinite, x), p.Position)
            @test 900 < minimum(p.Density) && maximum(p.Density) < 1100
        end
    end

    @testset "cell relative positions form exact pair vectors far from the origin" begin
        # `pair_vector` on `PosCell`s must reproduce the Float64 pair vector to
        # an ulp of the cell edge, where the Float32 difference of the rounded
        # positions carries an ulp of the position (a thousand times larger at
        # an offset of 1e4 with H = 0.04).
        G = SPHExampleGPU.GPUCellGrid
        H = 0.04f0
        for (D, R) in ((2, 1), (3, 1), (3, 2))
            offset = SVector{D, Float64}(ntuple(d -> (-1.0)^d * 1e4 + 3.0 * d, D))
            c0     = G.cell_coords(offset, G.bin_scale(CellGrid{D, R}(), 1 / H))
            grid   = CellGrid{D, R}(c0 .- Int32(40), ntuple(_ -> Int32(80), D), Int32(80^D))
            s      = cell_size(grid, H)
            inv    = G.bin_scale(grid, 1 / H)
            @test s === H / R
            n1, n2 = grid.dims[1], grid.dims[2]
            worst_pc = 0.0; worst_naive = 0.0; tested = 0
            for _ in 1:2000
                xi = offset + 0.2 .* randn(SVector{D, Float64})
                xj = xi + 0.03 .* randn(SVector{D, Float64})
                gi = G.cell_coords(xi, inv); gj = G.cell_coords(xj, inv)
                # `linear_cell` clamps into the margin: keep to the interior
                (G.in_grid(grid, gi .- Int32(R)) && G.in_grid(grid, gi .+ Int32(R)) &&
                 G.in_grid(grid, gj .- Int32(R)) && G.in_grid(grid, gj .+ Int32(R))) || continue
                ci = G.linear_cell(grid, gi)
                cj = G.linear_cell(grid, gj)
                d  = G.local_coords(grid, cj) .- G.local_coords(grid, ci)
                all(abs.(d) .<= R) || continue
                dy = Int32(d[2]); dz = D == 3 ? Int32(d[3]) : Int32(0)
                row = CellRow(dy * n1 + dz * n1 * n2, dy, dz)
                pcs = [pos_cell(xi, ci, grid, s), pos_cell(xj, cj, grid, s)]
                @test all(abs.(pcs[1].rel) .<= s)                    # bounded by the cell edge
                @test pcs[1].cell == ci && pcs[2].cell == cj
                v     = pair_vector(pcs, pcs[1], Int32(2), ci + row.off, row, s)
                exact = xi - xj
                naive = SVector{D, Float32}(xi) - SVector{D, Float32}(xj)
                @test v isa SVector{D, Float32}
                worst_pc    = max(worst_pc, norm(v - exact))
                worst_naive = max(worst_naive, norm(naive - exact))
                tested += 1
            end
            @test tested > 500
            @test worst_pc < 1e-5 * H
            @test worst_naive > 100 * worst_pc
            # plain positions: the reference is the position itself
            xs = [offset, offset + SVector{D, Float64}(ntuple(_ -> 0.01, D))]
            @test pair_vector(xs, xs[1], Int32(2), Int32(0), CellRow(Int32(0), Int32(0), Int32(0)), s) == xs[1] - xs[2]
            # a point that is not a particle (ghost node) is referred to its own cell
            cg = G.cell_coords(offset, inv)
            pg = pos_cell(offset, cg, s)
            @test pg.cell == 0 && all(abs.(pg.rel) .<= s)
        end
    end

    @testset "double positions: allocation, upload and download" begin
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        save = mktempdir()
        kw   = case.build(Float32, save)
        meta = kw.SimMetaData
        @test meta.GPUDoublePosition == false
        @test position_float_type(meta) === Float32
        meta.GPUDoublePosition = true
        @test position_float_type(meta) === Float64
        @test position_float_type(SimulationMetaData{2, Float64}(SimulationName = "m", SaveLocation = save,
                                                                  GPUDoublePosition = true)) === Float64
        host = AllocateDataStructures(kw.SimGeometry, meta)
        # only the positions (and ghost nodes) are Float64, as in DualSPHysics
        @test eltype(host.Position)    == SVector{2, Float64}
        @test eltype(host.GhostPoints) == SVector{2, Float64}
        @test eltype(host.Velocity)    == SVector{2, Float32}
        @test eltype(host.Density)     == Float32
        @test eltype(host.Acceleration) == SVector{2, Float32}
        @test eltype(host.GhostNormals) == SVector{2, Float32}
        # the input is read with its full precision
        host32 = AllocateDataStructures(kw.SimGeometry)
        @test eltype(host32.Position) == SVector{2, Float32}
        @test host32.ID == host.ID
        @test maximum(norm.(host32.Position .- host.Position)) < 1e-6
        @test any(host32.Position .!= host.Position)
        dev = upload_particles(host; position_type = Float64)
        @test position_type(dev) === Float64
        @test eltype(dev.Position) == SVector{2, Float64} && eltype(dev.Velocity) == SVector{2, Float32}
        @test eltype(dev.scratch.Position) == SVector{2, Float64}
        # a Float32 host array is converted on upload, and back on download
        dev2 = upload_particles(host32; position_type = Float64)
        @test position_type(dev2) === Float64
        @test Array(dev2.Position) == SVector{2, Float64}.(host32.Position)
        @test position_type(upload_particles(host32)) === Float32
        cl = CellListWorkspace{2, Float64}(length(host32))
        grid = SPHExampleGPU.SPHCellList.rebuild_cell_list!(dev2, cl, kw.SimKernel.H⁻¹)
        back = deepcopy(host32)
        download_particles!(back, dev2, grid, (:ID,))
        @test eltype(back.Position) == SVector{2, Float32}
        @test back[sortperm(back.ID)].Position == host32.Position    # host32 is in ID order
        sup = GPUSupportArrays{2, Float32}(length(host); position_type = Float64)
        @test uses_pos_cells(sup)
        @test eltype(sup.PosCells) == PosCell{2, Float32} && eltype(sup.Positionₙ⁺) == SVector{2, Float64}
        @test eltype(sup.Velocityₙ⁺) == SVector{2, Float32}
        @test !uses_pos_cells(GPUSupportArrays{2, Float32}(4))
        @test GPUSupportArrays{2, Float32}(4).PosCells === nothing
    end

    @testset "double positions run and match single precision positions nearby" begin
        # Near the origin the two representations differ at the Float32 ulp
        # level only. mDBC (ghost nodes referred to their cell), lanes, the
        # half width grid, moving bodies and shifting all go through the cell
        # relative pair vectors.
        for (name, kwargs) in (("StillWedge2D_MDBC_dp0.02", (;)),
                               ("DamBreak3D_dp0.02", (;)),
                               ("DamBreak3D_dp0.02", (; GPUCellSubdivision = 2, GPULanesPerParticle = 1)),
                               ("MovingSquare2D_dp0.04", (;)))
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            p32, m32 = run_gpu(case, Float32, 0.004; kwargs...)
            pd,  md  = run_gpu(case, Float32, 0.004; GPUDoublePosition = true, kwargs...)
            @test md.GPUDoublePosition
            @test eltype(pd.Position) == SVector{case.dims, Float64}
            @test eltype(pd.Velocity) == SVector{case.dims, Float32}
            @test eltype(pd.Density)  == Float32
            @test m32.Iteration == md.Iteration > 1
            @test pd.ID == p32.ID
            @test all(isfinite, pd.Density) && all(x -> all(isfinite, x), pd.Position)
            @test 900 < minimum(pd.Density) && maximum(pd.Density) < 1100
            # the impulsively started square amplifies any rounding difference
            tol = name == "MovingSquare2D_dp0.04" ? 1e-2 : 1e-4
            @test maximum(norm.(pd.Position .- p32.Position)) < tol
            @test relerr(pd.Density, p32.Density) < tol
        end
        # with Float64 arithmetic the option changes nothing at all
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        p1, _ = run_gpu(case, Float64, 0.004)
        p2, m2 = run_gpu(case, Float64, 0.004; GPUDoublePosition = true)
        @test m2.GPUDoublePosition
        @test p1.Position == p2.Position && p1.Density == p2.Density && p1.Velocity == p2.Velocity
    end

    @testset "double positions keep the precision far from the origin" begin
        # The same still wedge translated by 1e4 in both directions. Float32
        # positions are then quantised to about 1e-3 (a tenth of dx) and the
        # run degrades; Float64 positions with Float32 arithmetic on cell
        # relative coordinates stay as close to the Float64 reference as the
        # untranslated Float32 run.
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_DBC_dp0.01", BENCH_CASES)]
        off  = SVector(1.0e4, 1.0e4)
        function run_shifted(::Type{T}, simtime; kwargs...) where {T}
            save = mktempdir()
            kw   = case.build(T, save)
            kw.SimMetaData.SimulationTime = T(simtime)
            kw.SimMetaData.OutputTimes    = T(simtime)
            for (k, v) in kwargs
                setproperty!(kw.SimMetaData, k, v)
            end
            particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
            particles.Position .+= Ref(eltype(particles.Position)(off))
            logger = SimulationLogger(save; to_console = false)
            RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
            order = sortperm(particles.ID)
            return particles[order], kw.SimMetaData, save
        end
        simtime = 0.02
        ref,   mref, _    = run_shifted(Float64, simtime)
        plain, mp,   _    = run_shifted(Float32, simtime)
        dbl,   md,   save = run_shifted(Float32, simtime; GPUDoublePosition = true)
        near,  _          = run_gpu(case, Float32, simtime)
        ref0,  _          = run_gpu(case, Float64, simtime)
        @test mref.Iteration == mp.Iteration == md.Iteration > 100
        @test ref.ID == plain.ID == dbl.ID
        dx_plain = maximum(norm.(plain.Position .- ref.Position))
        dx_dbl   = maximum(norm.(dbl.Position .- ref.Position))
        dx_near  = maximum(norm.(near.Position .- ref0.Position))
        dρ_plain = relerr(plain.Density, ref.Density)
        dρ_dbl   = relerr(dbl.Density, ref.Density)
        dρ_near  = relerr(near.Density, ref0.Density)
        @info "offset 1e4: max |Δx| plain Float32 = $dx_plain, Float64 positions = $dx_dbl (nearby Float32: $dx_near); " *
              "max rel Δρ plain = $dρ_plain, Float64 positions = $dρ_dbl (nearby: $dρ_near)"
        @test dx_plain > 2e-4                     # the Float32 ulp at 1e4 is about 1e-3
        @test dx_dbl   < 5e-5
        @test dx_dbl   < dx_plain / 20
        @test dx_dbl   < 10 * dx_near
        @test dρ_plain > 5e-4
        @test dρ_dbl   < 2e-4
        @test dρ_dbl   < 10 * dρ_near
        @test all(abs.(first.(dbl.Position) .- 1e4) .< 10)   # the particles stayed where they were put
        # the positions are written to the file with their Float64 precision
        h5open(joinpath(save, md.SimulationName * ".vtkhdf"), "r") do fid
            pts = read(fid["VTKHDF"]["Points"])
            @test eltype(pts) == Float64
            n = length(dbl)
            last_frame = pts[:, end - n + 1:end]
            @test maximum(abs.(sort(last_frame[1, :]) .- sort(first.(dbl.Position)))) < 1e-9
            @test eltype(read(fid["VTKHDF"]["PointData"]["Velocity"])) == Float32
        end
    end

    @testset "final step corrector advances with the half step velocity" begin
        # The symplectic corrector of the CPU `FullTimeStep` moves the position
        # with `Velocityₙ⁺ * dt`. The GPU kernel is checked element by element
        # against that formula (with and without shifting) and against the
        # earlier averaged velocity scheme, which it must not reproduce.
        T = Float64
        V = SVector{2, T}
        n = 4_096
        rnd() = V(randn(T), randn(T))
        types  = rand([Fluid, Fixed, Moving], n)
        consts = SimulationConstants{T}(dx = 0.02, c₀ = 42.0, δᵩ = 0.1, CFL = 0.5)
        kern   = SPHKernelInstance{2, T}(WendlandC2(); dx = consts.dx)
        dt     = T(1e-4)
        for FlagShift in (false, true)
            Position     = [rnd() for _ in 1:n]
            Velocity     = [rnd() for _ in 1:n]
            Acceleration = [10 * rnd() for _ in 1:n]
            Velocityₙ⁺   = [rnd() for _ in 1:n]
            Positionₙ⁺   = Position .+ [1e-3 * rnd() for _ in 1:n]
            Density  = 1000 .+ 50 .* rand(T, n)
            Density[1:3:end] .-= 60     # below ρ₀, as the mDBC correction may leave a boundary density
            ρₙ⁺      = 1000 .+ 50 .* rand(T, n)
            dρdtI    = randn(T, n)
            ∇Cᵢ      = [rnd() for _ in 1:n]
            ∇◌rᵢ     = 4 .* rand(T, n) .- 1     # negative values must give no shift

            dP  = CuArray(Position);      dV  = CuArray(Velocity)
            dA  = CuArray(Acceleration);  dρ  = CuArray(Density)
            dPr = CUDA.zeros(T, n);       ddρ = CuArray(dρdtI)
            dρn = CuArray(ρₙ⁺);           dPn = CuArray(Positionₙ⁺)
            dVn = CuArray(Velocityₙ⁺);    dty = CuArray(types)
            dC  = CuArray(∇Cᵢ);           dr  = CuArray(∇◌rᵢ)
            red = ReductionWorkspace{SVector{3, T}}(n)
            launch_final_step!(dP, dV, dA, dρ, dPr, ddρ, dρn, dPn, dVn, dty, dC, dr,
                               HostStep(dt, zero(T)), kern, consts, red, Val(FlagShift))
            xg = Array(dP)
            vg = Array(dV)
            ρg = Array(dρ)
            Pg = Array(dPr)

            x_ref = similar(Position); v_ref = similar(Velocity); x_avg = similar(Position)
            ρ_ref = similar(Density); ρ_upd = similar(Density)
            for i in 1:n
                ML  = MotionLimiterValue(T, types[i])
                GF  = GravityFactorValue(T, types[i])
                # DensityEpsi! then LimitDensityAtBoundary! (the CPU order)
                epsi = -(dρdtI[i] / ρₙ⁺[i]) * dt
                ρ    = Density[i] * (2 - epsi) / (2 + epsi)
                ρ_upd[i] = ρ
                ρ_ref[i] = (types[i] != Fluid && ρ < consts.ρ₀) ? consts.ρ₀ : ρ
                acc = Acceleration[i] + ConstructGravitySVector(Acceleration[i], consts.g * GF)
                v   = Velocity[i] + acc * dt * ML
                δx  = zero(V)
                if FlagShift
                    A_FSC = (∇◌rᵢ[i] - 0) / (2 - 0)
                    δx = A_FSC < 0 ? zero(V) : -A_FSC * 2 * kern.h * norm(Velocityₙ⁺[i]) * dt * ∇Cᵢ[i]
                end
                v_ref[i] = v
                x_ref[i] = Position[i] + (Velocityₙ⁺[i] * dt + δx) * ML
                x_avg[i] = Position[i] + (((v + (v - acc * dt * ML)) / 2) * dt + δx) * ML
            end
            @test all(isapprox.(vg, v_ref; rtol = 1e-13, atol = 1e-15))
            @test all(isapprox.(xg, x_ref; rtol = 1e-13, atol = 1e-15))
            @test all(isapprox.(ρg, ρ_ref; rtol = 1e-13))
            @test all(isapprox.(Pg, EquationOfStateGamma7.(ρg, consts.c₀, consts.ρ₀); rtol = 1e-12, atol = 1e-8))
            # limiting after the update: a boundary particle whose updated
            # density is below ρ₀ ends exactly at ρ₀ (limiting before the
            # update would leave it at ρ₀ * (2 - ϵ) / (2 + ϵ) instead)
            clamped = (types .!= Fluid) .& (ρ_upd .< consts.ρ₀)
            @test count(clamped) > n ÷ 10
            @test all(ρg[clamped] .== consts.ρ₀)
            @test all(ρg[.!clamped] .!= consts.ρ₀)
            # the old averaged scheme differs for every moving particle
            moving = types .== Fluid
            @test maximum(norm.(xg[moving] .- x_avg[moving])) > 1e-6
            # non-fluid particles do not move
            @test all(xg[.!moving] .== Position[.!moving])
        end
    end

    @testset "device resident step state" begin
        # The finish kernel must reproduce the host formula for `dt` and the
        # displacement bookkeeping exactly, and the stop flag has to gate
        # every kernel of a step.
        S = SPHExampleGPU.GPUStepState
        T = Float64
        consts = SimulationConstants{T}(dx = 0.02, c₀ = 42.0, δᵩ = 0.1, CFL = 0.5)
        kern   = SPHKernelInstance{2, T}(WendlandC2(); dx = consts.dx)
        h      = kern.h
        red    = ReductionWorkspace{SVector{3, T}}(100_000)
        @test red.nblocks > 1
        partials = [SVector{3, T}(rand(), 1 + rand(), 0.2 * rand()) for _ in 1:red.nblocks]
        partials[min(7, red.nblocks)] = SVector{3, T}(3.0, 0.5, 0.25)   # extrema of all three components
        copyto!(red.partial, partials)
        visc, dt1, disp = 3.0, 0.5, 0.25
        dt_ref = consts.CFL * min(dt1, h / (consts.c₀ + visc))

        st = StepState{T}(; time = 0.0, dx = 0.0)
        S.set_output_time!(st, 1.0)
        launch_finish!(st, red, kern, consts)
        readback!(st)
        @test st.fh[S.F_DT]   == dt_ref
        @test st.fh[S.F_DISP] == 4 * disp
        @test st.fh[S.F_DX]   == 4 * disp
        @test st.ih[S.I_PHASE] == S.PHASE_DT_READY
        @test st.ih[S.I_STOP]  == S.STOP_REBUILD          # 4 * disp >= h
        # nothing runs while stopped
        launch_commit!(st)
        readback!(st)
        @test st.ih[S.I_ITER] == 0 && st.fh[S.F_TIME] == 0
        # after the rebuild the stored dt is kept and the step commits
        S.resume_after_rebuild!(st)
        launch_finish!(st, red, kern, consts)
        launch_commit!(st)
        readback!(st)
        @test st.fh[S.F_DX] == 0
        @test st.fh[S.F_DT] == dt_ref
        @test st.fh[S.F_TIME] == dt_ref
        @test st.ih[S.I_ITER] == 1
        @test st.ih[S.I_PHASE] == S.PHASE_NEED_DT
        @test st.ih[S.I_STOP]  == S.STOP_NONE
        # a second step accumulates the displacement bound
        copyto!(red.partial, fill(SVector{3, T}(0.1, 2.0, 1e-4), red.nblocks))
        launch_finish!(st, red, kern, consts)
        launch_commit!(st)
        readback!(st)
        @test st.fh[S.F_DX] == 4e-4
        @test st.fh[S.F_DT] == consts.CFL * min(2.0, h / (consts.c₀ + 0.1))
        @test st.fh[S.F_TIME] == dt_ref + st.fh[S.F_DT]
        @test st.ih[S.I_ITER] == 2
        # output time reached: stop before computing anything
        S.set_output_time!(st, 0.0)
        launch_finish!(st, red, kern, consts)
        launch_commit!(st)
        readback!(st)
        @test st.ih[S.I_STOP] == S.STOP_OUTPUT
        @test st.ih[S.I_ITER] == 2
        @test st.ih[S.I_PHASE] == S.PHASE_NEED_DT

        # batch size: estimate until rebuild / output, at most kmax, at least 1
        bs = SPHExampleGPU.SPHCellList.batch_size
        st.fh[S.F_DX] = 0.0; st.fh[S.F_DISP] = h / 10; st.fh[S.F_DT] = 1e-3; st.fh[S.F_TIME] = 0.0
        @test bs(st, h, 1.0, 32) == 11          # 10 steps until Δx reaches h, plus one
        @test bs(st, h, 0.0035, 32) == 4        # 3.5 steps until the output, plus one
        @test bs(st, h, 1.0, 4) == 4
        @test bs(st, h, 1.0, 1) == 1
        st.fh[S.F_DX] = h
        @test bs(st, h, 1.0, 32) == 1           # rebuild already due
        st.fh[S.F_DX] = 0.0; st.fh[S.F_DISP] = 0.0; st.fh[S.F_DT] = 0.0
        @test bs(st, h, 1.0, 32) == 32          # no information: fill the batch

        # a stopped state gates the particle kernels
        n = 2_048
        V = SVector{2, T}
        rnd() = V(randn(T), randn(T))
        P = [rnd() for _ in 1:n]
        dP = CuArray(P); dV = CuArray([rnd() for _ in 1:n]); dA = CuArray([rnd() for _ in 1:n])
        dρ = CuArray(1000 .+ rand(T, n)); dPr = CUDA.zeros(T, n); ddρ = CuArray(randn(T, n))
        dρn = CuArray(1000 .+ rand(T, n)); dPn = CuArray(P); dVn = CuArray([rnd() for _ in 1:n])
        dty = CuArray(fill(Fluid, n)); dC = CuArray([rnd() for _ in 1:n]); dr = CuArray(rand(T, n))
        red2 = ReductionWorkspace{SVector{3, T}}(n)
        launch_final_step!(dP, dV, dA, dρ, dPr, ddρ, dρn, dPn, dVn, dty, dC, dr, st, kern, consts, red2, Val(false))
        @test Array(dP) == P
        S.set_output_time!(st, 1.0)
        launch_final_step!(dP, dV, dA, dρ, dPr, ddρ, dρn, dPn, dVn, dty, dC, dr, st, kern, consts, red2, Val(false))
        @test Array(dP) != P
    end

    @testset "single neighbour scheme with mDBC runs, applies the correction and replays as a graph" begin
        # The boundary densities of the still wedge only leave ρ₀ through the
        # mDBC correction (the boundary particles do not move and their
        # continuity update follows the corrected density), so a run with
        # `SimpleMDBC` must have corrected boundary densities after the first
        # steps. Graph replay in batches must reproduce direct launches with
        # a read back after every step bitwise.
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        p1, m1 = run_gpu(case, Float64, 0.02; time_stepping = SingleNeighborTimeStepping())
        p2, m2 = run_gpu(case, Float64, 0.02; time_stepping = SingleNeighborTimeStepping(),
                         GPUUseGraph = false, GPUMaxStepsPerSync = 1)
        @test m1.TimeSteppingMode isa SingleNeighborTimeStepping
        @test m1.Iteration == m2.Iteration > 1
        @test m1.TotalTime == m2.TotalTime
        @test p1.ID == p2.ID
        @test p1.Position == p2.Position
        @test p1.Velocity == p2.Velocity
        @test p1.Density  == p2.Density
        bound = p1.Type .== Fixed
        @test any(bound)
        @test all(isfinite, p1.Density) && all(isfinite, p1.Pressure)
        @test any(abs.(p1.Density[bound] .- 1000) .> 1e-3)
    end

    @testset "graph replay and batching reproduce direct launches" begin
        # The same steps, once as replayed CUDA graphs in batches and once
        # launched directly with a read back after every step, must give the
        # same bits (deterministic sort keeps the particle order identical).
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        p1, m1 = run_gpu(case, Float64, 0.004)
        p2, m2 = run_gpu(case, Float64, 0.004; GPUUseGraph = false, GPUMaxStepsPerSync = 1)
        @test m1.Iteration == m2.Iteration > 1
        @test m1.TotalTime == m2.TotalTime
        @test p1.ID == p2.ID
        @test p1.Position == p2.Position
        @test p1.Velocity == p2.Velocity
        @test p1.Density  == p2.Density
    end

    @testset "models use the densities passed by the kernel" begin
        # The kernel hands the models ρᵢ, ρⱼ and their reciprocals of the state
        # being evaluated. The built in models must not reach back into a
        # particle array (which in the corrector loop would hold the wrong
        # state), so `SimParticles = nothing` has to work.
        T  = Float64
        consts = SimulationConstants{T}(dx = 0.02, c₀ = 42.0, δᵩ = 0.1, α = 0.02, ν₀ = 1e-4)
        kern   = SPHKernelInstance{2, T}(WendlandC2(); dx = consts.dx)
        xᵢⱼ    = SVector{2, T}(0.011, -0.007)
        vᵢⱼ    = SVector{2, T}(0.3, 0.1)
        d²     = dot(xᵢⱼ, xᵢⱼ)
        ∇W     = ∇Wᵢⱼ(kern, sqrt(d²) * kern.h⁻¹, xᵢⱼ)
        ρᵢ, ρⱼ = T(1003.5), T(998.2)
        types  = [Fluid, Fluid]
        for model in (ZeroViscosity(), ArtificialViscosity(), Laminar(), LaminarSPS())
            Πᵢ, Πⱼ = compute_viscosity(model, kern, consts, nothing, xᵢⱼ, vᵢⱼ, ∇W, d²,
                                       ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2)
            @test all(isfinite, Πᵢ) && all(isfinite, Πⱼ)
            @test Πⱼ ≈ -Πᵢ
        end
        for model in (ZeroDensityDiffusion(), ZeroGravityLinearDensityDiffusion(),
                      LinearDensityDiffusion(), ComplexDensityDiffusion())
            Dᵢ, Dⱼ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                               ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, types)
            @test isfinite(Dᵢ) && Dⱼ == -Dᵢ
        end
        # The gated models are exactly zero unless both particles are fluid and
        # non zero (the gate is a select of the full term) when both are.
        for model in (LinearDensityDiffusion(), ComplexDensityDiffusion())
            Dᵢ, Dⱼ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                               ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, [Fluid, Fixed])
            @test Dᵢ == 0 && Dⱼ == 0
            Dᵢ, Dⱼ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                               ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, [Fixed, Fluid])
            @test Dᵢ == 0 && Dⱼ == 0
            Dᵢ, _ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                              ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, [Fluid, Fluid])
            @test Dᵢ != 0
        end
        # Laminar viscosity is antisymmetric in the density arguments and
        # depends on them, so passing different densities must change it.
        a, _ = compute_viscosity(Laminar(), kern, consts, nothing, xᵢⱼ, vᵢⱼ, ∇W, d²,
                                 ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2)
        b, _ = compute_viscosity(Laminar(), kern, consts, nothing, xᵢⱼ, vᵢⱼ, ∇W, d²,
                                 2ρᵢ, 2ρⱼ, inv(2ρᵢ), inv(2ρⱼ), 1, 2)
        @test a ≈ 2b
    end

    if HAVE_CPU
        @testset "matches CPU reference: $(name) ($(nameof(typeof(scheme))))" for (name, simtime, scheme) in (
                ("StillWedge2D_MDBC_dp0.02", 0.02,  SymplecticTimeStepping()),
                ("MovingSquare2D_dp0.04",    0.01,  SymplecticTimeStepping()),
                ("DamBreak3D_dp0.02",        0.005, SymplecticTimeStepping()),
                ("Duckling3D_MDBC_dp0.01",   0.005, SymplecticTimeStepping()),
                ("StillWedge2D_MDBC_dp0.02", 0.02,  SingleNeighborTimeStepping()),
                ("DamBreak3D_dp0.02",        0.005, SingleNeighborTimeStepping()),
            )
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            ref  = run_cpu_reference(case, simtime; time_stepping = scheme)
            p, meta = run_gpu(case, Float64, simtime; time_stepping = scheme)

            @test meta.Iteration == ref.Iteration
            @test p.ID == ref.ID
            dρ = relerr(p.Density, ref.Density)
            dx = maximum(norm.(p.Position .- eachcol(ref.Position)))
            dv = maximum(norm.(p.Velocity .- eachcol(ref.Velocity)))
            @info "CPU vs GPU ($name): steps=$(meta.Iteration) max rel Δρ=$(dρ) max |Δx|=$(dx) max |Δv|=$(dv)"
            @test dρ < 1e-8
            @test dx < 1e-9
            @test dv < 1e-7
        end
    else
        @warn "CPU package with the mode type API not found at $(REPO); skipping CPU comparison tests " *
              "(set SPHEXAMPLE_CPU_REF to a checkout that has it)"
    end
end
