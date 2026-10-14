module GPUMemoryTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5

include(joinpath(@__DIR__, "..", "benchmark", "memory_buffers.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

const DRIVER = SPHExampleGPU.SPHCellList

function fixture(D, T; double_position = false, shifting = NoShifting,
                 kernel_output = NoKernelOutput, boundary = NoMDBC)
    points = vec([SVector{D, T}(ntuple(d -> T(Tuple(index)[d]) * T(0.02), D))
                  for index in CartesianIndices(ntuple(_ -> 0:3, D))])
    geometry = [SPHGeometry{D, T}(points; Density = 1000, GroupMarker = 1, Type = Fluid)]
    metadata = SimulationMetaData{D, T, shifting, kernel_output, boundary}(
        SimulationName = "Memory", SaveLocation = tempdir(), GPUDoublePosition = double_position)
    host = AllocateDataStructures(geometry, metadata)
    return (; host, metadata, geometry)
end

function hdf_datasets(path)
    datasets = Dict{String, Any}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            key = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, key)
            else
                data = read(item)
                datasets[key] = (eltype(data), size(data), data)
            end
            close(item)
        end
    end
    h5open(path, "r") do file
        visit(file, "")
    end
    return datasets
end

function simulate(build, T; compact, double_position, graph, async, scheme)
    return mktempdir() do save
        kwargs = build(T, save)
        metadata = kwargs.SimMetaData
        metadata.GPUCompactBuffers = compact
        metadata.GPUDoublePosition = double_position
        metadata.GPUUseGraph = graph
        metadata.GPUAsyncOutput = async
        metadata.GPUOutputQueueBytes = 1024^2
        metadata.SimulationTime = T(0.006)
        metadata.OutputTimes = T(0.002)
        metadata.ExportGridCells = true
        host = AllocateDataStructures(kwargs.SimGeometry, metadata)
        logger = SimulationLogger(save)
        try
            redirect_stdout(devnull) do
                RunSimulation(; merge(kwargs, (; SimTimeStepping = scheme))...,
                    SimParticles = host, SimLogger = logger)
            end
        finally
            close(logger.LoggerIo)
        end
        order = sortperm(host.ID)
        fields = map(name -> (eltype(getproperty(host, name)), getproperty(host, name)[order]),
            DRIVER.DOWNLOADABLE_FIELDS)
        return (; steps = metadata.Iteration, time = metadata.TotalTime,
            position = (eltype(host.Position), host.Position[order]), fields,
            particles = hdf_datasets(joinpath(save, metadata.SimulationName * ".vtkhdf")),
            cells = hdf_datasets(joinpath(save, metadata.SimulationName * "_GridCells.vtkhdf")))
    end
end

function updated_case(T, save)
    normal = SVector{2, T}(0, 0.1)
    wall = SVector{2, T}(0.5, 0)
    ghost = wall + normal
    fluid = [ghost + SVector{2, T}(T(i) * T(0.04), T(j) * T(0.04)) for i in -2:2 for j in -2:2]
    geometry = [
        SPHGeometry{2, T}(Particles = particle_struct_array([wall], T(1000);
            GhostPoints = [ghost], GhostNormals = [normal]), GroupMarker = 1, Type = Moving,
            Motion = MotionDetails{2, T}(Velocity = T(0.1), StartTime = -one(T), Duration = one(T),
                Direction = SVector{2, T}(1, 0))),
        SPHGeometry{2, T}(fluid; Density = 999, GroupMarker = 2, Type = Fluid),
    ]
    constants = SimulationConstants{T}(dx = T(0.04), c₀ = T(20), g = zero(T))
    metadata = SimulationMetaData{2, T, PlanarShifting, StoreKernelOutput, UpdatedMDBC}(
        SimulationName = "MemoryUpdated", SaveLocation = save,
        VisualizeInParaview = false, OpenLogFile = false)
    return (; SimGeometry = geometry, SimMetaData = metadata, SimConstants = constants,
        SimKernel = SPHKernelInstance{2, T}(WendlandC2(); dx = constants.dx),
        SimViscosity = Laminar(), SimDensityDiffusion = ZeroDensityDiffusion())
end

function floating_case(T, save; D)
    points = [SVector{D, T}(ntuple(d -> one(T) + T((i >> (d - 1)) & 1) * T(0.02), D))
              for i in 0:(2^D - 1)]
    geometry = [SPHGeometry{D, T}(points; Density = 1000, GroupMarker = 1, Type = Floating,
        Floating = FloatingDetails{T}(RelativeWeight = T(1.2)))]
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20))
    metadata = SimulationMetaData{D, T}(SimulationName = "MemoryFloating", SaveLocation = save,
        VisualizeInParaview = false, OpenLogFile = false)
    return (; SimGeometry = geometry, SimMetaData = metadata, SimConstants = constants,
        SimKernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx),
        SimViscosity = ArtificialViscosity(), SimDensityDiffusion = ZeroDensityDiffusion())
end

@testset "GPU compact buffers" begin
    @testset "low-level defaults and exact allocation savings" begin
        for D in (2, 3), T in (Float32, Float64), double_position in (false, true)
            f = fixture(D, T; double_position)
            full = upload_particles(f.host; position_type = position_float_type(f.metadata))
            full_support = GPUSupportArrays{D, T}(length(f.host);
                position_type = position_float_type(f.metadata))
            compact = upload_particles(f.host, f.metadata)
            support = GPUSupportArrays{D, T}(length(f.host), f.metadata)
            @test eltype(full.ID) === Int
            @test eltype(full.GroupMarker) === UInt
            @test eltype(compact.ID) === Int32
            @test eltype(compact.GroupMarker) === UInt32
            wrapped = GPUParticles{D, T, position_float_type(f.metadata), typeof(full.scratch)}(
                (getproperty(full, field) for field in fieldnames(typeof(full)))...)
            @test typeof(wrapped) === typeof(full)
            @test wrapped.ID === full.ID
            for field in (:GhostPoints, :GhostNormals, :Kernel, :KernelGradient)
                @test length(getproperty(full, field)) == length(f.host)
                @test isempty(getproperty(compact, field))
            end
            @test isempty(compact.GhostIndex)
            @test isempty(compact.scratch.GhostPoints)
            @test isempty(compact.scratch.GhostNormals)
            @test isempty(compact.scratch.Acceleration)
            @test isempty(compact.scratch.Pressure)
            @test length(full.scratch.Acceleration) == length(f.host)
            @test length(full.scratch.Pressure) == length(f.host)
            @test isempty(support.∇Cᵢ) && isempty(support.∇◌rᵢ)
            @test length(full_support.∇Cᵢ) == length(f.host)
            @test length(full_support.∇◌rᵢ) == length(f.host)
            @test support.boundary_data === nothing
            @test !hasproperty(support, :InvDensity)
            @test !hasproperty(support, :InvDensityₙ⁺)
            @test uses_pos_cells(support) == (position_float_type(f.metadata) !== T)
            n = length(f.host)
            support_bytes = n * (2 * sizeof(T) + sizeof(SVector{D, T}) + sizeof(SVector{D, position_float_type(f.metadata)}))
            if uses_pos_cells(support)
                support_bytes += 2 * n * sizeof(PosCell{D, T})
            end
            @test GPUMemoryBenchmark.device_buffer_bytes(support) == support_bytes
            expected = n * (2 * sizeof(eltype(full.GhostPoints)) + 2 * sizeof(eltype(full.GhostNormals)) +
                sizeof(T) + sizeof(SVector{D, T}) + sizeof(T) + sizeof(SVector{D, T}) +
                2 * (sizeof(Int) - sizeof(Int32)) + 2 * (sizeof(UInt) - sizeof(UInt32)) +
                sizeof(T) + sizeof(SVector{D, T}))
            full_bytes = GPUMemoryBenchmark.device_buffer_bytes((full, full_support))
            compact_bytes = GPUMemoryBenchmark.device_buffer_bytes((compact, support))
            @test full_bytes - compact_bytes == expected
            @test compact_bytes < 0.75 * full_bytes
            f.metadata.GPUCompactBuffers = false
            legacy = upload_particles(f.host, f.metadata)
            legacy_support = GPUSupportArrays{D, T}(n, f.metadata)
            @test GPUMemoryBenchmark.device_buffer_bytes((legacy, legacy_support)) == full_bytes
        end
    end

    @testset "active modes retain required buffers" begin
        for D in (2, 3), T in (Float32, Float64), boundary in (SimpleMDBC, UpdatedMDBC)
            f = fixture(D, T; double_position = true, shifting = PlanarShifting,
                kernel_output = StoreKernelOutput, boundary)
            gpu = upload_particles(f.host, f.metadata)
            support = GPUSupportArrays{D, T}(length(f.host), f.metadata)
            for field in (:GhostPoints, :GhostNormals, :Kernel, :KernelGradient)
                @test length(getproperty(gpu, field)) == length(f.host)
            end
            @test length(gpu.scratch.GhostPoints) == length(f.host)
            @test length(gpu.scratch.GhostNormals) == length(f.host)
            @test length(support.∇Cᵢ) == length(f.host)
            @test length(support.∇◌rᵢ) == length(f.host)
            @test (support.boundary_data !== nothing) == (boundary === UpdatedMDBC)
        end
    end

    @testset "guarded metadata, reordering and both download paths" begin
        for wide_id in (false, true), wide_group in (false, true)
            f = fixture(2, Float32; double_position = true)
            f.host.ID[1] = Int(typemax(Int32)) + Int(wide_id)
            f.host.GroupMarker[1] = UInt(typemax(UInt32)) + UInt(wide_group)
            gpu = upload_particles(f.host, f.metadata)
            @test eltype(gpu.ID) === (wide_id ? Int : Int32)
            @test eltype(gpu.GroupMarker) === (wide_group ? UInt : UInt32)
            @test eltype(gpu.scratch.ID) === eltype(gpu.ID)
            @test eltype(gpu.scratch.GroupMarker) === eltype(gpu.GroupMarker)
            cells = CellListWorkspace{2, Float64}(length(f.host))
            for _ in 1:2
                grid = DRIVER.rebuild_cell_list!(gpu, cells, 10f0; constants = SimulationConstants{Float32}(dx = 0.02))
                back = deepcopy(f.host)
                download_particles!(back, gpu, grid)
                original_order, order = sortperm(f.host.ID), sortperm(back.ID)
                for field in (:ID, :GroupMarker, :Position, :Velocity, :Density)
                    @test getproperty(back, field)[order] == getproperty(f.host, field)[original_order]
                    @test eltype(getproperty(back, field)) === eltype(getproperty(f.host, field))
                end
                @test_throws ArgumentError download_particles!(back, gpu, grid, (:Kernel,))
                @test_throws ArgumentError DRIVER.OutputDownload(gpu, (:GhostPoints,); cells = false)
                download = DRIVER.OutputDownload(gpu, (:ID, :GroupMarker); cells = true)
                try
                    DRIVER.enqueue_download!(download, gpu)
                    DRIVER.finish_download!(download, back, grid)
                    @test back.ID == Array(gpu.ID)
                    @test back.GroupMarker == Array(gpu.GroupMarker)
                    @test eltype(back.ID) === Int
                    @test eltype(back.GroupMarker) === UInt
                finally
                    DRIVER.free!(download)
                end
            end
        end
        f = fixture(2, Float32)
        f.host.ID[1] = typemax(Int)
        f.host.GroupMarker[1] = typemax(UInt)
        gpu = upload_particles(f.host, f.metadata)
        @test Array(gpu.ID) == f.host.ID
        @test Array(gpu.GroupMarker) == f.host.GroupMarker
        @test DRIVER.metadata_storage_type(Int32, Int, Int[typemin(Int32), typemax(Int32)]) === Int32
        @test DRIVER.metadata_storage_type(Int32, Int, Int[typemin(Int32) - Int(1)]) === Int
        @test DRIVER.metadata_storage_type(Int32, Int, Int[]) === Int32
    end

    @testset "missing mode buffers fail before launching steps" begin
        f = fixture(2, Float32)
        gpu = upload_particles(f.host, f.metadata)
        gpu.Density = similar(gpu.Density, length(f.host) - 1)
        back = deepcopy(f.host)
        @test_throws ArgumentError download_particles!(back, gpu, CellGrid{2}())
        @test back == f.host
        for mode in (:kernel_output, :shifting, :boundary)
            options = mode === :kernel_output ? (; kernel_output = StoreKernelOutput) :
                      mode === :shifting ? (; shifting = PlanarShifting) : (; boundary = SimpleMDBC)
            f = fixture(2, Float32; options...)
            gpu = upload_particles(f.host; kernel_output = false, ghost_data = false)
            support = GPUSupportArrays{2, Float32}(length(f.host); shifting = false)
            cells = CellListWorkspace{2, Float32}(length(f.host))
            reduction = ReductionWorkspace{SVector{3, Float32}}(length(f.host))
            motion = MotionArrays(f.geometry, f.host)
            constants = SimulationConstants{Float32}(dx = 0.02)
            kernel = SPHKernelInstance{2, Float32}(WendlandC2(); dx = constants.dx)
            @test_throws ArgumentError SimulationLoop(ZeroDensityDiffusion(), ZeroViscosity(),
                kernel, f.metadata, constants, gpu, cells, support, reduction, motion, StepState{Float32}())
        end
    end

    @testset "full simulations and exported datasets remain exact" begin
        builders = (
            (GPUMemoryBenchmark.generated_wedge_case, Float32, true, true, true),
            (still_wedge_mdbc, Float32, false, false, false),
            (moving_square_2d, Float64, false, false, false),
            (dambreak_3d, Float32, true, true, false),
            (updated_case, Float64, false, true, true),
            ((T, save) -> floating_case(T, save; D = 2), Float32, true, true, true),
            ((T, save) -> floating_case(T, save; D = 3), Float32, true, false, false),
        )
        for (build, T, double_position, graph, async) in builders,
            scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            reference = simulate(build, T; compact = false, double_position, graph, async, scheme)
            result = simulate(build, T; compact = true, double_position, graph, async, scheme)
            @test reference.steps > 1
            @test result == reference
        end
    end
end

end
