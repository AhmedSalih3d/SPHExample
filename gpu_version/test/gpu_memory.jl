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

function simulate(build, T; compact, double_position, graph, async, scheme, queue_bytes = 1024^2)
    return mktempdir() do save
        kwargs = build(T, save)
        metadata = kwargs.SimMetaData
        metadata.GPUCompactBuffers = compact
        metadata.GPUDoublePosition = double_position
        metadata.GPUUseGraph = graph
        metadata.GPUAsyncOutput = async
        metadata.GPUOutputQueueBytes = queue_bytes
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
        active_fields = filter(name -> !(name in DRIVER.OPTIONAL_PARTICLE_FIELDS && !hasproperty(host, name)),
            DRIVER.DOWNLOADABLE_FIELDS)
        fields = Dict(name => (eltype(getproperty(host, name)), getproperty(host, name)[order])
            for name in active_fields)
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
            @test eltype(compact.GroupMarker) === UInt8
            @test eltype(f.host.ID) === Int32
            @test eltype(f.host.GroupMarker) === UInt8
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
            @test !hasproperty(compact, :Pressure) && !hasproperty(full, :Pressure)
            @test !hasproperty(compact.scratch, :Pressure) && !hasproperty(full.scratch, :Pressure)
            @test !hasproperty(f.host, :Pressure)
            @test length(full.scratch.Acceleration) == length(f.host)
            @test isempty(support.∇Cᵢ) && isempty(support.∇◌rᵢ)
            @test length(full_support.∇Cᵢ) == length(f.host)
            @test length(full_support.∇◌rᵢ) == length(f.host)
            @test support.boundary_data === nothing
            @test support.InvDensity === nothing && support.Pressure === nothing
            @test full_support.InvDensity === nothing && full_support.Pressure === nothing
            @test !hasproperty(support, :InvDensityₙ⁺)
            @test uses_pos_cells(support) == (position_float_type(f.metadata) !== T)
            n = length(f.host)
            support_bytes = n * (2 * sizeof(T) + sizeof(SVector{D, T}) + sizeof(SVector{D, position_float_type(f.metadata)}))
            if uses_pos_cells(support)
                @test support.PosCells === support.PosCellsₙ⁺
                @test full_support.PosCells !== full_support.PosCellsₙ⁺
                support_bytes += n * sizeof(PosCell{D, T})
            end
            @test GPUMemoryBenchmark.device_buffer_bytes(support) == support_bytes
            expected = n * (2 * sizeof(eltype(full.GhostPoints)) + 2 * sizeof(eltype(full.GhostNormals)) +
                sizeof(T) + sizeof(SVector{D, T}) + sizeof(T) + sizeof(SVector{D, T}) +
                2 * (sizeof(Int) - sizeof(Int32)) + 2 * (sizeof(UInt) - sizeof(UInt8)) +
                sizeof(SVector{D, T}) +
                (uses_pos_cells(support) ? sizeof(PosCell{D, T}) : 0))
            full_bytes = GPUMemoryBenchmark.device_buffer_bytes((full, full_support))
            compact_bytes = GPUMemoryBenchmark.device_buffer_bytes((compact, support))
            @test full_bytes - compact_bytes == expected
            @test compact_bytes < 0.75 * full_bytes
            f.metadata.GPUCompactBuffers = false
            full_host = AllocateDataStructures(f.geometry, f.metadata)
            @test eltype(full_host.ID) === Int
            @test eltype(full_host.GroupMarker) === UInt
            legacy = upload_particles(full_host, f.metadata)
            legacy_support = GPUSupportArrays{D, T}(n, f.metadata)
            @test GPUMemoryBenchmark.device_buffer_bytes((legacy, legacy_support)) == full_bytes
            TP = position_float_type(f.metadata)
            @test !hasproperty(f.host, :Cells)
            @test !hasproperty(f.host, :Kernel) && !hasproperty(f.host, :KernelGradient)
            @test !hasproperty(f.host, :GhostPoints) && !hasproperty(f.host, :GhostNormals)
            for field in (:Cells, :Kernel, :KernelGradient, :GhostPoints, :GhostNormals)
                @test hasproperty(full_host, field)
            end
            expected_host_saving = n * (sizeof(CartesianIndex{D}) + sizeof(T) + sizeof(SVector{D, T}) +
                sizeof(SVector{D, TP}) + sizeof(SVector{D, T}) +
                (sizeof(Int) - sizeof(Int32)) + (sizeof(UInt) - sizeof(UInt8)))
            @test GPUMemoryBenchmark.host_particle_bytes(full_host) -
                  GPUMemoryBenchmark.host_particle_bytes(f.host) == expected_host_saving
        end
    end

    @testset "active modes retain required buffers" begin
        for D in (2, 3), T in (Float32, Float64), boundary in (SimpleMDBC, UpdatedMDBC)
            f = fixture(D, T; double_position = true, shifting = PlanarShifting,
                kernel_output = StoreKernelOutput, boundary)
            @test !hasproperty(f.host, :Cells)
            for field in (:Kernel, :KernelGradient, :GhostPoints, :GhostNormals)
                @test hasproperty(f.host, field)
            end
            f.metadata.ExportGridCells = true
            with_cells = AllocateDataStructures(f.geometry, f.metadata)
            @test hasproperty(with_cells, :Cells)
            gpu = upload_particles(f.host, f.metadata)
            support = GPUSupportArrays{D, T}(gpu, f.metadata)
            for field in (:Kernel, :KernelGradient)
                @test length(getproperty(gpu, field)) == length(f.host)
            end
            @test isempty(gpu.GhostPoints) && isempty(gpu.GhostNormals)
            @test isempty(gpu.scratch.GhostPoints) && isempty(gpu.scratch.GhostNormals)
            @test length(gpu.BoundarySlot) == length(f.host)
            @test all(iszero, Array(gpu.BoundarySlot))
            @test length(support.∇Cᵢ) == length(f.host)
            @test length(support.∇◌rᵢ) == length(f.host)
            @test (support.boundary_data !== nothing) == (boundary === UpdatedMDBC)
            if boundary === UpdatedMDBC
                @test isempty(support.boundary_data.active)
                @test isempty(support.boundary_data.velocity)
                @test isempty(support.boundary_data.tangent_velocity)
                @test support.boundary_data.slots === gpu.BoundarySlot
            end
        end
    end

    @testset "boundary slots, exact savings and frame-local ghost downloads" begin
        for D in (2, 3), T in (Float32, Float64), TP in unique((T, Float64)),
            updated in (false, true)
            V = SVector{D, TP}
            normal = SVector{D, T}(ntuple(d -> d == D ? T(0.1) : zero(T), D))
            points = [V(ntuple(d -> d == 1 ? TP(k) / 10 : zero(TP), D)) for k in 10:-1:1]
            geometry = [SPHGeometry{D, T}(points; Density = 1000, GroupMarker = 1, Type = Fluid)]
            metadata = SimulationMetaData{D, T, NoShifting, NoKernelOutput, SimpleMDBC}(
                SimulationName = "BoundarySlots", SaveLocation = tempdir(), GPUDoublePosition = TP !== T)
            host = AllocateDataStructures(geometry, metadata)
            host.Type[[2, 7, 9]] .= Fixed
            host.GhostPoints[[2, 7, 9]] = points[[2, 7, 9]] .+ Ref(V(normal))
            host.GhostNormals[[2, 7, 9]] .= Ref(normal)
            full = upload_particles(host; position_type = TP)
            gpu = upload_particles(host; position_type = TP, boundary_only = true)
            dense_support = GPUSupportArrays{D, T}(length(host); position_type = TP, updated_mdbc = updated)
            support = GPUSupportArrays{D, T}(length(host); position_type = TP, updated_mdbc = updated,
                boundary_slots = gpu.BoundarySlot, boundary_count = length(gpu.GhostIndex))
            n, b = length(host), length(gpu.GhostIndex)
            ghost_bytes = sizeof(V) + sizeof(normal)
            state_bytes = updated ? sizeof(Bool) + 2 * sizeof(normal) : 0
            expected = (2n - b) * ghost_bytes + (n - b) * state_bytes - 2n * sizeof(Int32)
            @test GPUMemoryBenchmark.device_buffer_bytes((full, dense_support)) -
                GPUMemoryBenchmark.device_buffer_bytes((gpu, support)) == expected
            @test length(gpu.GhostPoints) == b == 3
            @test isempty(gpu.scratch.GhostPoints) && isempty(gpu.scratch.GhostNormals)
            map_pointer = UInt(pointer(gpu.BoundarySlot))
            ghost_pointer = UInt(pointer(gpu.GhostPoints))
            owner_pointer = UInt(pointer(gpu.GhostIndex))
            cells = CellListWorkspace{D, TP}(n)
            for _ in 1:3
                grid = DRIVER.rebuild_cell_list!(gpu, cells, T(2))
                owners, slots = Array(gpu.GhostIndex), Array(gpu.BoundarySlot)
                @test slots[owners] == Int32.(1:b)
                @test all(iszero, slots[Array(gpu.Type) .== Fluid])
                @test UInt(pointer(gpu.BoundarySlot)) == map_pointer
                @test UInt(pointer(gpu.GhostPoints)) == ghost_pointer
                @test UInt(pointer(gpu.GhostIndex)) == owner_pointer
                back = deepcopy(host)
                download_particles!(back, gpu, grid)
                @test back.GhostPoints[sortperm(back.ID)] == host.GhostPoints[sortperm(host.ID)]
                @test back.GhostNormals[sortperm(back.ID)] == host.GhostNormals[sortperm(host.ID)]
                download = DRIVER.OutputDownload(gpu, (:ID, :GhostPoints, :GhostNormals); cells = false)
                try
                    DRIVER.enqueue_download!(download, gpu)
                    # Mutate/rebuild after capture: the writer must use this frame's owners.
                    copyto!(gpu.Position, reverse(Array(gpu.Position)))
                    DRIVER.rebuild_cell_list!(gpu, cells, T(2))
                    async_back = deepcopy(host)
                    DRIVER.finish_download!(download, async_back, grid)
                    @test async_back.ID == back.ID
                    @test async_back.GhostPoints == back.GhostPoints
                    @test async_back.GhostNormals == back.GhostNormals
                finally
                    DRIVER.free!(download)
                end
            end
        end
    end

    @testset "guarded metadata, reordering and both download paths" begin
        for wide_id in (false, true), group_type in (UInt8, UInt16, UInt32, UInt)
            f = fixture(2, Float32; double_position = true)
            f.metadata.GPUCompactBuffers = false
            host = AllocateDataStructures(f.geometry, f.metadata)
            f.metadata.GPUCompactBuffers = true
            host.ID[1] = Int(typemax(Int32)) + Int(wide_id)
            host.GroupMarker[1] = group_type === UInt8  ? UInt(typemax(UInt8)) :
                                  group_type === UInt16 ? UInt(typemax(UInt8)) + UInt(1) :
                                  group_type === UInt32 ? UInt(typemax(UInt16)) + UInt(1) :
                                                          UInt(typemax(UInt32)) + UInt(1)
            gpu = upload_particles(host, f.metadata)
            @test eltype(gpu.ID) === (wide_id ? Int : Int32)
            @test eltype(gpu.GroupMarker) === group_type
            @test eltype(gpu.scratch.ID) === eltype(gpu.ID)
            @test eltype(gpu.scratch.GroupMarker) === eltype(gpu.GroupMarker)
            cells = CellListWorkspace{2, Float64}(length(host))
            for _ in 1:2
                grid = DRIVER.rebuild_cell_list!(gpu, cells, 10f0; constants = SimulationConstants{Float32}(dx = 0.02))
                back = deepcopy(host)
                download_particles!(back, gpu, grid)
                original_order, order = sortperm(host.ID), sortperm(back.ID)
                for field in (:ID, :GroupMarker, :Position, :Velocity, :Density)
                    @test getproperty(back, field)[order] == getproperty(host, field)[original_order]
                    @test eltype(getproperty(back, field)) === eltype(getproperty(host, field))
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
        f.metadata.GPUCompactBuffers = false
        host = AllocateDataStructures(f.geometry, f.metadata)
        f.metadata.GPUCompactBuffers = true
        host.ID[1] = typemax(Int)
        host.GroupMarker[1] = typemax(UInt)
        gpu = upload_particles(host, f.metadata)
        @test Array(gpu.ID) == host.ID
        @test Array(gpu.GroupMarker) == host.GroupMarker
        @test DRIVER.metadata_storage_type(Int32, Int, Int[typemin(Int32), typemax(Int32)]) === Int32
        @test DRIVER.metadata_storage_type(Int32, Int, Int[typemin(Int32) - Int(1)]) === Int
        @test DRIVER.metadata_storage_type(Int32, Int, Int[]) === Int32
        @test DRIVER.group_marker_storage_type(UInt[1, 255]) === UInt8
        @test DRIVER.group_marker_storage_type(UInt[1, 256]) === UInt16
        @test DRIVER.group_marker_storage_type(UInt[1, 65536]) === UInt32
        @test DRIVER.group_marker_storage_type(UInt[UInt(typemax(UInt32)) + UInt(1)]) === UInt
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

    @testset "compact ghost exports match full buffers in both output modes" begin
        function ghost_output_case(T, save)
            kwargs = updated_case(T, save)
            kwargs.SimMetaData.OutputVariables = ["ID", "Type", "GhostPoints", "GhostNormals"]
            return kwargs
        end
        for double_position in (false, true),
            scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            reference = simulate(ghost_output_case, Float32; compact = false,
                double_position, graph = true, async = false, scheme)
            for async in (false, true)
                result = simulate(ghost_output_case, Float32; compact = true,
                    double_position, graph = true, async, scheme, queue_bytes = 0)
                @test result.steps == reference.steps
                @test result.time == reference.time
                @test result.position == reference.position
                @test result.cells == reference.cells
                for name in keys(result.fields)
                    if name in (:ID, :GroupMarker)
                        @test result.fields[name][2] == reference.fields[name][2]
                    else
                        @test result.fields[name] == reference.fields[name]
                    end
                end
                for key in keys(reference.particles)
                    if key in ("/VTKHDF/PointData/ID", "/VTKHDF/PointData/GroupMarker")
                        @test result.particles[key][2] == reference.particles[key][2]
                        @test result.particles[key][3] == reference.particles[key][3]
                    else
                        @test result.particles[key] == reference.particles[key]
                    end
                end
                types = result.particles["/VTKHDF/PointData/Type"][3]
                @test length(types) > length(result.position[2])
                fluid = vec(types .== Int8(Fluid))
                wall = vec(types .== Int8(Moving))
                for name in ("GhostPoints", "GhostNormals")
                    ghosts = result.particles["/VTKHDF/PointData/$name"][3]
                    @test all(iszero, ghosts[:, fluid])
                    @test any(!iszero, ghosts[:, wall])
                end
            end
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
            @test result.steps == reference.steps
            @test result.time == reference.time
            @test result.position == reference.position
            @test result.cells == reference.cells
            for key in keys(reference.particles)
                if key in ("/VTKHDF/PointData/ID", "/VTKHDF/PointData/GroupMarker")
                    @test result.particles[key][2] == reference.particles[key][2]
                    @test result.particles[key][3] == reference.particles[key][3]
                else
                    @test result.particles[key] == reference.particles[key]
                end
            end
            @test issubset(keys(result.fields), keys(reference.fields))
            for name in keys(result.fields)
                if name in (:ID, :GroupMarker)
                    @test result.fields[name][2] == reference.fields[name][2]
                else
                    @test result.fields[name] == reference.fields[name]
                end
            end
            for name in setdiff(keys(reference.fields), keys(result.fields))
                @test all(iszero, reference.fields[name][2])
            end
        end
    end

    @testset "on-demand pressure output and probes without stored Pressure" begin
        for single_file in (true, false), async in (false, true)
            mktempdir() do save
                kwargs = updated_case(Float32, save)
                meta = kwargs.SimMetaData
                meta.ExportSingleVTKHDF = single_file
                meta.GPUAsyncOutput = async
                meta.SimulationTime = 0.004f0
                meta.OutputTimes = 0.002f0
                meta.OutputVariables = ["Pressure"]
                measurements = single_file ? MeasurementConfig(
                    pressure_probes = [MeasurementProbe("probe", (0.5, 0.1))],
                ) : nothing
                host = AllocateDataStructures(kwargs.SimGeometry, meta)
                @test !hasproperty(host, :Pressure)
                logger = SimulationLogger(save)
                try
                    redirect_stdout(devnull) do
                        RunSimulation(; kwargs..., SimTimeStepping = SymplecticTimeStepping(),
                            SimParticles = host, SimLogger = logger, SimMeasurements = measurements)
                    end
                finally
                    close(logger.LoggerIo)
                end
                @test !hasproperty(host, :Pressure)
                n = length(host)
                wall = findfirst(==(Moving), host.Type)
                expected_last = [particle_pressure(host.Density[i], host.Type[i],
                    kwargs.SimConstants.c₀, kwargs.SimConstants.ρ₀, UpdatedMDBC()) for i in eachindex(host)]
                if single_file
                    data = hdf_datasets(joinpath(save, meta.SimulationName * ".vtkhdf"))
                    pressures = data["/VTKHDF/Particles/PointData/Pressure"][3]
                    @test length(pressures) >= 2 * n
                    @test pressures[(end - n + 1):end] ≈ expected_last rtol = 1f-5
                    @test pressures[end - n + wall] < 0
                    probe_series = vec(data["/VTKHDF/PressureProbes/PointData/Pressure"][3])
                    @test length(probe_series) >= 2 && all(isfinite, probe_series)
                else
                    files = sort(filter(f -> endswith(f, ".vtkhdf"), readdir(save; join = true)))
                    @test length(files) >= 2
                    last_data = hdf_datasets(last(files))
                    pressures = last_data["/VTKHDF/PointData/Pressure"][3]
                    @test collect(keys(last_data)) ⊇ ["/VTKHDF/PointData/Pressure"]
                    @test !haskey(last_data, "/VTKHDF/PointData/Density")
                    @test pressures ≈ expected_last rtol = 1f-5
                    @test pressures[wall] < 0
                end
            end
        end
    end
end

end
