module FrozenGridTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5

const DRIVER = SPHExampleGPU.SPHCellList
const GRID = SPHExampleGPU.GPUCellGrid
const KERNELS = SPHExampleGPU.GPUKernels
const BODIES = SPHExampleGPU.GPUFloating

function escaping_simulation(D, T, compact, graph, log_mode, scheme)
    mktempdir() do save
        V = SVector{D, T}
        vector(x, y) = V(ntuple(d -> d == 1 ? T(x) : (d == 2 ? T(y) : zero(T)), D))
        geometry = [
            SPHGeometry{D, T}(Particles = particle_struct_array([vector(0.2, -1)], T(950);
                Velocity = [vector(3, 0)]), GroupMarker = 1, Type = Fluid),
            SPHGeometry{D, T}([vector(0.2, 1)]; Density = 1000, GroupMarker = 2, Type = Moving,
                Motion = MotionDetails{D, T}(Velocity = 3, StartTime = -1, Duration = 2,
                    Direction = vector(1, 0))),
            SPHGeometry{D, T}([vector(-0.2, -0.2), vector(0.2, 0.2)];
                Density = 1000, GroupMarker = 3, Type = Fixed),
        ]
        metadata = SimulationMetaData{D, T, NoShifting, NoKernelOutput, NoMDBC, log_mode}(
            SimulationName = "FrozenGrid", SaveLocation = save, SimulationTime = T(0.12),
            OutputTimes = T(0.02), GPUFixedGrid = true, GPUUseGraph = graph,
            GPUCompactBuffers = compact, GPUDoublePosition = T === Float32,
            GPUAsyncOutput = graph, GPUCacheInteractionFields = true, ExportGridCells = true,
            VisualizeInParaview = false, OpenLogFile = false)
        constants = SimulationConstants{T}(dx = 0.02, g = 0, c₀ = 20)
        kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
        particles = AllocateDataStructures(geometry, metadata)
        ids = copy(particles.ID[1:2])
        logger = SimulationLogger(save)
        try
            redirect_stdout(devnull) do
                RunSimulation(; SimGeometry = geometry, SimMetaData = metadata,
                    SimConstants = constants, SimKernel = kernel, SimLogger = logger,
                    SimParticles = particles, SimTimeStepping = scheme,
                    SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion())
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        @test metadata.TotalTime > metadata.SimulationTime
        frozen = findall(==(Frozen), particles.Type)
        @test sort(particles.ID[frozen]) == sort(ids)
        @test all(iszero, particles.Velocity[frozen])
        @test all(iszero, particles.Acceleration[frozen])
        @test particles.Density[only(findall(==(ids[1]), particles.ID))] == T(950)
        log = read(joinpath(save, "SimulationOutput.log"), String)
        logged_ids = [parse(Int, m.captures[1]) for m in eachmatch(r"Frozen particle ID=(\d+),", log)]
        @test sort(logged_ids) == sort(ids)
        @test occursin("total frozen=2", log)
        @test occursin("locked limits=+x", log)
        @test occursin("step=", log) && occursin("time=", log) && occursin("position=", log)
        h5open(joinpath(save, "FrozenGrid.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            n = length(particles)
            frame_ids = reshape(read(root["PointData/ID"]), n, :)
            frame_types = reshape(read(root["PointData/Type"]), n, :)
            positions = reshape(read(root["Points"]), 3, n, :)
            for id in ids
                frozen_positions = [positions[:, i, frame] for frame in axes(frame_ids, 2)
                    for i in axes(frame_ids, 1) if frame_ids[i, frame] == id && frame_types[i, frame] == UInt8(Frozen)]
                @test length(frozen_positions) >= 2
                @test !isempty(frozen_positions) && all(==(first(frozen_positions)), frozen_positions)
            end
        end
    end
end

if CUDA.functional()
    @testset "escaped particles freeze, remain exported and log once" begin
        for (D, T, compact, graph, log_mode) in (
                (2, Float32, true, true, StoreLog), (2, Float64, false, false, NoLog),
                (3, Float32, true, false, NoLog), (3, Float64, false, true, StoreLog)),
            scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            escaping_simulation(D, T, compact, graph, log_mode, scheme)
        end
    end

    @testset "a small escape freezes before the final output without a natural rebuild" begin
        mktempdir() do save
            T = Float64
            constants = SimulationConstants{T}(dx = 0.02, g = 0, c₀ = 20)
            kernel = SPHKernelInstance{2, T}(WendlandC2(); dx = constants.dx)
            geometry = [SPHGeometry{2, T}([SVector(0.0, 0.0)]; Density = 1000, Type = Fluid, GroupMarker = 1),
                SPHGeometry{2, T}([SVector(0.49kernel.H, 0.0)]; Density = 1000, Type = Moving, GroupMarker = 2,
                    Motion = MotionDetails{2, T}(Velocity = 0.02kernel.H / 0.001, StartTime = -1,
                        Duration = 2, Direction = SVector(1.0, 0.0)))]
            metadata = SimulationMetaData{2, T}(SimulationName = "FinalEscape", SaveLocation = save,
                GPUFixedGrid = true, SimulationTime = 0.001, OutputTimes = 0.001,
                GPUUseGraph = true, GPUMaxStepsPerSync = 32, VisualizeInParaview = false, OpenLogFile = false)
            particles = AllocateDataStructures(geometry, metadata)
            logger = SimulationLogger(save)
            try
                redirect_stdout(devnull) do
                    RunSimulation(; SimGeometry = geometry, SimMetaData = metadata, SimParticles = particles,
                        SimConstants = constants, SimKernel = kernel, SimLogger = logger,
                        SimTimeStepping = SingleNeighborTimeStepping(),
                        SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion())
                end
                escaped = only(findall(==(2), particles.ID))
                @test particles.Type[escaped] == Frozen
                @test iszero(particles.Velocity[escaped])
                @test occursin("Frozen particle ID=2,", read(joinpath(save, "SimulationOutput.log"), String))
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
        end
    end

    @testset "frozen neighbors exert no force, diffusion or shifting" begin
        for D in (2, 3), T in (Float32, Float64), compact in (false, true)
            mktempdir() do save
                constants = SimulationConstants{T}(dx = 0.02, g = 0, c₀ = 20)
                kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
                V = SVector{D, T}
                axis = V(ntuple(d -> d == 1 ? one(T) : zero(T), D))
                geometry = [SPHGeometry{D, T}([zero(V), axis * T(0.1) * kernel.H];
                    Density = T[1000, 1100], Type = Fluid, GroupMarker = 1)]
                metadata = SimulationMetaData{D, T}(SimulationName = "InactiveNeighbors", SaveLocation = save,
                    GPUFixedGrid = true, GPUCompactBuffers = compact, GPUDoublePosition = T === Float32)
                host = AllocateDataStructures(geometry, metadata)
                Pressure!(host.Pressure, host.Density, constants)
                gpu = upload_particles(host, metadata)
                support = GPUSupportArrays{D, T}(gpu, metadata)
                cells = CellListWorkspace{D, position_float_type(metadata)}(2; fixed_grid = true)
                logger = SimulationLogger(save)
                try
                    DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; constants, support, kernel)
                    moved = Array(gpu.Position)
                    for invalid in (T(NaN), T(Inf), T(2)^40)
                        bad = copy(moved)
                        bad[2] = SVector{D, eltype(eltype(gpu.Position))}(
                            ntuple(d -> d == 1 ? invalid : zero(T), D))
                        copyto!(gpu.Position, bad)
                        @test_throws ErrorException DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹;
                            constants, support, kernel, logger)
                        @test all(==(Fluid), Array(gpu.Type))
                    end
                    moved[2] = axis * T(0.51) * kernel.H
                    copyto!(gpu.Position, moved)
                    redirect_stdout(devnull) do
                        DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; constants, support, kernel, logger)
                    end
                    @test count(==(Frozen), Array(gpu.Type)) == 1
                    step = HostStep(T(0.001), zero(T))
                    # The inactive corner bin is within the live particle's stencil.
                    KERNELS.launch_interactions!(support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
                        support.∇Cᵢ, support.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Pressure,
                        gpu.Velocity, gpu.Type, cells.CellStart, gpu.CellID, cells.grid_dev, step,
                        LinearDensityDiffusion(), ArtificialViscosity(), kernel, constants, Val(false), Val(false);
                        lanes = Val(4), pos_cells = support.PosCells)
                    @test all(iszero, Array(support.dρdtI))
                    @test all(iszero, Array(gpu.Acceleration))
                    # Freezing every entry retains a valid grid and neutral timestep reduction.
                    moved = Array(gpu.Position) .+ Ref(axis * kernel.H * 2)
                    copyto!(gpu.Position, moved)
                    redirect_stdout(devnull) do
                        DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; constants, support, kernel, logger)
                    end
                    @test all(==(Frozen), Array(gpu.Type))
                    @test cells.frozen_total == 2
                    @test cells.grid.ncells > 0
                finally
                    close(logger.LoggerIo)
                end
            end
        end
    end

    @testset "frozen ghost owners preserve density and pressure" begin
        for D in (2, 3), boundary in (SimpleMDBC, UpdatedMDBC), compact in (false, true)
            mktempdir() do save
                T = Float32
                V = SVector{D, T}
                constants = SimulationConstants{T}(dx = 0.02, g = 0, c₀ = 20)
                kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
                axis = V(ntuple(d -> d == 1 ? one(T) : zero(T), D))
                normal = axis * T(0.1) * kernel.H
                geometry = [SPHGeometry{D, T}([zero(V)]; Density = 1000, Type = Fluid, GroupMarker = 1),
                    SPHGeometry{D, T}(Particles = particle_struct_array([normal], T(950);
                        GhostPoints = [2normal], GhostNormals = [normal]), Type = Moving, GroupMarker = 2)]
                metadata = SimulationMetaData{D, T, NoShifting, NoKernelOutput, boundary}(
                    SimulationName = "FrozenGhost", SaveLocation = save, GPUFixedGrid = true,
                    GPUCompactBuffers = compact, GPUDoublePosition = true)
                host = AllocateDataStructures(geometry, metadata)
                gpu = upload_particles(host, metadata)
                support = GPUSupportArrays{D, T}(gpu, metadata)
                cells = CellListWorkspace{D, Float64}(2; fixed_grid = true)
                logger = SimulationLogger(save)
                try
                    DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; constants, support, kernel,
                        boundary_mode = boundary())
                    moved = Array(gpu.Position)
                    moved[2] = axis * T(0.51) * kernel.H
                    copyto!(gpu.Position, moved)
                    copyto!(gpu.Pressure, T[0, -37])
                    redirect_stdout(devnull) do
                        DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; constants, support, kernel,
                            boundary_mode = boundary(), logger)
                    end
                    owner = only(Array(gpu.GhostIndex))
                    @test Array(gpu.Type)[owner] == Frozen
                    before = (Array(gpu.Density), Array(gpu.Pressure))
                    step = HostStep(T(0.001), zero(T))
                    if boundary === SimpleMDBC
                        KERNELS.launch_mdbc!(gpu.Density, gpu.Pressure, gpu.Position, gpu.GhostPoints,
                            gpu.GhostIndex, gpu.Type, cells.CellStart, cells.grid_dev, step, kernel, constants;
                            lanes = Val(4), pos_cells = support.PosCells,
                            compact_ghosts = Val(gpu.BoundarySlot !== nothing))
                    else
                        KERNELS.launch_updated_mdbc!(gpu.Density, gpu.Pressure, gpu.Position, gpu.Velocity,
                            gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cells.CellStart,
                            cells.grid_dev, step, kernel, constants, support.boundary_data;
                            lanes = Val(4), pos_cells = support.PosCells,
                            compact_ghosts = Val(gpu.BoundarySlot !== nothing))
                    end
                    @test Array(gpu.Density) == before[1]
                    @test Array(gpu.Pressure) == before[2]
                    @test before[1][owner] == T(950)
                    @test before[2][owner] == T(-37)
                finally
                    close(logger.LoggerIo)
                end
            end
        end
    end

    @testset "a floating-body limit freezes the entire rigid body" begin
        for D in (2, 3), compact in (false, true)
            mktempdir() do save
                T = Float64
                V = SVector{D, T}
                body_points = [zero(V), (V(ntuple(d -> d == axis ? 0.04 : 0.0, D)) for axis in 1:D)...]
                geometry = [SPHGeometry{D, T}(body_points; Density = 1000, GroupMarker = 1, Type = Floating,
                        Floating = FloatingDetails{T}(RelativeWeight = 1.2)),
                    SPHGeometry{D, T}([V(ntuple(_ -> sign, D)) for sign in (-1, 1)];
                        Density = 1000, GroupMarker = 1, Type = Fixed)]
                metadata = SimulationMetaData{D, T}(SimulationName = "FrozenBody", SaveLocation = save,
                    GPUFixedGrid = true, GPUCompactBuffers = compact)
                constants = SimulationConstants{T}(dx = 0.02, g = 9.81, c₀ = 20)
                kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
                host = AllocateDataStructures(geometry, metadata)
                gpu = upload_particles(host, metadata)
                support = GPUSupportArrays{D, T}(gpu, metadata)
                floating = FloatingArrays(geometry, host, constants)
                cells = CellListWorkspace{D, T}(length(host); fixed_grid = true)
                logger = SimulationLogger(save)
                try
                    DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; constants, support, kernel, floating)
                    positions = Array(gpu.Position)
                    types = Array(gpu.Type)
                    max_cell = cells.grid.origin[1] + cells.grid.dims[1] - 2
                    shift = V(ntuple(d -> d == 1 ? (max_cell + 0.25) * kernel.H : 0.0, D))
                    positions[types .== Floating] .+= Ref(shift)
                    copyto!(gpu.Position, positions)
                    copyto!(floating.center, Array(floating.center) .+ Ref(shift))
                    fill!(floating.velocity, V(ntuple(_ -> 1.0, D)))
                    fill!(floating.force, one(T))
                    redirect_stdout(devnull) do
                        DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; constants, support, kernel, floating, logger)
                    end
                    @test count(==(Frozen), Array(gpu.Type)) == length(body_points)
                    @test Array(floating.frozen) == Int32[1]
                    @test all(iszero, Array(floating.velocity))
                    @test all(iszero, Array(floating.force))
                    @test length(unique(Array(floating.indices))) == length(body_points)
                    before = Array(gpu.Position)
                    center = Array(floating.center)
                    step = HostStep(0.01, 1.0)
                    for _ in 1:3
                        BODIES.launch_floating_derivative!(floating, step, constants.g)
                        BODIES.launch_floating_update!(floating, step, constants.g, true)
                        BODIES.launch_floating_particles!(floating, gpu.Position, gpu.Velocity, gpu.Position,
                            gpu.Type, gpu.GroupMarker, step, true)
                    end
                    @test Array(gpu.Position) == before
                    @test Array(floating.center) == center
                    @test all(iszero, Array(floating.acceleration))
                    @test occursin("floating body crossed a locked limit", read(joinpath(save, "SimulationOutput.log"), String))
                finally
                    close(logger.LoggerIo)
                end
            end
        end
    end
end

end
