module StationaryPredictorTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

include(joinpath(@__DIR__, "..", "benchmark", "memory_buffers.jl"))

const DRIVER = SPHExampleGPU.SPHCellList
const FLOATING = SPHExampleGPU.GPUFloating
const STEP = SPHExampleGPU.GPUStepState

function fixture(D, T; double_position = false, types = (Fixed, Fluid, Moving, Floating),
                 move_particles = true)
    point(x) = SVector{D, T}(ntuple(d -> d == 1 ? T(x) : T(0.02), D))
    velocity = SVector{D, T}(ntuple(d -> d == 1 ? T(-0.1) : zero(T), D))
    body = [point(0.06) + SVector{D, T}(ntuple(d -> T((i >> (d - 1)) & 1) * T(0.02), D))
            for i in 0:(2^D - 1)]
    geometry = SPHGeometry{D, T}[]
    for (group, type) in enumerate(types)
        positions = type == Floating ? body :
                    type == Fixed ? [point(0.12), point(0)] :
                    type == Moving ? [point(0.10)] : [point(0.02), point(0.04)]
        motion = type == Moving ? MotionDetails{D, T}(Velocity = T(0.1), StartTime = -one(T),
            Duration = one(T), Direction = SVector{D, T}(ntuple(d -> d == 1 ? one(T) : zero(T), D)),
            MoveParticles = move_particles) : nothing
        floating = type == Floating ? FloatingDetails{T}(RelativeWeight = T(1.2)) : nothing
        push!(geometry, SPHGeometry{D, T}(Particles = particle_struct_array(positions, T(1020);
            Velocity = fill(velocity, length(positions))), GroupMarker = group, Type = type,
            Motion = motion, Floating = floating))
    end
    metadata = SimulationMetaData{D, T, PlanarShifting, StoreKernelOutput}(
        SimulationName = "StationaryPredictors", SaveLocation = tempdir(),
        GPUDoublePosition = double_position)
    host = AllocateDataStructures(geometry, metadata)
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T), m₀ = T(1000 * 0.02^D))
    host.Pressure .= EquationOfStateGamma7.(host.Density, constants.c₀, constants.ρ₀)
    kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
    return (; host, metadata, geometry, constants, kernel)
end

function context(f, D, T; compact, forces, single, lanes)
    f.metadata.GPUCompactBuffers = compact
    gpu = upload_particles(f.host, f.metadata)
    cl = CellListWorkspace{D, position_float_type(f.metadata)}(length(f.host))
    sup = GPUSupportArrays{D, T}(gpu, f.metadata)
    red = ReductionWorkspace{SVector{3, T}}(length(f.host))
    motion = MotionArrays(f.geometry, f.host)
    floating = FLOATING.FloatingArrays(f.geometry, f.host, f.constants;
        position_type = position_float_type(f.metadata))
    DRIVER.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹; support = sup, floating, kernel = f.kernel,
        constants = f.constants, preserve_pressure = true)
    state = StepState{T}()
    STEP.set_output_time!(state, one(T))
    DRIVER.launch_step_reduction!(red, gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Position, f.kernel)
    ctx = (; gpu, cl, sup, red, motion, floating, state, SimKernel = f.kernel, SimConstants = f.constants,
        SimDensityDiffusion = LinearDensityDiffusion(), SimViscosity = LaminarSPS(),
        FlagKernel = Val(true), FlagShift = Val(true), UseMDBC = false, SingleNeighbor = single,
        threads = 128, lanes = Val(lanes), bforces = Val(forces), HourGlass = f.metadata.HourGlass,
        boundary_mode = NoMDBC(), radial = Val(forces), scalar_sps = Val(forces))
    if single
        DRIVER.enqueue_state_derivative!(ctx, HostStep(T(1e-4), zero(T)), false, "mdbc", "initial")
    end
    return ctx
end

function snapshot(ctx)
    gpu = ctx.gpu
    order = sortperm(Array(gpu.ID))
    fields = (:Position, :Velocity, :Density, :Acceleration, :Pressure, :Kernel, :KernelGradient,
              :ID, :Type, :GroupMarker)
    state = NamedTuple{fields}(Tuple(Array(getproperty(gpu, field))[order] for field in fields))
    return (; state, density_rate = Array(ctx.sup.dρdtI)[order], density_half = Array(ctx.sup.ρₙ⁺)[order],
        reduction = Array(ctx.red.partial), time = Array(ctx.state.f), flags = Array(ctx.state.i),
        body = FLOATING.floating_state(ctx.floating))
end

@testset "stationary boundary predictors" begin
    @testset "exact predictor savings by sharing reorder scratch" begin
        for D in (2, 3), T in (Float32, Float64), double_position in (false, true),
            types in ((Fixed,), (Fluid,), (Moving,), (Fixed, Fluid, Moving, Floating), ())
            f = fixture(D, T; double_position, types)
            n = length(f.host)
            cells = CellListWorkspace{D, position_float_type(f.metadata)}(n)
            gpu = upload_particles(f.host, f.metadata)
            compact = GPUSupportArrays{D, T}(gpu, f.metadata)
            full = GPUSupportArrays{D, T}(n, f.metadata)
            fixed = count(==(Fixed), f.host.Type)
            @test isempty(compact.Positionₙ⁺)
            @test isempty(compact.Velocityₙ⁺)
            @test length(compact.ρₙ⁺) == n
            @test length(compact.dρdtI) == n
            @test compact.ReusePredictors
            half_position, half_velocity = DRIVER.predictor_arrays(compact, gpu)
            @test half_position === gpu.scratch.Position
            @test half_velocity === gpu.scratch.Velocity
            bytes = GPUMemoryBenchmark.device_buffer_bytes
            saving = n * (sizeof(eltype(full.Positionₙ⁺)) + sizeof(eltype(full.Velocityₙ⁺)))
            @test bytes((gpu, full, cells)) - bytes((gpu, compact, cells)) == saving
            if D == 3 && T == Float32 && double_position
                @test saving == 36 * n
                @test saving >= 36 * fixed
            end
            f.metadata.GPUCompactBuffers = false
            legacy = GPUSupportArrays{D, T}(gpu, f.metadata)
            @test !legacy.ReusePredictors
            @test length(legacy.Positionₙ⁺) == n
            @test length(legacy.Velocityₙ⁺) == n
        end
        f = fixture(2, Float32; types = (Fixed, Fluid))
        gpu = upload_particles(f.host, f.metadata)
        f.metadata.GPUDoublePosition = true
        @test_throws ArgumentError GPUSupportArrays{2, Float32}(gpu, f.metadata)
    end

    @testset "shared predictors follow repeated cell reorders" begin
        points = [SVector{3, Float64}(0.02i, 0.02j, 0.02k) for i in 0:15 for j in 0:15 for k in 0:15]
        moving = [i % 3 == 0 for i in eachindex(points)]
        geometry = [SPHGeometry{3, Float32}(points[moving]; Density = 1000, GroupMarker = 1, Type = Fluid),
                    SPHGeometry{3, Float32}(points[.!moving]; Density = 1000, GroupMarker = 2, Type = Fixed)]
        metadata = SimulationMetaData{3, Float32}(SimulationName = "PredictorMap", SaveLocation = tempdir(),
            GPUDoublePosition = true)
        particles = AllocateDataStructures(geometry, metadata)
        gpu = upload_particles(particles, metadata)
        cells = CellListWorkspace{3, Float64}(length(particles))
        support = GPUSupportArrays{3, Float32}(gpu, metadata)
        for _ in 1:3
            DRIVER.rebuild_cell_list!(gpu, cells, 10f0; support, constants = SimulationConstants{Float32}(dx = 0.02),
                kernel = (; H = 0.1f0))
            types = Array(gpu.Type)
            position_half, velocity_half = DRIVER.predictor_arrays(support, gpu)
            @test position_half === gpu.scratch.Position
            @test velocity_half === gpu.scratch.Velocity
            @test Array(position_half)[types .== Fixed] == Array(gpu.Position)[types .== Fixed]
            @test Array(velocity_half)[types .== Fixed] == Array(gpu.Velocity)[types .== Fixed]
            @test Array(gpu.Position) == particles.Position[Array(gpu.ID)]
        end
    end

    @testset "half step skips fixed predictor writes but advances DBC density" begin
        f = fixture(3, Float32; double_position = true, types = (Fixed, Fluid))
        ctx = context(f, 3, Float32; compact = true, forces = true, single = false, lanes = 1)
        (; gpu, sup, cl, motion) = ctx
        position_half, velocity_half = DRIVER.predictor_arrays(sup, gpu)
        position_before = Array(gpu.Position)
        velocity_before = Array(gpu.Velocity)
        fixed = findall(==(Fixed), Array(gpu.Type))
        fluid = findall(==(Fluid), Array(gpu.Type))
        sentinel_position = SVector{3, Float64}(-99, -99, -99)
        sentinel_velocity = SVector{3, Float32}(-88, -88, -88)
        fill!(position_half, sentinel_position)
        fill!(velocity_half, sentinel_velocity)
        fill!(sup.dρdtI, 100f0)
        DRIVER.launch_half_step!(position_half, velocity_half, sup.ρₙ⁺, gpu.Pressure,
            gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, sup.dρdtI,
            gpu.Type, gpu.GroupMarker, motion, HostStep(1f-4, 0f0), f.constants;
            pos_cells = sup.PosCellsₙ⁺, CellID = gpu.CellID, grid = cl.grid_dev, SimKernel = f.kernel,
            boundary_mode = NoMDBC(), reuse_predictors = true)
        @test all(==(sentinel_position), Array(position_half)[fixed])
        @test all(==(sentinel_velocity), Array(velocity_half)[fixed])
        @test all(!=(sentinel_position), Array(position_half)[fluid])
        @test all(!=(sentinel_velocity), Array(velocity_half)[fluid])
        @test all(Array(sup.ρₙ⁺)[fixed] .> Array(gpu.Density)[fixed])
        @test Array(gpu.Position) == position_before
        @test Array(gpu.Velocity) == velocity_before
    end

    @testset "DBC density, forces, motion, floating, sorting and graphs" begin
        for D in (2, 3), T in (Float32, Float64), double_position in (false, true),
            forces in (false, true), single in (false, true)
            f = fixture(D, T; double_position, move_particles = !single)
            full = context(f, D, T; compact = false, forces, single, lanes = single ? 4 : 1)
            compact = context(f, D, T; compact = true, forces, single, lanes = single ? 4 : 1)
            original_position = copy(f.host.Position)
            original_velocity = copy(f.host.Velocity)
            for pass in 1:2
                for _ in 1:3
                    if single
                        DRIVER.launch_step_graph!(full)
                        DRIVER.launch_step_graph!(compact)
                    else
                        DRIVER.enqueue_step!(full, false)
                        DRIVER.enqueue_step!(compact, false)
                    end
                end
                result = snapshot(compact)
                reference = snapshot(full)
                for field in (:state, :density_rate, :density_half, :time, :flags, :body)
                    @test result[field] == reference[field]
                end
                @test result.reduction == reference.reduction
                fixed = findall(==(Fixed), result.state.Type)
                @test result.state.Position[fixed] == original_position[fixed]
                @test result.state.Velocity[fixed] == original_velocity[fixed]
                @test any(!=(T(1020)), result.state.Density[fixed])
                @test any(!iszero, result.density_rate[fixed])
                @test any(!iszero, result.state.Acceleration[fixed]) == forces
                for ctx in (full, compact)
                    DRIVER.rebuild_cell_list!(ctx.gpu, ctx.cl, f.kernel.H⁻¹;
                        support = ctx.sup, floating = ctx.floating, constants = f.constants, kernel = f.kernel,
                        discard_acceleration = true)
                    if single
                        DRIVER.enqueue_state_derivative!(ctx, ctx.state, false, "mdbc", "rebuild")
                    end
                end
                types = Array(compact.gpu.Type)
                position_half, velocity_half = DRIVER.predictor_arrays(compact.sup, compact.gpu)
                @test Array(position_half)[types .== Fixed] == Array(compact.gpu.Position)[types .== Fixed]
                @test Array(velocity_half)[types .== Fixed] == Array(compact.gpu.Velocity)[types .== Fixed]
                @test sort(Array(compact.gpu.ID)) == f.host.ID
            end
            if single
                @test !isempty(compact.state.graphs)
            end
        end

        @testset "entirely stationary input needs no predictor slots" begin
            for D in (2, 3), T in (Float32, Float64), double_position in (false, true)
                f = fixture(D, T; double_position, types = (Fixed,))
                full = context(f, D, T; compact = false, forces = true, single = false, lanes = 1)
                compact = context(f, D, T; compact = true, forces = true, single = false, lanes = 1)
                DRIVER.enqueue_step!(full, false)
                DRIVER.enqueue_step!(compact, false)
                @test snapshot(compact) == snapshot(full)
                @test isempty(compact.sup.Positionₙ⁺)
                @test isempty(compact.sup.Velocityₙ⁺)
            end
        end
    end
end

end
