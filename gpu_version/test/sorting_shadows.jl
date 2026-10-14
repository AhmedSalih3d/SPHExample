module SortingShadowTests

using Test, SPHExampleGPU, CUDA, StaticArrays

const DRIVER = SPHExampleGPU.SPHCellList
const KERNELS = SPHExampleGPU.GPUKernels
const STEP = SPHExampleGPU.GPUStepState
const BODIES = SPHExampleGPU.GPUFloating

function fixture(D, T; double_position = false)
    point(x) = SVector{D, T}(ntuple(d -> d == 1 ? T(x) : T(0.02), D))
    fluid = vec([SVector{D, T}(ntuple(d -> T(Tuple(index)[d]) * T(0.02), D))
                 for index in CartesianIndices(ntuple(_ -> 1:3, D))])
    body = [point(0.10) + SVector{D, T}(ntuple(d -> T((i >> (d - 1)) & 1) * T(0.02), D))
            for i in 0:(2^D - 1)]
    geometry = [
        SPHGeometry{D, T}(reverse(fluid); Density = 1000, GroupMarker = 1, Type = Fluid),
        SPHGeometry{D, T}([point(0.14), point(0)]; Density = 1000, GroupMarker = 2, Type = Fixed),
        SPHGeometry{D, T}([point(0.08)]; Density = 1000, GroupMarker = 3, Type = Moving,
            Motion = MotionDetails{D, T}(Velocity = T(0.1), StartTime = -one(T), Duration = one(T),
                Direction = SVector{D, T}(ntuple(d -> d == 1 ? one(T) : zero(T), D)))),
        SPHGeometry{D, T}(body; Density = 1000, GroupMarker = 4, Type = Floating,
            Floating = FloatingDetails{T}(RelativeWeight = T(1.2))),
    ]
    metadata = SimulationMetaData{D, T}(SimulationName = "SortingShadows", SaveLocation = tempdir(),
        GPUDoublePosition = double_position, SimulationTime = one(T),
        OutputTimes = T[0, 0.001, 0.002], GPUMaxStepsPerSync = 4)
    host = AllocateDataStructures(geometry, metadata)
    for i in eachindex(host)
        host.Density[i] = T(998 + i % 7)
        host.Acceleration[i] = SVector{D, T}(ntuple(d -> T(i * d), D))
    end
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), m₀ = T(1000 * 0.02^D))
    kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
    Pressure!(host.Pressure, host.Density, constants)
    return (; host, metadata, geometry, constants, kernel)
end

function context(f, D, T; reconstruct, scheme, graph, forces)
    metadata = deepcopy(f.metadata)
    metadata.TimeSteppingMode = scheme
    metadata.GPUUseGraph = graph
    metadata.GPUBoundaryForces = forces
    metadata.OutputIterationCounter = 1
    gpu = upload_particles(f.host; position_type = position_float_type(metadata), reconstruct_fields = reconstruct)
    sup = GPUSupportArrays{D, T}(gpu, metadata)
    cl = CellListWorkspace{D, position_type(gpu)}(length(gpu))
    red = ReductionWorkspace{SVector{3, T}}(length(gpu))
    motion = MotionArrays(f.geometry, f.host)
    floating = BODIES.FloatingArrays(f.geometry, f.host, f.constants; position_type = position_type(gpu))
    state = StepState{T}(; dx = one(T) + f.kernel.h)
    return (; metadata, gpu, sup, cl, red, motion, floating, state)
end

function advance!(ctx, f)
    (; metadata, gpu, sup, cl, red, motion, floating, state) = ctx
    SimulationLoop(ZeroDensityDiffusion(), ZeroViscosity(), f.kernel, metadata, f.constants,
        gpu, cl, sup, red, motion, state; floating)
    return nothing
end

function snapshot(ctx)
    order = sortperm(Array(ctx.gpu.ID))
    fields = (:Position, :Velocity, :Density, :Pressure, :Acceleration, :Type, :ID, :GroupMarker)
    return (; particles = map(f -> Array(getproperty(ctx.gpu, f))[order], fields),
        derivative = Array(ctx.sup.dρdtI)[order], reduction = Array(ctx.red.partial),
        state = Array(ctx.state.f), flags = Array(ctx.state.i),
        bodies = BODIES.floating_state(ctx.floating), rebuilds = ctx.cl.nrebuilds)
end

@testset "reconstructible sorting fields" begin
    @testset "EOS, standalone acceleration and exact shadow bytes" begin
        for D in (2, 3), T in (Float32, Float64), double_position in (false, true),
            mode in (NoMDBC(), SimpleMDBC(), UpdatedMDBC())
            f = fixture(D, T; double_position)
            gpu = upload_particles(f.host; position_type = position_float_type(f.metadata), reconstruct_fields = true)
            cl = CellListWorkspace{D, position_type(gpu)}(length(gpu))
            support = GPUSupportArrays{D, T}(gpu, f.metadata)
            full = upload_particles(f.host; position_type = position_type(gpu))
            bytes(s) = sum(a -> length(a) * sizeof(eltype(a)), s)
            @test bytes(full.scratch) - bytes(gpu.scratch) == length(gpu) * (D + 1) * sizeof(T)
            @test isempty(gpu.scratch.Acceleration) && isempty(gpu.scratch.Pressure)
            @test_throws ArgumentError DRIVER.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
            @test cl.nrebuilds == 0
            acceleration_pointer, pressure_pointer = pointer(gpu.Acceleration), pointer(gpu.Pressure)
            for _ in 1:3
                before = Array(gpu.Acceleration)
                ids_before = Array(gpu.ID)
                fill!(gpu.Pressure, T(NaN))
                grid = DRIVER.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹;
                    constants = f.constants, boundary_mode = mode, support)
                ids = Array(gpu.ID)
                expected_acceleration = before[indexin(ids, ids_before)]
                @test pointer(gpu.Acceleration) == acceleration_pointer
                @test pointer(gpu.Pressure) == pressure_pointer
                @test Array(gpu.Acceleration) == expected_acceleration
                densities, types = Array(gpu.Density), Array(gpu.Type)
                expected_pressure = [mode isa UpdatedMDBC && types[i] in (Fixed, Moving) ?
                    f.constants.c₀^2 * (densities[i] - f.constants.ρ₀) :
                    EquationOfStateGamma7(densities[i], f.constants.c₀, f.constants.ρ₀) for i in eachindex(ids)]
                @test Array(gpu.Pressure) ≈ expected_pressure rtol = T(1e-5)
                back = deepcopy(f.host)
                download_particles!(back, gpu, grid, (:ID, :Acceleration, :Pressure))
                @test back.Acceleration == expected_acceleration
                @test back.Pressure == Array(gpu.Pressure)
                fixed = findall(==(Fixed), types)
                @test Array(gpu.scratch.Velocity)[fixed] == Array(gpu.Velocity)[fixed]
            end
        end
    end

    @testset "initial reduction, carried derivatives, floating forces and output stops" begin
        for D in (2, 3), T in (Float32, Float64), double_position in (false, true),
            scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping()), graph in (false, true),
            forces in (false, true)
            f = fixture(D, T; double_position)
            f.host.Acceleration[1] = SVector{D, T}(ntuple(_ -> T(1e8), D))
            full = context(f, D, T; reconstruct = false, scheme, graph, forces)
            compact = context(f, D, T; reconstruct = true, scheme, graph, forces)
            initial_values = [KERNELS.step_values(f.host.Position[i], f.host.Velocity[i],
                f.host.Acceleration[i], zero(f.host.Position[i]), f.kernel.h, f.kernel.η²) for i in eachindex(f.host)]
            expected_dt = f.constants.CFL * min(minimum(v[2] for v in initial_values),
                f.kernel.h / (f.constants.c₀ + maximum(v[1] for v in initial_values)))
            for interval in 1:3
                if interval > 1
                    for ctx in (full, compact)
                        ctx.metadata.OutputIterationCounter = interval
                        STEP.set_slot!(ctx.state.f, ctx.state.fh, STEP.F_DX, one(T) + f.kernel.h)
                    end
                    # The completed-step reduction is valid; these unsorted
                    # fields must not be read before the rebuild refresh.
                    fill!(compact.gpu.Acceleration, SVector{D, T}(ntuple(_ -> T(NaN), D)))
                    fill!(compact.gpu.Pressure, T(NaN))
                end
                advance!(full, f)
                advance!(compact, f)
                @test snapshot(compact) == snapshot(full)
                @test compact.cl.nrebuilds >= interval
                if interval == 1
                    @test compact.metadata.Iteration == 1
                    @test compact.metadata.TotalTime == expected_dt
                end
            end
            @test graph ? !isempty(compact.state.graphs) : isempty(compact.state.graphs)
        end
    end
end

end
