module FinalStateCacheTests

using Test, SPHExampleGPU, CUDA, StaticArrays

const DRIVER = SPHExampleGPU.SPHCellList
const KERNELS = SPHExampleGPU.GPUKernels
const STEP = SPHExampleGPU.GPUStepState

function fixture(D, T; double_position = false, compact = true, single = false, graph = false)
    dx = T(0.02)
    points = vec([SVector{D,T}(ntuple(d -> dx * Tuple(index)[d], D))
        for index in CartesianIndices(ntuple(_ -> 1:4, D))])
    geometry = [
        SPHGeometry{D,T}(reverse(points); Density = [T(1001 + i % 13) for i in eachindex(points)], Type = Fluid, GroupMarker = 1),
        SPHGeometry{D,T}([zero(first(points)), 6 * first(points)]; Density = T[990, 1010], Type = Fixed, GroupMarker = 2),
    ]
    meta = SimulationMetaData{D,T,PlanarShifting,StoreKernelOutput}(
        SimulationName = "FinalStateCaches", SaveLocation = tempdir(),
        GPUCacheInteractionFields = true, GPUDoublePosition = double_position,
        GPUCompactBuffers = compact, GPUUseGraph = graph, GPUMaxStepsPerSync = 4,
        SimulationTime = T(0.1), OutputTimes = T[0.001, 0.002, 0.003])
    meta.TimeSteppingMode = single ? SingleNeighborTimeStepping() : SymplecticTimeStepping()
    meta.OutputIterationCounter = 1
    host = AllocateDataStructures(geometry, meta)
    constants = SimulationConstants{T}(; dx, c₀ = T(20), g = zero(T), m₀ = T(1000) * dx^D)
    kernel = SPHKernelInstance{D,T}(WendlandC2(); dx)
    gpu = upload_particles(host, meta)
    sup = GPUSupportArrays{D,T}(gpu, meta)
    cl = CellListWorkspace{D,position_float_type(meta)}(length(gpu))
    red = ReductionWorkspace{SVector{3,T}}(length(gpu))
    motion = MotionArrays(geometry, host)
    floating = SPHExampleGPU.GPUFloating.FloatingArrays(geometry, host, constants;
        position_type = position_type(gpu))
    state = StepState{T}(; dx = one(T) + kernel.h)
    ctx = (; gpu, sup, cl, red, motion, floating, state, SimKernel = kernel, SimConstants = constants,
        SimDensityDiffusion = LinearDensityDiffusion(), SimViscosity = ArtificialViscosity(),
        FlagKernel = Val(true), FlagShift = Val(true), UseMDBC = false, SingleNeighbor = single,
        threads = 128, lanes = Val(4), bforces = Val(true), HourGlass = meta.HourGlass,
        boundary_mode = NoMDBC())
    return (; ctx, meta)
end

function rebuild!(ctx)
    DRIVER.rebuild_cell_list!(ctx.gpu, ctx.cl, ctx.SimKernel.H⁻¹; support = ctx.sup,
        kernel = ctx.SimKernel, constants = ctx.SimConstants)
    DRIVER.prepare_state_caches!(ctx, HostStep(zero(eltype(ctx.gpu.Density)), zero(eltype(ctx.gpu.Density))))
    return nothing
end

function snapshot(ctx)
    order = sortperm(Array(ctx.gpu.ID))
    return (; fields = map(name -> Array(getproperty(ctx.gpu, name))[order],
        (:Position, :Velocity, :Density, :Acceleration, :Type, :ID, :Kernel, :KernelGradient)),
        derivative = Array(ctx.sup.dρdtI)[order], partial = Array(ctx.red.partial),
        state = Array(ctx.state.f), flags = Array(ctx.state.i))
end

function check_caches(ctx, density)
    T = eltype(density)
    @test Array(ctx.sup.InvDensity) ≈ inv.(density) rtol = 2eps(T)
    @test Array(ctx.sup.Pressure) ≈ EquationOfStateGamma7.(density, ctx.SimConstants.c₀, ctx.SimConstants.ρ₀) rtol = 20eps(T) atol = 20eps(T)
end

@testset "accepted-state interaction caches" begin
    @testset "standalone final destinations, frozen entries and multiple blocks" begin
        for T in (Float32, Float64), n in (7, 513), alias in (false, true)
            constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T))
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            V = SVector{2,T}
            position, velocity, acceleration = (CUDA.zeros(V, n) for _ in 1:3)
            types = CuArray([i % 3 == 0 ? Frozen : i % 3 == 1 ? Fluid : Fixed for i in 1:n])
            full = T[990 + i % 13 for i in 1:n]
            half = T[i % 3 == 0 ? full[i] : 1010 + i % 7 for i in 1:n]
            density_half = CuArray(half)
            rate = CuArray(T[i % 2 == 0 ? -5e4 : 1e4 for i in 1:n])
            red = ReductionWorkspace{SVector{3,T}}(n)
            finish!(density, inverse; kwargs...) = KERNELS.launch_final_step!(
                position, velocity, acceleration, density, rate, density_half, position, velocity,
                types, nothing, nothing, HostStep(T(0.001), zero(T)), kernel, constants, red, Val(false);
                boundary_mode = NoMDBC(), inverse_density = inverse, kwargs...)
            baseline, candidate = CuArray(full), CuArray(full)
            predictor_inverse = CuArray(inv.(half))
            next_inverse = alias ? predictor_inverse : CUDA.fill(T(NaN), n)
            next_pressure = CUDA.fill(T(NaN), n)
            finish!(baseline, predictor_inverse)
            @test Array(predictor_inverse) == inv.(half)
            finish!(candidate, predictor_inverse; next_inverse_density = next_inverse, next_pressure)
            accepted = Array(candidate)
            @test accepted == Array(baseline)
            @test accepted[3:3:n] == full[3:3:n]
            @test all(>=(constants.ρ₀), accepted[2:3:n])
            @test Array(next_inverse) ≈ inv.(accepted) rtol = 2eps(T)
            @test Array(next_pressure) ≈ EquationOfStateGamma7.(accepted, constants.c₀, constants.ρ₀) rtol = 20eps(T) atol = 20eps(T)
            @test_throws DimensionMismatch finish!(candidate, predictor_inverse; next_inverse_density = CUDA.zeros(T, n + 1))
            @test_throws DimensionMismatch finish!(candidate, predictor_inverse; next_pressure = CUDA.zeros(T, n + 1))
        end
    end

    @testset "alias-safe final update, graph replay and reorder" begin
        for D in (2, 3), T in (Float32, Float64), double_position in (false, true),
            compact in (false, true), single in (false, true), graph in (false, true)
            baseline = fixture(D, T; double_position, compact, single, graph).ctx
            candidate = merge(fixture(D, T; double_position, compact, single, graph).ctx,
                (; cache_final_state = true))
            for ctx in (baseline, candidate)
                rebuild!(ctx)
                STEP.resume_after_rebuild!(ctx.state)
                STEP.set_output_time!(ctx.state, one(T))
                KERNELS.launch_step_reduction!(ctx.red, ctx.gpu.Position, ctx.gpu.Velocity,
                    ctx.gpu.Acceleration, ctx.gpu.Position, ctx.SimKernel)
                single && DRIVER.enqueue_state_derivative!(ctx, ctx.state, false, "mdbc", "init")
            end
            for pass in 1:2
                for ctx in (baseline, candidate)
                    if graph
                        DRIVER.launch_step_graph!(ctx, 3)
                    else
                        for _ in 1:3
                            DRIVER.enqueue_step!(ctx, false)
                        end
                    end
                end
                @test snapshot(candidate) == snapshot(baseline)
                check_caches(candidate, Array(candidate.gpu.Density))
                check_caches(baseline, Array(baseline.sup.ρₙ⁺))
                for ctx in (baseline, candidate)
                    # Deliberately stale caches cannot survive a reorder refresh.
                    fill!(ctx.sup.InvDensity, T(NaN))
                    fill!(ctx.sup.Pressure, T(NaN))
                    rebuild!(ctx)
                    check_caches(ctx, Array(ctx.gpu.Density))
                    STEP.resume_after_rebuild!(ctx.state)
                    single && DRIVER.enqueue_state_derivative!(ctx, ctx.state, false, "mdbc", "rebuild")
                end
            end
        end
    end

    @testset "driver startup, forced rebuilds and output stops" begin
        for single in (false, true), graph in (false, true), prebuilt in (false, true)
            (; ctx, meta) = fixture(2, Float32; double_position = true, single, graph)
            if prebuilt
                DRIVER.rebuild_cell_list!(ctx.gpu, ctx.cl, ctx.SimKernel.H⁻¹;
                    support = ctx.sup, kernel = ctx.SimKernel, constants = ctx.SimConstants)
                STEP.set_slot!(ctx.state.f, ctx.state.fh, STEP.F_DX, 0f0)
            end
            for interval in 1:3
                if interval > 1
                    STEP.set_slot!(ctx.state.f, ctx.state.fh, STEP.F_DX, 1f0)
                end
                fill!(ctx.sup.InvDensity, NaN32)
                fill!(ctx.sup.Pressure, NaN32)
                meta.OutputIterationCounter = interval
                DRIVER.SimulationLoop(ctx.SimDensityDiffusion, ctx.SimViscosity, ctx.SimKernel,
                    meta, ctx.SimConstants, ctx.gpu, ctx.cl, ctx.sup, ctx.red, ctx.motion, ctx.state;
                    floating = ctx.floating)
                @test meta.TotalTime >= meta.OutputTimes[interval]
                @test all(isfinite, Array(ctx.gpu.Density))
                check_caches(ctx, Array(ctx.gpu.Density))
            end
            @test ctx.cl.nrebuilds >= 3 - Int(prebuilt)
        end
    end
end

end
