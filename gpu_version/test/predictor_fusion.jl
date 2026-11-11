module PredictorFusionTests

using Test, SPHExampleGPU, CUDA, StaticArrays
using SPHExampleGPU.SimulationLoggerConfiguration: device_buffer_bytes

const DRIVER = SPHExampleGPU.SPHCellList
const KERNELS = SPHExampleGPU.GPUKernels
const STEP = SPHExampleGPU.GPUStepState

function fixture(D, T; compact, cache, double_position, fused, lanes)
    dx = T(0.02)
    points = vec([SVector{D,T}(ntuple(d -> dx * Tuple(index)[d], D))
        for index in CartesianIndices(ntuple(_ -> 1:4, D))])
    geometry = [
        SPHGeometry{D,T}(reverse(points); Density = [T(1001 + i % 13) for i in eachindex(points)], Type = Fluid, GroupMarker = 1),
        SPHGeometry{D,T}([zero(first(points)), 6 * first(points)]; Density = T[990, 1010], Type = Fixed, GroupMarker = 2),
        SPHGeometry{D,T}([8 * first(points)]; Density = T[980], Type = Frozen, GroupMarker = 3),
    ]
    meta = SimulationMetaData{D,T,PlanarShifting,StoreKernelOutput}(
        SimulationName = "PredictorFusion", SaveLocation = tempdir(),
        GPUCacheInteractionFields = cache, GPUDoublePosition = double_position, GPUCompactBuffers = compact)
    host = AllocateDataStructures(geometry, meta)
    constants = SimulationConstants{T}(; dx, c₀ = T(20), m₀ = T(1000) * dx^D)
    kernel = SPHKernelInstance{D,T}(WendlandC2(); dx)
    gpu = upload_particles(host, meta)
    sup = GPUSupportArrays{D,T}(gpu, meta; fuse_predictor = fused)
    cl = CellListWorkspace{D,position_float_type(meta)}(length(gpu))
    red = ReductionWorkspace{SVector{3,T}}(length(gpu))
    state = StepState{T}()
    motion = MotionArrays(geometry, host)
    ctx = (; gpu, sup, cl, red, state, motion, SimKernel = kernel, SimConstants = constants,
        SimDensityDiffusion = LinearDensityDiffusion(), SimViscosity = ArtificialViscosity(),
        FlagKernel = Val(true), FlagShift = Val(true), UseMDBC = false, SingleNeighbor = false,
        threads = 128, lanes = Val(lanes), bforces = Val(true), HourGlass = meta.HourGlass,
        boundary_mode = NoMDBC(), cache_final_state = cache, fuse_predictor = fused,
        interaction_fastmath = false)
    return (; ctx, meta)
end

function rebuild!(ctx)
    DRIVER.rebuild_cell_list!(ctx.gpu, ctx.cl, ctx.SimKernel.H⁻¹; support = ctx.sup,
        kernel = ctx.SimKernel, constants = ctx.SimConstants)
    DRIVER.prepare_state_caches!(ctx, HostStep(zero(eltype(ctx.gpu.Density)), zero(eltype(ctx.gpu.Density))))
    STEP.resume_after_rebuild!(ctx.state)
    STEP.set_output_time!(ctx.state, 1.0)
    KERNELS.launch_step_reduction!(ctx.red, ctx.gpu.Position, ctx.gpu.Velocity,
        ctx.gpu.Acceleration, ctx.gpu.Position, ctx.SimKernel)
end

function snapshot(ctx)
    order = sortperm(Array(ctx.gpu.ID))
    return map(name -> Array(getproperty(ctx.gpu, name))[order],
        (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :Kernel, :KernelGradient))
end

@testset "gather/predictor fusion" begin
    @testset "separate inputs, graph replay and reordering" begin
        for D in (2, 3), T in (Float32, Float64), compact in (false, true), cache in (false, true), double_position in (false, true)
            T === Float64 && double_position && continue
            lanes = cache ? (compact ? 16 : 4) : (compact ? 32 : 1)
            baseline = fixture(D, T; compact, cache, double_position, fused = false, lanes).ctx
            fused = fixture(D, T; compact, cache, double_position, fused = true, lanes).ctx
            sup = fused.sup
            @test sup.PosCells === nothing || !Base.mightalias(sup.PosCells, sup.PosCellsₙ⁺)
            @test !cache || (!Base.mightalias(sup.InvDensity, sup.InvDensityₙ⁺) && !Base.mightalias(sup.Pressure, sup.Pressureₙ⁺))
            expected_extra = length(fused.gpu) * (cache ? 2sizeof(T) : 0)
            if compact && double_position
                expected_extra += length(fused.gpu) * sizeof(SPHExampleGPU.GPUCellGrid.PosCell{D,T})
            end
            @test device_buffer_bytes(sup) - device_buffer_bytes(baseline.sup) == expected_extra
            for pass in 1:2
                for ctx in (baseline, fused)
                    rebuild!(ctx)
                    STEP.invalidate_graphs!(ctx.state)
                    before = snapshot(ctx)
                    DRIVER.prepare_step_graph!(ctx)
                    @test snapshot(ctx) == before
                end
                for _ in 1:3
                    DRIVER.launch_step_graph!(baseline)
                    DRIVER.launch_step_graph!(fused)
                end
                a, b = snapshot(fused), snapshot(baseline)
                for i in eachindex(a)
                    if i in (5, 6)
                        @test a[i] == b[i]
                    else
                        @test a[i] ≈ b[i] rtol = 100eps(T) atol = 100eps(T)
                    end
                end
                @test Array(fused.state.i) == Array(baseline.state.i)
                @test Array(fused.state.f) ≈ Array(baseline.state.f) rtol = 100eps(T) atol = 100eps(T)
                position, velocity = DRIVER.predictor_arrays(sup, fused.gpu)
                types = Array(fused.gpu.Type)
                fixed = findall(==(Fixed), types)
                frozen = findall(==(Frozen), types)
                @test Array(position)[fixed] == Array(fused.gpu.Position)[fixed]
                @test all(iszero, Array(velocity)[frozen])
                @test Array(fused.gpu.Density)[frozen] == T[980]
                if cache
                    half = Array(sup.ρₙ⁺)
                    @test Array(sup.InvDensityₙ⁺) ≈ inv.(half) rtol = 2eps(T)
                    @test Array(sup.Pressureₙ⁺) ≈ EquationOfStateGamma7.(half, fused.SimConstants.c₀, fused.SimConstants.ρ₀) rtol = 20eps(T) atol = 20eps(T)
                    @test Array(sup.InvDensity) ≈ inv.(Array(fused.gpu.Density)) rtol = 2eps(T)
                end
            end
            bad_output = merge(DRIVER.predictor_outputs(fused), (; position = fused.gpu.Position))
            @test_throws ArgumentError DRIVER.launch_solver_interactions!(fused,
                fused.gpu.Position, fused.gpu.Density, fused.gpu.Velocity, sup.PosCells, HostStep(T(1e-4), 0.0);
                predictor_output = bad_output)
        end
    end

    @testset "derivative precision boundary" begin
        V = SVector{2,Float32}
        constants = SimulationConstants{Float32}(dx = 0.02, c₀ = 20)
        position, velocity, density = [V(1, 2)], [V(0.3, 0.7)], Float32[1001]
        out = (; position = copy(position), velocity = copy(velocity), density = copy(density),
            source_position = position, source_velocity = velocity, source_density = density,
            types = [Fluid], pos_cells = nothing, cell_ids = nothing, grid = nothing, H = 0.04f0,
            inverse_density = nothing, pressure = nothing, reuse_predictors = Val(false))
        acc = SVector(1 / 3, 2 / 7)
        rate = 1 / 9
        step = HostStep(0.01f0, 0.0)
        @test KERNELS.predict_after_gather!(out, 1, acc, rate, step, constants) isa V
        expected = merge(out, (; position = copy(position), velocity = copy(velocity), density = copy(density)))
        KERNELS.predict_after_gather!(expected, 1, V(acc), Float32(rate), step, constants)
        @test out.position == expected.position
        @test out.velocity == expected.velocity
        @test out.density == expected.density
    end

    @testset "driver eligibility" begin
        meta = SimulationMetaData{2,Float32}(SimulationName = "FusionEligibility", SaveLocation = tempdir(),
            TimeSteppingMode = SymplecticTimeStepping())
        stationary = (; motion = (; active = false), floating = nothing, external_forces = (),
            rotation = nothing, open_boundary = nothing)
        enabled(; kwargs...) = DRIVER.predictor_fusion_enabled(meta; merge(stationary, (; kwargs...))...)
        @test enabled()
        @test !enabled(motion = (; active = true))
        @test !enabled(floating = (; active = true))
        @test !enabled(external_forces = (1,))
        @test !enabled(rotation = 1)
        @test !enabled(open_boundary = 1)
        meta.GPUInteractionFastMath = false
        @test !enabled()
        meta.GPUInteractionFastMath = true
        meta.GPUFusePredictor = false
        @test !enabled()
        meta.GPUFusePredictor = true
        meta.TimeSteppingMode = SingleNeighborTimeStepping()
        @test !enabled()
    end
end

end
