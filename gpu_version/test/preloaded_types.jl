module PreloadedTypeBenchmarkTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, HDF5
include(joinpath(@__DIR__, "..", "benchmark", "preloaded_types.jl"))
const B = PreloadedTypeBenchmark

function frame(; displacement = 0.0, velocity_change = 0.0, density_change = 0.0, time = 0.1)
    position = [0.0 1.0 8.0; 0.0 2.0 8.0; 0.0 0.0 0.0]
    position[1, 2] += displacement
    velocity = [1.0 1.0 99.0; 0.0 0.0 99.0; 0.0 0.0 0.0]
    velocity[1, 2] += velocity_change
    density = [1000.0, 1000.0 + density_change, 1000.0]
    pressure = [1.0, 2.0, 3.0]
    metrics = B.frame_metrics(position, velocity, density, pressure, [1, 2], 0.5, 2)
    return (; ids = [1, 2, 3], types = UInt8[UInt8(Fluid), UInt8(Fluid), UInt8(Fixed)],
        position, velocity, density, pressure, metrics, time)
end

@testset "preloaded experiment input and diagnostic quantities" begin
    @test_throws ArgumentError B.main(["--unknown"])
    @test_throws ArgumentError B.main(["--rounds"])
    @test_throws ArgumentError B.main(["--rounds=2"])
    @test_throws ArgumentError B.main(["--time=0"])
    @test_throws ArgumentError B.main(["--time=NaN"])
    @test_throws ArgumentError B.main(["not-a-case"])
    @test_throws ErrorException B.replace_once("abcabc", "abc", "x")
    @test_throws ErrorException B.replace_once("abc", "missing", "x")
    @test B.replace_once("abc", "b", "x") == "axc"
    original = frame()
    @test original.metrics.fluid_count == 2
    @test original.metrics.fluid_mass == 1
    @test original.metrics.kinetic_energy == 0.5
    @test original.metrics.com_x == 0.5
    @test original.metrics.momentum_x == 1
    @test original.metrics.front_x == 1
    @test original.metrics.surface_max == 2
    run = (; dx = 0.02, c0 = 20.0, rho0 = 1000.0)
    exact = B.compare_frames(original, original, run)
    @test exact.exact
    @test exact.position_max_dx == exact.com_delta_dx == 0
    changed = B.compare_frames(original, frame(; displacement = 0.01, velocity_change = 0.02,
        density_change = 1, time = 0.1001), run)
    @test !changed.exact
    @test changed.position_max_dx ≈ 0.5
    @test changed.position_rms_dx ≈ 0.5 / sqrt(2)
    @test changed.com_delta_dx ≈ 0.25
    @test changed.velocity_max_c0 ≈ 0.001
    @test changed.density_max_rho0 == 0.001
    @test changed.time_delta ≈ 0.0001
    @test changed.kinetic_relative_delta ≈ 0.0202
    @test_throws ErrorException B.compare_frames(original, merge(original, (; ids = [3, 2, 1])), run)
end

@testset "installed preloaded-type option and explicit fallbacks" begin
    for D in (2, 3), T in (Float32, Float64)
        meta = SimulationMetaData{D,T}(SimulationName = "Preloaded", SaveLocation = tempdir(), GPULanesPerParticle = 1)
        kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.02))
        config() = effective_configuration(meta, kernel, LinearDensityDiffusion(), ArtificialViscosity(); particle_count = 8)
        @test meta.GPUPreloadInteractionTypes
        @test config().effective.preload_interaction_types
        meta.GPUPreloadInteractionTypes = false
        @test !config().effective.preload_interaction_types
        meta.GPUPreloadInteractionTypes = true
        enabled = config()
        @test enabled.requested.preload_interaction_types
        @test enabled.capabilities.preload_interaction_types
        @test enabled.effective.preload_interaction_types
        @test enabled.effective.density_diffusion_operator === :reference
        @test isempty(enabled.fallbacks)
        meta.GPURadialInteractions = true
        @test !config().effective.preload_interaction_types
        @test only(config().fallbacks).reason === :radial_diffusion_selected
        meta.GPURadialInteractions = false
        unsupported = effective_configuration(meta, kernel, ZeroDensityDiffusion(), ZeroViscosity(); particle_count = 8)
        @test !unsupported.capabilities.preload_interaction_types
        @test !unsupported.effective.preload_interaction_types
        @test only(unsupported.fallbacks).reason === :unsupported_diffusion_model
        if D == 2
            fine = [SPHGeometry{2,T}([SVector{2,T}(0.02, 0.02)]; Density = T(1000), Type = Fluid, GroupMarker = 1)]
            meta.VariableResolution = VariableResolutionConfig(
                VariableResolutionBox((0.0, 0.0), (1.0, 1.0)); fine_geometry = fine)
            resolved = config()
            @test !resolved.effective.preload_interaction_types
            @test any(f -> f.option === :GPUPreloadInteractionTypes && f.reason === :variable_resolution_backend,
                resolved.fallbacks)
        end
    end
end

function test_experimental_algebra(kernels)
    @testset "preloaded types retain the ordered reference diffusion formula" begin
        for T in (Float32, Float64), D in (2, 3)
            constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(35))
            kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx)
            x = SVector{D,T}(ntuple(d -> T(0.01 * d), D))
            gradient = ∇Wᵢⱼ(kernel, norm(x) * kernel.h⁻¹, x)
            for ti in (Fluid, Fixed, Moving, Floating, Frozen), tj in (Fluid, Fixed, Moving, Floating, Frozen),
                swapped in (false, true)
                types = [ti, tj]
                i, j = swapped ? (2, 1) : (1, 2)
                r, g = swapped ? (-x, -gradient) : (x, gradient)
                ra, rb = swapped ? (T(1007), T(995)) : (T(995), T(1007))
                args = (kernel, constants, nothing, r, g, dot(x, x), ra, rb, inv(ra), inv(rb), i, j, types)
                for model in (LinearDensityDiffusion(), ZeroDensityDiffusion())
                    actual = kernels.preloaded_diffusion(model, args..., (ti == Fluid) & (tj == Fluid))
                    @test actual == compute_density_diffusion(model, args...)
                    if model isa LinearDensityDiffusion
                        installed = compute_density_diffusion(model, args[1:end-1]..., (ti == Fluid) & (tj == Fluid))
                        @test installed == actual
                    end
                end
            end
            for (i, j) in ((0, 1), (1, 0), (3, 1), (1, 3))
                @test_throws BoundsError compute_density_diffusion(LinearDensityDiffusion(), kernel, constants,
                    nothing, x, gradient, dot(x, x), T(995), T(1007), inv(T(995)), inv(T(1007)),
                    i, j, [Fluid, Fluid])
            end
        end
    end
end

function test_device(candidate)
    @testset "isolated prototype executes both compiler modes" begin
        case = only(B.select_cases(["StillWedge2D_MDBC_dp0.02"]))
        for strict in (false, true)
            reference = (; B.CURRENT..., strict)
            optimized = (; reference..., name = "preloaded", driver = candidate.driver)
            a = B.measure(reference, case, 0.003)
            b = B.measure(optimized, case, 0.003)
            differences = B.compare_runs(a, b)
            @test a.work == b.work
            @test length(differences) == 2
            @test maximum(d.position_max_dx for d in differences) <= 1e-4
            @test maximum(d.density_max_rho0 for d in differences) <= 1e-5
            installed = (; reference..., name = "installed", preload = true)
            c = B.measure(installed, case, 0.003)
            @test c.work == b.work
            @test maximum(d.position_max_dx for d in B.compare_runs(b, c)) <= 1e-4
        end
    end
end

function switch_run(graph, scheme; refresh_control = false)
    D, T = 2, Float32
    points = [SVector{D,Float64}(0.02i, 0.02j) for i in 1:4, j in 1:4][:]
    geometry = [SPHGeometry{D,T}(points; Density = T[995 + i % 11 for i in eachindex(points)], Type = Fluid, GroupMarker = 1)]
    meta = SimulationMetaData{D,T}(SimulationName = "PreloadedSwitch", SaveLocation = tempdir(),
        SimulationTime = 0.01, OutputTimes = 0.001, OutputIterationCounter = 1,
        GPUDoublePosition = true, GPUUseGraph = graph, GPUMaxStepsPerSync = 4, GPULanesPerParticle = 4)
    meta.TimeSteppingMode = scheme
    host = AllocateDataStructures(geometry, meta)
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20))
    kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx)
    gpu = upload_particles(host, meta)
    support = GPUSupportArrays{D,T}(gpu, meta)
    cells = CellListWorkspace{D,Float64}(length(gpu))
    reduction = ReductionWorkspace{SVector{3,T}}(length(gpu))
    motion = MotionArrays(geometry, host)
    state = StepState{T}(dx = one(T) + kernel.h)
    previous_graphs = CUDA.CuGraphExec[]
    GK, GS = SPHExampleGPU.GPUKernels, SPHExampleGPU.GPUStepState
    for enabled in (false, true, false)
        meta.GPUPreloadInteractionTypes = enabled
        if refresh_control && state.primed && state.preload_interaction_types != enabled
            step = HostStep(state)
            GK.launch_prepare_interactions!(gpu.Position, step;
                pos_cells = support.PosCells, CellID = gpu.CellID, grid = cells.grid_dev, H = kernel.H)
            GK.launch_interactions!(support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
                support.∇Cᵢ, support.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
                cells.CellStart, gpu.CellID, cells.grid_dev, step, LinearDensityDiffusion(), ArtificialViscosity(),
                kernel, constants, Val(false), Val(false); lanes = Val(4), pos_cells = support.PosCells,
                preload_types = Val(enabled), frozen_particles = Val(false),
                include_gravity = Val(scheme isa SingleNeighborTimeStepping))
            GS.invalidate_graphs!(state)
            state.preload_interaction_types = enabled
        end
        SimulationLoop(LinearDensityDiffusion(), ArtificialViscosity(), kernel, meta, constants,
            gpu, cells, support, reduction, motion, state)
        @test state.preload_interaction_types == enabled
        @test state.configuration.effective.preload_interaction_types == enabled
        if graph
            @test !isempty(state.graphs)
            @test all(exec -> all(old -> exec !== old, previous_graphs), values(state.graphs))
            previous_graphs = collect(values(state.graphs))
        end
        meta.OutputIterationCounter += 1
    end
    order = sortperm(Array(gpu.ID))
    return (; steps = meta.Iteration, time = meta.TotalTime,
        fields = map(n -> Array(getproperty(gpu, n))[order], (:Position, :Velocity, :Density, :Acceleration)))
end

candidate = B.experimental_modules()
Base.invokelatest(test_experimental_algebra, candidate.kernels)
if CUDA.functional()
    Base.invokelatest(test_device, candidate)
    @testset "preloaded option invalidates graphs and refreshes carried derivatives" begin
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            direct = switch_run(false, scheme)
            @test direct == switch_run(true, scheme)
            if scheme isa SingleNeighborTimeStepping
                @test direct == switch_run(false, scheme; refresh_control = true)
            end
        end
    end
else
    @info "CUDA unavailable; skipping experimental preloaded-type device smoke tests."
end

end
