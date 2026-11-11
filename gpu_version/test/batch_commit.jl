module BatchCommitTests

using Test, SPHExampleGPU, CUDA, StaticArrays
using SPHExampleGPU.GPUStepState

const K = SPHExampleGPU.GPUKernels

@testset "commit and next timestep share a launch" begin
    for T in (Float32, Float64), nblocks in (1, 2, 32, 512), clip in (false, true),
        scenario in (:active, :output, :rebuild, :stopped)
        constants = SimulationConstants{T}(dx = 0.02, c₀ = 20)
        kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
        red = ReductionWorkspace{SVector{3,T}}(256nblocks)
        displacement = scenario === :rebuild ? T(0.25) : zero(T)
        partials = fill(SVector{3,T}(0, Inf, 0), nblocks)
        partials[end] = SVector{3,T}(3, 0.5, displacement)
        copyto!(red.partial, partials)
        states = [StepState{T}(dt = T(0.001), time = 0.25, iteration = 17) for _ in 1:2]
        for state in states
            set_output_time!(state, scenario === :output ? 0.2505 : 1.0)
            state.ih[I_PHASE] = PHASE_DT_READY
            state.ih[I_CLIP_OUTPUT] = Int32(clip)
            scenario === :stopped && (state.ih[I_STOP] = STOP_OUTPUT)
            copyto!(state.i, state.ih)
        end
        K.launch_commit!(states[1])
        K.launch_finish!(states[1], red, kernel, constants)
        K.launch_finish!(states[2], red, kernel, constants; commit_previous = true)
        @test isequal(Array(states[1].f), Array(states[2].f))
        @test Array(states[1].i) == Array(states[2].i)
    end
end

function simulate(D, T; graph, single, compact, moving, floating = false)
    return mktempdir() do save
        dx = T(0.02)
        width = D == 2 ? 17 : 7
        points = vec([SVector{D,T}(ntuple(d -> dx * Tuple(index)[d], D))
            for index in CartesianIndices(ntuple(_ -> 1:width, D))])
        wall = [SVector{D,T}(ntuple(d -> d == 1 ? T(2) : dx * i, D)) for i in 1:2]
        motion = moving ? MotionDetails{D,T}(Velocity = T(5),
            Direction = SVector{D,T}(ntuple(d -> d == 1 ? one(T) : zero(T), D)),
            StartTime = -1, Duration = 2) : nothing
        geometry = [SPHGeometry{D,T}(reverse(points); Density = T[1000 + i % 7 for i in eachindex(points)],
                Type = Fluid, GroupMarker = 1),
            SPHGeometry{D,T}(wall; Density = T(1000), Type = moving ? Moving : Fixed,
                GroupMarker = 2, Motion = motion)]
        if floating
            body = vec([SVector{D,T}(ntuple(d -> T(3) + dx * Tuple(index)[d], D))
                for index in CartesianIndices(ntuple(_ -> 1:2, D))])
            initial_velocity = SVector{D,T}(ntuple(d -> T(d) / 10, D))
            push!(geometry, SPHGeometry{D,T}(body; Density = T(1000), Type = Floating,
                GroupMarker = 3, Floating = FloatingDetails{T}(RelativeWeight = 1.2, InitialVelocity = initial_velocity)))
        end
        meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
            SimulationName = "BatchCommit", SaveLocation = save, SimulationTime = 0.02,
            OutputTimes = 0.002, GPUUseGraph = graph, GPUMaxStepsPerSync = 128,
            GPUCompactBuffers = compact, GPUCacheInteractionFields = compact,
            GPUDoublePosition = T === Float32, GPUTimestepDiagnosticsEvery = 1,
            GPUClipTimeStepToOutput = true, VisualizeInParaview = false, OpenLogFile = false)
        constants = SimulationConstants{T}(; dx, c₀ = 20, g = 0, m₀ = T(1000) * dx^D)
        kernel = SPHKernelInstance{D,T}(WendlandC2(); dx)
        particles = AllocateDataStructures(geometry, meta)
        logger = SimulationLogger(save)
        result = try
            redirect_stdout(devnull) do
                RunSimulation(; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
                    SimKernel = kernel, SimParticles = particles, SimLogger = logger,
                    SimDensityDiffusion = LinearDensityDiffusion(), SimViscosity = ArtificialViscosity(),
                    SimTimeStepping = single ? SingleNeighborTimeStepping() : SymplecticTimeStepping())
            end
        finally
            close(logger.LoggerIo)
        end
        order = sortperm(particles.ID)
        fields = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type))
        @test length(particles) > 256
        @test result.configuration.execution.graph_capture_failures == 0
        @test (result.configuration.execution.graph_batches > 0) == graph
        @test all(isfinite, particles.Density) && all(>(0), particles.Density)
        return (; fields, time = meta.TotalTime, steps = meta.Iteration,
            diagnostics = read(joinpath(save, "BatchCommit_Timestep.csv"), String))
    end
end

@testset "batched commit preserves graph stops, motion, diagnostics and integration" begin
    for D in (2, 3), T in (Float32, Float64), single in (false, true), compact in (false, true), moving in (false, true)
        reference = simulate(D, T; graph = false, single, compact, moving)
        actual = simulate(D, T; graph = true, single, compact, moving)
        @test isequal(actual, reference)
    end
end

@testset "floating placement completes before the next timestep" begin
    for D in (2, 3), T in (Float32, Float64), single in (false, true)
        reference = simulate(D, T; graph = false, single, compact = true, moving = false, floating = true)
        actual = simulate(D, T; graph = true, single, compact = true, moving = false, floating = true)
        @test isequal(actual, reference)
    end
end

end
