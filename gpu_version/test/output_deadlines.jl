module OutputDeadlineTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5, CSV
using SPHExampleGPU.GPUStepState
using SPHExampleGPU.GPUKernels: launch_finish!, launch_commit!
using SPHExampleGPU.GPUReductions: ReductionWorkspace

const STEP = SPHExampleGPU.GPUStepState

@testset "output deadline option defaults to full adaptive steps" begin
    for T in (Float32, Float64)
        meta = SimulationMetaData{2,T}(SimulationName = "Deadlines", SaveLocation = tempdir())
        @test !meta.GPUClipTimeStepToOutput
        exact = SimulationMetaData{2,T}(SimulationName = "Deadlines", SaveLocation = tempdir(),
            GPUClipTimeStepToOutput = true)
        @test exact.GPUClipTimeStepToOutput
    end
end

if CUDA.functional()
    @testset "device deadlines survive graph replay and rebuilds" begin
        for T in (Float32, Float64), graph in (false, true)
            constants = SimulationConstants{T}(dx = 0.02, c₀ = 20, CFL = 0.2)
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            reduction = ReductionWorkspace{SVector{3,T}}(32)
            fill!(reduction.partial, SVector{3,T}(0, 1, 0))
            natural_dt = constants.CFL * kernel.h / constants.c₀
            state = StepState{T}()
            deadline = T(2.5) * natural_dt
            set_output_time!(state, deadline; clip_timestep = true)
            function step!()
                launch_finish!(state, reduction, kernel, constants)
                launch_commit!(state)
            end
            step!() # Warm launches before capture; the first step is not clipped.
            readback!(state)
            @test state.fh[STEP.F_DT] ≈ natural_dt rtol = eps(T)
            executable = graph ? CUDA.instantiate(CUDA.capture(step!)) : nothing
            advance!() = graph ? CUDA.launch(executable) : step!()
            for _ in 1:8
                advance!()
            end
            readback!(state)
            @test state.fh[STEP.F_TIME] == deadline
            @test 0 < state.fh[STEP.F_DT] < natural_dt
            @test state.ih[STEP.I_ITER] == 3
            @test state.ih[STEP.I_STOP] == STOP_OUTPUT
            @test state.ih[STEP.I_PHASE] == PHASE_NEED_DT

            # The same executable must observe a changed option. At equality,
            # the default still takes a full step, while clipping takes none.
            set_output_time!(state, deadline)
            advance!()
            advance!()
            readback!(state)
            @test state.fh[STEP.F_TIME] > deadline
            @test state.ih[STEP.I_ITER] == 4
            @test state.ih[STEP.I_STOP] == STOP_OUTPUT
            set_output_time!(state, state.fh[STEP.F_TIME]; clip_timestep = true)
            advance!()
            readback!(state)
            @test state.ih[STEP.I_ITER] == 4
            @test state.ih[STEP.I_STOP] == STOP_OUTPUT

            # A deadline one representable instant away must advance once,
            # without an epsilon tolerance that drops a real step.
            deadline = nextfloat(state.fh[STEP.F_TIME])
            set_output_time!(state, deadline; clip_timestep = true)
            advance!()
            advance!()
            readback!(state)
            @test state.fh[STEP.F_TIME] == deadline
            @test state.ih[STEP.I_ITER] == 5

            # A rebuild interrupts a clipped step before integration. Its dt
            # must survive the rebuild even if the reduction workspace changes.
            rebuilt = StepState{T}(dx = one(T) + kernel.h)
            deadline = natural_dt / T(3)
            set_output_time!(rebuilt, deadline; clip_timestep = true)
            launch_finish!(rebuilt, reduction, kernel, constants)
            readback!(rebuilt)
            @test rebuilt.ih[STEP.I_STOP] == STOP_REBUILD
            @test rebuilt.ih[STEP.I_PHASE] == PHASE_DT_READY
            @test rebuilt.fh[STEP.F_DT] == deadline
            fill!(reduction.partial, SVector{3,T}(100, 1, 0))
            resume_after_rebuild!(rebuilt)
            launch_finish!(rebuilt, reduction, kernel, constants)
            launch_commit!(rebuilt)
            readback!(rebuilt)
            @test rebuilt.fh[STEP.F_TIME] == deadline
            @test rebuilt.fh[STEP.F_DT] == deadline
            @test rebuilt.ih[STEP.I_ITER] == 1
        end
    end

    function simulate(::Type{T}; scheme, graph, single_file = true, clip = true, schedule = T(0.00045)) where {T}
        return mktempdir() do save
            body = [SVector{2,T}(x, y) for x in (0, 0.02), y in (1, 1.02)][:]
            geometry = [
                SPHGeometry{2,T}(body; Density = 1000, Type = Floating, GroupMarker = 1,
                    Floating = FloatingDetails{T}(RelativeWeight = 1.2)),
                SPHGeometry{2,T}([SVector{2,T}(3, 1)]; Density = 1000, Type = Fluid, GroupMarker = 2),
                SPHGeometry{2,T}([SVector{2,T}(6, 1)]; Density = 1000, Type = Moving, GroupMarker = 3,
                    Motion = MotionDetails{2,T}(Velocity = 0.1, Direction = SVector{2,T}(1, 0),
                        StartTime = -1, Duration = 2)),
            ]
            meta = SimulationMetaData{2,T}(SimulationName = "Deadlines", SaveLocation = save,
                SimulationTime = T(0.00131), OutputTimes = schedule,
                GPUClipTimeStepToOutput = clip, GPUUseGraph = graph, GPUMaxStepsPerSync = graph ? 32 : 1,
                GPUAsyncOutput = graph, GPUOutputQueueBytes = 1024, ExportSingleVTKHDF = single_file,
                VisualizeInParaview = false, OpenLogFile = false)
            constants = SimulationConstants{T}(dx = 0.02, c₀ = 20, CFL = 0.2)
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            particles = AllocateDataStructures(geometry, meta)
            logger = SimulationLogger(save)
            try
                redirect_stdout(devnull) do
                    RunSimulation(SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
                        SimKernel = kernel, SimParticles = particles, SimLogger = logger,
                        SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion(),
                        SimTimeStepping = scheme)
                end
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            files = single_file ? ["Deadlines.vtkhdf"] :
                sort(filter(name -> occursin(r"^Deadlines_\d+\.vtkhdf$", name), readdir(save)))
            times = Float64[]
            positions = Matrix{Float64}[]
            # The CSV uses Float32's shortest round-trip decimal spelling.
            floating_times = Float64[T(row.Time) for row in CSV.File(joinpath(save, "Deadlines_Floating.csv"))]
            for (index, name) in enumerate(files)
                h5open(joinpath(save, name), "r") do file
                    root = file["VTKHDF"]
                    # Numbered files contain static datasets; the body log
                    # supplies their frame times. Combined files store times.
                    append!(times, single_file ? read(root["Steps/Values"]) : [floating_times[index]])
                    points = read(root["Points"])
                    ids = read(root["PointData/ID"])
                    n = length(particles)
                    for start in 1:n:length(ids)
                        columns = start:(start + n - 1)
                        push!(positions, points[:, columns[sortperm(ids[columns])]])
                    end
                end
            end
            @test times == floating_times
            order = sortperm(particles.ID)
            return (; times, points = hcat(positions...), steps = meta.Iteration, time = meta.TotalTime,
                position = particles.Position[order], velocity = particles.Velocity[order], density = particles.Density[order])
        end
    end

    @testset "exact saved states and step counts in both integration schemes" begin
        for T in (Float32, Float64), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            reference = simulate(T; scheme, graph = false)
            expected = Float64.([zero(T), T(0.00045), T(0.00045) * 2, T(0.00131)])
            @test reference.times == expected
            @test reference.time == T(0.00131)
            @test reference.steps == 6
            for single_file in (false, true)
                result = simulate(T; scheme, graph = true, single_file)
                @test result == reference
            end
            # All saved positions must be integrated at their own timestamp:
            # the body and isolated fluid fall, and the moving wall translates.
            n = length(reference.position)
            for (frame, time) in enumerate(reference.times)
                columns = (frame - 1) * n .+ (1:n)
                heights = sort(reference.points[2, columns[1:5]])
                expected_heights = sort([1, 1, 1, 1.02, 1.02] .- 9.81 * time^2 / 2)
                @test heights ≈ expected_heights atol = 10eps(T) rtol = 0
                @test reference.points[1, columns[6]] ≈ 6 + 0.1time atol = 10eps(T) rtol = 0
            end

            sparse = simulate(T; scheme, graph = true, schedule = T[0.00131, 0.002])
            @test sparse.times == Float64[0, T(0.00131)]
            @test sparse.steps == 4
            vector = simulate(T; scheme, graph = true, schedule = T[0.00045, 0.0009])
            @test vector == reference # The driver supplies the final deadline.
            uncut = simulate(T; scheme, graph = true, clip = false)
            @test uncut.time > T(0.00131)
            @test uncut.steps == 4
            @test all(uncut.times[2:end] .> expected[2:end])
        end
    end
else
    @testset "exact output stepping requires CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
