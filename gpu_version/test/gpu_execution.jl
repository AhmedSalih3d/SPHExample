module GPUExecutionTests

using Test
using SPHExampleGPU
using HDF5

include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

function simulate(case; graph, batch, double_position, scheme)
    return mktempdir() do save
        kwargs = case.build(Float32, save)
        meta = kwargs.SimMetaData
        meta.SimulationTime = 0.06f0
        meta.OutputTimes = 0.01f0
        meta.GPUUseGraph = graph
        meta.GPUMaxStepsPerSync = batch
        meta.GPUDoublePosition = double_position
        meta.GPUOutputQueueBytes = 1024^2
        particles = AllocateDataStructures(kwargs.SimGeometry, meta)
        logger = SimulationLogger(save)
        redirect_stdout(devnull) do
            RunSimulation(; merge(kwargs, (; SimTimeStepping = scheme))...,
                SimLogger = logger, SimParticles = particles)
        end
        order = sortperm(particles.ID)
        output = h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do file
            root = file["VTKHDF"]
            (; points = read(root["Points"]), times = read(root["Steps/Values"]),
               density = read(root["PointData/Density"]))
        end
        return (; steps = meta.Iteration, time = meta.TotalTime,
                position = particles.Position[order], density = particles.Density[order],
                velocity = particles.Velocity[order], output)
    end
end

@testset "batched graphs preserve integration and saved frames" begin
    for index in (1, 3, 4), double_position in (false, true),
        scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        case = BENCH_CASES[index]
        reference = simulate(case; graph = false, batch = 1, double_position, scheme)
        result = simulate(case; graph = true, batch = 128, double_position, scheme)
        @test result.steps == reference.steps
        @test result.time == reference.time
        @test result.position == reference.position
        @test result.density == reference.density
        @test result.velocity == reference.velocity
        @test result.output == reference.output
    end
end

# Both populations stay separated spatially: the analytic trajectories test
# motion ordering and rigid placement when final integration commits in one block.
function tiny_motion_and_floating(T, save)
    body = [SVector{2,T}(x, y) for x in (0, 0.02), y in (1, 1.02)][:]
    wall = [SVector{2,T}(2, 1), SVector{2,T}(2, 1.02)]
    geometry = [SPHGeometry{2,T}(body; Density = 1000, Type = Floating, GroupMarker = 1,
                    Floating = FloatingDetails{T}(RelativeWeight = 1.2)),
                SPHGeometry{2,T}(wall; Density = 1000, Type = Moving, GroupMarker = 2,
                    Motion = MotionDetails{2,T}(Velocity = 0.1, Direction = SVector{2,T}(1, 0),
                        StartTime = -1, Duration = 2))]
    constants = SimulationConstants{T}(dx = 0.02, c₀ = 20, CFL = 0.2)
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
        SimulationName = "TinyMotionFloating", SaveLocation = save,
        VisualizeInParaview = false, OpenLogFile = false)
    return (; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
        SimKernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx),
        SimViscosity = Laminar(), SimDensityDiffusion = ZeroDensityDiffusion())
end

@testset "single-block commit with prescribed motion and floating bodies" begin
    case = BenchCase("TinyMotionFloating", 2, tiny_motion_and_floating)
    for double_position in (false, true), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        reference = simulate(case; graph = false, batch = 1, double_position, scheme)
        result = simulate(case; graph = true, batch = 128, double_position, scheme)
        @test result == reference
        @test result.position[1][2] ≈ 1 - 9.81f0 * result.time^2 / 2 atol = 2e-6
        # Each Float32 half increment rounds at x=2; bound accumulated ulps.
        motion_tolerance = double_position ? 2e-6 : result.steps * eps(2f0)
        @test result.position[5][1] ≈ 2 + 0.1f0 * result.time atol = motion_tolerance
        @test result.velocity[5] == SVector(0.1f0, 0f0)
    end
end

end
