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
        result = simulate(case; graph = true, batch = 32, double_position, scheme)
        @test result.steps == reference.steps
        @test result.time == reference.time
        @test result.position == reference.position
        @test result.density == reference.density
        @test result.velocity == reference.velocity
        @test result.output == reference.output
    end
end

end
