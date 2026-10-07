using SPHExampleGPU
using CUDA
using TimerOutputs
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))

"""Benchmark the generated StillWedge case without changing its physics or outputs."""
function run_wedge(; lanes = 0, threads = 128, batch = 32, graph = true,
                   simtime = 0.5, subdivision = 1, boundary_forces = true,
                   queue_bytes = 256 * 1024^2)
    constants = SimulationConstants{Float32}(
        dx = 0.02, c₀ = 42.48576250492629, δᵩ = 0.1, CFL = 0.5)
    polygons = still_wedge_2d_polygons()
    sampled = sample_particles([
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ], constants.dx)
    geometry = [
        SPHGeometry{2, Float32}(sampled[1].positions;
            Density = constants.ρ₀, GroupMarker = 1, Type = Fixed),
        SPHGeometry{2, Float32}(sampled[2].positions;
            Density = hydrostatic_density(sampled[2].positions, constants),
            GroupMarker = 2, Type = Fluid),
    ]
    save = mktempdir()
    meta = SimulationMetaData{2, Float32, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(
        SimulationName = "StillWedge", SaveLocation = save,
        SimulationTime = simtime, OutputTimes = 0.01,
        VisualizeInParaview = false, OpenLogFile = false,
        ExportSingleVTKHDF = true, ExportGridCells = true,
        GPUDoublePosition = true, GPULanesPerParticle = lanes,
        GPUInteractionThreads = threads, GPUMaxStepsPerSync = batch,
        GPUUseGraph = graph, GPUCellSubdivision = subdivision,
        GPUBoundaryForces = boundary_forces, GPUOutputQueueBytes = queue_bytes)
    particles = AllocateDataStructures(geometry, meta)
    logger = SimulationLogger(save)
    elapsed = try
        redirect_stdout(devnull) do
            @elapsed RunSimulation(
                SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
                SimKernel = SPHKernelInstance{2, Float32}(WendlandC2(); dx = constants.dx),
                SimLogger = logger, SimParticles = particles,
                SimViscosity = ArtificialViscosity(),
                SimDensityDiffusion = LinearDensityDiffusion(),
                SimTimeStepping = SymplecticTimeStepping(), ParticleNormalsPath = nothing)
        end
    finally
        rm(save; recursive = true, force = true)
    end
    order = sortperm(particles.ID)
    loop = TimerOutputs.time(meta.HourGlass["00 SimulationLoop"]) / 1e9
    return (; elapsed, loop, steps = meta.Iteration, time = meta.TotalTime,
            position = particles.Position[order], density = particles.Density[order],
            velocity = particles.Velocity[order])
end

function main()
    settings = [
        (; lanes = 0, threads = 128),
        (; lanes = 4, threads = 128),
        (; lanes = 8, threads = 128),
        (; lanes = 16, threads = 128),
        (; lanes = 32, threads = 256),
        (; lanes = 32, threads = 64),
        (; lanes = 32, threads = 128, subdivision = 2),
    ]
    for setting in settings
        run_wedge(; setting..., simtime = 0.001)
    end
    reference = run_wedge()
    for repeat in 1:3, setting in settings
        result = run_wedge(; setting...)
        @printf("repeat=%d %-65s wall=%.4fs loop=%.4fs steps=%d position_error=%.3g density_error=%.3g\n",
                repeat, string(setting), result.elapsed, result.loop, result.steps,
                maximum(norm.(result.position .- reference.position)),
                maximum(abs.(result.density .- reference.density)))
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
