using SPHExampleGPU
using CUDA
using TimerOutputs
using Printf
using LinearAlgebra
using Statistics
using HDF5

let path = joinpath(@__DIR__, "..", "example", "StillWedgeMDBC.jl")
    source = read(path, String)
    source = replace(source, r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n")
    include_string(@__MODULE__, source, path)
end

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

function run_example(time_stepping; simtime = 4.0)
    return mktempdir() do save
        result = nothing
        elapsed = redirect_stdout(devnull) do
            @elapsed result = run_still_wedge(; time_stepping, simulation_time = simtime,
                save_location = save, visualize = false, open_log_file = false)
        end
        particles, meta = result.particles, result.metadata
        order = sortperm(particles.ID)
        output = h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do file
            root = file["VTKHDF"]
            (; times = read(root["Steps/Values"]),
               points = size(root["Points"]),
               fields = Dict(name => size(root["PointData"][name]) for name in keys(root["PointData"])))
        end
        grid_output = h5open(joinpath(save, meta.SimulationName * "_GridCells.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            (; times = read(root["Steps/Values"]), points = size(root["Points"]),
               cells = read(root["NumberOfCells"]))
        end
        c₀ = Float32(42.48576250492629)
        ρ₀ = 1000f0
        pressures = [particle_pressure(particles.Density[i], particles.Type[i], c₀, ρ₀) for i in order]
        return (; elapsed, loop = TimerOutputs.time(meta.HourGlass["00 SimulationLoop"]) / 1e9,
            steps = meta.Iteration, time = meta.TotalTime, output, grid_output,
            position = particles.Position[order], density = particles.Density[order],
            velocity = particles.Velocity[order], pressure = pressures,
            acceleration = particles.Acceleration[order])
    end
end

function compare_schemes(; rounds = 5, minimum_speedup = nothing,
                         output_path = joinpath(@__DIR__, "still_wedge_scheme_results.csv"))
    schemes = (SymplecticTimeStepping(), SingleNeighborTimeStepping())
    warmups = map(run_example, schemes)
    times = [Float64[], Float64[]]
    loops = [Float64[], Float64[]]
    open(output_path, "w") do out
        println(out, "round,scheme,total_s,loop_s,steps,final_time,particle_frames,grid_frames,position_abs,velocity_abs,density_abs,pressure_abs,acceleration_abs")
        for round in 1:rounds
            results = Vector{typeof(warmups[1])}(undef, 2)
            for index in (isodd(round) ? (1, 2) : (2, 1))
                results[index] = run_example(schemes[index])
            end
            reference = results[1]
            for index in 1:2
                result = results[index]
                push!(times[index], result.elapsed)
                push!(loops[index], result.loop)
                result.steps == reference.steps || error("schemes took different numbers of steps")
                result.output.points == reference.output.points || error("output point shapes differ")
                result.output.fields == reference.output.fields || error("output field shapes differ")
                length(result.output.times) == 401 || error("four-second run must export 401 frames")
                all(isfinite, result.output.times) && all(>(0), diff(result.output.times)) ||
                    error("output times must be finite and increasing")
                length(result.grid_output.times) == 400 && length(result.grid_output.cells) == 400 ||
                    error("four-second run must export 400 nonempty grid frames")
                result.grid_output.times == result.output.times[2:end] || error("particle and grid frame times differ")
                result.grid_output.points[1] == 3 && all(>=(0), result.grid_output.cells) ||
                    error("invalid exported grid dimensions or cell counts")
                all(>(0), result.density) || error("nonpositive final density")
                for field in (result.position, result.velocity, result.acceleration)
                    all(value -> all(isfinite, value), field) || error("nonfinite final vector field")
                end
                all(isfinite, result.density) && all(isfinite, result.pressure) ||
                    error("nonfinite final scalar field")
                differences = (
                    maximum(norm.(result.position .- reference.position)),
                    maximum(norm.(result.velocity .- reference.velocity)),
                    maximum(abs.(result.density .- reference.density)),
                    maximum(abs.(result.pressure .- reference.pressure)),
                    maximum(norm.(result.acceleration .- reference.acceleration)),
                )
                println(out, join((round, nameof(typeof(schemes[index])), result.elapsed, result.loop,
                    result.steps, result.time, length(result.output.times), length(result.grid_output.times), differences...), ','))
            end
            @printf("round=%d symplectic=%.6f s single-neighbor=%.6f s speedup=%.3fx\n",
                round, times[1][end], times[2][end], times[1][end] / times[2][end])
            flush(out)
            flush(stdout)
        end
    end
    speedup = median(times[1] ./ times[2])
    @printf("Median complete simulation %.6f -> %.6f s; loop %.6f -> %.6f s; paired speedup %.3fx (%.1f%% less time)\n",
        median(times[1]), median(times[2]), median(loops[1]), median(loops[2]), speedup,
        100 * (1 - median(times[2] ./ times[1])))
    println("Different integration schemes: numerical trajectories are not identical. Results: ", output_path)
    minimum_speedup === nothing || speedup >= minimum_speedup ||
        error("measured speedup $speedup is below the required $minimum_speedup")
    return (; speedup, symplectic = median(times[1]), single_neighbor = median(times[2]))
end

function main(args = ARGS)
    if "--compare-schemes" in args
        all(arg -> arg in ("--compare-schemes", "--assert-1.5x-throughput"), args) ||
            throw(ArgumentError("unknown scheme-comparison option"))
        compare_schemes(; minimum_speedup = "--assert-1.5x-throughput" in args ? 1.5 : nothing)
        return
    end
    isempty(args) || throw(ArgumentError("expected --compare-schemes [--assert-1.5x-throughput]"))
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
