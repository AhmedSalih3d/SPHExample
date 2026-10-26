module MotionSamplingBenchmark

using SPHExampleGPU, CUDA, StaticArrays, Statistics, Printf, HDF5, SHA, TimerOutputs

# Last driver before shared stage sampling. Read-only git objects keep the
# baseline independent of both working-tree edits and the current branch tip.
const BASELINE_COMMIT = "2067c583521dd06d317b56a33beb957f0889684b"
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function baseline_source(name)
    command = Cmd(Cmd(["git", "show", "$BASELINE_COMMIT:gpu_version/src/$name"]); dir = ROOT)
    return replace(read(command, String), "\r\n" => "\n")
end

function baseline_driver()
    rotation = baseline_source("PrescribedRotation.jl")
    start = first(findfirst("prepare_rotation(::Nothing", rotation))
    rotation = """
        module MotionSamplingBaselineRotation
        using CUDA, StaticArrays
        using ..SimulationGeometry: Moving
        using ..GPUStepState: step_active, step_time, step_dt
        using ..GPUKernels: store_pos_cell!
        using ..PrescribedRotation: RotationMotion, rotation_state, rotated_position
        """ * rotation[start:end]
    Base.include_string(SPHExampleGPU, rotation, @__FILE__)

    forces = baseline_source("ExternalForces.jl")
    start = first(findfirst("# A device-safe binary search", forces))
    forces = """
        module MotionSamplingBaselineForces
        using Adapt, CUDA, StaticArrays
        using LinearAlgebra: cross
        using ..SimulationGeometry: Fluid
        using ..GPUStepState: step_active, step_time, step_dt
        using ..ExternalForces: ExternalForce, embed, project
        """ * forces[start:end]
    Base.include_string(SPHExampleGPU, forces, @__FILE__)

    driver = replace(baseline_source("SPHCellList.jl"),
        "module SPHCellList" => "module MotionSamplingBaselineDriver",
        "using ..PrescribedRotation:" => "using ..MotionSamplingBaselineRotation:",
        "using ..ExternalForces:" => "using ..MotionSamplingBaselineForces:")
    Base.include_string(SPHExampleGPU, driver, @__FILE__)
    return Base.invokelatest(getproperty, SPHExampleGPU, :MotionSamplingBaselineDriver)
end

const BASELINE = baseline_driver()

let path = joinpath(@__DIR__, "..", "example", "SloshingTank.jl")
    source = first(split(read(path, String), "# Including this file starts the simulation."))
    include_string(@__MODULE__, source, path)
end

function fingerprint(path)
    datasets = Dict{String,Tuple{DataType,Tuple,String}}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            key = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, key)
            else
                data = read(item)
                datasets[key] = (eltype(data), size(data), bytes2hex(sha256(reinterpret(UInt8, vec(data)))))
            end
        end
    end
    h5open(file -> visit(file["VTKHDF"], ""), path, "r")
    return datasets
end

function measure(driver, mode, ::Type{T}; dx, duration, graph, single, double_position) where {T}
    mktempdir() do save
        case = sloshing_tank_geometry(; dx, FloatType = T, driving_mode = mode)
        rotation = mode == :rotation ? sloshing_tank_rotation() : nothing
        force = mode == :acceleration ? sloshing_tank_force(T) : nothing
        meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName = "MotionSampling", SaveLocation = save,
            SimulationTime = duration, OutputTimes = duration / 3,
            VisualizeInParaview = false, OpenLogFile = false,
            GPUUseGraph = graph, GPUDoublePosition = double_position,
            GPUCellSubdivision = 1, GPUDeterministicSort = true,
            GPUSyncTimers = false, ExportGridCells = false)
        particles = AllocateDataStructures(case.geometry, meta)
        logger = SimulationLogger(save)
        GC.gc()
        CUDA.synchronize()
        result = nothing
        elapsed = try
            redirect_stdout(devnull) do
                @elapsed begin
                    result = driver.RunSimulation(SimGeometry = case.geometry, SimMetaData = meta,
                        SimParticles = particles, SimConstants = case.constants, SimKernel = case.kernel,
                        SimLogger = logger, SimTimeStepping = single ? SingleNeighborTimeStepping() : SymplecticTimeStepping(),
                        SimViscosity = ArtificialViscosity(), SimDensityDiffusion = ComplexDensityDiffusion(),
                        SimRotation = rotation, SimExternalForces = force)
                    CUDA.synchronize()
                end
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        result.final_state.complete || error("Incomplete simulation")
        result.configuration.execution.graph_capture_failures == 0 || error("Graph capture failed")
        graph && result.configuration.execution.graph_batches == 0 && error("No graph replay")
        order = sortperm(particles.ID)
        fields = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker))
        all(isfinite, particles.Density) && all(>(0), particles.Density) || error("Invalid density")
        all(p -> all(isfinite, p), particles.Position) || error("Invalid position")
        all(v -> all(isfinite, v), particles.Velocity) || error("Invalid velocity")
        d = result.diagnostics
        work = (; steps = meta.Iteration, time = meta.TotalTime, rebuilds = d.cell_list_rebuilds,
            readbacks = d.step_state_readbacks, frames = d.output_frames, frozen = d.frozen_particles)
        return (; elapsed, loop = TimerOutputs.time(meta.HourGlass["00 SimulationLoop"]) / 1e9,
            work, fields, output = fingerprint(joinpath(save, "MotionSampling.vtkhdf")),
            particles = length(particles))
    end
end

function equivalent(a, b)
    a.work == b.work || error("Step/rebuild/readback/output counts or time changed: $(a.work) vs $(b.work)")
    isequal(a.fields, b.fields) || error("Accepted particle state changed")
    isequal(a.output, b.output) || error("Exported datasets changed")
    return nothing
end

function main(args = ARGS)
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "9"))
    duration = parse(Float64, value("--duration", "0.1"))
    dx = parse(Float64, value("--dx", "0.002"))
    rounds >= 3 || throw(ArgumentError("At least three alternating pairs are required"))
    isfinite(duration) && 0 < duration <= 8.35 || throw(ArgumentError("Duration must be in (0, 8.35]"))
    T = "--float64" in args ? Float64 : Float32
    graph, single = !("--no-graph" in args), "--single-neighbor" in args
    double_position = "--double-position" in args
    modes = Symbol.(split(value("--modes", "rotation,acceleration"), ','))
    all(in((:rotation, :acceleration)), modes) || throw(ArgumentError("Unknown driving mode"))
    output = value("--output", joinpath(@__DIR__, "motion_sampling_results.csv"))
    drivers = (BASELINE, SPHExampleGPU.SPHCellList)
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; baseline=", BASELINE_COMMIT,
        "; physics=", T, "; double_position=", double_position, "; graph=", graph,
        "; single=", single, "; dx=", dx, "; duration=", duration,
        "; shared_forces=true",
        "; R=1; deterministic sorting; 167001 history samples; unsynchronized timers")
    open(output, "w") do io
        println(io, "mode,round,variant,total_s,loop_s,particles,steps,rebuilds,readbacks,frames,time,exact_state,exact_output")
        for mode in modes
            for _ in 1:2
                warm = map(driver -> measure(driver, mode, T; dx, duration, graph, single, double_position), drivers)
                equivalent(warm...)
                println("warm mode=", mode, " particles=", warm[1].particles, " work=", warm[1].work, " exact=true")
                flush(stdout)
            end
            times, loops = [Float64[], Float64[]], [Float64[], Float64[]]
            for round in 1:rounds
                indices = isodd(round) ? (1, 2) : (2, 1)
                results = Dict(i => measure(drivers[i], mode, T; dx, duration, graph, single, double_position) for i in indices)
                equivalent(results[1], results[2])
                for i in 1:2
                    r, w = results[i], results[i].work
                    push!(times[i], r.elapsed)
                    push!(loops[i], r.loop)
                    println(io, join((mode, round, i == 1 ? "particle" : "stage", r.elapsed, r.loop,
                        r.particles, w.steps, w.rebuilds, w.readbacks, w.frames, w.time, true, true), ','))
                end
                flush(io)
                @printf("mode=%s pair=%d particle=%.6fs stage=%.6fs ratio=%.4f exact=true\n",
                    mode, round, results[1].elapsed, results[2].elapsed, results[2].elapsed / results[1].elapsed)
                flush(stdout)
            end
            ratios = times[2] ./ times[1]
            @printf("mode=%s complete medians particle=%.6fs stage=%.6fs paired_ratio=%.4f IQR=[%.4f,%.4f] range=[%.4f,%.4f] faster=%d/%d loop_ratio=%.4f\n",
                mode, median(times[1]), median(times[2]), median(ratios), quantile(ratios, 0.25), quantile(ratios, 0.75),
                minimum(ratios), maximum(ratios), count(<(1), ratios), rounds, median(loops[2] ./ loops[1]))
            flush(stdout)
        end
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    MotionSamplingBenchmark.main()
end
