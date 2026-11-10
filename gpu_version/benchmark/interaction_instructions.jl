# Paired complete runs, including output, against the pre-optimization source.
# julia --project=gpu_version -t 2,0 gpu_version/benchmark/interaction_instructions.jl
module InteractionInstructionBenchmark

using SPHExampleGPU, CUDA, Statistics, Printf, LinearAlgebra, TimerOutputs, HDF5, CSV, SHA

include(joinpath(@__DIR__, "cases.jl"))

const BASELINE = "aa91e13"
const SOURCE = normpath(joinpath(@__DIR__, "..", "src"))

function baseline_modules()
    names = Dict(name => "InstructionBaseline" * name for name in
        ("SPHDensityDiffusionModels", "GPUKernels", "SPHCellList"))
    function source(file)
        text = replace(read(`git -C $REPO_ROOT show $(BASELINE * ":gpu_version/src/" * file)`, String), "\r\n" => "\n")
        # Resolve included driver/kernel fragments from the same revision.
        return replace(text, r"include\(\"([^\"]+)\"\)" =>
            m -> source(match(r"\"([^\"]+)\"", m)[1]))
    end
    for name in ("SPHDensityDiffusionModels", "GPUKernels", "SPHCellList")
        text = replace(source(name * ".jl"), "module $name" => "module $(names[name])")
        for (original, renamed) in names
            text = replace(text, "using ..$original" => "using ..$renamed", "$original." => "$renamed.")
        end
        Base.include_string(SPHExampleGPU, text, joinpath(SOURCE, name * ".jl"))
    end
    return (; driver = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(names["SPHCellList"])),
        diffusion = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(names["SPHDensityDiffusionModels"])),
        kernels = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(names["GPUKernels"])))
end

function measure(variant, case, duration; strict = false)
    return mktempdir() do save
        kw = case.build(Float32, save)
        model = kw.SimDensityDiffusion
        kw = merge(kw, (; SimDensityDiffusion = getproperty(variant.diffusion, nameof(typeof(model)))(
            (getfield(model, i) for i in 1:fieldcount(typeof(model)))...)))
        meta = kw.SimMetaData
        meta.SimulationTime = Float32(duration)
        meta.OutputTimes = Float32(min(0.01, duration))
        meta.GPUDoublePosition = true
        meta.GPUCellSubdivision = 1
        meta.GPURadialInteractions = false
        meta.GPUScalarSPSViscosity = false
        meta.GPUCacheInteractionFields = false
        meta.GPUInteractionFastMath = !strict
        meta.GPUUseGraph = true
        meta.GPUMaxStepsPerSync = 32
        meta.OutputVariables = ["Density", "Pressure", "Velocity", "Acceleration", "ID", "Type"]
        host = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        GC.gc()
        CUDA.synchronize()
        seconds = try
            redirect_stdout(devnull) do
                @elapsed begin
                    variant.driver.RunSimulation(; kw..., SimParticles = host, SimLogger = logger)
                    CUDA.synchronize()
                end
            end
        finally
            close(logger.LoggerIo)
        end
        order = sortperm(host.ID)
        state = map(n -> getproperty(host, n)[order], (:Position, :Velocity, :Density, :Acceleration))
        all(field -> all(v -> v isa Number ? isfinite(v) : all(isfinite, v), field), state) ||
            error("Nonfinite particle state")
        all(>(0), state[3]) || error("Nonpositive particle density")
        output = h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do file
            root = file["VTKHDF"]
            datasets = Dict{String,Tuple{DataType,Tuple,String}}()
            for path in ["Points"; ["PointData/" * name for name in keys(root["PointData"])]]
                data = read(root[path])
                datasets[path] = (eltype(data), size(data), bytes2hex(sha256(reinterpret(UInt8, vec(data)))))
            end
            (; times = read(root["Steps/Values"]), counts = read(root["NumberOfPoints"]),
                datasets)
        end
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration, rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9, state, work, time = meta.TotalTime,
            ids = host.ID[order], types = host.Type[order], output, particles = length(host))
    end
end

function compare(a, b)
    a.work == b.work || error("Step/rebuild/readback counts changed: $(a.work) versus $(b.work)")
    a.ids == b.ids && a.types == b.types || error("Particle identity/type changed")
    a.output == b.output || error("Output frame counts, timestamps, dataset shapes or values changed")
    errors = map((x, y) -> maximum(norm.(x .- y)), a.state, b.state)
    isequal(a.state, b.state) || error("Accepted particle state changed: maximum differences $errors")
    time_error = abs(a.time - b.time)
    a.time == b.time || error("Accepted physical time changed")
    a.output.times == b.output.times || error("Output timestamps changed")
    return (; position = errors[1], velocity = errors[2], density = errors[3], acceleration = errors[4],
        time_error, exact = true, exact_output = true)
end

function run_comparison(variants, cases; rounds, duration, long_duration, strict, output)
    records = NamedTuple[]
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; CUDA.jl=", pkgversion(CUDA),
        "; baseline=", BASELINE, "; Float32 physics / Float64 positions; R=1; fastmath=", !strict)
    for case in cases
        time = duration === nothing ? (case.dims == 3 ? 0.03 : 0.2) : duration
        for _ in 1:2
            warm = map(v -> measure(v, case, time; strict), variants)
            compare(warm...)
        end
        times = [Float64[] for _ in variants]
        for round in 1:rounds
            results = Vector{NamedTuple}(undef, length(variants))
            for i in (isodd(round) ? eachindex(variants) : reverse(eachindex(variants)))
                results[i] = measure(variants[i], case, time; strict)
                push!(times[i], results[i].seconds)
            end
            errors = compare(results[1], results[2])
            for i in eachindex(variants)
                r = results[i]
                push!(records, merge((; case = case.name, phase = "timed", strict, round, variant = i == 1 ? "baseline" : "optimized",
                    particles = r.particles, duration = time, seconds = r.seconds, loop_seconds = r.loop_seconds),
                    r.work, (; time = r.time, frames = length(r.output.times)), errors))
            end
            CSV.write(output, records)
            @printf("%s round=%d baseline=%.6fs optimized=%.6fs ratio=%.4f errors=%s\n",
                case.name, round, results[1].seconds, results[2].seconds,
                results[2].seconds / results[1].seconds, string(errors))
            flush(stdout)
        end
        ratios = times[2] ./ times[1]
        @printf("%s paired ratio median=%.4f range=[%.4f, %.4f]; runtime reduction=%.2f%%; faster=%d/%d\n",
            case.name, median(ratios), minimum(ratios), maximum(ratios),
            100 * (1 - median(ratios)), count(<(1), ratios), rounds)
        if long_duration !== nothing
            results = map(v -> measure(v, case, long_duration; strict), variants)
            errors = compare(results...)
            println(case.name, " long_time=", long_duration, " work=", results[1].work,
                " errors=", errors)
            for i in eachindex(variants)
                r = results[i]
                push!(records, merge((; case = case.name, phase = "validation", strict, round = 0,
                    variant = i == 1 ? "baseline" : "optimized", particles = r.particles,
                    duration = long_duration, seconds = r.seconds, loop_seconds = r.loop_seconds),
                    r.work, (; time = r.time, frames = length(r.output.times)), errors))
            end
            CSV.write(output, records)
        end
    end
    return nothing
end

function main(args)
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "7"))
    rounds >= 3 || throw(ArgumentError("At least three paired rounds are required"))
    duration = value("--time", nothing)
    duration = duration === nothing ? nothing : parse(Float64, duration)
    long_duration = value("--long-time", nothing)
    long_duration = long_duration === nothing ? nothing : parse(Float64, long_duration)
    all(t -> t === nothing || isfinite(t) && t > 0, (duration, long_duration)) ||
        throw(ArgumentError("Durations must be finite and positive"))
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(isempty(names) ? ["DamBreak3D_dp0.0085", "DamBreak3D_dp0.02", "StillWedge2D_MDBC_dp0.02"] : names)
    isempty(cases) && throw(ArgumentError("No matching benchmark cases"))
    output = value("--output", joinpath(@__DIR__, "interaction_instructions_results.csv"))
    baseline = baseline_modules()
    current = (; driver = SPHExampleGPU.SPHCellList, diffusion = SPHExampleGPU.SPHDensityDiffusionModels,
        kernels = SPHExampleGPU.GPUKernels)
    Base.invokelatest(run_comparison, (baseline, current), cases;
        rounds, duration, long_duration, strict = "--strict" in args, output)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    InteractionInstructionBenchmark.main(ARGS)
end
