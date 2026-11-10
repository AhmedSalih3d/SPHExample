# Experimental modules only: including this file never changes the production solver.
# julia --project=gpu_version -t 2,0 gpu_version\benchmark\preloaded_types.jl
module PreloadedTypeBenchmark

using SPHExampleGPU, CUDA, LinearAlgebra, Statistics, Printf, TimerOutputs, HDF5, CSV

include(joinpath(@__DIR__, "cases.jl"))

const SOURCE = normpath(joinpath(@__DIR__, "..", "src"))
const CURRENT = (; name = "current", driver = SPHExampleGPU.SPHCellList, strict = false, perturb = false)

function replace_once(source, old, new)
    count(old, source) == 1 || error("Expected exactly one source marker: $old")
    return replace(source, old => new; count = 1)
end

function preload_source(source, diffusion_source)
    start = findfirst("Base.@propagate_inbounds function compute_density_diffusion(\n        ::LinearDensityDiffusion,", diffusion_source)
    start === nothing && error("Cannot locate the reference linear diffusion implementation")
    body = diffusion_source[first(start):end]
    stop = findfirst("\nend", body)
    stop === nothing && error("Cannot locate the end of linear diffusion")
    body = body[1:last(stop)]
    body = replace_once(body, "Base.@propagate_inbounds function compute_density_diffusion(",
        "@inline function preloaded_diffusion(")
    body = replace_once(body, "        ParticleType\n)", "        ParticleType,\n        both_fluid::Bool\n)")
    body = replace_once(body, "        both_fluid = (ParticleType[i] == Fluid) & (ParticleType[j] == Fluid)\n", "")
    body = replace_once(body, "linear_density_difference(", "SPHDensityDiffusionModels.linear_density_difference(")
    fallback = """
    @inline preloaded_diffusion(model, kernel, constants, particles, x, gradient, d2, ra, rb,
                                ira, irb, i, j, types, both_fluid) =
        compute_density_diffusion(model, kernel, constants, particles, x, gradient, d2, ra, rb, ira, irb, i, j, types)
    """
    source = replace_once(source, "function interaction_kernel!", body * "\n" * fallback * "\nfunction interaction_kernel!")
    source = replace_once(source, "D1, D2 = compute_density_diffusion(SimDensityDiffusion",
        "D1, D2 = preloaded_diffusion(SimDensityDiffusion")
    return replace_once(source, "ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja, ParticleType)",
        "ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja, ParticleType, (typeᵢ == Fluid) & (ParticleType[j] == Fluid))")
end

function experimental_modules()
    isdefined(SPHExampleGPU, :PreloadedTypeDriver) &&
        error("Experimental modules are already loaded; start a fresh Julia process")
    read_source(file) = replace(read(`git -C $REPO_ROOT show $("3473acb:gpu_version/src/" * file)`, String), "\r\n" => "\n")
    function includes(text)
        return replace(text, r"include\(\"([^\"]+)\"\)" =>
            m -> includes(read_source(match(r"\"([^\"]+)\"", m)[1])))
    end
    kernels = preload_source(read_source("GPUKernels.jl"), read_source("SPHDensityDiffusionModels.jl"))
    kernels = replace_once(includes(kernels), "module GPUKernels", "module PreloadedTypeKernels")
    driver = replace_once(includes(read_source("SPHCellList.jl")), "module SPHCellList", "module PreloadedTypeDriver")
    driver = replace(driver, "using ..GPUKernels" => "using ..PreloadedTypeKernels",
        "GPUKernels." => "PreloadedTypeKernels.")
    Base.include_string(SPHExampleGPU, kernels, joinpath(SOURCE, "GPUKernels.jl"))
    Base.include_string(SPHExampleGPU, driver, joinpath(SOURCE, "SPHCellList.jl"))
    return (; kernels = Base.invokelatest(getproperty, SPHExampleGPU, :PreloadedTypeKernels),
        driver = Base.invokelatest(getproperty, SPHExampleGPU, :PreloadedTypeDriver))
end

function frame_metrics(position, velocity, density, pressure, fluid, mass, vertical_axis)
    p = Float64.(position[:, fluid])
    v = Float64.(velocity[:, fluid])
    rho = Float64.(density[fluid])
    pres = Float64.(pressure[fluid])
    com = vec(mean(p; dims = 2))
    momentum = mass .* vec(sum(v; dims = 2))
    return (; fluid_count = length(fluid), fluid_mass = mass * length(fluid),
        com_x = com[1], com_y = com[2], com_z = com[3],
        momentum_x = momentum[1], momentum_y = momentum[2], momentum_z = momentum[3],
        kinetic_energy = 0.5 * mass * sum(abs2, v),
        density_mean = mean(rho), density_min = minimum(rho), density_max = maximum(rho),
        pressure_rms = sqrt(mean(abs2, pres)), pressure_max = maximum(pres),
        front_x = maximum(p[1, :]), front_x99 = quantile(p[1, :], 0.99),
        surface_max = maximum(p[vertical_axis, :]))
end

function read_frames(save, meta, constants, dims)
    return h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do file
        root = file["VTKHDF"]
        counts = Int.(read(root["NumberOfPoints"]))
        offsets = Int.(read(root["Steps/PointOffsets"]))
        times = read(root["Steps/Values"])
        length(counts) == length(offsets) == length(times) || error("Inconsistent output frame coverage")
        frames = NamedTuple[]
        for index in eachindex(times)
            columns = offsets[index] + 1:offsets[index] + counts[index]
            ids = root["PointData/ID"][columns]
            length(unique(ids)) == length(ids) || error("Duplicate particle IDs")
            order = sortperm(ids)
            types = root["PointData/Type"][columns][order]
            position = root["Points"][:, columns][:, order]
            velocity = root["PointData/Velocity"][:, columns][:, order]
            density = root["PointData/Density"][columns][order]
            pressure = root["PointData/Pressure"][columns][order]
            all(a -> all(isfinite, a), (position, velocity, density, pressure)) || error("Nonfinite frame")
            all(>(0), density) || error("Nonpositive density")
            fluid = findall(==(UInt8(Fluid)), types)
            isempty(fluid) && error("No fluid particles")
            push!(frames, (; time = times[index], ids = ids[order], types, position, velocity, density, pressure,
                metrics = frame_metrics(position, velocity, density, pressure, fluid, Float64(constants.m₀), dims)))
        end
        return frames
    end
end

function measure(variant, case, duration; interval = 0.01, precision = Float32)
    return mktempdir() do save
        kw = case.build(precision, save)
        kw.SimDensityDiffusion isa LinearDensityDiffusion ||
            throw(ArgumentError("This experiment measures reference LinearDensityDiffusion only"))
        meta = kw.SimMetaData
        meta.SimulationTime = precision(duration)
        meta.OutputTimes = precision(min(interval, duration))
        meta.GPUDoublePosition = true
        meta.GPUCellSubdivision = 1
        meta.GPURadialInteractions = false
        meta.GPUScalarSPSViscosity = false
        meta.GPUCacheInteractionFields = false
        meta.GPUInteractionFastMath = !variant.strict
        meta.GPUPreloadInteractionTypes = get(variant, :preload, false)
        meta.GPUUseGraph = true
        meta.GPUMaxStepsPerSync = 32
        meta.OutputVariables = ["Density", "Pressure", "Velocity", "ID", "Type"]
        host = AllocateDataStructures(kw.SimGeometry, meta)
        if variant.perturb
            for i in eachindex(host.Density)
                host.Type[i] == Fluid && (host.Density[i] = nextfloat(host.Density[i]))
            end
        end
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
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration, rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9, work,
            frames = read_frames(save, meta, kw.SimConstants, case.dims), particles = length(host),
            dx = Float64(kw.SimConstants.dx), c0 = Float64(kw.SimConstants.c₀),
            rho0 = Float64(kw.SimConstants.ρ₀), gravity = abs(Float64(kw.SimConstants.g)))
    end
end

function compare_frames(a, b, run)
    a.ids == b.ids && a.types == b.types || error("Particle identities/types changed")
    fluid = findall(==(UInt8(Fluid)), a.types)
    dp = vec(sqrt.(sum(abs2, a.position[:, fluid] .- b.position[:, fluid]; dims = 1)))
    dv = vec(sqrt.(sum(abs2, Float64.(a.velocity[:, fluid]) .- Float64.(b.velocity[:, fluid]); dims = 1)))
    drho = Float64.(a.density[fluid]) .- Float64.(b.density[fluid])
    dpres = Float64.(a.pressure[fluid]) .- Float64.(b.pressure[fluid])
    metrics_a, metrics_b = a.metrics, b.metrics
    com_error = norm([getproperty(metrics_a, s) - getproperty(metrics_b, s) for s in (:com_x, :com_y, :com_z)])
    momentum_error = norm([getproperty(metrics_a, s) - getproperty(metrics_b, s)
        for s in (:momentum_x, :momentum_y, :momentum_z)])
    return (; time_delta = b.time - a.time,
        position_max_dx = maximum(dp) / run.dx, position_rms_dx = sqrt(mean(abs2, dp)) / run.dx,
        position_p99_dx = quantile(dp, 0.99) / run.dx,
        velocity_max_c0 = maximum(dv) / run.c0,
        density_max_rho0 = maximum(abs, drho) / run.rho0,
        pressure_rms_delta = sqrt(mean(abs2, dpres)), pressure_max_delta = maximum(abs, dpres),
        com_delta_dx = com_error / run.dx,
        momentum_delta = momentum_error,
        kinetic_delta = abs(metrics_a.kinetic_energy - metrics_b.kinetic_energy),
        kinetic_relative_delta = metrics_a.kinetic_energy == 0 ? missing :
            abs(metrics_a.kinetic_energy - metrics_b.kinetic_energy) / metrics_a.kinetic_energy,
        front_delta_dx = abs(metrics_a.front_x - metrics_b.front_x) / run.dx,
        front_x99_delta_dx = abs(metrics_a.front_x99 - metrics_b.front_x99) / run.dx,
        surface_delta_dx = abs(metrics_a.surface_max - metrics_b.surface_max) / run.dx,
        exact = isequal((a.position, a.velocity, a.density, a.pressure), (b.position, b.velocity, b.density, b.pressure)))
end

function compare_runs(a, b)
    length(a.frames) == length(b.frames) || error("Output frame count changed")
    all(f -> f.metrics.fluid_mass == first(a.frames).metrics.fluid_mass, [a.frames; b.frames]) ||
        error("Fluid mass/count changed")
    return [compare_frames(x, y, a) for (x, y) in zip(a.frames, b.frames)]
end

function run_experiment(candidate, cases; rounds, duration, timed_duration, output_prefix, controls)
    optimized = (; CURRENT..., name = "preloaded", driver = candidate.driver, preload = get(candidate, :preload, false))
    timing_rows, observation_rows, comparison_rows = NamedTuple[], NamedTuple[], NamedTuple[]
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; CUDA.jl=", pkgversion(CUDA),
        "; R=1; Float32 physics / Float64 positions; baseline=current checkout; ",
        get(candidate, :preload, false) ? "installed opt-in" : "isolated prototype")
    for case in cases
        time = timed_duration === nothing ? (case.dims == 3 ? 0.03 : 0.2) : timed_duration
        variants = (CURRENT, optimized)
        for _ in 1:2, v in variants
            measure(v, case, time)
        end
        times = [Float64[] for _ in variants]
        for round in 1:rounds
            results = Vector{NamedTuple}(undef, 2)
            for i in (isodd(round) ? (1, 2) : (2, 1))
                r = results[i] = measure(variants[i], case, time)
                push!(times[i], r.seconds)
                push!(timing_rows, merge((; case = case.name, variant = variants[i].name, phase = "timing",
                    duration = time, round, particles = r.particles, seconds = r.seconds, loop_seconds = r.loop_seconds), r.work))
            end
            results[1].work == results[2].work || error("Timed variants performed different step/rebuild/readback work")
            compare_runs(results...)
            @printf("%s pair=%d current=%.6fs preloaded=%.6fs ratio=%.4f\n",
                case.name, round, results[1].seconds, results[2].seconds, results[2].seconds / results[1].seconds)
            CSV.write(output_prefix * "_timings.csv", timing_rows)
            flush(stdout)
        end
        println(case.name, " median paired ratio=", median(times[2] ./ times[1]))
        long_time = duration === nothing ? (case.dims == 3 ? 0.6 : 4.0) : duration
        baseline = measure(CURRENT, case, long_time)
        validation_variants = controls ? (optimized, (; CURRENT..., name = "strict", strict = true),
            (; CURRENT..., name = "ulp", perturb = true)) : (optimized,)
        for v in (CURRENT, validation_variants...)
            r = v === CURRENT ? baseline : measure(v, case, long_time)
            push!(timing_rows, merge((; case = case.name, variant = v.name, phase = "validation",
                duration = long_time, round = 0, particles = r.particles, seconds = r.seconds,
                loop_seconds = r.loop_seconds), r.work))
            differences = compare_runs(baseline, r)
            for (index, frame) in enumerate(r.frames)
                push!(observation_rows, merge((; case = case.name, variant = v.name, frame = index,
                    time = frame.time, dx = r.dx, c0 = r.c0, rho0 = r.rho0), frame.metrics))
                push!(comparison_rows, merge((; case = case.name, variant = v.name, frame = index,
                    reference_time = baseline.frames[index].time), differences[index]))
            end
            println(case.name, " ", v.name, " duration=", long_time, " work=", r.work,
                " final=", last(differences))
            CSV.write(output_prefix * "_timings.csv", timing_rows)
            CSV.write(output_prefix * "_observables.csv", observation_rows)
            CSV.write(output_prefix * "_differences.csv", comparison_rows)
            flush(stdout)
        end
    end
end

function main(args)
    prefixes = ("--rounds=", "--time=", "--timed-duration=", "--output-prefix=")
    for arg in args
        startswith(arg, "--") && !(arg in ("--no-controls", "--installed")) && !any(p -> startswith(arg, p), prefixes) &&
            throw(ArgumentError("Unknown or incomplete option: $arg"))
    end
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "5"))
    rounds >= 3 || throw(ArgumentError("At least three paired rounds required"))
    parse_duration(x) = x === nothing ? nothing : parse(Float64, x)
    duration = parse_duration(value("--time", nothing))
    timed_duration = parse_duration(value("--timed-duration", nothing))
    all(t -> t === nothing || isfinite(t) && t > 0, (duration, timed_duration)) ||
        throw(ArgumentError("Durations must be finite and positive"))
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(isempty(names) ? ["DamBreak3D_dp0.0085", "StillWedge2D_MDBC_dp0.02"] : names)
    isempty(cases) && throw(ArgumentError("No matching cases"))
    output_prefix = value("--output-prefix", joinpath(@__DIR__, "preloaded_types"))
    candidate = "--installed" in args ? (; driver = SPHExampleGPU.SPHCellList, preload = true) : experimental_modules()
    Base.invokelatest(run_experiment, candidate, cases; rounds, duration, timed_duration, output_prefix,
        controls = !("--no-controls" in args))
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    PreloadedTypeBenchmark.main(ARGS)
end
