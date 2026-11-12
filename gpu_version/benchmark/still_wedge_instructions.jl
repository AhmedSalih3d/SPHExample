using SPHExampleGPU, CUDA, HDF5, SHA, Statistics, Printf, TimerOutputs, LinearAlgebra

const SOURCE_DIR = normpath(joinpath(@__DIR__, "..", "src"))
const EXAMPLE_PATH = normpath(joinpath(@__DIR__, "..", "example", "StillWedgeMDBC.jl"))

function checked_driver(; propagate = false)
    names = Dict(name => "CheckedRadial" * name for name in
        ("SPHDensityDiffusionModels", "GPUKernels", "GPUConfiguration", "SPHCellList"))
    function source(file)
        text = read(joinpath(SOURCE_DIR, file), String)
        return replace(text, r"include\(\"([^\"]+)\"\)" =>
            m -> source(match(r"\"([^\"]+)\"", m)[1]))
    end
    for name in ("SPHDensityDiffusionModels", "GPUKernels", "GPUConfiguration", "SPHCellList")
        text = source(name * ".jl")
        if name == "SPHDensityDiffusionModels" && !propagate
            needle = "Base.@propagate_inbounds function radial_density_diffusion(model::Union{LinearDensityDiffusion, ZeroGravityLinearDensityDiffusion}"
            occursin(needle, text) || error("optimized annotation not found")
            text = replace(text, needle => replace(needle, "Base.@propagate_inbounds" => "@inline"); count = 1)
        end
        text = replace(text, "module $name" => "module $(names[name])"; count = 1)
        for (old, new) in names
            text = replace(text, "using ..$old" => "using ..$new", "$old." => "$new.")
        end
        Base.include_string(SPHExampleGPU, text, joinpath(SOURCE_DIR, name * ".jl"))
    end
    return (; driver = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(names["SPHCellList"])),
        diffusion = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(names["SPHDensityDiffusionModels"])))
end

function example_runner(variant)
    scope = Module(gensym(:WedgeRun))
    Core.eval(scope, :(include(path) = Base.include($scope, path)))
    Core.eval(scope, quote
        function RunSimulation(; kwargs...)
            model = kwargs[:SimDensityDiffusion]
            replacement = $(variant.diffusion).LinearDensityDiffusion()
            initial = map(name -> copy(getproperty(kwargs[:SimParticles], name)), (:Position, :Density, :Type))
            result = $(variant.driver).RunSimulation(; merge(NamedTuple(kwargs), (; SimDensityDiffusion = replacement))...)
            return (; particles = kwargs[:SimParticles], metadata = kwargs[:SimMetaData], initial, result)
        end
    end)
    text = read(EXAMPLE_PATH, String)
    text = replace(text, r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n")
    Base.include_string(scope, text, EXAMPLE_PATH)
    return Base.invokelatest(getproperty, scope, :run_still_wedge)
end

function fingerprint(save)
    result = Dict{String,Tuple{DataType,Tuple,String}}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            path = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, path)
            else
                value = read(item)
                result[path] = (eltype(value), size(value), bytes2hex(sha256(reinterpret(UInt8, vec(value)))))
            end
        end
    end
    for file in ("StillWedge.vtkhdf", "StillWedge_GridCells.vtkhdf")
        h5open(joinpath(save, file), "r") do handle
            visit(handle, file)
        end
    end
    return result
end

function measure(run, duration)
    mktempdir() do save
        result = nothing
        GC.gc(); CUDA.synchronize()
        seconds = redirect_stdout(devnull) do
            @elapsed begin
                result = Base.invokelatest(run; simulation_time = duration, save_location = save,
                    visualize = false, open_log_file = false, generate_pdf_report = false)
                CUDA.synchronize()
            end
        end
        particles, meta = result.particles, result.metadata
        order = sortperm(particles.ID)
        state = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker))
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration,
            rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        return (; seconds, time = meta.TotalTime, work, state, initial = result.initial, output = fingerprint(save))
    end
end

function compare(a, b)
    exact = a.time == b.time && a.work == b.work && isequal(a.state, b.state) && a.output == b.output
    if !exact
        state_counts = map((x,y) -> count(p -> !isequal(p[1], p[2]), zip(x, y)), a.state, b.state)
        max_norms = map((x,y) -> maximum(norm.(x .- y)), a.state[1:4], b.state[1:4])
        output_diffs = count(key -> get(a.output, key, nothing) != get(b.output, key, nothing), union(keys(a.output), keys(b.output)))
        error("comparison differs: time=$(a.time), $(b.time); work=$(a.work), $(b.work); initial=$(map((x,y) -> count(p -> !isequal(p[1], p[2]), zip(x, y)), a.initial, b.initial)); state counts=$state_counts; max norms=$max_norms; HDF5 datasets differing=$output_diffs")
    end
    return nothing
end

function main(args)
    option(key, default) = begin
        values = filter(a -> startswith(a, key * "="), args)
        isempty(values) ? default : split(only(values), '='; limit = 2)[2]
    end
    rounds = parse(Int, option("--rounds", "9"))
    duration = parse(Float64, option("--duration", "4"))
    checked = example_runner(checked_driver(; propagate = "--clone-optimized" in args))
    optimized = example_runner((; driver = SPHExampleGPU.SPHCellList,
        diffusion = SPHExampleGPU.SPHDensityDiffusionModels))
    profile = option("--profile", "")
    if profile != ""
        profile in ("checked", "optimized") || error("--profile must be checked or optimized")
        measure(profile == "checked" ? checked : optimized, duration)
        return nothing
    end
    if "--ptx" in args
        for (name, run) in (("checked", checked), ("optimized", optimized))
            path = joinpath(@__DIR__, "still_wedge_$(name).ptx")
            open(path, "w") do io
                CUDA.@device_code_ptx io=io measure(run, duration)
            end
            println("PTX $name: $path")
        end
        return nothing
    end
    timing_only = "--timing-only" in args
    if timing_only
        measure(checked, duration)
        measure(optimized, duration)
        times = (Float64[], Float64[])
        for round in 1:rounds
            results = Vector{NamedTuple}(undef, 2)
            for i in (isodd(round) ? (1, 2) : (2, 1))
                results[i] = measure((checked, optimized)[i], duration)
                push!(times[i], results[i].seconds)
            end
            results[1].work == results[2].work || error("work counts differ in timing round $round")
            @printf("round=%d checked=%.6fs optimized=%.6fs ratio=%.4f work=%s exact=%s\n",
                round, results[1].seconds, results[2].seconds,
                results[2].seconds / results[1].seconds, string(results[1].work),
                string(results[1].time == results[2].time && isequal(results[1].state, results[2].state) && results[1].output == results[2].output))
            flush(stdout)
        end
        ratios = times[2] ./ times[1]
        @printf("paired median=%.4f range=[%.4f, %.4f] faster=%d/%d checked median=%.6fs optimized median=%.6fs\n",
            median(ratios), minimum(ratios), maximum(ratios), count(<(1), ratios), rounds,
            median(times[1]), median(times[2]))
        return nothing
    end
    warm_checked = measure(checked, duration)
    repeat_checked = measure(checked, duration)
    println("Checking baseline repeatability")
    compare(warm_checked, repeat_checked)
    measure(optimized, duration)
    times = (Float64[], Float64[])
    for round in 1:rounds
        results = Vector{NamedTuple}(undef, 2)
        for i in (isodd(round) ? (1, 2) : (2, 1))
            results[i] = measure((checked, optimized)[i], duration)
            push!(times[i], results[i].seconds)
        end
        println("Checking round $round")
        compare(results...)
        @printf("round=%d checked=%.6fs optimized=%.6fs ratio=%.4f work=%s datasets=%d exact=true\n",
            round, results[1].seconds, results[2].seconds,
            results[2].seconds / results[1].seconds, string(results[1].work), length(results[1].output))
        flush(stdout)
    end
    ratios = times[2] ./ times[1]
    @printf("paired median=%.4f range=[%.4f, %.4f] faster=%d/%d\n",
        median(ratios), minimum(ratios), maximum(ratios), count(<(1), ratios), rounds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
