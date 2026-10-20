using SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, Statistics, Printf, HDF5, SHA, TimerOutputs

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))

const RADIAL_BASELINE_REVISION = "911a76c"

function load_radial_baseline()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    source(file) = replace(read(`git -C $root show $(RADIAL_BASELINE_REVISION * ":gpu_version/src/" * file)`, String),
        "\r\n" => "\n")
    module_names = Dict(name => string("RadialBaseline", name) for name in
        ("SPHKernels", "SPHViscosityModels", "SPHDensityDiffusionModels", "GPUKernels", "SPHCellList"))
    for name in ("SPHKernels", "SPHViscosityModels", "SPHDensityDiffusionModels", "GPUKernels", "SPHCellList")
        text = source(name * ".jl")
        text = replace(text, "module $name" => "module $(module_names[name])")
        if name == "GPUKernels"
            text = replace(text, "include(\"UpdatedMDBC.jl\")" => source("UpdatedMDBC.jl"))
        end
        for dependency in keys(module_names)
            text = replace(text, "using ..$dependency" => "using ..$(module_names[dependency])",
                "$dependency." => "$(module_names[dependency]).")
        end
        Base.include_string(SPHExampleGPU, text, joinpath(@__DIR__, "..", "src", name * ".jl"))
    end
    return (; driver = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(module_names["SPHCellList"])),
        kernels = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(module_names["SPHKernels"])),
        viscosity = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(module_names["SPHViscosityModels"])),
        diffusion = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(module_names["SPHDensityDiffusionModels"])))
end

function generated_wedge_case(T, save)
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(42.48576250492629), δᵩ = T(0.1), CFL = T(0.5))
    geometry = still_wedge_2d_geometry(constants)
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "StillWedge", SaveLocation = save, SimulationTime = T(4), OutputTimes = T(0.01),
        GPUDoublePosition = true, GPUOutputQueueBytes = 8 * 1024^2,
        VisualizeInParaview = false, OpenLogFile = false, ExportGridCells = true)
    return (; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
        SimKernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx),
        SimViscosity = ArtificialViscosity(), SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(), ParticleNormalsPath = nothing)
end

function remap_baseline(kwargs, baseline)
    kernel = kwargs.SimKernel
    D = typeof(kernel).parameters[2]
    T = typeof(kernel).parameters[3]
    original_kernel = baseline.kernels.SPHKernelInstance{D,T}(
        getproperty(baseline.kernels, nameof(typeof(kernel.kernel)))(); h = kernel.h, k = kernel.k)
    all(name -> getproperty(original_kernel, name) == getproperty(kernel, name),
        filter(!=(:kernel), fieldnames(typeof(kernel)))) || error("baseline kernel constants changed")
    return merge(kwargs, (; SimKernel = original_kernel,
        SimViscosity = getproperty(baseline.viscosity, nameof(typeof(kwargs.SimViscosity)))(),
        SimDensityDiffusion = getproperty(baseline.diffusion, nameof(typeof(kwargs.SimDensityDiffusion)))()))
end

function read_radial_output(save, meta)
    particles = h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do file
        root = file["VTKHDF"]
        fields = Dict(name => read(root["PointData"][name]) for name in keys(root["PointData"]))
        points = read(root["Points"])
        offsets, counts = read(root["Steps/PointOffsets"]), read(root["NumberOfPoints"])
        ids = fields["ID"]
        order = Int[]
        for (offset, count) in zip(offsets, counts)
            columns = Int(offset + 1):Int(offset + count)
            length(unique(ids[columns])) == count || error("duplicate exported particle IDs")
            append!(order, columns[sortperm(ids[columns])])
        end
        length(order) == length(ids) || error("exported frame coverage changed")
        sorted_fields = Dict(name => ndims(data) == 2 ? data[:, order] : data[order] for (name, data) in fields)
        (; times = read(root["Steps/Values"]), points = points[:, order], fields = sorted_fields)
    end
    grid = h5open(joinpath(save, meta.SimulationName * "_GridCells.vtkhdf"), "r") do file
        root = file["VTKHDF"]
        (; times = read(root["Steps/Values"]), cells = read(root["NumberOfCells"]),
           points = read(root["Points"]), connectivity = read(root["Connectivity"]))
    end
    length(particles.times) == length(grid.times) + 1 || error("particle/grid frame counts differ")
    particles.times[2:end] == grid.times || error("particle/grid output timestamps differ")
    return (; particles, grid)
end

function measure_radial_run(case, variant; baseline = nothing, duration, subdivision, precision = Float32,
                            save_location = nothing, retain_output = true)
    function run_in_folder(save)
        result = nothing
        seconds = redirect_stdout(devnull) do
            @elapsed begin
                kwargs = case.build(precision, save)
                meta = kwargs.SimMetaData
                meta.SimulationTime = precision(duration)
                meta.OutputTimes = precision(0.01)
                meta.ExportGridCells = true
                meta.GPUDoublePosition = precision === Float32
                meta.GPUCellSubdivision = subdivision
                meta.GPULanesPerParticle = case.dims == 3 ? 1 : 0
                meta.GPURadialInteractions = variant === :radial
                if variant in (:scalar_sps, :radial_scalar_sps)
                    meta.GPUScalarSPSViscosity = true
                    meta.GPURadialInteractions = variant === :radial_scalar_sps
                end
                meta.GPUMaxStepsPerSync = 32
                meta.GPUUseGraph = true
                meta.GPUOutputQueueBytes = 8 * 1024^2
                meta.OutputVariables = ["Density", "Pressure", "Velocity", "Acceleration", "ID", "Type", "GroupMarker"]
                driver = variant === :baseline ? baseline.driver : SPHExampleGPU.SPHCellList
                variant === :baseline && (kwargs = remap_baseline(kwargs, baseline))
                host = AllocateDataStructures(kwargs.SimGeometry, meta)
                logger = SimulationLogger(save)
                try
                    driver.RunSimulation(; kwargs..., SimParticles = host, SimLogger = logger)
                    CUDA.synchronize()
                finally
                    close(logger.LoggerIo)
                end
                result = (; host, meta, dx = kwargs.SimConstants.dx, c₀ = kwargs.SimConstants.c₀)
            end
        end
        order = sortperm(result.host.ID)
        pressures = [particle_pressure(result.host.Density[i], result.host.Type[i],
            kwargs.SimConstants.c₀, kwargs.SimConstants.ρ₀) for i in order]
        state = (result.host.Position[order], result.host.Velocity[order],
            result.host.Density[order], pressures, result.host.Acceleration[order])
        all(field -> all(value -> value isa Number ? isfinite(value) : all(isfinite, value), field), state) ||
            error("nonfinite final particle state")
        all(>(0), result.host.Density) || error("nonpositive final density")
        return (; seconds, loop = loop_time(result.meta.HourGlass), steps = result.meta.Iteration,
            time = result.meta.TotalTime, state,
            output = retain_output ? read_radial_output(save, result.meta) : nothing,
            name = result.meta.SimulationName, result.dx, result.c₀)
    end
    return save_location === nothing ? mktempdir(run_in_folder) : run_in_folder(save_location)
end

function radial_state_differences(result, reference)
    position_errors = norm.(result.state[1] .- reference.state[1])
    velocity_errors = norm.(result.state[2] .- reference.state[2])
    density_errors = abs.(result.state[3] .- reference.state[3])
    position = maximum(position_errors)
    velocity = maximum(velocity_errors)
    density = maximum(density_errors)
    pressure = maximum(abs.(result.state[4] .- reference.state[4]))
    acceleration = maximum(norm.(result.state[5] .- reference.state[5]))
    return (; position, velocity, density, pressure, acceleration,
        position_dx = position / result.dx,
        position_rms_dx = sqrt(mean(abs2, position_errors)) / result.dx,
        velocity_rms = sqrt(mean(abs2, velocity_errors)),
        density_rms = sqrt(mean(abs2, density_errors)),
        exact_state = result.state == reference.state && result.time == reference.time)
end

function radial_differences(result, reference)
    length(result.output.particles.times) == length(reference.output.particles.times) || error("output frame count changed")
    size(result.output.particles.points) == size(reference.output.particles.points) || error("output point dimensions changed")
    keys(result.output.particles.fields) == keys(reference.output.particles.fields) || error("output fields changed")
    for name in ("ID", "Type", "GroupMarker")
        result.output.particles.fields[name] == reference.output.particles.fields[name] || error("particle identities changed")
    end
    return merge(radial_state_differences(result, reference), (;
        output_position = maximum(abs.(result.output.particles.points .- reference.output.particles.points)),
        output_density = maximum(abs.(result.output.particles.fields["Density"] .- reference.output.particles.fields["Density"])),
        output_time = maximum(abs.(result.output.particles.times .- reference.output.particles.times)),
        exact_output = result.output == reference.output))
end

function grid_fingerprint(path)
    result = Dict{String,Tuple}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            key = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, key)
            else
                dimensions = size(item)
                context = SHA.SHA2_256_CTX()
                if ndims(item) == 1
                    for start in 1:1_048_576:length(item)
                        data = item[start:min(start + 1_048_575, length(item))]
                        eltype(data) <: AbstractFloat && !all(isfinite, data) && error("nonfinite grid output")
                        SHA.update!(context, reinterpret(UInt8, vec(data)))
                    end
                elseif ndims(item) == 2
                    width = max(1, 1_048_576 ÷ max(1, dimensions[1]))
                    for start in 1:width:dimensions[2]
                        data = item[:, start:min(start + width - 1, dimensions[2])]
                        eltype(data) <: AbstractFloat && !all(isfinite, data) && error("nonfinite grid output")
                        SHA.update!(context, reinterpret(UInt8, vec(data)))
                    end
                else
                    error("unexpected grid dataset dimensions for $key")
                end
                result[key] = (eltype(item), dimensions, bytes2hex(SHA.digest!(context)))
            end
        end
    end
    h5open(path, "r") do file
        visit(file, "")
    end
    return result
end

function streamed_output_differences(original_folder, radial_folder, name)
    data = h5open(joinpath(original_folder, name * ".vtkhdf"), "r") do original_file
        h5open(joinpath(radial_folder, name * ".vtkhdf"), "r") do radial_file
            reference, actual = original_file["VTKHDF"], radial_file["VTKHDF"]
            times, actual_times = read(reference["Steps/Values"]), read(actual["Steps/Values"])
            counts, actual_counts = read(reference["NumberOfPoints"]), read(actual["NumberOfPoints"])
            counts == actual_counts || error("particle frame counts changed")
            length(times) == length(actual_times) == length(counts) || error("output time counts changed")
            all(isfinite, times) && all(isfinite, actual_times) || error("nonfinite output timestamps")
            offsets, actual_offsets = read(reference["Steps/PointOffsets"]), read(actual["Steps/PointOffsets"])
            fields = sort(collect(keys(reference["PointData"])))
            fields == sort(collect(keys(actual["PointData"]))) || error("output fields changed")
            exact = times == actual_times && offsets == actual_offsets
            position_error, density_error = 0.0, 0.0
            for frame in eachindex(counts)
                columns = Int(offsets[frame] + 1):Int(offsets[frame] + counts[frame])
                actual_columns = Int(actual_offsets[frame] + 1):Int(actual_offsets[frame] + counts[frame])
                ids = reference["PointData/ID"][columns]
                actual_ids = actual["PointData/ID"][actual_columns]
                length(unique(ids)) == length(ids) && length(unique(actual_ids)) == length(actual_ids) ||
                    error("duplicate exported particle IDs")
                order, actual_order = sortperm(ids), sortperm(actual_ids)
                ids[order] == actual_ids[actual_order] || error("particle identities changed")
                points = reference["Points"][:, columns][:, order]
                actual_points = actual["Points"][:, actual_columns][:, actual_order]
                all(isfinite, points) && all(isfinite, actual_points) || error("nonfinite output positions")
                position_error = max(position_error, maximum(abs.(actual_points .- points)))
                exact &= points == actual_points
                for field in fields
                    dataset, actual_dataset = reference["PointData"][field], actual["PointData"][field]
                    size(dataset) == size(actual_dataset) || error("output dimensions changed for $field")
                    values = ndims(dataset) == 2 ? dataset[:, columns][:, order] : dataset[columns][order]
                    actual_values = ndims(actual_dataset) == 2 ?
                        actual_dataset[:, actual_columns][:, actual_order] : actual_dataset[actual_columns][actual_order]
                    if eltype(values) <: AbstractFloat
                        all(isfinite, values) && all(isfinite, actual_values) || error("nonfinite output $field")
                        if field == "Density"
                            all(>(0), values) && all(>(0), actual_values) || error("nonpositive exported density")
                            density_error = max(density_error, maximum(abs.(actual_values .- values)))
                        end
                    else
                        values == actual_values || error("particle identity fields changed")
                    end
                    exact &= values == actual_values
                end
            end
            return (; output_position = position_error, output_density = density_error,
                output_time = maximum(abs.(actual_times .- times)), exact_output = exact)
        end
    end
    grid_exact = grid_fingerprint(joinpath(original_folder, name * "_GridCells.vtkhdf")) ==
        grid_fingerprint(joinpath(radial_folder, name * "_GridCells.vtkhdf"))
    return merge(data, (; exact_output = data.exact_output && grid_exact))
end

function long_radial_comparison(case, duration, subdivision; variant = :radial)
    return mktempdir() do folder
        original_folder, radial_folder = joinpath(folder, "reference"), joinpath(folder, "radial")
        mkpath(original_folder)
        mkpath(radial_folder)
        reference = measure_radial_run(case, :reference; duration, subdivision,
            save_location = original_folder, retain_output = false)
        radial = measure_radial_run(case, variant; duration, subdivision,
            save_location = radial_folder, retain_output = false)
        output = streamed_output_differences(original_folder, radial_folder, radial.name)
        delta = merge(radial_state_differences(radial, reference), output)
        return (; reference, radial, delta)
    end
end

function check_radial_accuracy(delta, maximum_position_dx)
    maximum_position_dx === nothing && return nothing
    delta.position_dx <= maximum_position_dx ||
        error("optimized interaction maximum position difference $(delta.position_dx) dx exceeds the required $maximum_position_dx dx")
    return nothing
end

function with_benchmark_viscosity(case, viscosity)
    viscosity === nothing && return case
    build(T, save) = merge(case.build(T, save), (; SimViscosity = viscosity))
    return BenchCase(case.name * "_" * string(nameof(typeof(viscosity))), case.dims, build)
end

"""
Compare baseline, default and opt-in radial interactions at identical physics,
precision, integration and output settings. Complete warm wall time includes
input/allocation, upload, graph construction, simulation, all particle/grid output
and file closing. Verification and Julia/package loading are excluded.
`scalar_sps=true` also compares the independently selected SPS contraction and
both stages together; the speed assertion then applies to scalar SPS alone.
A viscosity override is applied identically to every variant and labeled in
the case name; it must not be interpreted as speeding up the original example
with a different viscosity. Long-run comparisons use scalar SPS alone when
selected. Accuracy assertions apply to every optimized variant.
"""
function benchmark_radial(; rounds = 5, duration = 0.03,
                          names = ["DamBreak3D_dp0.0085"], long_time = nothing,
                          minimum_reduction = nothing, maximum_position_dx = nothing,
                          viscosity::Union{Nothing,SPHViscosity} = nothing,
                          scalar_sps::Bool = false,
                          output_path = joinpath(@__DIR__, "radial_interactions_results.csv"))
    rounds >= 3 || throw(ArgumentError("at least three alternating rounds are required"))
    isfinite(duration) && duration > 0 || throw(ArgumentError("duration must be finite and positive"))
    long_time === nothing || isfinite(long_time) && long_time > duration ||
        throw(ArgumentError("long_time must exceed the benchmark duration"))
    minimum_reduction === nothing || isfinite(minimum_reduction) && 0 < minimum_reduction < 1 ||
        throw(ArgumentError("minimum_reduction must be between zero and one"))
    maximum_position_dx === nothing || isfinite(maximum_position_dx) && maximum_position_dx >= 0 ||
        throw(ArgumentError("maximum_position_dx must be finite and nonnegative"))
    cases = [BENCH_CASES; BenchCase("GeneratedStillWedge", 2, generated_wedge_case)]
    selected = filter(case -> any(name -> occursin(name, case.name), names), cases)
    isempty(selected) && throw(ArgumentError("no matching benchmark case"))
    selected = map(case -> with_benchmark_viscosity(case, viscosity), selected)
    baseline = load_radial_baseline()
    println("Baseline ", RADIAL_BASELINE_REVISION, "; GPU ", CUDA.name(CUDA.device()), "; Julia ", VERSION)
    open(output_path, "w") do out
        println(out, "case,duration,round,variant,total_s,loop_s,steps,final_time,exact_state,exact_output,position_abs,position_dx,position_rms_dx,velocity_abs,velocity_rms,density_abs,density_rms,pressure_abs,acceleration_abs,output_position_abs,output_density_abs,output_time_abs")
        for case in selected
            subdivision = case.dims == 3 ? 2 : 1
            variants = startswith(case.name, "GeneratedStillWedge") ? (:baseline, :reference) :
                scalar_sps ? (:baseline, :reference, :radial, :scalar_sps, :radial_scalar_sps) : (:baseline, :reference, :radial)
            times = [Float64[] for _ in variants]
            for variant in variants
                Base.invokelatest(measure_radial_run, case, variant; baseline, duration, subdivision)
            end
            for round in 1:rounds
                results = Vector{Any}(undef, length(variants))
                for index in (isodd(round) ? eachindex(variants) : reverse(eachindex(variants)))
                    results[index] = Base.invokelatest(measure_radial_run, case, variants[index]; baseline, duration, subdivision)
                end
                reference = results[2]
                for index in eachindex(variants)
                    result = results[index]
                    delta = radial_differences(result, reference)
                    if variants[index] in (:baseline, :reference)
                        delta.exact_state && delta.exact_output || error("default path changed the baseline state or output")
                    end
                    push!(times[index], result.seconds)
                    println(out, join((case.name, duration, round, variants[index], result.seconds, result.loop,
                        result.steps, result.time, delta.exact_state, delta.exact_output,
                        delta.position, delta.position_dx, delta.position_rms_dx, delta.velocity, delta.velocity_rms,
                        delta.density, delta.density_rms, delta.pressure, delta.acceleration,
                        delta.output_position, delta.output_density, delta.output_time), ','))
                    flush(out)
                    variants[index] in (:baseline, :reference) || check_radial_accuracy(delta, maximum_position_dx)
                end
                flush(out)
            end
            for index in eachindex(variants)
                ratio = median(times[index] ./ times[2])
                @printf("%s %-10s total=%.6f s ratio_to_reference=%.4f reduction=%.2f%%\n",
                    case.name, string(variants[index]), median(times[index]), ratio, 100 * (1 - ratio))
            end
            if minimum_reduction !== nothing && :radial in variants
                radial_index = findfirst(==(scalar_sps ? :scalar_sps : :radial), variants)
                ratio = median(times[radial_index] ./ times[2])
                ratio <= 1 - minimum_reduction ||
                    error("$(variants[radial_index]) total-time reduction is below $(100 * minimum_reduction)% for $(case.name); measurements saved to $output_path")
            end
            if long_time !== nothing && !startswith(case.name, "GeneratedStillWedge")
                optimized_variant = scalar_sps ? :scalar_sps : :radial
                comparison = Base.invokelatest(long_radial_comparison, case, long_time, subdivision; variant = optimized_variant)
                (; reference, radial) = comparison
                for (variant, result) in ((:reference, reference), (optimized_variant, radial))
                    delta = variant === optimized_variant ? comparison.delta :
                        merge(radial_state_differences(reference, reference),
                            (; exact_output = true, output_position = 0.0, output_density = 0.0, output_time = 0.0))
                    println(out, join((case.name, long_time, 0, variant, result.seconds, result.loop,
                        result.steps, result.time, delta.exact_state, delta.exact_output,
                        delta.position, delta.position_dx, delta.position_rms_dx, delta.velocity, delta.velocity_rms,
                        delta.density, delta.density_rms, delta.pressure, delta.acceleration,
                        delta.output_position, delta.output_density, delta.output_time), ','))
                end
                delta = comparison.delta
                @printf("Long %.3f s: steps=%d/%d |position|=%.6g m (%.4g dx), RMS=%.4g dx, |velocity|=%.6g m/s, |density|=%.6g, |pressure|=%.6g Pa\n",
                    long_time, reference.steps, radial.steps, delta.position, delta.position_dx,
                    delta.position_rms_dx, delta.velocity, delta.density, delta.pressure)
                flush(out)
                check_radial_accuracy(delta, maximum_position_dx)
            end
            flush(stdout)
        end
    end
    println("Results: ", output_path, ". Reassociated arithmetic is not bitwise equivalent; validate accuracy for the intended case.")
end

function main_radial(args)
    rounds, duration, long_time, minimum_reduction, maximum_position_dx = 5, 0.03, nothing, nothing, nothing
    viscosity = nothing
    scalar_sps = false
    names = String[]
    output_path = joinpath(@__DIR__, "radial_interactions_results.csv")
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--rounds", "--time", "--long-time", "--assert-min-reduction", "--assert-max-position-dx", "--viscosity", "--output")
            index < length(args) || throw(ArgumentError("missing value for $arg"))
            value = args[index + 1]
            if arg == "--rounds"
                rounds = parse(Int, value)
            elseif arg == "--time"
                duration = parse(Float64, value)
            elseif arg == "--long-time"
                long_time = parse(Float64, value)
            elseif arg == "--assert-min-reduction"
                minimum_reduction = parse(Float64, value)
            elseif arg == "--assert-max-position-dx"
                maximum_position_dx = parse(Float64, value)
            elseif arg == "--viscosity"
                value == "laminar-sps" || throw(ArgumentError("expected --viscosity laminar-sps"))
                viscosity = LaminarSPS()
            else
                output_path = value
            end
            index += 2
        elseif arg == "--scalar-sps"
            scalar_sps = true
            index += 1
        elseif startswith(arg, "--")
            throw(ArgumentError("unknown option $arg"))
        else
            push!(names, arg)
            index += 1
        end
    end
    isempty(names) && push!(names, "DamBreak3D_dp0.0085")
    return benchmark_radial(; rounds, duration, long_time, minimum_reduction, maximum_position_dx, viscosity, scalar_sps, names, output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_radial(ARGS)
end
