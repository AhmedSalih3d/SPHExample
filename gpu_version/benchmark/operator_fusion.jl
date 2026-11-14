# Compare default radial force fusion with the previous vector-force arithmetic.
module OperatorFusionBenchmark

using SPHExampleGPU, CUDA, Statistics, Printf, LinearAlgebra, CSV
include("still_wedge_instructions.jl")
include("interaction_instructions.jl")
using .InteractionInstructionBenchmark: select_cases

function replace_once(source, before, after)
    length(findall(before, source)) == 1 || error("Expected exactly one fusion source anchor: $before")
    return replace(source, before => after; count = 1)
end

function unfused_source(source)
    source = replace(source, "\r\n" => "\n")
    source = replace_once(source,
        "elseif (Radial || FuseReferenceForces) && SimViscosity isa ArtificialViscosity\n" *
        "                                visc = SPHViscosityModels.radial_viscosity(SimViscosity, SimKernel, SimConstants,\n" *
        "                                    xᵢⱼ, viscous_velocity, one(T), ρᵢ, ρⱼ, inverse_distance)\n" *
        "                            elseif ScalarSPS",
        "elseif ScalarSPS")
    return replace_once(source,
        "if (Radial || FuseReferenceForces) && SimViscosity isa ArtificialViscosity && viscous_dt === nothing\n" *
        "                                # Combine the collinear forces before applying their shared gradient.\n" *
        "                                acc += (-mass_j * (Pfac + f_ab) + visc) * ∇ᵢWᵢⱼ\n" *
        "                            else\n" *
        "                                dvdt = -mass_j * (Pfac + f_ab) * ∇ᵢWᵢⱼ\n" *
        "                                acc += dvdt + visc\n" *
        "                            end",
        "dvdt = -mass_j * (Pfac + f_ab) * ∇ᵢWᵢⱼ\n                            acc += dvdt + visc")
end

function unit_wall_source(source)
    return replace_once(source,
        "if SimViscosity isa ArtificialViscosity && (is_wall(typeᵢ) || is_wall(ParticleType[j]))",
        "if SimViscosity isa ArtificialViscosity && T(SimViscosity.boundary_factor) != one(T) && (is_wall(typeᵢ) || is_wall(ParticleType[j]))")
end

function resources_source(source)
    source = replace(source, "\r\n" => "\n")
    source = replace_once(source, "module GPUKernels",
        "module GPUKernels\nconst FUSION_KERNEL_RESOURCES = Dict{DataType, NamedTuple}()")
    source = replace_once(source,
        "@cuda always_inline=always_inline fastmath=fastmath threads=threads blocks=cld(n * K, threads) interaction_kernel!(",
        "compiled = @cuda always_inline=always_inline fastmath=fastmath threads=threads blocks=cld(n * K, threads) interaction_kernel!(")
    return replace_once(source,
        "inv(SimConstants.ρ₀), predictor_output, Val(fuse_reference_forces))\n    return nothing",
        """inv(SimConstants.ρ₀), predictor_output, Val(fuse_reference_forces))
    if !haskey(FUSION_KERNEL_RESOURCES, typeof(compiled))
        memory = CUDA.memory(compiled)
        FUSION_KERNEL_RESOURCES[typeof(compiled)] = (;
            pass = predictor_output === nothing ? "corrector" : "predictor",
            step = step isa HostStep ? "host" : "device", threads, lanes = K,
            registers_per_thread = CUDA.registers(compiled),
            local_bytes_per_thread = memory.local, shared_bytes_per_block = memory.shared)
    end
    return nothing""")
end

function run_registers(variants, cases; duration, strict, radial, cached, candidate, output)
    configure(meta) = configure_case(meta; radial, cached)
    records = NamedTuple[]
    for case in cases
        results = map(variants) do variant
            empty!(variant.kernels.FUSION_KERNEL_RESOURCES)
            result = InteractionInstructionBenchmark.measure(variant, case, duration; strict, configure)
            isempty(variant.kernels.FUSION_KERNEL_RESOURCES) && error("No interaction kernels observed")
            for resource in values(variant.kernels.FUSION_KERNEL_RESOURCES)
                push!(records, merge((; case = case.name, variant = variant === variants[1] ? "baseline" : "fused",
                    candidate, radial, cached, strict, particles = result.particles), resource))
            end
            result
        end
        dx = mktempdir(save -> case.build(Float32, save).SimConstants.dx)
        println(case.name, " resource-run comparison: ", differences(results..., dx))
    end
    sort!(records; by = r -> (r.case, r.pass, r.step, r.variant))
    CSV.write(output, records)
    for record in records
        println(record)
    end
    println("Register queries are separate from uninstrumented timing runs.")
end

function configure_case(meta; radial = true, cached = true)
    meta.GPURadialInteractions = radial
    meta.GPUPreloadInteractionTypes = !radial
    meta.GPUCacheInteractionFields = cached
    meta.GPUCellSubdivision = 1
    meta.GPUXCellSubdivision = 4
    meta.GPUOutputQueueBytes = 8 * 1024^2
    return nothing
end

function differences(a, b, dx)
    a.work == b.work || error("Step/rebuild/readback counts changed: $(a.work) versus $(b.work)")
    a.ids == b.ids && a.types == b.types || error("Particle identities or types changed")
    a.output.counts == b.output.counts || error("Output particle counts changed")
    keys(a.output.datasets) == keys(b.output.datasets) || error("Output fields changed")
    for key in keys(a.output.datasets)
        a.output.datasets[key][1:2] == b.output.datasets[key][1:2] || error("Output shape/type changed: $key")
    end
    length(a.output.times) == length(b.output.times) || error("Output frame count changed")
    errors = map((x, y) -> maximum(norm.(x .- y)), a.state, b.state)
    return (; position_dx = errors[1] / dx, velocity = errors[2], density = errors[3],
        acceleration = errors[4], time_error = abs(a.time - b.time),
        output_time_error = maximum(abs.(a.output.times .- b.output.times)),
        exact_state = isequal(a.state, b.state) && a.time == b.time, exact_output = a.output == b.output)
end

function check_accuracy(errors, maximum_position_dx)
    isfinite(errors.position_dx) && errors.position_dx <= maximum_position_dx ||
        error("Position drift $(errors.position_dx) dx exceeds $maximum_position_dx dx")
    return nothing
end

function run_cases(variants, cases; rounds, duration, strict, radial, cached, maximum_position_dx, candidate, output)
    configure(meta) = configure_case(meta; radial, cached)
    records = NamedTuple[]
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; CUDA.jl=", pkgversion(CUDA),
        "; Float32 physics / Float64 positions; R=1 Rx=4; radial=", radial,
        "; cached=", cached, "; fastmath=", !strict, "; candidate=", candidate)
    for case in cases
        dx = mktempdir(save -> case.build(Float32, save).SimConstants.dx)
        measure(variant) = InteractionInstructionBenchmark.measure(variant, case, duration; strict, configure)
        # Compare the unmodified clone to the installed solver, not to the old arithmetic.
        original = (; driver = SPHExampleGPU.SPHCellList, diffusion = SPHExampleGPU.SPHDensityDiffusionModels)
        baseline = measure(variants[1])
        optimized = measure(variants[2])
        InteractionInstructionBenchmark.compare(candidate == "force" ? optimized : baseline, measure(original))
        errors = differences(baseline, optimized, dx)
        println(case.name, " warmup comparison: ", errors)
        check_accuracy(errors, maximum_position_dx)
        require_exact = (strict && !radial) || candidate == "unit-wall"
        !require_exact || (errors.exact_state && errors.exact_output || error("Exact comparison failed"))
        # Compile both cloned call paths before the first measured pair.
        warm_errors = differences(measure(variants[1]), measure(variants[2]), dx)
        check_accuracy(warm_errors, maximum_position_dx)
        !require_exact || (warm_errors.exact_state && warm_errors.exact_output || error("Exact comparison failed"))
        ratios = Float64[]
        for round in 1:rounds
            results = Vector{NamedTuple}(undef, 2)
            for i in (isodd(round) ? (1, 2) : (2, 1))
                results[i] = measure(variants[i])
            end
            a, b = results
            errors = differences(a, b, dx)
            check_accuracy(errors, maximum_position_dx)
            !require_exact || (errors.exact_state && errors.exact_output || error("Exact comparison failed"))
            push!(ratios, b.seconds / a.seconds)
            for i in 1:2
                r = results[i]
                push!(records, merge((; case = case.name, round, variant = i == 1 ? "baseline" : "fused",
                    duration, radial, cached, strict, candidate, particles = r.particles,
                    seconds = r.seconds, loop_seconds = r.loop_seconds), r.work, errors))
            end
            CSV.write(output, records)
            @printf("%s round=%d baseline=%.6fs fused=%.6fs ratio=%.4f work=%s errors=%s\n",
                case.name, round, a.seconds, b.seconds, ratios[end], string(a.work), string(errors))
            flush(stdout)
        end
        @printf("summary %s paired median=%.4f range=[%.4f,%.4f] faster=%d/%d\n",
            case.name, median(ratios), minimum(ratios), maximum(ratios), count(<(1), ratios), rounds)
    end
end

function main(args)
    for arg in args
        startswith(arg, "--") || continue
        arg in ("--strict", "--reference", "--uncached", "--registers") && continue
        any(key -> startswith(arg, key * "="),
            ("--rounds", "--duration", "--max-position-dx", "--candidate", "--output")) ||
            throw(ArgumentError("Unknown option: $arg"))
    end
    option(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, option("--rounds", "7"))
    duration = parse(Float64, option("--duration", "0.03"))
    maximum_position_dx = parse(Float64, option("--max-position-dx", "0.1"))
    candidate = option("--candidate", "force")
    candidate in ("force", "unit-wall") || throw(ArgumentError("Candidate must be force or unit-wall"))
    rounds >= 3 || throw(ArgumentError("At least three paired rounds are required"))
    isfinite(duration) && duration > 0 || throw(ArgumentError("Duration must be finite and positive"))
    isfinite(maximum_position_dx) && maximum_position_dx >= 0 ||
        throw(ArgumentError("Position threshold must be finite and nonnegative"))
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(isempty(names) ? ["DamBreak3D_dp0.02", "DamBreak3D_dp0.0085"] : names)
    isempty(cases) && throw(ArgumentError("No matching benchmark cases"))
    resources = "--registers" in args
    instrumentation = resources ? resources_source : identity
    baseline = checked_driver(; propagate = true,
        kernel_transform = instrumentation ∘ (candidate == "force" ? unfused_source : identity), prefix = "FusionBaseline")
    optimized = checked_driver(; propagate = true,
        kernel_transform = instrumentation ∘ (candidate == "force" ? identity : unit_wall_source), prefix = "FusionCandidate")
    if resources
        return Base.invokelatest(run_registers, (baseline, optimized), cases; duration, candidate,
            strict = "--strict" in args, radial = !("--reference" in args), cached = !("--uncached" in args),
            output = option("--output", joinpath(@__DIR__, "operator_fusion_registers.csv")))
    end
    Base.invokelatest(run_cases, (baseline, optimized), cases; rounds, duration, maximum_position_dx, candidate,
        strict = "--strict" in args, radial = !("--reference" in args), cached = !("--uncached" in args),
        output = option("--output", joinpath(@__DIR__, "operator_fusion_results.csv")))
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    OperatorFusionBenchmark.main(ARGS)
end
