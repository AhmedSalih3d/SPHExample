using CSV

include("still_wedge_candidate_ranges.jl")

function replace_one(source, old, new)
    count(old, source) == 1 || error("Expected one source anchor: $old")
    return replace(source, old => new)
end

function masked_kernel_source(source; branchless_walk = false)
    source = replace(source, "\r\n" => "\n")
    source = replace_one(source, "density_budget = nothing) where {D, T, FlagKernel",
        "density_budget = nothing, candidate_masks = nothing, candidate_guard = nothing, mask_invalid = nothing) where {D, T, FlagKernel")
    source = replace_one(source, "pair_signal_precleared::Bool = false)",
        "pair_signal_precleared::Bool = false, candidate_masks = nothing, candidate_guard = nothing, mask_invalid = nothing)")
    source = replace_one(source, "cleaning, pair_signal, density_budget)\n    return nothing",
        "cleaning, pair_signal, density_budget, candidate_masks, candidate_guard, mask_invalid)\n    return nothing")
    source = replace_one(source, "        for row in neighbor_rows(periodicity, grid, c, s, pair_position",
        "        use_candidate_masks = candidate_masks !== nothing && (mask_invalid === nothing || mask_invalid[1] == 0)\n" *
        "        for row in neighbor_rows(periodicity, grid, c, s, pair_position")
    select_row_walk = branchless_walk ?
        "            walk_mask = candidate_masks !== nothing && jhi - jlo < 8 * sizeof(eltype(candidate_masks))\n" :
        "            walk_mask = use_candidate_masks\n"
    source = replace_one(source, "            j = jlo + lane\n            while j <= jhi",
        select_row_walk * """
                    pending = candidate_masks === nothing ? UInt64(0) : zero(eltype(candidate_masks))
                    j = jlo + lane
                    if walk_mask
                        bit = one(pending)
                        lane_bits = (typemax(typeof(pending)) ÷ ((bit << K) - bit)) << lane
                        pending = (use_candidate_masks ? candidate_masks[row.dy + Int32(2), source_i] : typemax(typeof(pending))) & lane_bits
                        j = jlo + Int32(trailing_zeros(pending))
                    end
                    while j <= jhi""")
    source = replace_one(source, "                j += Int32(K)",
        """
                        if !walk_mask
                            j += Int32(K)
                        else
                            pending &= pending - one(pending)
                            j = jlo + Int32(trailing_zeros(pending))
                        end""")
    source = replace_one(source, "        acc = predict_after_gather!(predictor_output, i, acc, dρdt, step, SimConstants)",
        "        acc = predict_after_gather!(predictor_output, i, acc, dρdt, step, SimConstants)\n" *
        "        if predictor_output !== nothing && candidate_guard !== nothing\n" *
        "            validate_mask_position!(candidate_guard, predictor_output.position[i], i)\n" *
        "        end")
    source = replace_one(source, "batch_end = nothing) where {FlagShift, CommitStep}",
        "batch_end = nothing, candidate_guard = nothing) where {FlagShift, CommitStep}")
    source = replace_one(source, "        Position[i] = x\n        v = damped_velocity",
        "        Position[i] = x\n        validate_mask_position!(candidate_guard, x, i)\n        v = damped_velocity")
    source = replace_one(source, "batch_end = nothing)\n    commit_step && red.nblocks",
        "batch_end = nothing, candidate_guard = nothing)\n    commit_step && red.nblocks")
    source = replace_one(source, "damping, Val(commit_step), Int32(n), cleaning, pair_signal, batch_end)",
        "damping, Val(commit_step), Int32(n), cleaning, pair_signal, batch_end, candidate_guard)")
    source = replace_one(source, "function final_step_kernel!(",
        """
        @inline validate_mask_position!(::Nothing, position, i) = nothing
        @inline function validate_mask_position!(guard, position, i)
            @inbounds delta = position - guard.reference[i]
            if !(sum(abs2, delta) <= guard.radius_squared[1])
                CUDA.atomic_or!(pointer(guard.invalid), Int32(1))
            end
            return nothing
        end

        function final_step_kernel!(""")
    return scalar_budget_kernel_source(source)
end

function scalar_budget_kernel_source(source)
    source = replace_one(source, "pair_signal = nothing) where {T, CommitPrevious}",
        "pair_signal = nothing, mask_budget = nothing) where {T, CommitPrevious}")
    source = replace_one(source,
        "            threadIdx().x == Int32(1) && GPUStepState.prepare_stage_samples!(stage_sampler, step)",
        """
                    if threadIdx().x == Int32(1)
                        update_mask_budget!(mask_budget, step, f[F_DT], h)
                        GPUStepState.prepare_stage_samples!(stage_sampler, step)
                    end""")
    source = replace_one(source, "            f[F_DT]   = dt\n            record_diagnostic!",
        "            f[F_DT]   = dt\n            set_mask_speed!(mask_budget, visc)\n            record_diagnostic!")
    source = replace_one(source, "                GPUStepState.prepare_stage_samples!(stage_sampler, step)\n            end\n        end",
        "                update_mask_budget!(mask_budget, step, dt, h)\n" *
        "                GPUStepState.prepare_stage_samples!(stage_sampler, step)\n            end\n        end")
    source = replace(source, "commit_previous::Bool = false, cleaning = nothing, pair_signal = nothing) where {T}" =>
        "commit_previous::Bool = false, cleaning = nothing, pair_signal = nothing, mask_budget = nothing) where {T}")
    source = replace_one(source, "commit_previous, cleaning, pair_signal)...)",
        "commit_previous, cleaning, pair_signal, mask_budget)...)")
    source = replace_one(source, "Val(commit_previous), cleaning, pair_signal)",
        "Val(commit_previous), cleaning, pair_signal, mask_budget)")
    return replace_one(source, "function finish_kernel!(",
        """
        @inline set_mask_speed!(::Nothing, speed) = nothing
        @inline set_mask_speed!(budget, speed) = (@inbounds budget.path_speed[2] = speed; nothing)
        @inline update_mask_budget!(::Nothing, step, dt, h) = nothing
        @inline function update_mask_budget!(budget, step::DeviceStep{T}, dt, h) where {T}
            @inbounds begin
                iteration = step.i[I_ITER]
                if budget.last_iteration[1] != iteration
                    budget.last_iteration[1] = iteration
                    budget.path_speed[1] += 0.5 * Float64(T(dt)) * budget.path_speed[2]
                    path = budget.path_speed[1] + 0.25 * step.f[F_DX]
                    # Allow for rounding in the Float32 speed and accumulated gap.
                    if !(1.002 * path + 2e-5 * Float64(h) <= budget.radius[1])
                        budget.invalid[1] = Int32(1)
                    end
                end
            end
            return nothing
        end

        function finish_kernel!(""")
end

function masked_driver_source(source)
    source = replace(source, "\r\n" => "\n")
    anchor = "                if sup.PeriodicImages !== nothing\n                    configure_periodic_images!"
    source = replace_one(source, anchor,
        "                Main.refresh_candidate_masks(gpu, cl, SimKernel, SimMetaData.Iteration)\n" * anchor)
    anchor = "        native_rigid=Val(native_rigid),interaction_mass,predictor_output,pair_signal_precleared,"
    source = replace_one(source, anchor,
        anchor * "\n        Main.mask_interaction_options(gpu.Position, lanes, cleaning_half, predictor_output)...,")
    source = replace_one(source, "            timestep_controller = floats ? rigid_timestep_controller(floating, SimConstants) : nothing)",
        "            mask_budget = Main.candidate_mask_budget(gpu.Position),\n" *
        "            timestep_controller = floats ? rigid_timestep_controller(floating, SimConstants) : nothing)")
    source = replace_one(source, "pair_signal, commit_previous = true))",
        "pair_signal, commit_previous = true, mask_budget = Main.candidate_mask_budget(gpu.Position)))")
    return replace_one(source, "        batch_end)\n    observe_density_stage!",
        "        batch_end, candidate_guard = Main.mask_final_guard(gpu.Position))\n    observe_density_stage!")
end

function prepare_masks!(device, gpu, cl, cutoff, allowance)
    positions = Array(gpu.Position)
    types = Array(gpu.Type)
    cells = Array(gpu.CellID)
    starts = Array(cl.CellStart)
    Mask = eltype(device)
    masks = zeros(Mask, size(device))
    for i in eachindex(positions)
        reach_i = types[i] == Fixed ? 0.0 : allowance
        for row in CELL_GRID.cell_rows(cl.grid)
            lo, hi = CELL_GRID.row_range(cl.grid, starts, cells[i], row)
            if hi - lo >= 8 * sizeof(Mask)
                Mask === UInt32 && continue
                error("A stencil row exceeds the experimental mask size")
            end
            bits = zero(Mask)
            for j in lo:hi
                i == j && continue
                reach_j = types[j] == Fixed ? 0.0 : allowance
                limit = cutoff * (1 + 1e-5) + reach_i + reach_j
                if sum(abs2, positions[i] - positions[j]) <= limit^2
                    bits |= one(Mask) << (j - lo)
                end
            end
            masks[row.dy + 2, i] = bits
        end
    end
    copyto!(device, masks)
    return nothing
end

function main_candidate_masks(trace, output, rounds, guard_fraction = nothing, ::Type{Mask} = UInt64;
                              scalar = false) where {Mask}
    # This closed benchmark replays allowances derived from the same workload.
    # It is deliberately not used as an unchecked production pruning rule.
    allowance = Dict{Int,Float64}()
    first_steps = Dict{Int,Int}()
    for row in CSV.File(trace)
        first_steps[row.rebuild] = min(get(first_steps, row.rebuild, row.step), row.step)
        allowance[row.rebuild] = max(get(allowance, row.rebuild, 0.0), row.accepted_fraction, row.predictor_fraction)
    end
    buffers = Ref{Any}(nothing)
    invalid_intervals = NamedTuple[]
    function mask_buffers(positions)
        if buffers[] === nothing
            buffers[] = (; masks = CUDA.zeros(Mask, 3, length(positions)),
                reference = scalar ? nothing : similar(positions), accepted = CUDA.zeros(Int32, 1),
                predictor = CUDA.zeros(Int32, 1), radius_squared = CUDA.zeros(Float64, 1),
                budget = scalar ? (; path_speed = CUDA.zeros(Float64, 2), last_iteration = CUDA.zeros(Int32, 1),
                    radius = CUDA.zeros(Float64, 1), invalid = CUDA.zeros(Int32, 1)) : nothing)
        end
        return buffers[]
    end
    function guard(positions, phase)
        (guard_fraction === nothing || scalar) && return nothing
        storage = mask_buffers(positions)
        # A device scalar keeps graph captures valid across rebuilds and startup.
        return (; storage.reference, invalid = getproperty(storage, phase), storage.radius_squared)
    end
    global mask_interaction_options = function (positions, ::Val{K}, half, predictor_output) where {K}
        K in (1, 2, 4, 8, 16, 32) || error("Unsupported mask lane count")
        storage = mask_buffers(positions)
        guard_fraction === nothing && return (; candidate_masks = storage.masks)
        # Standalone derivative preparation does not produce a predictor.
        half || predictor_output !== nothing || return (;)
        scalar && return (; candidate_masks = storage.masks, mask_invalid = storage.budget.invalid)
        return (; candidate_masks = storage.masks, candidate_guard = guard(positions, :predictor),
            mask_invalid = half ? storage.predictor : storage.accepted)
    end
    global candidate_mask_budget = positions -> mask_buffers(positions).budget
    global mask_final_guard = positions -> guard(positions, :accepted)
    function record_flags!(storage, rebuild)
        if scalar
            push!(invalid_intervals, (; rebuild, scalar = only(Array(storage.budget.invalid))))
        else
            push!(invalid_intervals, (; rebuild, accepted = only(Array(storage.accepted)),
                predictor = only(Array(storage.predictor))))
        end
    end
    global refresh_candidate_masks = function (gpu, cl, kernel, step)
        if guard_fraction === nothing
            first_steps[cl.nrebuilds] == step + 1 || error("Rebuild schedule differs from trace")
        end
        cutoff = Float64(kernel.H)
        storage = mask_buffers(gpu.Position)
        if guard_fraction !== nothing
            cl.nrebuilds == 1 || record_flags!(storage, cl.nrebuilds - 1)
            if scalar
                fill!(storage.budget.invalid, 0)
                fill!(storage.budget.last_iteration, -1)
                copyto!(storage.budget.path_speed, 1, [0.0], 1, 1)
                fill!(storage.budget.radius, cutoff * guard_fraction)
            else
                fill!(storage.accepted, 0)
                fill!(storage.predictor, 0)
                copyto!(storage.reference, gpu.Position)
                fill!(storage.radius_squared, (cutoff * guard_fraction)^2)
            end
        end
        fraction = guard_fraction === nothing ? allowance[cl.nrebuilds] + 1e-4 : guard_fraction + 1e-4
        prepare_masks!(storage.masks, gpu, cl, cutoff, cutoff * fraction)
    end
    variant = checked_driver(; propagate = true, prefix = "CandidateMask",
        kernel_transform = source -> masked_kernel_source(source; branchless_walk = Mask === UInt32),
        driver_transform = masked_driver_source, source_revision = "517dda05")
    baseline = example_runner((; driver = SPHExampleGPU.SPHCellList, diffusion = SPHExampleGPU.SPHDensityDiffusionModels))
    masked = example_runner(variant)
    runs = (baseline, masked)
    warm = measure(baseline, 4.0)
    compare(warm, measure(masked, 4.0))
    if guard_fraction !== nothing
        record_flags!(buffers[], warm.work.rebuilds)
        println("Guard flags from warmup: ", invalid_intervals)
    end
    records = map(1:rounds) do round
        results = Vector{NamedTuple}(undef, 2)
        for i in (isodd(round) ? (1, 2) : (2, 1))
            results[i] = measure(runs[i], 4.0)
        end
        compare(results...)
        control, candidate = results
        for result in results
            result.execution.graph_batches > 0 && result.execution.graph_capture_failures == 0 ||
                error("The mask comparison requires successful CUDA graph execution")
        end
        ratio = candidate.seconds / control.seconds
        @printf("round=%d baseline=%.6fs masked=%.6fs ratio=%.4f exact=true\n",
            round, control.seconds, candidate.seconds, ratio)
        flush(stdout)
        (; round, baseline_seconds = control.seconds, masked_seconds = candidate.seconds, ratio,
            baseline_graph_batches = control.execution.graph_batches, masked_graph_batches = candidate.execution.graph_batches)
    end
    write_records(output, records)
    @printf("Median paired ratio %.4f\n", median([row.ratio for row in records]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    fraction = length(ARGS) >= 4 && ARGS[4] != "trace" ? parse(Float64, ARGS[4]) : nothing
    mask_type = length(ARGS) >= 5 && ARGS[5] == "32" ? UInt32 : UInt64
    scalar = length(ARGS) == 6 && ARGS[6] == "scalar"
    scalar && fraction === nothing && error("The scalar guard requires an explicit displacement allowance")
    main_candidate_masks(ARGS[1], ARGS[2], parse(Int, ARGS[3]), fraction, mask_type; scalar)
end
