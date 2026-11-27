include("still_wedge_instructions.jl")

const CELL_GRID = SPHExampleGPU.GPUCellGrid
const STEP_STATE = SPHExampleGPU.GPUStepState
const DISPLACEMENT_FRACTIONS = (0.0, 0.025, 0.05, 0.1, 0.125, 0.2, 0.25)
const TRACE_THREADS = 128
const TRACE_STEPS = 16_384

function displacement_kernel!(partial, speed, control, reference, position, velocity, state, phase, n)
    STEP_STATE.step_active(state) || return nothing
    step = @inbounds state.i[STEP_STATE.I_ITER] + Int32(1)
    step <= size(partial, 2) || return nothing
    thread = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + thread
    distance = 0.0
    speed_squared = 0.0
    if i <= n
        @inbounds delta = position[i] - reference[i]
        distance = sum(abs2, delta)
        if velocity !== nothing
            @inbounds v = velocity[i]
            speed_squared = sum(x -> abs2(Float64(x)), v)
        end
    end
    for shift in (16, 8, 4, 2, 1)
        distance = max(distance, CUDA.shfl_down_sync(0xffffffff, distance, shift))
        if velocity !== nothing
            speed_squared = max(speed_squared, CUDA.shfl_down_sync(0xffffffff, speed_squared, shift))
        end
    end
    if (thread - 1) % 32 == 0
        warp = (blockIdx().x - 1) * (blockDim().x ÷ 32) + (thread - 1) ÷ 32 + 1
        @inbounds partial[warp, step, phase] = distance
        if velocity !== nothing
            @inbounds speed[warp, step] = speed_squared
            if warp == 1
                @inbounds control[1, step] = STEP_STATE.step_dt(state)
                @inbounds control[2, step] = state.f[STEP_STATE.F_DX]
            end
        end
    end
    return nothing
end

function record_displacement!(buffers, position, velocity, state, phase)
    n = length(position)
    @cuda threads=TRACE_THREADS blocks=cld(n, TRACE_THREADS) displacement_kernel!(
        buffers.partial, buffers.speed, buffers.control, buffers.reference, position,
        velocity, state, Int32(phase), Int32(n))
    return nothing
end

# Count ordered row visits at real rebuilds without changing either device gather.
function count_ranges!(records, gpu, workspace, kernel, time, step)
    positions = Array(gpu.Position)
    types = Array(gpu.Type)
    cells = Array(gpu.CellID)
    grid = workspace.grid
    starts = Array(workspace.CellStart)
    cutoff = Float64(kernel.H)
    for fraction in DISPLACEMENT_FRACTIONS
        displacement = fraction * cutoff
        raw = kept = envelope = rows = nonempty = kept_rows = 0
        for i in eachindex(positions)
            reach_i = types[i] == Fixed ? 0.0 : displacement
            for row in CELL_GRID.cell_rows(grid)
                lo, hi = CELL_GRID.row_range(grid, starts, cells[i], row)
                rows += 1
                nonempty += lo <= hi
                first_kept, last_kept = hi + 1, lo - 1
                for j in lo:hi
                    raw += 1
                    reach_j = types[j] == Fixed ? 0.0 : displacement
                    limit = cutoff + reach_i + reach_j
                    if sum(abs2, positions[i] - positions[j]) <= limit^2
                        kept += 1
                        first_kept = min(first_kept, j)
                        last_kept = j
                    end
                end
                if first_kept <= last_kept
                    kept_rows += 1
                    envelope += last_kept - first_kept + 1
                end
            end
        end
        push!(records, (; rebuild = workspace.nrebuilds, time, step, fraction, raw, kept, envelope,
            rows, nonempty, kept_rows))
    end
    return nothing
end

function main_candidate_ranges(output, trace_output)
    records = NamedTuple[]
    anchor = "                if sup.PeriodicImages !== nothing\n                    configure_periodic_images!"
    function instrument(source)
        source = replace(source, "\r\n" => "\n")
        count(anchor, source) == 1 || error("Rebuild instrumentation anchor changed")
        source = replace(source, anchor =>
            "                Main.capture_rebuild(gpu, cl, SimKernel, SimMetaData.TotalTime, SimMetaData.Iteration)\n" * anchor)
        first_gather = "        # Only the corrector's pair signal is consumed by the next timestep."
        second_gather = "    # Corrector: every term, including the viscosity and density diffusion"
        count(first_gather, source) == count(second_gather, source) == 1 || error("Gather instrumentation anchor changed")
        return replace(source,
            first_gather => "        Main.capture_displacement(gpu.Position, gpu.Velocity, state, 1)\n" * first_gather,
            second_gather => "    Main.capture_displacement(position_half, nothing, state, 2)\n" * second_gather)
    end
    trace = Ref{Any}(nothing)
    cutoff = Ref(0.0)
    function get_trace(position)
        if trace[] === nothing
            warps = cld(length(position), TRACE_THREADS) * (TRACE_THREADS ÷ 32)
            trace[] = (; reference = similar(position), partial = CUDA.fill(-1.0, warps, TRACE_STEPS, 2),
                speed = CUDA.fill(-1.0, warps, TRACE_STEPS), control = CUDA.zeros(Float64, 2, TRACE_STEPS))
        end
        return trace[]
    end
    global capture_rebuild = function (gpu, workspace, kernel, time, step)
        count_ranges!(records, gpu, workspace, kernel, time, step)
        cutoff[] = Float64(kernel.H)
        copyto!(get_trace(gpu.Position).reference, gpu.Position)
        return nothing
    end
    global capture_displacement = function (position, velocity, state, phase)
        record_displacement!(get_trace(position), position, velocity, state, phase)
    end
    variant = checked_driver(; propagate = true, driver_transform = instrument, prefix = "CandidateRange")
    baseline = example_runner((; driver = SPHExampleGPU.SPHCellList, diffusion = SPHExampleGPU.SPHDensityDiffusionModels))
    reference = measure(baseline, 4.0)
    sampled = measure(example_runner(variant), 4.0)
    compare(reference, sampled)
    @assert length(records) == sampled.work.rebuilds * length(DISPLACEMENT_FRACTIONS)
    write_records(output, records)
    @assert sampled.work.steps <= TRACE_STEPS
    partial = Array(trace[].partial)
    speed = Array(trace[].speed)
    control = Array(trace[].control)
    rebuild_steps = [row.step for row in records if iszero(row.fraction)]
    predictor_path = 0.0
    trace_records = map(1:sampled.work.steps) do step
        accepted = view(partial, :, step, 1)
        predictor = view(partial, :, step, 2)
        @assert minimum(accepted) >= 0 && minimum(predictor) >= 0 "Missing gather measurement on step $step"
        rebuild = searchsortedlast(rebuild_steps, step - 1)
        if step - 1 == rebuild_steps[rebuild]
            predictor_path = 0.0
        end
        speeds = view(speed, :, step)
        @assert minimum(speeds) >= 0 && control[1, step] > 0
        # F_DX sums four times the previous predictor-to-final position gap.
        # Add each step's maximum accepted-to-predictor excursion separately.
        predictor_path += 0.5 * control[1, step] * sqrt(maximum(speeds))
        predictor_bound = (predictor_path + 0.25 * control[2, step]) / cutoff[]
        measured_predictor = sqrt(maximum(predictor)) / cutoff[]
        @assert measured_predictor <= predictor_bound + 1e-6 "Measured predictor exceeded scalar bound"
        (; step, rebuild, accepted_fraction = sqrt(maximum(accepted)) / cutoff[],
            predictor_fraction = measured_predictor, scalar_predictor_fraction = predictor_bound)
    end
    @assert all(<(0), view(partial, :, sampled.work.steps + 1:TRACE_STEPS, :)) "Inactive graph steps wrote trace data"
    write_records(trace_output, trace_records)
    println("Exact instrumented result: ", sampled.work, ", ", length(sampled.output), " datasets")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_candidate_ranges(ARGS...)
end
