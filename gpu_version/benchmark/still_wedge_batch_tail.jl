include("still_wedge_instructions.jl")

function instrument_batch_size(source, trim_queued_margin)
    source = replace(source, "\r\n" => "\n")
    anchor = "function launch_step_batch!(ctx, K::Int, use_graph::Bool, timed::Bool)\n"
    count(anchor, source) == 1 || error("Batch instrumentation anchor changed")
    source = replace(source, anchor => anchor * "    Main.count_submitted_steps(K)\n")
    if trim_queued_margin
        anchor = "    est_output = (deadline - start) / dt + 1"
        count(anchor, source) == 1 || error("Queued batch sizing anchor changed")
        source = replace(source, anchor => "    est_output = (deadline - start) / dt")
    end
    return source
end

function main_batch_tail(output, rounds)
    submitted = Ref(0)
    global count_submitted_steps = steps -> (submitted[] += steps)
    function run(trim, prefix)
        example_runner(checked_driver(; propagate = true, prefix,
            driver_transform = source -> instrument_batch_size(source, trim)))
    end
    runs = (run(false, "OriginalTail"), run(true, "ShorterTail"))
    function sample(index)
        submitted[] = 0
        result = measure(runs[index], 4.0)
        @assert result.execution.graph_capture_failures == 0 && result.execution.graph_batches > 0
        @assert submitted[] >= result.work.steps
        return (; result, submitted = submitted[])
    end
    function verify(control, candidate)
        a, b = control.result, candidate.result
        @assert a.work.steps == b.work.steps && a.work.rebuilds == b.work.rebuilds
        # Readbacks and graph launches may change with batch sizes.
        compare(a, merge(b, (; work = a.work)))
    end
    installed = example_runner((; driver = SPHExampleGPU.SPHCellList, diffusion = SPHExampleGPU.SPHDensityDiffusionModels))
    compare(measure(installed, 4.0), sample(1).result)
    verify(sample(1), sample(2))
    records = map(1:rounds) do round
        samples = Vector{NamedTuple}(undef, 2)
        for index in (isodd(round) ? (1, 2) : (2, 1))
            samples[index] = sample(index)
        end
        verify(samples...)
        control, candidate = samples
        a, b = control.result, candidate.result
        row = (; round, baseline_seconds = a.seconds, trimmed_seconds = b.seconds, ratio = b.seconds / a.seconds,
            accepted_steps = a.work.steps, rebuilds = a.work.rebuilds,
            baseline_submitted = control.submitted, trimmed_submitted = candidate.submitted,
            baseline_graph_batches = a.execution.graph_batches, trimmed_graph_batches = b.execution.graph_batches,
            baseline_readbacks = a.work.readbacks, trimmed_readbacks = b.work.readbacks)
        println(row)
        flush(stdout)
        row
    end
    write_records(output, records)
    @printf("Median paired ratio %.4f\n", median([row.ratio for row in records]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_batch_tail(ARGS[1], parse(Int, ARGS[2]))
end
