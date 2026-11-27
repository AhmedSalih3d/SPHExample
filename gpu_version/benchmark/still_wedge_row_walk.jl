include("still_wedge_instructions.jl")

function row_walk_source(source)
    source = replace(source, "\r\n" => "\n")
    begin_row = "            j = jlo + lane\n            while j <= jhi"
    next_candidate = "                j += Int32(K)"
    count(begin_row, source) == count(next_candidate, source) == 1 ||
        error("Interaction row instrumentation anchors changed")
    return replace(source,
        begin_row => """
                    walk_bits = K <= 32 && 32 % K == 0 && jhi - jlo < 32
                    pending = UInt32(0)
                    j = jlo + lane
                    if walk_bits
                        pending = (typemax(UInt32) ÷ ((UInt32(1) << K) - UInt32(1))) << lane
                        j = jlo + Int32(trailing_zeros(pending))
                    end
                    while j <= jhi""",
        next_candidate => """
                        if walk_bits
                            pending &= pending - UInt32(1)
                            j = jlo + Int32(trailing_zeros(pending))
                        else
                            j += Int32(K)
                        end""")
end

function main_row_walk(output, rounds; prototype = false)
    installed = (; driver = SPHExampleGPU.SPHCellList, diffusion = SPHExampleGPU.SPHDensityDiffusionModels)
    variant = prototype ? checked_driver(; propagate = true, prefix = "RowWalk",
        kernel_transform = row_walk_source, source_revision = "517dda05") : installed
    runs = map(example_runner, (installed, variant))
    sample(i) = measure(runs[i], 4.0; short_row_bit_walk = i == 2 && !prototype ? nothing : false)
    compare(sample(1), sample(2))
    records = map(1:rounds) do round
        results = Vector{NamedTuple}(undef, 2)
        for i in (isodd(round) ? (1, 2) : (2, 1))
            results[i] = sample(i)
        end
        compare(results...)
        baseline, bitwalk = results
        @assert bitwalk.execution.graph_capture_failures == 0 && bitwalk.execution.graph_batches > 0
        @assert !baseline.effective.short_row_bit_walk
        prototype || @assert bitwalk.effective.short_row_bit_walk
        row = (; round, baseline_seconds = baseline.seconds, bitwalk_seconds = bitwalk.seconds,
            ratio = bitwalk.seconds / baseline.seconds)
        println(row)
        flush(stdout)
        row
    end
    write_records(output, records)
    @printf("Median paired ratio %.4f\n", median([row.ratio for row in records]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_row_walk(ARGS[1], parse(Int, ARGS[2]); prototype = "--prototype" in ARGS)
end
