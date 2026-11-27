include("still_wedge_instructions.jl")

function main_still_wedge_blocks(output, rounds)
    threads = (128, 64, 96, 256)
    driver = (; driver = SPHExampleGPU.SPHCellList, diffusion = SPHExampleGPU.SPHDensityDiffusionModels)
    anchor = "        GPUXCellSubdivision = 4,"
    runs = map(threads) do block_size
        function configure(source)
            count(anchor, source) == 1 || error("StillWedge configuration anchor changed")
            return replace(source, anchor => anchor * "\n        GPUInteractionThreads = $block_size,")
        end
        example_runner(driver; example_transform = configure)
    end
    baseline = measure(runs[1], 4.0)
    for i in 2:length(runs)
        compare(baseline, measure(runs[i], 4.0))
    end
    records = NamedTuple[]
    for round in 1:rounds
        sequence = circshift(collect(eachindex(runs)), round - 1)
        if iseven(cld(round, length(runs)))
            reverse!(sequence)
        end
        results = Vector{NamedTuple}(undef, length(runs))
        for i in sequence
            results[i] = measure(runs[i], 4.0)
        end
        for i in eachindex(runs)
            compare(results[1], results[i])
            result = results[i]
            @assert result.execution.graph_capture_failures == 0 && result.execution.graph_batches > 0
            ratio = result.seconds / results[1].seconds
            push!(records, (; round, threads = threads[i], seconds = result.seconds, ratio))
            @printf("round=%d threads=%d seconds=%.6f ratio=%.4f exact=true\n",
                round, threads[i], result.seconds, ratio)
        end
        flush(stdout)
    end
    write_records(output, records)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_still_wedge_blocks(ARGS[1], parse(Int, ARGS[2]))
end
