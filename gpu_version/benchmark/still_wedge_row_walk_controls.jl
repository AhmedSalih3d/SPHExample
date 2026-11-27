include("still_wedge_instructions.jl")
include("batch_commit.jl")
include("cases.jl")

function measure_control(driver, case, duration)
    simulate = Base.invokelatest(getproperty, driver, :RunSimulation)
    return mktempdir() do save
        kw = case.build(Float32, save)
        meta = kw.SimMetaData
        @assert !meta.GPUShortRowBitWalk
        meta.SimulationTime = duration
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        result = nothing
        GC.gc()
        CUDA.synchronize()
        seconds = try
            redirect_stdout(devnull) do
                @elapsed begin
                    result = Base.invokelatest(simulate; kw..., SimParticles = particles, SimLogger = logger)
                    CUDA.synchronize()
                end
            end
        finally
            close(logger.LoggerIo)
        end
        execution = result.configuration.execution
        @assert execution.graph_capture_failures == 0 && execution.graph_batches > 0
        @assert !result.configuration.effective.short_row_bit_walk
        order = sortperm(particles.ID)
        fields = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker))
        return (; seconds, fields, particles = length(order), time = meta.TotalTime,
            steps = result.performance.steps, rebuilds = result.diagnostics.cell_list_rebuilds,
            readbacks = result.diagnostics.step_state_readbacks,
            output = BatchCommitBenchmark.output_digests(save))
    end
end

function main_row_walk_controls(output, rounds)
    drivers = (BatchCommitBenchmark.baseline_driver("517dda05"), SPHExampleGPU.SPHCellList)
    records = NamedTuple[]
    for index in (1, 4, 6)
        case = BENCH_CASES[index]
        sample(i) = measure_control(drivers[i], case, 0.2)
        for round in 0:rounds
            results = Vector{NamedTuple}(undef, 2)
            for i in (isodd(round) ? (1, 2) : (2, 1))
                results[i] = sample(i)
            end
            a, b = results
            @assert Base.structdiff(a, (; seconds = 0.0)) == Base.structdiff(b, (; seconds = 0.0))
            if round > 0
                row = (; case = case.name, round, baseline_seconds = a.seconds, installed_seconds = b.seconds,
                    ratio = b.seconds / a.seconds, a.particles, a.steps, a.rebuilds, a.readbacks)
                push!(records, row)
                println(row)
                flush(stdout)
            end
        end
    end
    write_records(output, records)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_row_walk_controls(ARGS[1], parse(Int, ARGS[2]))
end
