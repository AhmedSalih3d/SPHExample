module GraphPreparationBenchmark

using SPHExampleGPU, CUDA, HDF5, SHA, Statistics, Printf, TimerOutputs

const DRIVER = SPHExampleGPU.SPHCellList
const PREPARATION_ENABLED = Ref(true)

function install_switch!()
    # Keep the same driver and kernels for both variants; switch only preparation.
    source = replace(read(joinpath(@__DIR__, "..", "src", "SPHCellList.jl"), String), "\r\n" => "\n")
    start = first(findfirst("function prepare_step_graph!(ctx::NamedTuple)", source))
    stop = last(findnext("\nend\n", source, start))
    method = replace(source[start:stop], "prepare_step_graph!" => "benchmark_prepare_step_graph!")
    Base.include_string(DRIVER, method, @__FILE__)
    @eval DRIVER prepare_step_graph!(ctx::NamedTuple) =
        Main.GraphPreparationBenchmark.PREPARATION_ENABLED[] ? benchmark_prepare_step_graph!(ctx) : nothing
    return nothing
end

function load_example!()
    path = normpath(joinpath(@__DIR__, "..", "example", "StillWedgeMDBC.jl"))
    source = replace(read(path, String), "\r\n" => "\n")
    # The working example may contain interactive top-level calls; load only definitions.
    guard = first(findfirst("\nif abspath(PROGRAM_FILE)", source))
    Base.include_string(@__MODULE__, source[1:prevind(source, guard)], path)
    return nothing
end

function output_fingerprint(save)
    result = Dict{String,Tuple{DataType,Tuple,String}}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            key = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, key)
            else
                data = read(item)
                bytes = reinterpret(UInt8, vec(data))
                result[key] = (eltype(data), size(data), bytes2hex(sha256(bytes)))
            end
        end
    end
    for name in ("StillWedge.vtkhdf", "StillWedge_GridCells.vtkhdf")
        h5open(joinpath(save, name), "r") do file
            visit(file["VTKHDF"], name)
        end
    end
    return result
end

function measure(prepared, simtime)
    PREPARATION_ENABLED[] = prepared
    return mktempdir() do save
        result = nothing
        elapsed = redirect_stdout(devnull) do
            @elapsed begin
                result = Base.invokelatest(run_still_wedge; simulation_time = simtime,
                    save_location = save, visualize = false, open_log_file = false)
                CUDA.synchronize()
            end
        end
        p = result.final_state.particles
        order = sortperm(p.ID)
        state = map(name -> copy(getproperty(p, name)[order]),
            (:Position, :Velocity, :Density, :ID, :Type, :GroupMarker))
        d = result.diagnostics
        counts = (; result.performance.steps, d.cell_list_rebuilds, d.step_state_readbacks, d.output_frames,
            time = result.final_state.time)
        return (; elapsed, wall = result.performance.wall_time_seconds,
            loop = TimerOutputs.time(result.performance.phase_timings["00 SimulationLoop"]) / 1e9,
            counts, state, output = output_fingerprint(save), execution = result.configuration.execution,
            fallbacks = Any[result.configuration.fallbacks...])
    end
end

function csv_row(io, round, prepared, result)
    c, e = result.counts, result.execution
    println(io, join((round, prepared ? "prepared" : "baseline", result.elapsed, result.wall, result.loop,
        c.steps, c.cell_list_rebuilds, c.step_state_readbacks, c.output_frames, c.time,
        e.graph_batches, e.direct_batches, e.graph_capture_failures), ','))
    flush(io)
end

function main(args = ARGS)
    cold = nothing
    rounds = 9
    simtime = 4.0
    output = joinpath(@__DIR__, "graph_preparation_results.csv")
    for arg in args
        if startswith(arg, "--cold=")
            cold = split(arg, '='; limit = 2)[2]
            cold in ("baseline", "prepared") || error("--cold requires baseline or prepared")
        elseif startswith(arg, "--rounds=")
            rounds = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--simtime=")
            simtime = parse(Float64, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        else
            error("unknown argument: $arg")
        end
    end
    rounds >= 3 || error("use at least three alternating paired rounds")
    simtime > 0 || error("simulation time must be positive")
    install_switch!()
    load_example!()
    println("GPU=", CUDA.name(CUDA.device()), " Julia=", VERSION,
        " Float32 physics, Float64 positions, 3027 particles, symplectic, simtime=", simtime,
        " threads=", Threads.nthreads())
    mkpath(dirname(abspath(output)))
    open(output, "w") do io
        println(io, "round,variant,total_s,solver_wall_s,loop_s,steps,rebuilds,readbacks,frames,final_time,graph_batches,direct_batches,capture_failures")
        if cold !== nothing
            result = Base.invokelatest(measure, cold == "prepared", simtime)
            csv_row(io, 0, cold == "prepared", result)
            println("COLD variant=", cold, " total_s=", result.elapsed, " wall_s=", result.wall,
                " loop_s=", result.loop, " counts=", result.counts, " execution=", result.execution,
                " fallbacks=", result.fallbacks)
            cold == "prepared" && result.execution.graph_capture_failures != 0 && error("prepared cold run failed capture")
            return
        end
        warmups = [Base.invokelatest(measure, false, simtime), Base.invokelatest(measure, true, simtime)]
        warmups[1].counts == warmups[2].counts && warmups[1].state == warmups[2].state &&
            warmups[1].output == warmups[2].output || error("warmup work or numerical output differs")
        totals = [Float64[], Float64[]]
        loops = [Float64[], Float64[]]
        for round in 1:rounds
            pair = Vector{typeof(warmups[1])}(undef, 2)
            for index in (isodd(round) ? (1, 2) : (2, 1))
                pair[index] = Base.invokelatest(measure, index == 2, simtime)
                csv_row(io, round, index == 2, pair[index])
            end
            pair[1].counts == pair[2].counts && pair[1].state == pair[2].state &&
                pair[1].output == pair[2].output || error("paired work or numerical output differs")
            for index in 1:2
                push!(totals[index], pair[index].elapsed)
                push!(loops[index], pair[index].loop)
                pair[index].execution.graph_capture_failures == 0 || error("warmed run failed capture")
            end
            @printf("round=%d total %.6f -> %.6f s, loop %.6f -> %.6f s, ratio=%.4f\n",
                round, pair[1].elapsed, pair[2].elapsed, pair[1].loop, pair[2].loop,
                pair[1].elapsed / pair[2].elapsed)
            flush(stdout)
        end
        ratios = totals[1] ./ totals[2]
        @printf("MEDIAN total %.6f -> %.6f s; loop %.6f -> %.6f s; paired ratio %.4f, IQR [%.4f, %.4f]\n",
            median(totals[1]), median(totals[2]), median(loops[1]), median(loops[2]),
            median(ratios), quantile(ratios, 0.25), quantile(ratios, 0.75))
        println("All paired particle states, exported datasets, steps, rebuilds, readbacks and frame counts matched exactly.")
    end
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    GraphPreparationBenchmark.main()
end
