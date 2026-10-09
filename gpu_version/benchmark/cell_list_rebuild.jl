# Warmed, interleaved HEAD/current rebuild and complete-driver benchmark.
# From gpu_version/:
#   julia --project=. benchmark/cell_list_rebuild.jl --rounds 5 --rebuilds 100
#   julia --project=. benchmark/cell_list_rebuild.jl --no-graph --double-position
#   julia --project=. benchmark/cell_list_rebuild.jl --profile
# Rare-rebuild control (explicitly labeled in CSV):
#   julia --project=. benchmark/cell_list_rebuild.jl --allow-single-rebuild --small StillWedge2D_MDBC_dp0.02 --large StillWedge2D_MDBC_dp0.02
# The baseline is loaded from the requested git revision into a separate module;
# no tracked source is replaced. Both variants write the same particle/grid
# outputs. Raw measurements are written before any correctness assertion.
# Defaults use prescribed motion to exercise repeated natural rebuilding.
# StillWedge2D_MDBC_dp0.02 at duration 0.15 produced only its initial rebuild
# and is intentionally rejected by the complete-run activity guard.

using SPHExampleGPU
using CUDA
using StaticArrays
using TimerOutputs
using Statistics
using Printf
using HDF5

const BENCHMARK_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

function option(args, name, fallback, parsefn = identity)
    index = findfirst(==(name), args)
    index === nothing && return fallback
    index < length(args) || error("Missing value after $name")
    return parsefn(args[index + 1])
end

function configuration(args)
    return (;
        revision = option(args, "--baseline", "HEAD"),
        rounds = option(args, "--rounds", 5, x -> parse(Int, x)),
        rebuilds = option(args, "--rebuilds", 100, x -> parse(Int, x)),
        T = "--float64" in args ? Float64 : Float32,
        double_position = "--double-position" in args,
        graph = !("--no-graph" in args),
        subdivision = option(args, "--subdivision", 1, x -> parse(Int, x)),
        batch = option(args, "--batch", 32, x -> parse(Int, x)),
        duration = option(args, "--duration", 0.15, x -> parse(Float64, x)),
        profile = "--profile" in args,
        allow_single_rebuild = "--allow-single-rebuild" in args,
        profile_rebuilds = option(args, "--profile-rebuilds", 10, x -> parse(Int, x)),
        small = option(args, "--small", "MovingSquare2D_dp0.04"),
        large = option(args, "--large", "MovingSquare2D_dp0.02"),
        output = abspath(option(args, "--output", joinpath(@__DIR__, "cell_list_rebuild_results.csv"))),
    )
end

function load_baseline(revision, directory)
    repository = strip(read(`git -C $BENCHMARK_PACKAGE_ROOT rev-parse --show-toplevel`, String))
    prefix = replace(relpath(BENCHMARK_PACKAGE_ROOT, repository), '\\' => '/') * "/src/"
    files = split(read(`git -C $repository ls-tree -r --name-only $revision -- $prefix`, String), '\n'; keepempty = false)
    isempty(files) && error("No package source files in baseline $revision")
    for filename in files
        destination = joinpath(directory, relpath(filename, prefix))
        mkpath(dirname(destination))
        write(destination, read(`git -C $repository show $(revision * ":" * filename)`, String))
    end
    entrypoint = joinpath(directory, "SPHExampleGPU.jl")
    source = read(entrypoint, String)
    source = replace(source, "module SPHExampleGPU" => "module CellRebuildBaseline";
                     count = 1)
    # Preferences belong to the registered package; the benchmark's second
    # module compiles normally and has its own explicit warmup below.
    source = replace(source, "include(\"PrecompileWorkload.jl\")" => "")
    write(entrypoint, source)
    Base.include(Main, entrypoint)
    baseline = Base.invokelatest(getfield, Main, :CellRebuildBaseline)
    cases_file = joinpath(@__DIR__, "cases.jl")
    Core.eval(Main, :(
        module CellRebuildBaselineCases
            using ..CellRebuildBaseline
            include($cases_file)
        end
    ))
    Core.eval(Main, :(
        module CellRebuildCandidateCases
            using SPHExampleGPU
            include($cases_file)
        end
    ))
    hash = strip(read(`git -C $repository rev-parse $revision`, String))
    return baseline, hash
end

function selected_case(cases, name)
    selected = filter(c -> c.name == name, cases.BENCH_CASES)
    length(selected) == 1 || error("Expected one exact benchmark case named $name")
    return only(selected)
end

function configured_kwargs(case, cfg, save)
    kw = case.build(cfg.T, save)
    meta = kw.SimMetaData
    meta.SimulationTime = cfg.T(cfg.duration)
    meta.OutputTimes = cfg.T(cfg.duration / 3)
    meta.GPUUseGraph = cfg.graph
    meta.GPUMaxStepsPerSync = cfg.batch
    meta.GPUDoublePosition = cfg.double_position
    meta.GPUCellSubdivision = cfg.subdivision
    meta.GPUSyncTimers = false
    meta.GPUDeterministicSort = true
    meta.GPUAsyncOutput = true
    meta.ExportGridCells = true
    meta.GPUOutputQueueBytes = 8 * 1024^2
    meta.GPUOutputBufferBytes = 8 * 1024^2
    meta.OutputVariables = ["Velocity", "Density", "Pressure", "Acceleration", "ID", "Type", "GroupMarker"]
    if nameof(typeof(meta).parameters[5]) !== :NoMDBC
        append!(meta.OutputVariables, ["GhostPoints", "GhostNormals"])
    end
    return kw
end

mutable struct IsolatedRebuild{M, W, P, I, C, S, G}
    package::M
    workspace::W
    position::P
    position_scratch::P
    ids::I
    ids_scratch::I
    cells::C
    inverse_cutoff::S
    step::G
end

function isolated_setup(package, case, cfg)
    return mktempdir() do save
        kw = configured_kwargs(case, cfg, save)
        particles = package.AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
        points = particles.Position
        n = length(points)
        D = case.dims
        TP = eltype(eltype(points))
        workspace = package.CellListWorkspace{D, TP}(n; reach = cfg.subdivision)
        position = CuArray(points)
        ids = CuArray(Int32.(1:n))
        step = package.StepState{cfg.T}()
        step.ih[package.GPUStepState.I_STOP] = package.GPUStepState.STOP_REBUILD
        copyto!(step.i, step.ih)
        IsolatedRebuild(package, workspace, position, similar(position), ids, similar(ids),
                        CuVector{Int32}(undef, n), kw.SimKernel.H⁻¹, step)
    end
end

function isolated_rebuild!(variant, repeats; shared = false, baseline = false)
    for _ in 1:repeats
        workspace = variant.workspace
        srcs = (variant.position, variant.ids, workspace.CellIDScratch)
        dsts = (variant.position_scratch, variant.ids_scratch, variant.cells)
        if shared
            if baseline
                # The original driver reads the stopped timestep before doing
                # its additional bounding-box readback inside the rebuild.
                variant.package.readback!(variant.step)
                variant.package.update_cell_list!(workspace, variant.position, variant.inverse_cutoff, srcs, dsts)
            else
                grid_module = variant.package.GPUCellGrid
                grid_module.prepare_cell_grid!(workspace, variant.position, variant.inverse_cutoff; step = variant.step)
                variant.package.readback!(variant.step)
                grid_module.consume_cell_grid!(workspace, variant.step.ih) || error("Missing shared rebuild header")
                variant.package.update_cell_list!(workspace, variant.position, variant.inverse_cutoff, srcs, dsts; prepared = true)
            end
        else
            variant.package.update_cell_list!(workspace, variant.position, variant.inverse_cutoff, srcs, dsts)
        end
        variant.position, variant.position_scratch = variant.position_scratch, variant.position
        variant.ids, variant.ids_scratch = variant.ids_scratch, variant.ids
    end
    CUDA.synchronize()
    return nothing
end

workspace_counter(workspace, name, fallback = 0) =
    hasproperty(workspace, name) ? getproperty(workspace, name) : fallback

function rebuild_counters(variant, baseline)
    workspace = variant.workspace
    rebuilds = workspace.nrebuilds
    return (; rebuilds,
        state_readbacks = variant.step.readbacks,
        bbox_readbacks = workspace_counter(workspace, :bbox_host_readbacks, baseline ? rebuilds : 0),
        grid_status_readbacks = workspace_counter(workspace, :grid_status_readbacks),
        capacity_growths = workspace_counter(workspace, :capacity_growths, workspace.generation),
    )
end

function isolated_snapshot(variant)
    workspace = variant.workspace
    grid = workspace.grid
    return (; origin = grid.origin, dims = grid.dims, ncells = grid.ncells,
        cellstart = Array(view(workspace.CellStart, 1:(grid.ncells + 1))),
        cells = Array(variant.cells), ids = Array(variant.ids), position = Array(variant.position))
end

function timer_seconds(timer, name)
    elapsed = 0.0
    for (key, child) in timer.inner_timers
        elapsed += key == name ? TimerOutputs.time(child) / 1e9 : timer_seconds(child, name)
    end
    return elapsed
end

function logged_counter(log, expression; default = missing)
    found = match(expression, log)
    return found === nothing ? default : parse(Int, found.captures[1])
end

function read_datasets!(output, group, prefix = "")
    for name in keys(group)
        path = prefix * "/" * name
        object = group[name]
        try
            if object isa HDF5.Group
                read_datasets!(output, object, path)
            elseif object isa HDF5.Dataset
                output[path] = read(object)
            end
        finally
            close(object)
        end
    end
    return output
end

function output_snapshot(directory)
    output = Dict{String, Any}()
    for (root, _, files) in walkdir(directory), filename in files
        endswith(filename, ".vtkhdf") || continue
        h5open(joinpath(root, filename), "r") do file
            read_datasets!(output, file, replace(relpath(joinpath(root, filename), directory), '\\' => '/'))
        end
    end
    isempty(output) && error("RunSimulation produced no VTKHDF output")
    return output
end

function complete_run(package, case, cfg, baseline)
    return mktempdir() do save
        kw = configured_kwargs(case, cfg, save)
        particles = package.AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
        logger = package.SimulationLogger(save)
        CUDA.synchronize()
        measured = redirect_stdout(devnull) do
            @timed begin
                package.RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
                CUDA.synchronize()
            end
        end
        meta = kw.SimMetaData
        log = read(joinpath(save, "SimulationOutput.log"), String)
        rebuilds = logged_counter(log, r"Cell list rebuilds: (\d+)")
        state_readbacks = logged_counter(log, r"Host read backs of the step state: (\d+)")
        bbox_readbacks = baseline ? rebuilds :
            logged_counter(log, r"bounding-box host partial reads: (\d+)")
        grid_status_readbacks = baseline ? 0 :
            logged_counter(log, r"Device grid status reads: (\d+)")
        capacity_growths = baseline ? missing :
            logged_counter(log, r"capacity growths: (\d+)")
        order = sortperm(particles.ID)
        snapshot = (; ids = particles.ID[order], position = particles.Position[order],
            velocity = particles.Velocity[order], density = particles.Density[order],
            pressure = particles.Pressure[order], acceleration = particles.Acceleration[order],
            ghost_points = particles.GhostPoints[order], ghost_normals = particles.GhostNormals[order],
            time = meta.TotalTime, dt = meta.CurrentTimeStep, steps = meta.Iteration,
            output = output_snapshot(save))
        return (; wall_s = measured.time, loop_s = timer_seconds(meta.HourGlass, "00 SimulationLoop"),
            rebuild_host_s = timer_seconds(meta.HourGlass, "02a Actual Calculate IndexCounter"),
            prepare_host_s = timer_seconds(meta.HourGlass, "02a Prepare Cell Grid"),
            state_readback_host_s = timer_seconds(meta.HourGlass, "13 Read Back State"),
            host_bytes = measured.bytes, gc_s = measured.gctime, n = length(particles),
            steps = meta.Iteration, rebuilds, state_readbacks, bbox_readbacks,
            grid_status_readbacks, capacity_growths, snapshot)
    end
end

const RESULT_COLUMNS = (:phase, :case, :implementation, :round, :n, :wall_s, :loop_s,
    :rebuild_host_s, :prepare_host_s, :state_readback_host_s,
    :steps, :rebuilds, :state_readbacks, :bbox_readbacks,
    :grid_status_readbacks, :capacity_growths, :host_bytes, :gc_s,
    :precision, :position_precision, :graph, :subdivision, :batch, :duration,
    :baseline_revision, :device, :julia_version, :cuda_version)

function append_result!(io, phase, name, implementation, round, result, cfg)
    metadata = (; phase, case = name, implementation, round,
        precision = string(cfg.T), position_precision = cfg.double_position ? "Float64" : string(cfg.T),
        graph = cfg.graph, subdivision = cfg.subdivision, batch = cfg.batch, duration = cfg.duration,
        baseline_revision = cfg.baseline_hash, device = CUDA.name(CUDA.device()),
        julia_version = string(VERSION), cuda_version = string(pkgversion(CUDA)))
    row = merge(result, metadata)
    cells = map(RESULT_COLUMNS) do key
        value = getproperty(row, key)
        ismissing(value) ? "" : "\"" * replace(string(value), "\"" => "\"\"") * "\""
    end
    println(io, join(cells, ','))
    flush(io)
    @printf("%-9s %-28s %-9s round=%d wall=%.5fs steps=%d rebuilds=%d host_bytes=%d\n",
        phase, name, implementation, round, result.wall_s, result.steps, result.rebuilds, result.host_bytes)
    flush(stdout)
end

function assert_equal_snapshot(reference, candidate, context)
    for name in propertynames(reference)
        expected = getproperty(reference, name)
        actual = getproperty(candidate, name)
        if expected isa AbstractDict
            Set(keys(expected)) == Set(keys(actual)) || error("$context output dataset names differ")
            for key in keys(expected)
                isequal(expected[key], actual[key]) || error("$context output dataset differs: $key")
            end
        else
            isequal(expected, actual) || error("$context differs in $name")
        end
    end
    return nothing
end

function profile_shared_rebuilds!(variants, name, cfg)
    filename = splitext(cfg.output)[1] * "_profile.txt"
    open(filename, "a") do io
        println(io, "\nCase $name: $(cfg.profile_rebuilds) warmed shared rebuilds, baseline $(cfg.baseline_hash)")
        println(io, "Counts below exclude the profiler's start/end cuCtxSynchronize sentinels and warmup copy; the benchmark's final CUDA.synchronize remains included.")
        for (index, variant) in enumerate(variants)
            implementation = index == 1 ? "baseline" : "candidate"
            CUDA.synchronize()
            profile = CUDA.@profile external=false raw=true isolated_rebuild!(
                variant, cfg.profile_rebuilds; shared = true, baseline = index == 1)
            println(io, "\n$implementation")
            show(IOContext(io, :limit => false, :color => false, :displaysize => (2000, 240)), profile)
            println(io)
            host = profile.host
            first_sync = findfirst(==("cuCtxSynchronize"), host.name)
            last_sync = findlast(==("cuCtxSynchronize"), host.name)
            first_sync !== nothing && last_sync !== nothing && first_sync != last_sync ||
                error("Profiler synchronization sentinels not found")
            t0, t1 = host.stop[first_sync], host.start[last_sync]
            active = findall(i -> host.start[i] >= t0 && host.stop[i] <= t1, eachindex(host.name))
            active_ids = Set(host.id[active])
            api_counts = Dict{String, Int}()
            for i in active
                api_counts[host.name[i]] = get(api_counts, host.name[i], 0) + 1
            end
            synchronizes = sum(count for (api, count) in api_counts if occursin("Synchronize", api); init = 0)
            queries = sum(count for (api, count) in api_counts if occursin("Query", api); init = 0)
            device = profile.device
            copies = findall(eachindex(device.name)) do i
                device.id[i] in active_ids &&
                    occursin(r"\[copy .*device.* to .*(?:pinned|pageable|host).* memory\]", lowercase(device.name[i]))
            end
            bytes = sum(i -> ismissing(device.size[i]) ? 0 : device.size[i], copies; init = 0)
            summary = "$name $implementation: profiled_rebuilds=$(cfg.profile_rebuilds) synchronize_API_calls=$synchronizes query_API_calls=$queries D2H_copies=$(length(copies)) D2H_bytes=$bytes"
            println(io, summary)
            println(summary)
            println(io, "Captured host API counts:")
            for api in sort!(collect(keys(api_counts)))
                println(io, "  $api: $(api_counts[api])")
            end
            # Pool allocations count even when CUDA reuses reserved memory;
            # measure them separately from CUPTI tracing and host @timed bytes.
            CUDA.synchronize()
            device_bytes = CUDA.@allocated isolated_rebuild!(variant, cfg.profile_rebuilds;
                                                             shared = true, baseline = index == 1)
            allocation_summary = "$name $implementation: allocation_rebuilds=$(cfg.profile_rebuilds) device_allocated_bytes=$device_bytes"
            println(io, allocation_summary)
            println(allocation_summary)
            flush(io)
        end
    end
    assert_equal_snapshot(isolated_snapshot(variants[1]), isolated_snapshot(variants[2]), "$name profiled shared rebuild")
    println("CUPTI report: $filename")
    return nothing
end

function benchmark_case!(io, name, packages, cases, cfg)
    variants = [isolated_setup(packages[i], cases[i], cfg) for i in 1:2]
    # Both allocations and initial capacity growth happen before measurements.
    for (index, variant) in enumerate(variants)
        isolated_rebuild!(variant, 3)
        isolated_rebuild!(variant, 3; shared = true, baseline = index == 1)
    end
    assert_equal_snapshot(isolated_snapshot(variants[1]), isolated_snapshot(variants[2]), "$name warmup rebuild")
    isolated_times = [Float64[], Float64[]]
    shared_times = [Float64[], Float64[]]
    full_times = [Float64[], Float64[]]
    for phase in ("isolated", "shared"), round in 1:cfg.rounds
        order = isodd(round) ? (1, 2) : (2, 1)
        for i in order
            variant = variants[i]
            before = rebuild_counters(variant, i == 1)
            GC.gc()
            CUDA.synchronize()
            measured = @timed isolated_rebuild!(variant, cfg.rebuilds; shared = phase == "shared", baseline = i == 1)
            after = rebuild_counters(variant, i == 1)
            result = (; wall_s = measured.time, loop_s = 0.0, rebuild_host_s = measured.time,
                prepare_host_s = 0.0, state_readback_host_s = 0.0,
                steps = 0, n = length(variant.position), host_bytes = measured.bytes, gc_s = measured.gctime,
                rebuilds = after.rebuilds - before.rebuilds, state_readbacks = after.state_readbacks - before.state_readbacks,
                bbox_readbacks = after.bbox_readbacks - before.bbox_readbacks,
                grid_status_readbacks = after.grid_status_readbacks - before.grid_status_readbacks,
                capacity_growths = after.capacity_growths - before.capacity_growths)
            result.capacity_growths == 0 || error("Isolated timed region unexpectedly grew capacity")
            push!(phase == "shared" ? shared_times[i] : isolated_times[i], result.wall_s)
            append_result!(io, phase, name, i == 1 ? "baseline" : "candidate", round, result, cfg)
        end
        assert_equal_snapshot(isolated_snapshot(variants[1]), isolated_snapshot(variants[2]), "$name repeated rebuild round $round")
    end
    cfg.profile && profile_shared_rebuilds!(variants, name, cfg)
    variants = nothing
    GC.gc()
    CUDA.reclaim()
    println("Warming complete RunSimulation for $name")
    warm = [complete_run(packages[i], cases[i], cfg, i == 1) for i in 1:2]
    assert_equal_snapshot(warm[1].snapshot, warm[2].snapshot, "$name complete warmup")
    for result in warm
        result.rebuilds >= 1 || error("$name completed without a cell-list rebuild")
        cfg.allow_single_rebuild || result.rebuilds > 1 ||
            error("$name has only $(result.rebuilds) rebuild; increase --duration")
    end
    reference = warm[1].snapshot
    warm = nothing
    for round in 1:cfg.rounds
        order = isodd(round) ? (1, 2) : (2, 1)
        for i in order
            GC.gc()
            CUDA.reclaim()
            result = complete_run(packages[i], cases[i], cfg, i == 1)
            append_result!(io, cfg.allow_single_rebuild ? "rarecontrol" : "complete", name,
                           i == 1 ? "baseline" : "candidate", round, result, cfg)
            assert_equal_snapshot(reference, result.snapshot, "$name complete round $round variant $i")
            result.rebuilds >= 1 || error("Measured run completed without a cell-list rebuild")
            cfg.allow_single_rebuild || result.rebuilds > 1 || error("Measured run has insufficient rebuild activity")
            push!(full_times[i], result.wall_s)
        end
    end
    @printf("%s: median paired candidate/baseline isolated %.4f, shared %.4f, complete %.4f; exact states and VTKHDF datasets match\n",
        name, median(isolated_times[2] ./ isolated_times[1]), median(shared_times[2] ./ shared_times[1]),
        median(full_times[2] ./ full_times[1]))
    return nothing
end

function main(args = ARGS)
    cfg = configuration(args)
    cfg.rounds > 0 && cfg.rebuilds > 0 && cfg.duration > 0 && cfg.profile_rebuilds > 0 || error("Counts and duration must be positive")
    CUDA.functional() || error("A functional CUDA GPU is required")
    mktempdir() do directory
        baseline, hash = load_baseline(cfg.revision, directory)
        cfg = merge(cfg, (; baseline_hash = hash))
        println("GPU: $(CUDA.name(CUDA.device())); Julia $VERSION; CUDA.jl $(pkgversion(CUDA)); baseline $hash")
        println("Warmed AB/BA order; precision=$(cfg.T), double_position=$(cfg.double_position), graph=$(cfg.graph), duration=$(cfg.duration), outputs every $(cfg.duration / 3).")
        println("Host readback counts are logical API calls, not a CUDA profiler's total synchronization count. Baseline bounding-box reads equal rebuild count by the original implementation.")
        println("Isolated wall/rebuild time includes final synchronization. Complete rebuild_host_s is the existing host rebuild timer; complete wall includes upload, stepping, output, log, and file closing.")
        println("Shared phase models the existing timestep readback: original reads step then bbox; candidate prepares grid before step readback. Complete prepare_host_s measures CPU enqueue overhead, not device execution.")
        cfg.allow_single_rebuild && println("Explicit rare-rebuild control: single rebuilds are accepted and complete-run CSV rows are labeled rarecontrol.")
        mkpath(dirname(cfg.output))
        if cfg.profile
            write(splitext(cfg.output)[1] * "_profile.txt", "CUDA CUPTI reports; baseline $hash; GPU $(CUDA.name(CUDA.device())); Julia $VERSION; CUDA.jl $(pkgversion(CUDA))\n")
        end
        open(cfg.output, "w") do io
            println(io, join(string.(RESULT_COLUMNS), ','))
            packages = (baseline, SPHExampleGPU)
            case_modules = (Base.invokelatest(getfield, Main, :CellRebuildBaselineCases),
                            Base.invokelatest(getfield, Main, :CellRebuildCandidateCases))
            for name in unique((cfg.small, cfg.large))
                cases = ntuple(i -> Base.invokelatest(selected_case, case_modules[i], name), 2)
                Base.invokelatest(benchmark_case!, io, name, packages, cases, cfg)
            end
        end
        println("Raw measurements: $(cfg.output)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
