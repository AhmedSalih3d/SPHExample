# Controlled experiments against the committed kernels and driver. Run with:
# julia --project=. benchmark/performance_investigation.jl kernels
# julia --project=. benchmark/performance_investigation.jl profile
using SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, Statistics, Printf, HDF5, SHA
include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))
const CYLINDER_GENERATOR = joinpath(@__DIR__, "..", "example", "GenerateFloatingCylinder2D.jl")
# Load geometry definitions without running the example's file exports.
include_string(Main, first(split(read(CYLINDER_GENERATOR, String), "\noutput_dir = ")), CYLINDER_GENERATOR)

function generated_wedge(::Type{T}, save) where {T}
    constants = SimulationConstants{T}(dx = 0.02, c₀ = 42.48576250492629, δᵩ = 0.1, CFL = 0.5)
    geometry = still_wedge_2d_geometry(constants)
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "GeneratedStillWedge", SaveLocation = save,
        SimulationTime = 4, OutputTimes = 0.01, ExportGridCells = true,
        VisualizeInParaview = false, OpenLogFile = false)
    return (; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
        SimKernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx),
        SimViscosity = ArtificialViscosity(), SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(), ParticleNormalsPath = nothing)
end

function generated_cylinder(::Type{T}, save) where {T}
    constants = SimulationConstants{T}(dx = 0.025, c₀ = 30 * sqrt(9.81 * 0.8), CFL = 0.2)
    shapes = floating_cylinder_2d_shapes(dx = 0.025)
    regions = [ParticleRegion("Cylinder", shapes.cylinder, Floating; sampling = :conforming),
        ParticleRegion("Bound", shapes.tank, Fixed), ParticleRegion("Fluid", shapes.water, Fluid)]
    sampled = sample_particles(regions, constants.dx)
    level = maximum(last, sampled[3].positions)
    geometry = [SPHGeometry{2,T}(region.positions;
        Density = hydrostatic_density(region.positions, constants; water_level = level),
        Type = regions[i].type, GroupMarker = i,
        Floating = i == 1 ? FloatingDetails{T}(RelativeWeight = 1.2) : nothing)
        for (i, region) in enumerate(sampled)]
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "GeneratedFloatingCylinder", SaveLocation = save,
        SimulationTime = 0.5, OutputTimes = 0.05, ExportGridCells = false,
        VisualizeInParaview = false, OpenLogFile = false)
    return (; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
        SimKernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx, k = T(sqrt(2))),
        SimViscosity = LaminarSPS(), SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(), ParticleNormalsPath = nothing)
end

const PERFORMANCE_CASES = [BENCH_CASES;
    BenchCase("GeneratedStillWedge2D_dp0.02", 2, generated_wedge);
    BenchCase("FloatingCylinder2D_dp0.025", 2, generated_cylinder)]

const PERF_REVISION = get(ENV, "SPH_PERF_REVISION", "cb1d1cc")
const PERF_SRC = joinpath(@__DIR__, "..", "src")
revision_source(file) = read(`git show $(PERF_REVISION * ":gpu_version/src/" * file)`, String)

function kernel_variant(name, transform = identity)
    source = replace(transform(revision_source("GPUKernels.jl")), "module GPUKernels" => "module $name",
        "include(\"UpdatedMDBC.jl\")" => revision_source("UpdatedMDBC.jl"))
    Base.include_string(SPHExampleGPU, source, joinpath(PERF_SRC, "GPUKernels.jl"))
    return Base.invokelatest(getproperty, SPHExampleGPU, Symbol(name))
end

function driver_variant(name, kernels; transform = identity)
    source = transform(revision_source("SPHCellList.jl"))
    source = replace(source, "module SPHCellList" => "module $name",
        "using ..GPUKernels" => "using ..$(nameof(kernels))",
        "GPUKernels." => "$(nameof(kernels)).")
    Base.include_string(SPHExampleGPU, source, joinpath(PERF_SRC, "SPHCellList.jl"))
    return Base.invokelatest(getproperty, SPHExampleGPU, Symbol(name))
end

function setup_performance(case; driver = SPHExampleGPU.SPHCellList, T = Float32,
                           double_position = true, subdivision = 1, interval = 0.01)
    kw = case.build(T, mktempdir())
    meta = kw.SimMetaData
    meta.SimulationTime = T(100)
    meta.OutputTimes = T(interval)
    meta.GPUDoublePosition = double_position
    meta.GPUCellSubdivision = subdivision
    meta.GPUMaxStepsPerSync = 32
    meta.TimeSteppingMode = kw.SimTimeStepping
    meta.OutputIterationCounter = 1
    host = AllocateDataStructures(kw.SimGeometry, meta)
    if kw.ParticleNormalsPath !== nothing
        _, gp, gn = LoadBoundaryNormals(Val(case.dims), eltype(eltype(host.Position)), kw.ParticleNormalsPath)
        host.GhostPoints[eachindex(gp)] .= gp
        host.GhostNormals[eachindex(gn)] .= gn
    end
    Pressure!(host.Pressure, host.Density, kw.SimConstants)
    n = length(host)
    TP = double_position ? Float64 : T
    gpu = driver.upload_particles(host; position_type = TP)
    sup = driver.GPUSupportArrays{case.dims,T}(n; position_type = TP)
    red = ReductionWorkspace{SVector{3,T}}(n)
    cl = CellListWorkspace{case.dims,TP}(n; reach = subdivision)
    mot = driver.MotionArrays(kw.SimGeometry, host)
    floating = any(==(Floating), host.Type) ?
        FloatingArrays(kw.SimGeometry, host, kw.SimConstants; position_type = TP) : nothing
    st = StepState{T}(dx = one(T) + kw.SimKernel.h)
    return (; kw, gpu, sup, red, cl, mot, st, floating, driver)
end

function advance!(v)
    kw = v.kw
    v.driver.SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel,
        kw.SimMetaData, kw.SimConstants, v.gpu, v.cl, v.sup, v.red, v.mot, v.st; floating = v.floating)
    kw.SimMetaData.OutputIterationCounter += 1
    return nothing
end

function state_snapshot(v)
    order = sortperm(Array(v.gpu.ID))
    return (; fields = map(f -> Array(getproperty(v.gpu, f))[order],
            (:Position, :Velocity, :Density, :Pressure, :Acceleration)),
        steps = v.kw.SimMetaData.Iteration, time = v.kw.SimMetaData.TotalTime,
        rebuilds = v.cl.nrebuilds, readbacks = v.st.readbacks)
end

function launch_probe(kernels, kind, v; threads = 128)
    (; kw, gpu, sup, cl) = v
    step = HostStep(0f0, 0f0)
    lanes = Val(choose_lanes(length(gpu), Val(length(eltype(gpu.Position)))))
    if kind === :mdbc
        return () -> kernels.launch_mdbc!(gpu.Density, gpu.Pressure, gpu.Position,
            gpu.GhostPoints, gpu.GhostIndex, gpu.Type, cl.CellStart, cl.grid_dev,
            step, kw.SimKernel, kw.SimConstants; threads, lanes, pos_cells = sup.PosCells)
    end
    shift = Val(kw.SimMetaData isa SimulationMetaData{D,T,PlanarShifting} where {D,T})
    return () -> kernels.launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel,
        gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ, gpu.Position, gpu.Density, sup.InvDensity,
        gpu.Pressure, gpu.Velocity, gpu.Type, cl.CellStart, gpu.CellID, cl.grid_dev, step,
        kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel, kw.SimConstants,
        Val(false), shift; threads, lanes, pos_cells = sup.PosCells)
end

function probe_graph(launch; repeats = 64)
    launch(); CUDA.synchronize()
    graph = CUDA.capture() do
        for _ in 1:repeats
            launch()
        end
    end
    exec = CUDA.instantiate(graph)
    CUDA.launch(exec); CUDA.synchronize()
    return exec
end

function kernel_screen(cases)
    baseline = kernel_variant("PerfBaselineKernels")
    lazy = kernel_variant("PerfLazyRows", s -> replace(s, "for row in cell_rows(grid)" =>
        "for row in GPUCellGrid.CellRows{D,typeof(grid).parameters[2]}(grid.dims[1], grid.dims[1] * grid.dims[2])"))
    registers = kernel_variant("PerfRegisters64", s -> replace(s,
        "@cuda threads=threads" => "@cuda maxregs=64 threads=threads"))
    reused = kernel_variant("PerfReusedLU", s -> replace(s,
        "if abs(det(A)) >= 1e-3\n            sol  = A \\ b" =>
        "factor = D == 3 ? lu(A; check = false) : A\n        if abs(det(factor)) >= 1e-3\n            sol = factor \\ b"))
    typed = kernel_variant("PerfMDBCTypeFirst", s -> replace(s,
        "if xᵢⱼ² <= H²\n            if ParticleType[j] == Fluid" =>
        "if ParticleType[j] == Fluid\n            if xᵢⱼ² <= H²"))
    variants = [("baseline", baseline, 128), ("lazy_rows", lazy, 128),
        ("registers64", registers, 128), ("threads64", baseline, 64),
        ("threads256", baseline, 256), ("reused_lu", reused, 128), ("type_first", typed, 128)]
    Base.invokelatest(measure_kernel_screen, cases, variants)
end

function measure_kernel_screen(cases, variants)
    path = get(ENV, "SPH_PERF_KERNEL_RESULTS", joinpath(@__DIR__, "performance_kernel_screen.csv"))
    open(path, "w") do out
        println(out, "case,subdivision,kernel,variant,round,us,max_abs_difference,exact")
        for case in cases, subdivision in (1, 2)
            v = setup_performance(case; subdivision)
            for _ in 1:3
                advance!(v)
            end
            SPHExampleGPU.GPUKernels.launch_pos_cells!(v.sup.PosCells, v.gpu.Position,
                v.gpu.CellID, v.cl.grid_dev, HostStep(0f0, 0f0), v.kw.SimKernel)
            SPHExampleGPU.GPUKernels.launch_inv_density!(v.sup.InvDensity, v.gpu.Density, HostStep(0f0, 0f0))
            original_density, original_pressure = copy(v.gpu.Density), copy(v.gpu.Pressure)
            for kind in (:interaction, :mdbc)
                kind === :mdbc && isempty(v.gpu.GhostIndex) && continue
                selected = filter(x -> kind === :interaction ? !(x[1] in ("reused_lu", "type_first", "hand_solve")) : !(x[1] in ("lazy_rows", "const_loads")), variants)
                graphs, differences, exact = CUDA.CuGraphExec[], Float64[], Bool[]
                reference = nothing
                for (name, kernels, threads) in selected
                    copyto!(v.gpu.Density, original_density); copyto!(v.gpu.Pressure, original_pressure)
                    launch = launch_probe(kernels, kind, v; threads)
                    launch(); CUDA.synchronize()
                    fields = kind === :mdbc ? (Array(v.gpu.Density), Array(v.gpu.Pressure)) :
                        (Array(v.sup.dρdtI), Array(v.gpu.Acceleration), Array(v.sup.∇Cᵢ), Array(v.sup.∇◌rᵢ))
                    reference === nothing && (reference = fields)
                    push!(differences, maximum(maximum(norm.(a .- b); init = 0.0) for (a,b) in zip(fields, reference)))
                    push!(exact, fields == reference)
                    push!(graphs, probe_graph(launch))
                end
                times = [Float64[] for _ in selected]
                for round in 1:7
                    indices = isodd(round) ? eachindex(selected) : reverse(eachindex(selected))
                    for index in indices
                        elapsed = CUDA.@elapsed CUDA.launch(graphs[index])
                        us = 1e6 * elapsed / 64
                        push!(times[index], us)
                        println(out, join((case.name, subdivision, kind, selected[index][1], round, us,
                            differences[index], exact[index]), ','))
                    end
                end
                for index in eachindex(selected)
                    @printf("%s R=%d %-11s %-12s %8.2f us ratio=%.3f exact=%s max_abs=%.4g\n",
                        case.name, subdivision, string(kind), selected[index][1], median(times[index]),
                        median(times[index] ./ times[1]), exact[index], differences[index])
                end
                flush(stdout); flush(out)
            end
            v = nothing
            GC.gc(); CUDA.reclaim()
        end
    end
end

function profile_cases(cases)
    for case in cases
        v = setup_performance(case)
        for _ in 1:3
            advance!(v)
        end
        before = v.kw.SimMetaData.Iteration
        prof = redirect_stdout(devnull) do
            CUDA.@profile trace=true advance!(v)
        end
        steps = v.kw.SimMetaData.Iteration - before
        @assert steps > 0
        totals = Dict{String,Float64}()
        calls = Dict{String,Int}()
        for i in eachindex(prof.device.name)
            name = first(split(String(prof.device.name[i]), '('))
            totals[name] = get(totals, name, 0.0) + prof.device.stop[i] - prof.device.start[i]
            calls[name] = get(calls, name, 0) + 1
        end
        println(case.name, ": ", steps, " steps, ", length(v.gpu), " particles")
        for name in sort(collect(keys(totals)); by = n -> -totals[n])
            @printf("  %-45s %8.2f us/step %4d calls\n", first(name, 45), 1e6 * totals[name] / steps, calls[name])
        end
        flush(stdout)
    end
end

function graph_preparation_source(source)
    # Both graph capture and its direct-launch fallback must prepare the header.
    loop = "for _ in 1:steps\n                enqueue_step!(ctx, false)\n            end"
    source = replace(source, loop => loop * "\n            prepare_cell_grid!(ctx.cl, ctx.gpu.Position, ctx.SimKernel.H⁻¹; step = state)")
    return replace(source,
        "@timeit HourGlass \"02a Prepare Cell Grid\" prepare_cell_grid!(cl, gpu.Position, SimKernel.H⁻¹; step = state)" =>
        "if !use_graph\n            @timeit HourGlass \"02a Prepare Cell Grid\" prepare_cell_grid!(cl, gpu.Position, SimKernel.H⁻¹; step = state)\n        end")
end

function driver_screen(cases)
    kernels = kernel_variant("PerfDriverKernels")
    baseline = driver_variant("PerfBaselineDriver", kernels)
    candidate = driver_variant("PerfGraphPreparation", kernels; transform = graph_preparation_source)
    Base.invokelatest(measure_driver_screen, cases, [("baseline", baseline), ("graph_prepare", candidate)])
end

function measure_driver_screen(cases, drivers)
    open(joinpath(@__DIR__, "performance_driver_screen.csv"), "w") do out
        println(out, "case,variant,round,seconds,steps,rebuilds,readbacks,exact")
        for case in cases
            variants = [setup_performance(case; driver, interval = 0.05) for (_, driver) in drivers]
            for _ in 1:3, v in variants
                advance!(v)
            end
            times = [Float64[] for _ in variants]
            for round in 1:7
                indices = isodd(round) ? eachindex(variants) : reverse(eachindex(variants))
                for index in indices
                    v = variants[index]
                    elapsed = @elapsed begin advance!(v); CUDA.synchronize() end
                    push!(times[index], elapsed)
                end
                states = state_snapshot.(variants)
                for index in eachindex(variants)
                    state = states[index]
                    exact = state == states[1]
                    println(out, join((case.name, drivers[index][1], round, times[index][end],
                        state.steps, state.rebuilds, state.readbacks, exact), ','))
                    @assert exact "driver changed the accepted trajectory or work counts"
                end
            end
            for index in eachindex(variants)
                @printf("%s %-15s median %.6f s ratio=%.3f exact=true\n", case.name,
                    drivers[index][1], median(times[index]), median(times[index] ./ times[1]))
            end
            flush(out); flush(stdout)
            variants = nothing
            GC.gc(); CUDA.reclaim()
        end
    end
end

function output_digest(save, meta)
    h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do file
        root = file["VTKHDF"]
        arrays = [read(root["Points"]), read(root["Steps/Values"])]
        append!(arrays, [read(root["PointData/" * key]) for key in sort(collect(keys(root["PointData"])))])
        return [bytes2hex(sha256(reinterpret(UInt8, vec(a)))) for a in arrays]
    end
end

function full_run(case, driver; subdivision = 1)
    mktempdir() do save
        T = get(ENV, "SPH_PERF_PRECISION", "Float32") == "Float64" ? Float64 : Float32
        kw = case.build(T, save)
        meta = kw.SimMetaData
        meta.SimulationTime = startswith(case.name, "GeneratedStillWedge") ? 4f0 : case.dims == 3 ? 0.3f0 : 0.5f0
        meta.OutputTimes = startswith(case.name, "GeneratedStillWedge") ? 0.01f0 : meta.SimulationTime / 10
        meta.GPUDoublePosition = get(ENV, "SPH_PERF_DOUBLE_POSITION", "true") == "true"
        meta.GPUMaxStepsPerSync = 32
        meta.GPUCellSubdivision = subdivision
        meta.GPUOutputQueueBytes = 8 * 1024^2
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        wall = redirect_stdout(devnull) do
            @elapsed driver.RunSimulation(; kw..., SimParticles = particles, SimLogger = logger)
        end
        order = sortperm(particles.ID)
        state = map(f -> getproperty(particles, f)[order], (:Position, :Velocity, :Density, :Pressure, :Acceleration))
        return (; wall, loop = loop_time(meta.HourGlass), steps = meta.Iteration,
            time = meta.TotalTime, state, digest = output_digest(save, meta))
    end
end

function full_screen(cases)
    kernels = kernel_variant("PerfFullBaselineKernels")
    baseline = driver_variant("PerfFullBaselineDriver", kernels)
    Base.invokelatest(measure_full_screen, cases, baseline)
end

function repeatability_screen(cases)
    kernels = kernel_variant("PerfRepeatBaselineKernels")
    baseline = driver_variant("PerfRepeatBaselineDriver", kernels)
    Base.invokelatest(measure_full_screen, cases, baseline; candidate = baseline)
end

function measure_full_screen(cases, baseline; candidate = SPHExampleGPU.SPHCellList)
    drivers = [("baseline", baseline), ("candidate", candidate)]
    path = get(ENV, "SPH_PERF_RESULTS", joinpath(@__DIR__, "performance_full_runs.csv"))
    subdivision = parse(Int, get(ENV, "SPH_PERF_SUBDIVISION", "1"))
    rounds = parse(Int, get(ENV, "SPH_PERF_ROUNDS", "5"))
    open(path, "w") do out
        println(out, "case,subdivision,variant,round,wall_s,loop_s,steps,final_time,exact_state,exact_output,max_scaled_difference")
        for case in cases
            for (_, driver) in drivers
                full_run(case, driver; subdivision)
            end
            println("Warmed ", case.name); flush(stdout)
            times = [Float64[] for _ in drivers]
            loops = [Float64[] for _ in drivers]
            for round in 1:rounds
                results = Vector{Any}(undef, 2)
                for index in (isodd(round) ? (1, 2) : (2, 1))
                    GC.gc(); CUDA.reclaim()
                    results[index] = full_run(case, drivers[index][2]; subdivision)
                end
                reference = results[1]
                for index in eachindex(drivers)
                    result = results[index]
                    delta = maximum(maximum(norm.(a .- b)) / max(1, maximum(norm.(b)))
                        for (a,b) in zip(result.state, reference.state))
                    exact_state = result.state == reference.state && result.steps == reference.steps && result.time == reference.time
                    exact_output = result.digest == reference.digest
                    println(out, join((case.name, subdivision, drivers[index][1], round,
                        result.wall, result.loop, result.steps, result.time, exact_state, exact_output, delta), ','))
                    push!(times[index], result.wall); push!(loops[index], result.loop)
                    if !(exact_state && exact_output)
                        println("NUMERICAL MISMATCH ", case.name, " ", drivers[index][1], " delta=", delta,
                            " exact_state=", exact_state, " exact_output=", exact_output)
                    end
                end
                flush(out); flush(stdout)
            end
            @printf("%s R=%d wall %.4f -> %.4f s ratio=%.3f loop_ratio=%.3f\n", case.name,
                subdivision, median(times[1]), median(times[2]), median(times[2] ./ times[1]),
                median(loops[2] ./ loops[1]))
            flush(stdout)
        end
    end
end

function main()
    mode = isempty(ARGS) ? "kernels" : ARGS[1]
    names = length(ARGS) > 1 ? ARGS[2:end] : ["StillWedge2D_MDBC", "MovingSquare2D_dp0.02", "DamBreak3D_dp0.0085", "Duckling3D_MDBC_dp0.005"]
    cases = filter(c -> any(name -> occursin(name, c.name), names), PERFORMANCE_CASES)
    isempty(cases) && error("no matching benchmark case")
    println("Baseline ", PERF_REVISION, "; GPU ", CUDA.name(CUDA.device()), "; Julia ", VERSION)
    if mode == "kernels"
        kernel_screen(cases)
    elseif mode == "drivers"
        driver_screen(cases)
    elseif mode == "full"
        full_screen(cases)
    elseif mode == "advanced"
        advanced_screen(cases)
    elseif mode == "repeatability"
        repeatability_screen(cases)
    elseif mode == "exact"
        baseline = kernel_variant("PerfExactBaseline")
        Base.invokelatest(measure_kernel_screen, cases,
            [("baseline", baseline, 128), ("candidate", SPHExampleGPU.GPUKernels, 128)])
    else
        profile_cases(cases)
    end
end

include("performance_candidates.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
