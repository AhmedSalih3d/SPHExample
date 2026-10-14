# Compare the current solver with the cached-density baseline in isolated modules.
# Run from the repository root: julia --project=gpu_version gpu_version/benchmark/inverse_density.jl
module InverseDensityBenchmark

using SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, Statistics, Printf

include(joinpath(@__DIR__, "cases.jl"))
include(joinpath(@__DIR__, "memory_buffers.jl"))

const CACHED_REVISION = "d4ae12b99f2adc247ea1e79b896ececa105fca02"
const REPOSITORY = normpath(joinpath(@__DIR__, "..", ".."))
baseline_source(file) = replace(read(`git -C $REPOSITORY show $(CACHED_REVISION * ":gpu_version/src/" * file)`, String), "\r\n" => "\n")

function replace_once(source, before, after)
    length(findall(before, source)) == 1 || error("Expected exactly one source match for: $before")
    return replace(source, before => after)
end

function load_variant(name; shared = false)
    source_dir = joinpath(@__DIR__, "..", "src")
    kernel_name = name * "Kernels"
    source = baseline_source("GPUKernels.jl")
    source = replace_once(source, "module GPUKernels", "module $kernel_name")
    source = replace_once(source, "include(\"UpdatedMDBC.jl\")",
        baseline_source("UpdatedMDBC.jl"))
    Base.include_string(SPHExampleGPU, source, joinpath(source_dir, "GPUKernels.jl"))
    kernels = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(kernel_name))

    source = baseline_source("SPHCellList.jl")
    source = replace_once(source, "module SPHCellList", "module $name")
    source = replace_once(source, "using ..GPUKernels", "using ..$kernel_name")
    source = replace(source, "GPUKernels." => "$kernel_name.")
    allocations = "        CUDA.zeros(T, n),\n        CUDA.zeros(T, n),\n        CUDA.zeros(T, n),\n" *
                  "        CUDA.zeros(SVector{D, T}, shifting ? n : 0),"
    if shared
        source = replace_once(source, "    return GPUSupportArrays{D, T, TP, typeof(pc), typeof(boundary_data)}(",
            "    inverse_density = CUDA.zeros(T, n)\n" *
            "    return GPUSupportArrays{D, T, TP, typeof(pc), typeof(boundary_data)}(")
        source = replace_once(source, allocations,
            "        CUDA.zeros(T, n),\n        inverse_density,\n        inverse_density,\n" *
            "        CUDA.zeros(SVector{D, T}, shifting ? n : 0),")
    end
    Base.include_string(SPHExampleGPU, source, joinpath(source_dir, "SPHCellList.jl"))
    driver = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(name))
    return (; name, kernels, driver)
end

function uniform_3d(::Type{T}, save) where {T}
    dx = T(0.02)
    constants = SimulationConstants{T}(; dx, c₀ = 33.14, CFL = 0.2, g = 0, m₀ = 1000 * dx^3)
    positions = vec([SVector{3,T}(dx * i, dx * j, dx * k) for i in 0:31, j in 0:31, k in 0:31])
    density = [constants.ρ₀ * (1 + T(0.002) * sin(x[1]) * cos(x[2])) for x in positions]
    geometry = [SPHGeometry{3,T}(positions; Density = density, GroupMarker = 1, Type = Fluid)]
    meta = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "InverseDensity3D", SaveLocation = save,
        VisualizeInParaview = false, OpenLogFile = false)
    return (; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
        SimKernel = SPHKernelInstance{3,T}(WendlandC2(); dx),
        SimViscosity = ArtificialViscosity(), SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(), ParticleNormalsPath = nothing)
end

const CASES = [
    BenchCase("GeneratedWedge2D_dp0.005", 2,
        (T, save) -> merge(GPUMemoryBenchmark.generated_wedge_case(T, save; dx = 0.005),
            (; SimTimeStepping = SymplecticTimeStepping()))),
    first(filter(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)),
    first(filter(c -> c.name == "MovingSquare2D_dp0.02", BENCH_CASES)),
    first(filter(c -> c.name == "DamBreak3D_dp0.02", BENCH_CASES)),
    BenchCase("Uniform3D_32cubed", 3, uniform_3d),
]

function setup(case, variant, ::Type{T}; double_position, subdivision, single_neighbor) where {T}
    kw = case.build(T, tempdir())
    meta = kw.SimMetaData
    meta.SimulationTime = T(100)
    meta.OutputTimes = T(0.01)
    meta.GPUDoublePosition = double_position
    meta.GPUCellSubdivision = subdivision
    meta.GPUMaxStepsPerSync = 32
    meta.GPUUseGraph = true
    meta.GPUSyncTimers = false
    meta.TimeSteppingMode = single_neighbor ? SingleNeighborTimeStepping() : kw.SimTimeStepping
    meta.OutputIterationCounter = 1
    host = AllocateDataStructures(kw.SimGeometry, meta)
    if kw.ParticleNormalsPath !== nothing
        _, ghosts, normals = LoadBoundaryNormals(Val(case.dims), eltype(eltype(host.Position)), kw.ParticleNormalsPath)
        host.GhostPoints[eachindex(ghosts)] .= ghosts
        host.GhostNormals[eachindex(normals)] .= normals
    end
    Pressure!(host.Pressure, host.Density, kw.SimConstants)
    gpu = variant.driver.upload_particles(host, meta)
    sup = variant.driver.GPUSupportArrays{case.dims,T}(length(host), meta)
    cl = CellListWorkspace{case.dims,position_float_type(meta)}(length(host); reach = subdivision)
    red = ReductionWorkspace{SVector{3,T}}(length(host))
    motion = variant.driver.MotionArrays(kw.SimGeometry, host)
    state = StepState{T}(dx = one(T) + kw.SimKernel.h)
    return (; case, variant, kw, gpu, sup, cl, red, motion, state)
end

function advance!(v)
    (; kw, gpu, sup, cl, red, motion, state, variant) = v
    variant.driver.SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel,
        kw.SimMetaData, kw.SimConstants, gpu, cl, sup, red, motion, state)
    kw.SimMetaData.OutputIterationCounter += 1
    return nothing
end

scheme(v) = v.kw.SimMetaData.TimeSteppingMode isa SingleNeighborTimeStepping ? "single" : "symplectic"

function snapshot(v)
    order = sortperm(Array(v.gpu.ID))
    fields = map(name -> Array(getproperty(v.gpu, name))[order],
        (:Position, :Velocity, :Density, :Pressure, :Acceleration))
    all(a -> all(x -> x isa Number ? isfinite(x) : all(isfinite, x), a), fields) ||
        error("Nonfinite particle state in $(v.variant.name)")
    all(>(0), fields[3]) || error("Nonpositive density in $(v.variant.name)")
    return (; fields, steps = v.kw.SimMetaData.Iteration, time = v.kw.SimMetaData.TotalTime,
        rebuilds = v.cl.nrebuilds, readbacks = v.state.readbacks)
end

function interaction_launch(v, kernels; prepare = false)
    (; kw, gpu, sup, cl) = v
    T = eltype(gpu.Density)
    step = HostStep(zero(T), zero(T))
    shift = Val(kw.SimMetaData isa SimulationMetaData{D,T,PlanarShifting} where {D,T})
    mdbc = !(kw.SimMetaData isa SimulationMetaData{D,T,S,K,NoMDBC} where {D,T,S,K})
    lanes = Val(choose_lanes(length(gpu), Val(v.case.dims)))
    cached = isdefined(kernels, :launch_inv_density!)
    density_args = cached ? (gpu.Density, sup.InvDensity, gpu.Pressure) : (gpu.Density, gpu.Pressure)
    return () -> begin
        if prepare
            if cached && mdbc
                kernels.launch_inv_density!(sup.InvDensity, gpu.Density, step)
            elseif cached
                kernels.launch_inv_density!(sup.InvDensity, gpu.Density, step;
                    pos_cells = sup.PosCells, Position = gpu.Position, CellID = gpu.CellID,
                    grid = cl.grid_dev, H = kw.SimKernel.H)
            elseif !mdbc
                kernels.launch_prepare_interactions!(gpu.Position, step;
                    pos_cells = sup.PosCells, CellID = gpu.CellID, grid = cl.grid_dev, H = kw.SimKernel.H)
            end
        end
        kernels.launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
            sup.∇Cᵢ, sup.∇◌rᵢ, gpu.Position, density_args...,
            gpu.Velocity, gpu.Type, cl.CellStart, gpu.CellID, cl.grid_dev, step,
            kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel, kw.SimConstants,
            Val(false), shift; threads = kw.SimMetaData.GPUInteractionThreads, lanes,
            boundary_forces = Val(kw.SimMetaData.GPUBoundaryForces), pos_cells = sup.PosCells)
    end
end

function probe_graph(launch, repeats)
    launch()
    CUDA.synchronize()
    graph = CUDA.capture() do
        for _ in 1:repeats
            launch()
        end
    end
    exec = CUDA.instantiate(graph)
    CUDA.launch(exec)
    CUDA.synchronize()
    return exec
end

function warm_graphs(graphs)
    start = time_ns()
    while (time_ns() - start) / 1e9 < 0.3
        for graph in graphs
            CUDA.launch(graph)
        end
        CUDA.synchronize()
    end
end

function interaction_screen(v, variants, out; rounds, repeats)
    # Use one evolved state and refresh the caches for its full-step density
    # and positions, rather than the predictor state of the preceding step.
    v.gpu.Pressure .= EquationOfStateGamma7.(v.gpu.Density, v.kw.SimConstants.c₀, v.kw.SimConstants.ρ₀)
    variants[1].kernels.launch_inv_density!(v.sup.InvDensity, v.gpu.Density,
        HostStep(zero(eltype(v.gpu.Density)), zero(eltype(v.gpu.Density)));
        pos_cells = v.sup.PosCells, Position = v.gpu.Position,
        CellID = v.gpu.CellID, grid = v.cl.grid_dev, H = v.kw.SimKernel.H)
    for prepare in (false, true)
        graphs = CUDA.CuGraphExec[]
        launches = Function[]
        reference = nothing
        for variant in variants[1:2]
            launch = interaction_launch(v, variant.kernels; prepare)
            launch()
            CUDA.synchronize()
            fields = map(Array, (v.sup.dρdtI, v.gpu.Acceleration, v.sup.∇Cᵢ, v.sup.∇◌rᵢ))
            reference === nothing && (reference = fields)
            isequal(fields, reference) || error("Interaction outputs differ for $(variant.name)")
            push!(launches, launch)
            push!(graphs, probe_graph(launch, repeats))
        end
        warm_graphs(graphs)
        elapsed = CUDA.@elapsed CUDA.launch(graphs[1])
        sample_repeats = clamp(ceil(Int, repeats * 0.02 / elapsed), repeats, 4096)
        if sample_repeats != repeats
            graphs = [probe_graph(launch, sample_repeats) for launch in launches]
            warm_graphs(graphs)
        end
        times = [Float64[] for _ in graphs]
        for round in 1:rounds
            indices = isodd(round) ? eachindex(graphs) : reverse(eachindex(graphs))
            for index in indices
                us = 1e6 * (CUDA.@elapsed CUDA.launch(graphs[index])) / sample_repeats
                push!(times[index], us)
                println(out, join((v.case.name, eltype(v.gpu.Density), v.kw.SimMetaData.GPUDoublePosition,
                    v.kw.SimMetaData.GPUCellSubdivision, scheme(v), prepare ? "prepare_interaction" : "interaction",
                    variants[index].name, round, us, length(v.gpu), 0, 0, 0, true), ','))
            end
        end
        @printf("  %-20s cached=%9.2f us on_demand=%9.2f us ratio=%.3f [%.3f, %.3f] exact=true\n",
            prepare ? "prepare+interaction" : "interaction", median(times[1]), median(times[2]),
            median(times[2] ./ times[1]), minimum(times[2] ./ times[1]), maximum(times[2] ./ times[1]))
    end
end

function measure_case(case, variants, ::Type{T}, out; double_position, subdivision, single_neighbor, rounds, repeats) where {T}
    runs = [setup(case, variant, T; double_position, subdivision, single_neighbor) for variant in variants]
    for _ in 1:2, v in runs
        advance!(v)
    end
    start = time_ns()
    while (time_ns() - start) / 1e9 < 2
        for v in runs
            advance!(v)
        end
    end
    states = snapshot.(runs)
    all(s -> isequal(s, states[1]), states) || error("Warmup trajectories or work counts differ")
    v = runs[1]
    @printf("%s %s positions=%s R=%d N=%d scheme=%s\n", case.name, T, eltype(eltype(v.gpu.Position)),
        subdivision, length(v.gpu), single_neighbor ? "single" : "symplectic")
    bytes = [GPUMemoryBenchmark.device_buffer_bytes((r.gpu, r.sup, r.cl, r.red, r.motion, r.state.f, r.state.i)) for r in runs]
    expected = length(v.gpu) * sizeof(T)
    bytes[1] - bytes[2] == 2 * expected || error("On-demand allocation savings incorrect")
    bytes[1] - bytes[3] == expected || error("Shared allocation savings incorrect")
    runs[3].sup.InvDensity === runs[3].sup.InvDensityₙ⁺ || error("Shared vectors do not alias")
    @printf("  reciprocal storage: cached=%.4f MiB on_demand=0 shared=%.4f MiB\n",
        2 * expected / 2.0^20, expected / 2.0^20)

    times = [Float64[] for _ in runs]
    for round in 1:rounds
        indices = isodd(round) ? eachindex(runs) : reverse(eachindex(runs))
        counts = Tuple{Int,Int,Int}[]
        for index in indices
            r = runs[index]
            before = (r.kw.SimMetaData.Iteration, r.cl.nrebuilds, r.state.readbacks)
            elapsed = @elapsed begin
                advance!(r)
                CUDA.synchronize()
            end
            after = (r.kw.SimMetaData.Iteration, r.cl.nrebuilds, r.state.readbacks)
            steps, rebuilds, readbacks = after .- before
            steps > 0 || error("Timed interval executed no solver steps")
            push!(counts, (steps, rebuilds, readbacks))
            push!(times[index], 1e6 * elapsed / steps)
            println(out, join((case.name, T, double_position, subdivision, scheme(r), "solver",
                variants[index].name, round, times[index][end], length(r.gpu), steps, rebuilds, readbacks, true), ','))
        end
        all(==(counts[1]), counts) || error("Measured work counts differ")
        states = snapshot.(runs)
        all(s -> isequal(s, states[1]), states) || error("Timed trajectories differ")
    end
    for index in eachindex(runs)
        ratios = times[index] ./ times[1]
        @printf("  solver %-22s %9.2f us/step ratio=%.3f [%.3f, %.3f] exact=true live=%.3f MiB\n",
            variants[index].name, median(times[index]), median(ratios), minimum(ratios), maximum(ratios),
            bytes[index] / 2.0^20)
    end
    interaction_screen(runs[1], variants, out; rounds, repeats)
    flush(out)
    flush(stdout)
    return nothing
end

function main(args)
    for arg in args
        startswith(arg, "--") || continue
        arg in ("--float64", "--native-positions", "--single-neighbor") ||
            startswith(arg, "--subdivision=") || startswith(arg, "--output=") ||
            error("Unknown benchmark option: $arg")
    end
    CUDA.functional() || error("This benchmark requires a functional NVIDIA GPU")
    T = "--float64" in args ? Float64 : Float32
    double_position = !("--native-positions" in args) && T === Float32
    single_neighbor = "--single-neighbor" in args
    subdivision_arg = findfirst(a -> startswith(a, "--subdivision="), args)
    subdivision = subdivision_arg === nothing ? 1 : parse(Int, split(args[subdivision_arg], '=')[2])
    subdivision in (1, 2, 3) || error("Subdivision must be 1, 2 or 3")
    output_arg = findfirst(a -> startswith(a, "--output="), args)
    output = output_arg === nothing ? tempname() * ".csv" : split(args[output_arg], '='; limit = 2)[2]
    names = filter(a -> !startswith(a, "--"), args)
    cases = isempty(names) ? CASES : filter(c -> any(occursin(name, c.name) for name in names), CASES)
    isempty(cases) && error("No benchmark cases match $(join(names, ", "))")
    println("GPU: ", CUDA.name(CUDA.device()), "; Julia ", VERSION, "; CUDA.jl ", pkgversion(CUDA))
    println("Cached baseline: ", CACHED_REVISION, "; on-demand: current solver; shared: baseline allocation experiment.")
    println("No fast-math, precision, physics or neighbor changes.")
    println("Ratios are variant/cached time; <1 is faster. Nine paired alternating rounds; warmup and >=64 graph replays targeting 20 ms per kernel sample.")
    println("CSV: ", output)
    cached = merge(load_variant("CachedDensity"), (; name = "cached"))
    on_demand = (; name = "OnDemandDensity", kernels = SPHExampleGPU.GPUKernels, driver = SPHExampleGPU.SPHCellList)
    shared = load_variant("SharedDensity"; shared = true)
    open(output, "w") do out
        println(out, "case,precision,double_position,subdivision,scheme,measurement,variant,round,microseconds,particles,steps,rebuilds,readbacks,exact")
        for case in cases
            Base.invokelatest(measure_case, case, [cached, on_demand, shared], T, out;
                double_position, subdivision, single_neighbor, rounds = 9, repeats = 64)
            GC.gc()
            CUDA.reclaim()
        end
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    InverseDensityBenchmark.main(ARGS)
end
