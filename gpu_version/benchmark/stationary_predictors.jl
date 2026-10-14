# Warmed, alternating comparisons with the pre-predictor-compaction solver.
# Run from the repository root:
# julia --project=gpu_version -t 2,0 gpu_version/benchmark/stationary_predictors.jl [user default large]
module StationaryPredictorBenchmark

using SPHExampleGPU, CUDA, StaticArrays, Statistics, Printf, LinearAlgebra

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const BASELINE = get(ENV, "SPH_PREDICTOR_BASELINE", "7966a05")
include(joinpath(@__DIR__, "memory_buffers.jl"))

revision_source(file) = read(`git -C $REPO show $(BASELINE * ":gpu_version/src/" * file)`, String)

function baseline_modules()
    name = "PredictorBaselineKernels"
    source = replace(revision_source("GPUKernels.jl"),
        "module GPUKernels" => "module $name",
        "include(\"UpdatedMDBC.jl\")" => revision_source("UpdatedMDBC.jl"))
    Base.include_string(SPHExampleGPU, source, joinpath(REPO, "gpu_version", "src", "GPUKernels.jl"))
    kernels = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(name))
    name = "PredictorBaselineDriver"
    source = replace(revision_source("SPHCellList.jl"),
        "module SPHCellList" => "module $name",
        "using ..GPUKernels" => "using ..PredictorBaselineKernels",
        "GPUKernels." => "PredictorBaselineKernels.")
    Base.include_string(SPHExampleGPU, source, joinpath(REPO, "gpu_version", "src", "SPHCellList.jl"))
    return (; kernels, driver = Base.invokelatest(getproperty, SPHExampleGPU, Symbol(name)), compact = false, name = "parent")
end

const PARENT = baseline_modules()

function setup(variant; dx, subdivision, radial, interval = 0.01)
    kw = GPUMemoryBenchmark.generated_wedge_case(Float32, tempdir(); dx)
    metadata = kw.SimMetaData
    metadata.GPUDoublePosition = true
    metadata.GPUCellSubdivision = subdivision
    metadata.GPURadialInteractions = radial
    metadata.GPUUseGraph = true
    metadata.GPUMaxStepsPerSync = 32
    metadata.TimeSteppingMode = SymplecticTimeStepping()
    metadata.SimulationTime = 100f0
    metadata.OutputTimes = Float32(interval)
    metadata.OutputIterationCounter = 1
    host = AllocateDataStructures(kw.SimGeometry, metadata)
    Pressure!(host.Pressure, host.Density, kw.SimConstants)
    driver = variant.driver
    gpu = driver.upload_particles(host, metadata)
    cl = CellListWorkspace{2, Float64}(length(host); reach = subdivision)
    sup = variant.compact ? driver.GPUSupportArrays{2, Float32}(gpu, metadata) :
                           driver.GPUSupportArrays{2, Float32}(length(host), metadata)
    red = ReductionWorkspace{SVector{3, Float32}}(length(host))
    motion = driver.MotionArrays(kw.SimGeometry, host)
    state = StepState{Float32}(; dx = 1f0 + kw.SimKernel.h)
    return (; variant, kw, host, gpu, cl, sup, red, motion, state)
end

function advance!(run)
    (; kw, gpu, cl, sup, red, motion, state, variant) = run
    variant.driver.SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel,
        kw.SimMetaData, kw.SimConstants, gpu, cl, sup, red, motion, state)
    kw.SimMetaData.OutputIterationCounter += 1
    return nothing
end

function snapshot(run)
    order = sortperm(Array(run.gpu.ID))
    return (; fields = map(name -> Array(getproperty(run.gpu, name))[order],
        (:Position, :Velocity, :Density, :Pressure, :Acceleration)),
        steps = run.kw.SimMetaData.Iteration, time = run.kw.SimMetaData.TotalTime,
        rebuilds = run.cl.nrebuilds)
end

function summarize(times, variants, label)
    baseline = times[1]
    for i in eachindex(times)
        ratios = times[i] ./ baseline
        @printf("%s %-10s median=%9.2f us/step paired_ratio=%.4f range=[%.4f, %.4f]\n",
            label, variants[i].name, median(times[i]), median(ratios), minimum(ratios), maximum(ratios))
    end
    flush(stdout)
end

function solver_case(variants; dx, subdivision, radial, rounds = 21, intervals = 10)
    label = "dx=$dx R=$subdivision radial=$radial"
    runs = [setup(variant; dx, subdivision, radial) for variant in variants]
    @printf("\n%s particles=%d fixed=%d\n", label, length(runs[1].host),
        count(==(Fixed), runs[1].host.Type))
    flush(stdout)
    for _ in 1:4, run in runs
        advance!(run)
    end
    reference = snapshot(runs[1])
    for run in runs[2:end]
        snapshot(run) == reference || error("Warmup states or work counts differ for $(run.variant.name)")
    end
    times = [Float64[] for _ in runs]
    for round in 1:rounds
        work = Tuple{Int, Int, Int}[]
        order = isodd(round) ? eachindex(runs) : reverse(eachindex(runs))
        for i in order
            run = runs[i]
            before = run.kw.SimMetaData.Iteration
            rebuilds = run.cl.nrebuilds
            readbacks = run.state.readbacks
            elapsed = @elapsed for _ in 1:intervals
                advance!(run)
            end
            steps = run.kw.SimMetaData.Iteration - before
            push!(times[i], 1e6 * elapsed / steps)
            push!(work, (steps, run.cl.nrebuilds - rebuilds, run.state.readbacks - readbacks))
        end
        all(==(first(work)), work) || error("Measured step/rebuild/readback counts differ")
        reference = snapshot(runs[1])
        for run in runs[2:end]
            snapshot(run) == reference || error("Measured states or work counts differ for $(run.variant.name)")
        end
    end
    summarize(times, variants, label)
    return runs
end

function phase_graph(launch; repeats = 256)
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

function phase_screen(runs; rounds = 15, repeats = 256)
    variants = getproperty.(runs, :variant)
    for phase in (:half, :corrector_interactions, :final)
        graphs = CUDA.CuGraphExec[]
        launches = Function[]
        for run in runs
            (; kw, gpu, sup, cl, motion, variant) = run
            kernels = variant.kernels
            step = HostStep(0f0, 0f0)
            position_half, velocity_half = variant.compact ? variant.driver.predictor_arrays(sup, gpu) :
                                                           (sup.Positionₙ⁺, sup.Velocityₙ⁺)
            half!() = kernels.launch_half_step!(position_half, velocity_half, sup.ρₙ⁺, gpu.Pressure,
                gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, sup.dρdtI,
                gpu.Type, gpu.GroupMarker, motion, step, kw.SimConstants;
                pos_cells = sup.PosCellsₙ⁺, CellID = gpu.CellID, grid = cl.grid_dev, SimKernel = kw.SimKernel,
                boundary_mode = NoMDBC(),
                (variant.compact ? (; reuse_predictors = true) : (;))...)
            half!()
            launch = if phase === :half
                half!
            elseif phase === :corrector_interactions
                () -> kernels.launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
                    sup.∇Cᵢ, sup.∇◌rᵢ, position_half, sup.ρₙ⁺, gpu.Pressure, velocity_half, gpu.Type,
                    cl.CellStart, gpu.CellID, cl.grid_dev, step,
                    kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel, kw.SimConstants,
                    Val(false), Val(false); lanes = Val(choose_lanes(length(gpu), Val(2))),
                    pos_cells = sup.PosCellsₙ⁺, boundary_forces = Val(true),
                    radial = Val(kw.SimMetaData.GPURadialInteractions))
            else
                () -> kernels.launch_final_step!(gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density,
                    gpu.Pressure, sup.dρdtI, sup.ρₙ⁺, position_half, velocity_half, gpu.Type,
                    sup.∇Cᵢ, sup.∇◌rᵢ, step, kw.SimKernel, kw.SimConstants, run.red, Val(false);
                    boundary_mode = NoMDBC())
            end
            push!(launches, launch)
            push!(graphs, phase_graph(launch; repeats))
        end
        for _ in 1:4, graph in graphs
            CUDA.launch(graph)
        end
        CUDA.synchronize()
        elapsed = median([CUDA.@elapsed CUDA.launch(graphs[1]) for _ in 1:3])
        sample_repeats = clamp(ceil(Int, repeats * 0.02 / elapsed), repeats, 4096)
        if sample_repeats != repeats
            graphs = [phase_graph(launch; repeats = sample_repeats) for launch in launches]
        end
        start = time_ns()
        while (time_ns() - start) / 1e9 < 0.3
            foreach(CUDA.launch, graphs)
            CUDA.synchronize()
        end
        times = [Float64[] for _ in graphs]
        for round in 1:rounds
            order = isodd(round) ? eachindex(graphs) : reverse(eachindex(graphs))
            for i in order
                push!(times[i], 1e6 * (CUDA.@elapsed CUDA.launch(graphs[i])) / sample_repeats)
            end
        end
        summarize(times, variants, string(phase))
    end
end

function main(args = ARGS)
    CUDA.functional() || error("Stationary predictor benchmarks require a functional CUDA device")
    println("GPU: ", CUDA.name(CUDA.device()), "; Julia ", VERSION, "; CUDA ", pkgversion(CUDA))
    println("Float32 physics / Float64 positions; CUDA graphs; solver only (no output or visualization)")
    println("Baseline revision: ", BASELINE, "; dense control uses the current kernels with full predictors")
    parent = PARENT
    current = (; kernels = SPHExampleGPU.GPUKernels, driver = SPHExampleGPU.SPHCellList,
        compact = true, name = "shared")
    full = (; kernels = SPHExampleGPU.GPUKernels, driver = SPHExampleGPU.SPHCellList,
        compact = false, name = "dense")
    variants = [parent, full, current]
    selections = isempty(args) ? ["user", "default", "large"] : args
    for selection in selections
        selection in ("user", "default", "large") || throw(ArgumentError("unknown benchmark case: $selection"))
        config = selection == "user" ? (; dx = 0.02, subdivision = 8, radial = true) :
                 selection == "large" ? (; dx = 0.005, subdivision = 1, radial = false) :
                 (; dx = 0.02, subdivision = 1, radial = false)
        runs = solver_case(variants; config...)
        phase_screen(runs)
        GC.gc()
        CUDA.reclaim()
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    StationaryPredictorBenchmark.main()
end
