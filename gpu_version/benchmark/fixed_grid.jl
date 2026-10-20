module FixedGridBenchmark

using SPHExampleGPU, CUDA, StaticArrays, Statistics, Printf

include(joinpath(@__DIR__, "cases.jl"))

function setup(; fixed_grid)
    kwargs = moving_square_2d(Float32, tempdir())
    metadata = kwargs.SimMetaData
    metadata.GPUFixedGrid = fixed_grid
    metadata.GPUDoublePosition = true
    metadata.TimeSteppingMode = SingleNeighborTimeStepping()
    metadata.SimulationTime = 10f0
    metadata.OutputTimes = 0.01f0
    metadata.OutputIterationCounter = 1
    host = AllocateDataStructures(kwargs.SimGeometry, metadata)
    gpu = upload_particles(host, metadata)
    cells = CellListWorkspace{2, Float64}(length(host); fixed_grid)
    support = GPUSupportArrays{2, Float32}(gpu, metadata)
    reduction = ReductionWorkspace{SVector{3, Float32}}(length(host))
    motion = MotionArrays(kwargs.SimGeometry, host)
    state = StepState{Float32}(; dx = 1f0 + kwargs.SimKernel.h)
    return (; kwargs, metadata, gpu, cells, support, reduction, motion, state)
end

function advance!(run)
    (; kwargs, metadata, gpu, cells, support, reduction, motion, state) = run
    SimulationLoop(kwargs.SimDensityDiffusion, kwargs.SimViscosity, kwargs.SimKernel,
        metadata, kwargs.SimConstants, gpu, cells, support, reduction, motion, state)
    metadata.OutputIterationCounter += 1
    return nothing
end

work(run) = (run.metadata.Iteration, run.cells.nrebuilds, run.state.readbacks,
             run.cells.grid_status_readbacks)

function snapshot(run)
    order = sortperm(Array(run.gpu.ID))
    fields = map(f -> Array(getproperty(run.gpu, f))[order],
                 (:Position, :Velocity, :Density, :Acceleration))
    return (; fields, work = work(run), time = run.metadata.TotalTime)
end

function main(; rounds = 9, intervals = 5)
    println("GPU: ", CUDA.name(CUDA.device()), "; Julia ", VERSION, "; CUDA.jl ", Base.pkgversion(CUDA),
        "; Float32 physics / Float64 positions; MovingSquare2D LaminarSPS / planar shifting")
    println("Paired whole-solver intervals, including natural rebuilds and readbacks; excludes setup, output and state verification.")
    println("Five warmup intervals per variant; alternating variant order; exact particle states and work counts required.")
    runs = [setup(; fixed_grid) for fixed_grid in (false, true)]
    for _ in 1:5, run in runs
        advance!(run)
    end
    snapshot(runs[1]) == snapshot(runs[2]) || error("Warm states or work counts differ")
    times = [Float64[], Float64[]]
    counts = NTuple{4, Int}[]
    for round in 1:rounds
        paired_work = NTuple{4, Int}[]
        for index in (isodd(round) ? (1, 2) : (2, 1))
            run = runs[index]
            before = work(run)
            CUDA.synchronize()
            elapsed = @elapsed begin
                for _ in 1:intervals
                    advance!(run)
                end
                CUDA.synchronize()
            end
            performed = work(run) .- before
            push!(paired_work, performed)
            push!(times[index], elapsed)
        end
        paired_work[1] == paired_work[2] || error("Measured work counts differ")
        snapshot(runs[1]) == snapshot(runs[2]) || error("Measured particle states differ")
        push!(counts, first(paired_work))
        @printf("round=%d dynamic_ms=%.3f fixed_ms=%.3f speedup=%.4fx work=%s\n",
            round, 1e3 * last(times[1]), 1e3 * last(times[2]),
            last(times[1]) / last(times[2]), string(first(paired_work)))
    end
    ratios = times[1] ./ times[2]
    totals = reduce((a, b) -> a .+ b, counts)
    totals[2] > 0 || error("No rebuilds occurred in measured intervals; use a longer run")
    @printf("particles=%d rounds=%d dynamic_ms=%.3f fixed_ms=%.3f median_paired_speedup=%.4fx range=[%.4f,%.4f]\n",
        length(runs[1].gpu), rounds, 1e3 * median(times[1]), 1e3 * median(times[2]),
        median(ratios), minimum(ratios), maximum(ratios))
    println("Matched timed counts per variant (steps, rebuilds, timestep readbacks, grid readbacks): ",
        totals)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    FixedGridBenchmark.main()
end
