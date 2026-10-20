module SortingShadowBenchmark

using SPHExampleGPU, CUDA, StaticArrays, Statistics, Printf

include(joinpath(@__DIR__, "memory_buffers.jl"))
const DRIVER = SPHExampleGPU.SPHCellList
const STEP = SPHExampleGPU.GPUStepState

function setup(kw; reconstruct, scheme)
    metadata = deepcopy(kw.SimMetaData)
    metadata.GPUDoublePosition = true
    metadata.TimeSteppingMode = scheme
    metadata.SimulationTime = 100f0
    metadata.OutputTimes = 0.01f0
    metadata.OutputIterationCounter = 1
    host = AllocateDataStructures(kw.SimGeometry, metadata)
    gpu = upload_particles(host; position_type = Float64, compact_metadata = true,
        kernel_output = false, ghost_data = false, reconstruct_fields = reconstruct)
    cl = CellListWorkspace{2, Float64}(length(host))
    sup = GPUSupportArrays{2, Float32}(gpu, metadata)
    red = ReductionWorkspace{SVector{3, Float32}}(length(host))
    motion = MotionArrays(kw.SimGeometry, host)
    state = StepState{Float32}(; dx = 1f0 + kw.SimKernel.h)
    return (; kw, metadata, gpu, cl, sup, red, motion, state)
end

function advance!(run)
    (; kw, metadata, gpu, cl, sup, red, motion, state) = run
    STEP.set_slot!(state.f, state.fh, STEP.F_DX, 1f0 + kw.SimKernel.h)
    SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel,
        metadata, kw.SimConstants, gpu, cl, sup, red, motion, state)
    metadata.OutputIterationCounter += 1
end

function snapshot(run)
    order = sortperm(Array(run.gpu.ID))
    return (; fields = map(f -> Array(getproperty(run.gpu, f))[order],
        (:Position, :Velocity, :Density, :Acceleration)),
        steps = run.metadata.Iteration, time = run.metadata.TotalTime,
        rebuilds = run.cl.nrebuilds, readbacks = run.state.readbacks)
end

function run_pair(dx, scheme)
    kw = GPUMemoryBenchmark.generated_wedge_case(Float32, tempdir(); dx)
    runs = [setup(kw; reconstruct, scheme) for reconstruct in (false, true)]
    for _ in 1:3, run in runs
        advance!(run)
    end
    snapshot(runs[1]) == snapshot(runs[2]) || error("Warm states differ")
    times = [Float64[], Float64[]]
    for round in 1:11
        work = Tuple{Int, Int, Int}[]
        for i in (isodd(round) ? (1, 2) : (2, 1))
            run = runs[i]
            before = (run.metadata.Iteration, run.cl.nrebuilds, run.state.readbacks)
            elapsed = @elapsed for _ in 1:2
                advance!(run)
            end
            after = (run.metadata.Iteration, run.cl.nrebuilds, run.state.readbacks)
            counts = after .- before
            push!(work, counts)
            push!(times[i], 1e6 * elapsed / counts[1])
        end
        work[1] == work[2] || error("Measured work differs")
        snapshot(runs[1]) == snapshot(runs[2]) || error("Measured states differ")
    end
    bytes(run) = GPUMemoryBenchmark.device_buffer_bytes((run.gpu, run.sup, run.cl))
    saved = bytes(runs[1]) - bytes(runs[2])
    @assert saved == 8 * length(runs[1].gpu)
    ratios = times[2] ./ times[1]
    @printf("dx=%.3f scheme=%s n=%d shadow_bytes_saved=%d full_us=%.2f removed_us=%.2f median_ratio=%.4f range=[%.4f,%.4f]\n",
        dx, nameof(typeof(scheme)), length(runs[1].gpu), saved, median(times[1]), median(times[2]),
        median(ratios), minimum(ratios), maximum(ratios))
    flush(stdout)
end

function main()
    println("GPU: ", CUDA.name(CUDA.device()), "; graph batches, forced interval-start rebuilds, 11 alternating rounds")
    for dx in (0.02, 0.005), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        run_pair(dx, scheme)
        GC.gc()
        CUDA.reclaim()
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    SortingShadowBenchmark.main()
end
