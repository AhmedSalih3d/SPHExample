module BoundaryStorageBenchmark

using SPHExampleGPU, CUDA, StaticArrays, Statistics, Printf

include(joinpath(@__DIR__, "memory_buffers.jl"))
include(joinpath(@__DIR__, "cases.jl"))

const DRIVER = SPHExampleGPU.SPHCellList
const STEP = SPHExampleGPU.GPUStepState

function cube_case(::Type{B}; side = 32) where {B}
    T = Float32
    dx = T(0.02)
    normal = SVector{3, T}(0, 0, 2dx)
    walls = [SVector{3, T}(i * dx, j * dx, 0) for i in 1:side for j in 1:side]
    fluid = [SVector{3, T}(i * dx, j * dx, k * dx) for i in 1:side for j in 1:side for k in 1:side]
    geometry = [
        SPHGeometry{3, T}(Particles = particle_struct_array(walls, T(1000);
            GhostPoints = walls .+ Ref(normal), GhostNormals = fill(normal, length(walls))),
            GroupMarker = 1, Type = Fixed),
        SPHGeometry{3, T}(fluid; Density = 1000, GroupMarker = 2, Type = Fluid),
    ]
    constants = SimulationConstants{T}(; dx, m₀ = 1000dx^3, c₀ = 20, g = 0)
    metadata = SimulationMetaData{3, T, NoShifting, NoKernelOutput, B}(
        SimulationName = "BoundaryCube", SaveLocation = tempdir())
    return (; SimGeometry = geometry, SimMetaData = metadata, SimConstants = constants,
        SimKernel = SPHKernelInstance{3, T}(WendlandC2(); dx),
        SimViscosity = Laminar(), SimDensityDiffusion = ZeroDensityDiffusion(),
        ParticleNormalsPath = nothing)
end

function setup(kw; boundary_only, scheme)
    metadata = deepcopy(kw.SimMetaData)
    metadata.GPUDoublePosition = true
    metadata.GPUUseGraph = true
    metadata.TimeSteppingMode = scheme
    metadata.SimulationTime = 100f0
    metadata.OutputTimes = 0.01f0
    metadata.OutputIterationCounter = 1
    host = AllocateDataStructures(kw.SimGeometry, metadata)
    D = length(eltype(host.Position))
    if kw.ParticleNormalsPath !== nothing
        _, ghosts, normals = LoadBoundaryNormals(Val(D), Float64, kw.ParticleNormalsPath)
        owners = findall(!=(Fluid), host.Type)
        host.GhostPoints[owners] = ghosts
        host.GhostNormals[owners] = normals
    end
    gpu = upload_particles(host; position_type = Float64, compact_metadata = true,
        kernel_output = false, reconstruct_fields = true, boundary_only)
    cl = CellListWorkspace{D, Float64}(length(host))
    sup = GPUSupportArrays{D, Float32}(gpu, metadata)
    red = ReductionWorkspace{SVector{3, Float32}}(length(host))
    motion = MotionArrays(kw.SimGeometry, host)
    state = StepState{Float32}(; dx = 1f0 + kw.SimKernel.h)
    return (; kw, metadata, host, gpu, cl, sup, red, motion, state)
end

function advance!(run)
    (; kw, metadata, gpu, cl, sup, red, motion, state) = run
    STEP.set_slot!(state.f, state.fh, STEP.F_DX, 1f0 + kw.SimKernel.h)
    SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel,
        metadata, kw.SimConstants, gpu, cl, sup, red, motion, state)
    metadata.OutputIterationCounter += 1
    return nothing
end

function snapshot(run)
    order = sortperm(Array(run.gpu.ID))
    return (; fields = map(f -> Array(getproperty(run.gpu, f))[order],
        (:Position, :Velocity, :Density, :Acceleration)),
        work = (run.metadata.Iteration, run.cl.nrebuilds, run.state.readbacks))
end

function run_pair(kw, label, scheme; rounds = 15)
    runs = [setup(kw; boundary_only, scheme) for boundary_only in (false, true)]
    for _ in 1:5, run in runs
        advance!(run)
    end
    snapshot(runs[1]) == snapshot(runs[2]) || error("Warm states or work counts differ")
    bytes(run) = GPUMemoryBenchmark.device_buffer_bytes((run.gpu, run.sup, run.cl))
    full_bytes, compact_bytes = bytes.(runs)
    times = [Float64[], Float64[]]
    for round in 1:rounds
        work = Tuple{Int, Int, Int}[]
        for i in (isodd(round) ? (1, 2) : (2, 1))
            run = runs[i]
            before = (run.metadata.Iteration, run.cl.nrebuilds, run.state.readbacks)
            elapsed = @elapsed for _ in 1:5
                advance!(run)
            end
            counts = (run.metadata.Iteration, run.cl.nrebuilds, run.state.readbacks) .- before
            push!(work, counts)
            push!(times[i], 1e6 * elapsed / counts[1])
        end
        work[1] == work[2] || error("Measured work counts differ")
        snapshot(runs[1]) == snapshot(runs[2]) || error("Measured states differ")
    end
    ratios = times[2] ./ times[1]
    @printf("%s %s n=%d boundaries=%d dense_MiB=%.4f compact_MiB=%.4f saved_MiB=%.4f saved_pct=%.2f dense_us=%.2f compact_us=%.2f paired_ratio=%.4f range=[%.4f,%.4f]\n",
        label, nameof(typeof(scheme)), length(runs[1].gpu), length(runs[1].gpu.GhostIndex),
        full_bytes / 2.0^20, compact_bytes / 2.0^20, (full_bytes - compact_bytes) / 2.0^20,
        100 * (1 - compact_bytes / full_bytes), median(times[1]), median(times[2]),
        median(ratios), minimum(ratios), maximum(ratios))
    flush(stdout)
    return nothing
end

function main()
    println("GPU: ", CUDA.name(CUDA.device()), "; Float32 physics / Float64 positions")
    println("Only boundary storage differs; shared predictors and other compact options are identical.")
    println("Warmed graph-batched solver, forced interval-start rebuilds, 15 alternating rounds; exact states and work counts checked.")
    for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        run_pair(still_wedge_mdbc(Float32, tempdir()), "StillWedge2D SimpleMDBC", scheme)
        GC.gc()
        CUDA.reclaim()
        for mode in (SimpleMDBC, UpdatedMDBC)
            run_pair(cube_case(mode), "Cube3D $mode", scheme)
            GC.gc()
            CUDA.reclaim()
        end
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    BoundaryStorageBenchmark.main()
end
