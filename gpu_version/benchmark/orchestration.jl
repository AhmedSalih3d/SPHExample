using SPHExampleGPU, CUDA, StaticArrays, Printf
using SPHExampleGPU.GPUReductions
using SPHExampleGPU.GPUCellGrid
using SPHExampleGPU.GPUStepState

function measure(driver, n, batch, scheme; output_interval = 0.02)
    T = Float32
    constants = SimulationConstants{T}(dx = 0.02, g = 0, c₀ = 20, CFL = 0.2)
    positions = [SVector{2,T}(0.02 * (i % 32), 0.02 * (i ÷ 32)) for i in 0:n-1]
    geometry = [SPHGeometry{2,T}(positions[3:end]; Density = 1000, Type = Fluid, GroupMarker = 1),
        SPHGeometry{2,T}([SVector{2,T}(2, 2), SVector{2,T}(2, 2.02)]; Density = 1000,
            Type = Moving, GroupMarker = 2, Motion = MotionDetails{2,T}(Velocity = 0.1,
                Direction = SVector{2,T}(1, 0), StartTime = -1, Duration = 2))]
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
        SimulationName = "Orchestration", SaveLocation = tempdir(), SimulationTime = 2,
        OutputTimes = output_interval, GPUDoublePosition = true, GPUMaxStepsPerSync = batch,
        VisualizeInParaview = false, OpenLogFile = false)
    meta.TimeSteppingMode = scheme
    particles = AllocateDataStructures(geometry, meta)
    Pressure!(particles.Pressure, particles.Density, constants)
    gpu = driver.upload_particles(particles)
    support = driver.GPUSupportArrays{2,T}(n; position_type = Float64)
    reduction = ReductionWorkspace{SVector{3,T}}(n)
    cells = CellListWorkspace{2,Float64}(n)
    motion = driver.MotionArrays(geometry, particles)
    kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
    state = StepState{T}(dx = 1 + kernel.h)
    loop() = driver.SimulationLoop(LinearDensityDiffusion(), ArtificialViscosity(), kernel,
        meta, constants, gpu, cells, support, reduction, motion, state)
    samples = Float64[]
    for k in 1:10
        before = meta.Iteration
        meta.OutputIterationCounter += 1
        elapsed = @elapsed begin loop(); CUDA.synchronize() end
        k > 3 && push!(samples, 1e6 * elapsed / (meta.Iteration - before))
    end
    sort!(samples)
    @printf("  %s n=%d batch=%d: readbacks=%d largest_graph=%d\n",
        nameof(driver), n, batch, state.readbacks, maximum(last.(keys(state.graphs)); init = 0))
    result = (; position = Array(gpu.Position), velocity = Array(gpu.Velocity),
        density = Array(gpu.Density), id = Array(gpu.ID))
    return samples[4], result, meta.Iteration, cells.nrebuilds
end

revision = isempty(ARGS) ? "c47b34b" : first(ARGS)
source = read(`git show $(revision * ":gpu_version/src/SPHCellList.jl")`, String)
source = replace(source, "module SPHCellList" => "module OrchestrationBaseline")
Base.include_string(SPHExampleGPU, source, "orchestration_baseline.jl")
baseline = getproperty(SPHExampleGPU, :OrchestrationBaseline)
println("Baseline driver revision: ", revision, "; baseline host batch=32; original graph cap=8")
println("RTX benchmark: median warm microseconds/step; stationary fluid and prescribed motion; sparse rebuilds; output interval 0.02 s")
for n in (128, 1024), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
    before, reference, iterations, rebuilds = Base.invokelatest(measure, baseline, n, 32, scheme)
    for batch in (8, 32, 128)
        after, result, steps, builds = measure(SPHExampleGPU.SPHCellList, n, batch, scheme)
        @assert result == reference && steps == iterations && builds == rebuilds
        @printf("n=%d %s batch=%d baseline=%.3f candidate=%.3f speedup=%.3fx steps=%d rebuilds=%d\n",
            n, nameof(typeof(scheme)), batch, before, after, before/after, steps, builds)
    end
end


# A short deadline bounds usable graph sizes regardless of the requested batch.
for batch in (32, 128)
    before, reference, iterations, rebuilds = Base.invokelatest(measure, baseline, 128, batch,
        SingleNeighborTimeStepping(); output_interval = 0.001)
    after, result, steps, builds = measure(SPHExampleGPU.SPHCellList, 128, batch,
        SingleNeighborTimeStepping(); output_interval = 0.001)
    @assert result == reference && steps == iterations && builds == rebuilds
    @printf("short deadline=0.001s n=128 SingleNeighborTimeStepping batch=%d baseline=%.3f candidate=%.3f speedup=%.3fx steps=%d rebuilds=%d\n",
        batch, before, after, before/after, steps, builds)
end
