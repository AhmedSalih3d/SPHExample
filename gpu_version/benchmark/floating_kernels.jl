# Usage: julia --project=. benchmark/floating_kernels.jl [baseline_GPUFloating.jl]
# Measures device time with graph replay to remove host launch overhead.
using SPHExampleGPU
using CUDA
using StaticArrays
using Printf

function graph_time(f; repetitions = 100, samples = 7)
    f()
    CUDA.synchronize()
    graph = CUDA.capture() do
        for _ in 1:repetitions
            f()
        end
    end
    executable = CUDA.instantiate(graph)
    CUDA.launch(executable)
    CUDA.synchronize()
    return minimum(CUDA.@elapsed(CUDA.launch(executable)) for _ in 1:samples) /
        repetitions
end

function benchmark_floating(mod, D; n = 250_000, nbody = 5000, nbodies = 1)
    T = Float32
    TP = Float64
    positions = [SVector{D, TP}(ntuple(d ->
        0.025 * ((i ÷ 20^(d - 1)) % 20), D)) for i in 1:n]
    geometry = SPHGeometry{D, T}[]
    for b in 1:nbodies
        members = (1 + (b - 1) * nbody ÷ nbodies):(b * nbody ÷ nbodies)
        push!(geometry, SPHGeometry{D, T}(positions[members];
            Density = 1000, Type = Floating, GroupMarker = b,
            Floating = FloatingDetails{T}(RelativeWeight = 1.2)))
    end
    push!(geometry, SPHGeometry{D, T}(positions[(nbody + 1):end];
        Density = 1000, Type = Fluid, GroupMarker = nbodies + 1))
    particles = AllocateDataStructures(geometry; position_type = TP)
    constants = SimulationConstants{T}()
    floating = mod.FloatingArrays(geometry, particles, constants; position_type = TP)
    gpu = upload_particles(particles)
    fill!(gpu.Acceleration, SVector{D, T}(ntuple(d -> T(d), D)))
    step = StepState{T}(dt = 1e-4)
    out = similar(gpu.Position)
    vel = similar(gpu.Velocity)
    forces() = begin
        mod.launch_floating_forces!(floating, gpu.Acceleration, gpu.Position,
            gpu.Type, gpu.GroupMarker, floating.center, constants.m₀, step)
        mod.launch_floating_update!(floating, step, zero(T), false)
    end
    placement() = mod.launch_floating_particles!(floating, out, vel, gpu.Position,
        gpu.Type, gpu.GroupMarker, step, false)
    stages() = begin
        forces()
        placement()
        mod.launch_floating_forces!(floating, gpu.Acceleration, out,
            gpu.Type, gpu.GroupMarker, floating.center_half, constants.m₀, step)
        mod.launch_floating_update!(floating, step, zero(T), true)
        mod.launch_floating_particles!(floating, out, vel, gpu.Position,
            gpu.Type, gpu.GroupMarker, step, true)
    end
    # Timings use bounded state; no artificial falling motion during warmup.
    result = (forces = graph_time(forces), placement = graph_time(placement),
              stages = graph_time(stages))
    @printf("%s %dD, %d/%d floating, %d bodies: force+update %.2f us, placement %.2f us, both stages %.2f us\n",
        nameof(mod), D, nbody, n, nbodies, 1e6 * result.forces,
        1e6 * result.placement, 1e6 * result.stages)
    CUDA.reclaim()
    return result
end

function main(args)
    optimized = SPHExampleGPU.GPUFloating
    baseline = nothing
    if !isempty(args)
        source = replace(read(only(args), String),
            "module GPUFloating" => "module FloatingBaseline")
        Base.include_string(SPHExampleGPU, source, "floating_baseline.jl")
        baseline = Base.invokelatest(getproperty, SPHExampleGPU, :FloatingBaseline)
    end
    println("GPU: ", CUDA.name(CUDA.device()))
    for D in (2, 3), nbodies in (1, 8)
        before = baseline === nothing ? nothing :
            Base.invokelatest(benchmark_floating, baseline, D; nbodies)
        after = benchmark_floating(optimized, D; nbodies)
        if before !== nothing
            @printf("  speedup: force+update %.2fx, placement %.2fx, stages %.2fx\n",
                before.forces / after.forces, before.placement / after.placement,
                before.stages / after.stages)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
