# Usage: julia --project=. benchmark/floating_steps.jl baseline_GPUFloating.jl [revision]
# Compare the complete falling-cylinder timestep loop, including cell rebuilds.
include("floating_kernels.jl")
using LinearAlgebra: norm

function cylinder_case()
    generator = joinpath(@__DIR__, "..", "example", "GenerateFloatingCylinder2D.jl")
    definitions = first(split(read(generator, String), "\noutput_dir = "))
    Base.include_string(Main, definitions, generator)
    build_shapes = Base.invokelatest(getproperty, Main, :floating_cylinder_2d_shapes)
    shapes = Base.invokelatest(build_shapes; dx = 0.025)
    constants = SimulationConstants{Float32}(dx = 0.025,
        c₀ = 30 * sqrt(9.81 * 0.8), CFL = 0.2)
    regions = [ParticleRegion("Cylinder", shapes.cylinder, Floating;
                   sampling = :conforming),
               ParticleRegion("Bound", shapes.tank, Fixed),
               ParticleRegion("Fluid", shapes.water, Fluid)]
    sampled = sample_particles(regions, constants.dx)
    level = maximum(last, sampled[3].positions)
    geometry = [SPHGeometry{2, Float32}(region.positions;
        Density = hydrostatic_density(region.positions, constants; water_level = level),
        Type = regions[i].type, GroupMarker = i,
        Floating = i == 1 ? FloatingDetails{Float32}(RelativeWeight = 1.2) : nothing)
        for (i, region) in enumerate(sampled)]
    return geometry, constants
end

function step_benchmark(driver, floating_mod, geometry, constants)
    T = Float32
    meta = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, NoLog}(
        SimulationName = "FloatingBenchmark", SaveLocation = tempdir(),
        SimulationTime = 1.0, OutputTimes = 0.02, GPUDoublePosition = true,
        VisualizeInParaview = false, OpenLogFile = false)
    meta.TimeSteppingMode = SymplecticTimeStepping()
    particles = AllocateDataStructures(geometry, meta)
    Pressure!(particles.Pressure, particles.Density, constants)
    gpu = driver.upload_particles(particles)
    n = length(gpu)
    support = driver.GPUSupportArrays{2, T}(n; position_type = Float64)
    reduction = ReductionWorkspace{SVector{3, T}}(n)
    cells = CellListWorkspace{2, Float64}(n)
    motion = driver.MotionArrays(geometry, particles)
    floating = floating_mod.FloatingArrays(geometry, particles, constants;
        position_type = Float64)
    kernel = SPHKernelInstance{2, T}(WendlandC2(); dx = constants.dx, k = T(sqrt(2)))
    state = StepState{T}(dx = one(T) + kernel.h)
    loop() = driver.SimulationLoop(LinearDensityDiffusion(), LaminarSPS(), kernel,
        meta, constants, gpu, cells, support, reduction, motion, state; floating)
    # Warm both pointer variants of the CUDA graph before measuring.
    for _ in 1:3
        meta.OutputIterationCounter += 1
        loop()
    end
    milliseconds = Float64[]
    for _ in 1:7
        before = meta.Iteration
        meta.OutputIterationCounter += 1
        elapsed = @elapsed begin
            loop()
            CUDA.synchronize()
        end
        push!(milliseconds, 1000 * elapsed / (meta.Iteration - before))
    end
    sort!(milliseconds)
    ms = milliseconds[4]
    @printf("%s: %d particles, %d floating, median %.3f ms/step, %d rebuilds\n",
        nameof(driver), n, sum(floating.count), ms, cells.nrebuilds)
    return ms, floating_state(floating)
end

function main_steps(args)
    baseline_source = replace(read(first(args), String),
        "module GPUFloating" => "module FloatingBaseline")
    Base.include_string(SPHExampleGPU, baseline_source, "floating_baseline.jl")
    revision = length(args) > 1 ? args[2] : "HEAD"
    driver_source = read(`git show $(revision * ":gpu_version/src/SPHCellList.jl")`, String)
    driver_source = replace(driver_source, "module SPHCellList" => "module BaselineCellList",
        "using ..GPUFloating" => "using ..FloatingBaseline")
    Base.include_string(SPHExampleGPU, driver_source, "baseline_cell_list.jl")
    baseline = Base.invokelatest(getproperty, SPHExampleGPU, :FloatingBaseline)
    driver = Base.invokelatest(getproperty, SPHExampleGPU, :BaselineCellList)
    geometry, constants = cylinder_case()
    before, old_state = Base.invokelatest(step_benchmark, driver, baseline,
        geometry, constants)
    GC.gc()
    CUDA.reclaim()
    after, new_state = step_benchmark(SPHExampleGPU.SPHCellList, SPHExampleGPU.GPUFloating,
        geometry, constants)
    @printf("Complete timestep speedup %.3fx; centre difference %.3g m\n",
        before / after, maximum(norm.(old_state.center .- new_state.center)))
end

main_steps(ARGS)
