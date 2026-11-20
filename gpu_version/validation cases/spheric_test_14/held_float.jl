# Measure the water load at the prescribed still-water draft of Float 1.
include(joinpath(@__DIR__, "case.jl"))

function held_float(dx, duration)
    release = merge(first(RELEASES), (; offset = 0.0))
    depth, half_width = 0.8, 1.2
    T = Float32
    constants = SimulationConstants{T}(3; dx, g = 9.81, c₀ = 25.0, CFL = 0.2)
    kernel = SPHKernelInstance{3,T}(WendlandC2(); h = T(1.3dx))
    geometry, target_z = sample_geometry(release, constants, kernel;
        dx, half_width, depth, pause_time = duration + 1)
    tag = "held_float_dp$(dx)_t$(duration)"
    save = joinpath(@__DIR__, "runs", tag)
    output = joinpath(@__DIR__, "results", tag)
    isfile(joinpath(save, "HeldFloat.vtkhdf")) && error("Run already exists at $save")
    metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        SimulationName = "HeldFloat", SaveLocation = save, SimulationTime = duration,
        OutputTimes = 0.02, ExportSingleVTKHDF = true, OutputVariables = ["Type"],
        VisualizeInParaview = false, OpenLogFile = false, GPUDoublePosition = true,
        GPUClipTimeStepToOutput = true)
    damping = DampingCylinder((0.0, 0.0, -depth), (0.0, 0.0, 0.2), 0.7, 1.2; rate = 12)
    mkpath(save)
    logger = SimulationLogger(save)
    host = AllocateDataStructures(geometry, metadata)
    result = try
        RunSimulation(SimGeometry = geometry, SimMetaData = metadata, SimConstants = constants,
            SimKernel = kernel, SimLogger = logger, SimParticles = host, SimViscosity = Laminar(),
            SimDensityDiffusion = LinearDensityDiffusion(), SimTimeStepping = SymplecticTimeStepping(),
            SimDamping = damping)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    result.final_state.complete && result.diagnostics.frozen_particles == 0 || error("Incomplete held-float run")
    rows = CSV.File(joinpath(save, "HeldFloat_Floating.csv"))
    center_z = getproperty(rows, Symbol("Center:2"))
    all(z -> isapprox(z, target_z; atol = 1e-8, rtol = 0), center_z) || error("Float moved during hold")
    time = rows.Time
    isapprox(last(time), duration; atol = 1e-6) || error("Incomplete force history")
    force_z = getproperty(rows, Symbol("Force:2"))
    all(isfinite, force_z) && any(!iszero, force_z) || error("No valid hydrodynamic load")
    weight = release.mass * Float64(constants.g)
    mkpath(output)
    CSV.write(joinpath(output, "vertical_force.csv"), (; time, fluid_force_n = force_z,
        weight_n = fill(weight, length(time)), excess_force_n = force_z .- weight, center_z_m = center_z))
    println("Force history saved to ", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Usage: held_float.jl [duration]")
    duration = isempty(ARGS) ? 3.0 : parse(Float64, only(ARGS))
    isfinite(duration) && duration >= 0.02 || error("Duration must be at least 0.02 s")
    for dx in (0.025, 0.0125)
        held_float(dx, duration)
    end
end
