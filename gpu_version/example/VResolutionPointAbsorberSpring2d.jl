# Heaving point absorber with a damped spring and body-following refinement.
# Input structs are below; geometry and comparison helpers are in Setup.jl.
# Running or including this file starts the simulation.
using SPHExampleGPU
using SPHExampleGPU.Constraints: FloatingConstraints, LinearSpring
using StaticArrays: SVector

include(joinpath(@__DIR__, "VResolutionPointAbsorberSpring2d", "Setup.jl"))

"""
    point_absorber_spring_case(; dx=0.01, ratio=2, duration=10.0, kwargs...)

Build the sloping wave tank, heave-only floating box and damped spring without
running a solver. The piston follows second-order wave theory with a smooth
one-period ramp. Geometry dimensions and the default smoothing-length ratio
`h/dp = 1.2sqrt(2)` follow DualSPHysics `vresolution/06_Special`; the solver uses
this package's ordinary DBC and density-diffusion settings.
"""
function point_absorber_spring_case(;
        dx = 0.01,
        ratio = 2,
        duration = 10.0,
        output_interval = 0.05,
        FloatType = Float32,
        h_coefficient = 1.2sqrt(2),
        viscosity = 0.05,
        diffusion = 0.1,
        CFL = 0.2,
        stiffness = 100.0,
        damping = 500.0,
        rest_length = 0.325,
        region = VariableResolutionBox((1.7, 0.2), (2.3, 0.6); ratio),
        save_location = joinpath(@__DIR__, "..", "output", "vresolution_point_absorber_spring2d"))
    Dimensions = 2
    T = FloatType
    T in (Float32, Float64) || throw(ArgumentError("FloatType must be Float32 or Float64"))
    isfinite(dx) && 0 < dx <= 0.02 || throw(ArgumentError("dx must satisfy 0 < dx <= 0.02; enlarge region when using coarse test spacings"))
    isfinite(duration) && duration > 0 && isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("duration and output_interval must be positive and finite"))
    ratio == region.ratio || throw(ArgumentError("ratio must match the refinement region"))

    depth = 0.4
    sound_speed = 20sqrt(9.81 * depth)
    spring = LinearSpring{Dimensions,T}(;
        Anchor     = (2.0, 0.0),
        Attachment = (2.0, 0.325),
        Stiffness  = stiffness,
        Damping    = damping,
        RestLength = rest_length,
    )
    constraints = FloatingConstraints{Dimensions,T}(;
        TranslationLocks = (true, false),
        RotationLocks    = (true,),
    )
    floating_details = FloatingDetails{T}(;
        RelativeWeight = 0.5,
        Constraints    = constraints,
        Springs        = (spring,),
    )

    function level(spacing; fine)
        constants = SimulationConstants{T}(Dimensions;
            dx  = spacing,
            c₀  = sound_speed,
            γ   = 7,
            g   = 9.81,
            α   = viscosity,
            δᵩ  = diffusion,
            CFL = CFL,
        )
        h = h_coefficient * spacing
        kernel = SPHKernelInstance{Dimensions,T}(WendlandC2();
            h = T(h), η² = T((0.1h)^2))
        return point_absorber_level(spacing, constants, kernel, region, floating_details; fine)
    end
    coarse = level(dx; fine = false)
    fine = level(dx / ratio; fine = true)

    # Sampled displacement and its analytic velocity drive the separate piston.
    times = collect(range(0.0, duration + 0.002; step = 0.002))
    states = point_absorber_piston_state.(times)
    trajectory = SampledTrajectory(times, [(s[1], 0.0) for s in states];
        velocities = [(s[2], 0.0) for s in states])
    motion = PrescribedMotion(10; translation = trajectory)

    return (; coarse, fine, region, motion, spring, depth, sound_speed,
        simulation_time = Float64(duration), output_interval = Float64(output_interval),
        save_location = abspath(save_location))
end

"""
    run_point_absorber_spring(; particle_reserve=40000, metadata_options=(;), kwargs...)

Run the case and save particles, the moving body/patch and force-coloured
helical spring in `PointAbsorberSpring2D.vtkhdf`. `spring_display` configures
display geometry only. CSV files contain body, spring, represented-mass
and bundled-reference comparison histories. Use a new, empty output directory.
"""
function run_point_absorber_spring(;
        particle_reserve = 40000,
        metadata_options = (;),
        spring_display = (;radius=0.02,turns=8,sections_per_turn=16,wire_radius=0.0015),
        kwargs...)
    any(k -> k in (:SimulationName, :SaveLocation, :VariableResolution), keys(metadata_options)) &&
        throw(ArgumentError("set case identity/location/refinement through the case keywords"))
    get(metadata_options, :ExportSingleVTKHDF, true) && !get(metadata_options, :ExportGridCells, false) ||
        throw(ArgumentError("this example saves one VTKHDF: use ExportSingleVTKHDF=true and ExportGridCells=false"))
    case = point_absorber_spring_case(; kwargs...)
    isdir(case.save_location) && !isempty(readdir(case.save_location)) &&
        throw(ArgumentError("save_location must be empty: $(case.save_location)"))
    mkpath(case.save_location)
    T = typeof(case.coarse.constants.dx)

    refinement = VariableResolutionConfig(case.region;
        fine_geometry  = case.fine.geometry,
        fine_constants = case.fine.constants,
        fine_kernel    = case.fine.kernel,
        follow         = 50,
        particle_reserve = particle_reserve,
    )
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        SimulationName      = "PointAbsorberSpring2D",
        SaveLocation        = case.save_location,
        SimulationTime      = case.simulation_time,
        OutputTimes         = case.output_interval,
        VariableResolution  = refinement,
        GPUUseGraph         = true,
        GPUClipTimeStepToOutput = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = false,
        VisualizeInParaview = false,
        OpenLogFile         = false,
        OutputVariables = ["Velocity", "Density", "Pressure", "Type", "ID", "GroupMarker",
            "Resolution", "Spacing", "Mass", "Owned", "IsBuffer"],
        metadata_options...,
    )
    viscosity = ArtificialViscosity()
    density_diffusion = LinearDensityDiffusion()
    time_stepping = SymplecticTimeStepping()
    visualize = metadata.VisualizeInParaview
    # Add the accepted spring/body histories before opening the combined file.
    metadata.VisualizeInParaview = false

    particles = AllocateDataStructures(case.coarse.geometry, metadata)
    logger = SimulationLogger(case.save_location)
    result = try
        RunSimulation(;
            SimGeometry         = case.coarse.geometry,
            SimConstants        = case.coarse.constants,
            SimKernel           = case.coarse.kernel,
            SimMetaData         = metadata,
            SimParticles        = particles,
            SimLogger           = logger,
            SimMotion           = case.motion,
            SimViscosity        = viscosity,
            SimDensityDiffusion = density_diffusion,
            SimTimeStepping     = time_stepping,
        )
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    point_absorber_outputs(case,result;spring_display)
    maximum_force = maximum(CSV.File(joinpath(case.save_location,"PointAbsorberSpring2D_Springs.csv"))) do row
        hypot(getproperty(row,Symbol("Force:0")),getproperty(row,Symbol("Force:1")))
    end
    metadata.VisualizeInParaview = visualize
    AutoOpenParaview(metadata,case.coarse.constants,metadata.OutputVariables;
        spring_force_range=(0.0,max(maximum_force,1.0)))
    return result
end

# Including this file starts the simulation.
run_point_absorber_spring()
