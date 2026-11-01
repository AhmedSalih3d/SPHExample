# A refined patch translates and rotates with a square body in a DBC tank.
# Set floating=false for prescribed motion. Including the file defines the case;
# executing it runs the floating example and saves one temporal VTKHDF file.
using SPHExampleGPU
using StaticArrays: SVector
using CSV, HDF5

"""
    run_attached_resolution_example(; floating=true, duration=0.5, dx=0.04,
        output_interval=0.01, save_location=...)

Run a body-following refined tank. The input structs below define the material,
body, patch, motion and output. Both domains advance on the GPU between output
frames. Use a new, empty output directory.
"""
function run_attached_resolution_example(;
        floating = true,
        duration = 0.5,
        dx = 0.04,
        output_interval = 0.01,
        save_location = joinpath(@__DIR__, "..", "output", "AttachedResolution"))
    Dimensions = 2
    FloatType = Float32
    body_marker = 5
    isfinite(dx) && dx > 0 || throw(ArgumentError("dx must be positive and finite"))
    isfinite(duration) && duration > 0 || throw(ArgumentError("duration must be positive and finite"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be positive and finite"))
    isdir(save_location) && !isempty(readdir(save_location)) &&
        throw(ArgumentError("save_location must be empty: $save_location"))

    constants = SimulationConstants{FloatType}(Dimensions;
        dx = dx,
        c₀ = 25,
        g  = 9.81,
    )
    kernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();
        h = FloatType(1.3dx))
    body_details = FloatingDetails{FloatType}(;
        RelativeWeight = 0.5,
    )
    region = VariableResolutionBox((-0.7, -0.85), (0.7, 0.55); ratio = 2)

    function domain_geometry(spacing)
        water = [SVector(x, y) for x in -1.5:spacing:1.5 for y in -0.9:spacing:0.3
            if !(abs(x) <= 0.16 && abs(y + 0.2) <= 0.16)]
        wall = [SVector(x, y) for x in (-1.5 - 3spacing):spacing:(1.5 + 3spacing)
            for y in (-0.9 - 3spacing):spacing:0.6 if x < -1.5 || x > 1.5 || y < -0.9]
        square = [SVector(x, y) for x in -0.12:spacing:0.12 for y in -0.32:spacing:-0.08]

        Tank = SPHGeometry{Dimensions,FloatType}(wall;
            Density = constants.ρ₀, Type = Fixed, GroupMarker = 1)
        Water = SPHGeometry{Dimensions,FloatType}(water;
            Density = constants.ρ₀, Type = Fluid, GroupMarker = 2)
        Body = SPHGeometry{Dimensions,FloatType}(square;
            Density     = constants.ρ₀,
            Type        = floating ? Floating : Moving,
            GroupMarker = body_marker,
            Floating    = floating ? body_details : nothing,
        )
        return [Tank, Water, Body]
    end
    coarse_geometry = domain_geometry(dx)
    fine_geometry = domain_geometry(dx / region.ratio)

    refinement = VariableResolutionConfig(region;
        fine_geometry = fine_geometry,
        follow = body_marker,
        particle_reserve = 20000,
    )
    metadata = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        SimulationName      = floating ? "FloatingRefinement" : "MovingRefinement",
        SaveLocation        = save_location,
        SimulationTime      = duration,
        OutputTimes         = output_interval,
        VariableResolution  = refinement,
        GPUUseGraph         = true,
        GPUClipTimeStepToOutput = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = false,
        VisualizeInParaview = false,
        OpenLogFile         = false,
        OutputVariables = ["Velocity", "Density", "Pressure", "Type", "GroupMarker",
            "Resolution", "Spacing", "Owned"],
    )
    motion = floating ? nothing : PrescribedMotion(body_marker;
        center = (0.0, -0.2),
        translation = MotionSequence([KinematicSegment(10; velocity = (0.3, 0.0))]),
        rotation = MotionSequence([KinematicSegment(10; velocity = 0.4)]),
    )
    viscosity = ArtificialViscosity()
    density_diffusion = LinearDensityDiffusion()
    time_stepping = SymplecticTimeStepping()

    mkpath(save_location)
    particles = AllocateDataStructures(coarse_geometry, metadata)
    logger = SimulationLogger(save_location)
    result = try
        RunSimulation(;
            SimGeometry         = coarse_geometry,
            SimConstants        = constants,
            SimKernel           = kernel,
            SimMetaData         = metadata,
            SimParticles        = particles,
            SimLogger           = logger,
            SimMotion           = motion,
            SimTimeStepping     = time_stepping,
            SimViscosity        = viscosity,
            SimDensityDiffusion = density_diffusion,
        )
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    save_attached_geometry(metadata, region, fine_geometry[3], motion)
    return result
end

# Animate the geometry from accepted body states, rather than a predicted pose.
function save_attached_geometry(metadata, region, body, motion)
    filepath = joinpath(metadata.SaveLocation, metadata.SimulationName * ".vtkhdf")
    times = h5open(filepath, "r") do io
        read(io["VTKHDF/Steps/Values"])
    end
    initial_center = sum(body.Particles.Position) / length(body.Particles)
    if motion === nothing
        rows = collect(CSV.File(joinpath(metadata.SaveLocation, metadata.SimulationName * "_Floating.csv")))
        transform = function (point, time)
            row = rows[searchsortedfirst(times, time)]
            angle = getproperty(row, :Angle)
            s, c = sincos(angle)
            x, y = point[1] - initial_center[1], point[2] - initial_center[2]
            return SVector(c*x - s*y + getproperty(row, Symbol("Center:0")),
                s*x + c*y + getproperty(row, Symbol("Center:1")), point[3])
        end
    else
        transform = (point, time) -> motion_position(motion, SVector(point[1], point[2]), time)
    end
    tank = polyline([(-1.5, 0.6), (-1.5, -0.9), (1.5, -0.9), (1.5, 0.6)];
        thickness = 0.02, side = :right)
    body_shape = rectangle((-0.12, -0.32), 0.24, 0.24)
    patch = rectangle(Tuple(region.lower), region.upper[1] - region.lower[1], region.upper[2] - region.lower[2])
    SavePolygonMotionSequence(filepath, (; tank, body = body_shape, refinement = patch);
        times, motions = (; body = transform, refinement = transform), block_name = "Geometry")
    return filepath
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_attached_resolution_example()
end
