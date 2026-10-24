# GPU version of example/MovingSquare2d.jl. Requires an NVIDIA GPU with CUDA.
#
# Particles are generated in this file, then the GPU solver runs.
# Including or running this file starts the simulation:
#     julia --project=gpu_version gpu_version/example/MovingSquare2d.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
import StaticArrays: SVector
using SPHExampleGPU
using Meshes
using StructArrays

function run_moving_square_2d()
    Dimensions = 2
    FloatType = Float32

    # ViscoBoundFactor should be 1, but need to understand how to implement it
    SimConstantsMovingSquare = SimulationConstants{FloatType}(
        dx = 0.02,
        c₀ = 28,
        δᵩ = 0.1,
        g = 0,
        Cb = 112000,
        α = 1e-6,
        CFL = 0.2,
    )

    SimMetaDataMovingSquare = SimulationMetaData{
        Dimensions,
        FloatType,
        PlanarShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName = "MovingSquare2D",
        SaveLocation = "C:/TestSimulations/MovingSquare2D_GPU",
        SimulationTime = 2.5,
        OutputTimes = 0.01,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        OpenLogFile = true,
    )
    moving_square_input_dir =
        normpath(joinpath(@__DIR__, "..", "input", "moving_square_2d_generated"))

    # Sample particles here; the generator script supplies shape definitions.
    polygons = moving_square_2d_polygons()
    regions = [
        ParticleRegion("Fixed", polygons.tank, Fixed),
        ParticleRegion("Square", polygons.square, Moving),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, SimConstantsMovingSquare.dx)
    positions(name) = only(r.positions for r in sampled if r.name == name)

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Fixed"), SimConstantsMovingSquare.ρ₀),
        # CSVFile     = joinpath(moving_square_input_dir,
        # "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Fixed.csv"),
        GroupMarker = 1,
        Type = Fixed,
        Motion = nothing,
    )

    Water = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Fluid"), SimConstantsMovingSquare.ρ₀),
        # CSVFile     = joinpath(moving_square_input_dir,
        # "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Fluid.csv"),
        GroupMarker = 2,
        Type = Fluid,
        Motion = nothing,
    )

    MovingSquare = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Square"), SimConstantsMovingSquare.ρ₀),
        # CSVFile     = joinpath(moving_square_input_dir,
        # "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Square.csv"),
        GroupMarker = 3,
        Type = Moving,
        Motion = MotionDetails{Dimensions, FloatType}(
            Velocity = 2.8,
            StartTime = 0.0,
            Duration = 3.0,
            Direction = SVector{Dimensions, FloatType}(1.0, 0.0),  # 2D direction vector with Float64 type
        ),
    )

    SimulationGeometry = [FixedBoundary; Water; MovingSquare]

    # Collect SPHGeometry instances into a vector
    SimulationGeometry = [FixedBoundary, Water, MovingSquare]
    # If save directory is not already made, make it
    if !isdir(SimMetaDataMovingSquare.SaveLocation)
        mkpath(SimMetaDataMovingSquare.SaveLocation)
    end

    SimLogger = SimulationLogger(SimMetaDataMovingSquare.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataMovingSquare)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(
        WendlandC2();
        dx = SimConstantsMovingSquare.dx,
        k = FloatType(sqrt(2)),
    )

    CleanUpSimulationFolder(SimMetaDataMovingSquare.SaveLocation)

    motion_preview = joinpath(SimMetaDataMovingSquare.SaveLocation, "MovingSquare2D_Motion.vtkhdf")
    SavePolygonMotionSequence(
        motion_preview,
        (; polygons.tank, polygons.square);
        motions = (; square = MovingSquare.Motion),
        times = 0.0:0.1:3.0,
    )
    OpenParaviewFile(motion_preview)

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataMovingSquare,
        SimConstants = SimConstantsMovingSquare,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimKernel = SimKernel,
        SimViscosity = LaminarSPS(),
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
    )
end

"""
    moving_square_2d_polygons(; tank_width = 10.0, tank_height = 5.0,
                               wall_thickness = 0.04,
                               square_lower_left = (1.0, 2.0), square_size = 1.0)

Return the closed tank, fluid region and initial moving square as `PolyArea`s,
in metres. The tank wall is a rectangular shell drawn outside the fluid domain
with `outline`; the fluid polygon has the square as a hole so fluid particles
do not overlap the moving body.
"""
function moving_square_2d_polygons(;
    tank_width = 10.0,
    tank_height = 5.0,
    wall_thickness = 0.04,
    square_lower_left = (1.0, 2.0),
    square_size = 1.0,
)
    if !(tank_width > 0 && tank_height > 0 && wall_thickness > 0 && square_size > 0)
        throw(
            ArgumentError(
                "tank dimensions, wall thickness, and square size " * "must be positive",
            ),
        )
    end

    square_x, square_y = square_lower_left
    if !(
        0 < square_x &&
        square_x + square_size < tank_width &&
        0 < square_y &&
        square_y + square_size < tank_height
    )
        throw(ArgumentError("the moving square must be strictly inside the tank"))
    end

    domain = rectangle((0.0, 0.0), tank_width, tank_height)
    tank = outline(domain; thickness = wall_thickness, side = :outward)
    body = square(square_lower_left, square_size)
    water = polygon(domain; holes = [body])

    return (; tank, water, square = body)
end

"""
    generate_moving_square_2d_example(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 28, g = 0))

Write the polygons (`MovingSquare2D_Geometry.vtkhdf`), sampled particles
(`MovingSquare2D_Dp<dx>_Particles.vtkhdf`), CSV input files and a ParaView
motion preview (`MovingSquare2D_Motion.vtkhdf`) to `output_dir`.

All particles start at `ρ₀`, matching the zero-gravity moving-square case.
Pass the simulation's `SimConstants` to use its reference density.
Returns the sampled particles per region, with their densities.
"""
function generate_moving_square_2d_example(
    output_dir;
    dx = 0.02,
    SimConstants = SimulationConstants{Float64}(; dx, c₀ = 28, g = 0),
)
    dx > 0 || throw(ArgumentError("particle spacing dx must be positive"))

    polygons = moving_square_2d_polygons()
    regions = [
        ParticleRegion("Fixed", polygons.tank, Fixed),
        ParticleRegion("Square", polygons.square, Moving),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    particles = map(sampled) do region
        density = fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    prefix = joinpath(output_dir, "MovingSquare2D_Dp$(dx)")
    SavePolygonVTKHDF(joinpath(output_dir, "MovingSquare2D_Geometry.vtkhdf"), polygons)

    next_id = 0
    for region in particles
        next_id = write_particle_csv(
            "$(prefix)_$(region.name).csv",
            region.positions;
            density = region.density,
            first_id = next_id,
        )
    end

    (; ρ₀, c₀) = SimConstants
    group_markers = Dict("Fixed" => 1, "Fluid" => 2, "Square" => 3)
    positions = SVector{3, Float64}[]
    density   = Float64[]
    types     = Int8[]
    markers   = Int[]
    for region in particles
        append!(positions, to_3d(region.positions))
        append!(density, region.density)
        append!(types, fill(Int8(region.type), length(region.positions)))
        append!(markers, fill(group_markers[region.name], length(region.positions)))
    end
    pressure = EquationOfStateGamma7.(density, c₀, ρ₀)
    SaveVTKHDF(
        "$(prefix)_Particles.vtkhdf",
        positions,
        ["Density", "Pressure", "Type", "GroupMarker"],
        density,
        pressure,
        types,
        markers,
    )

    preview_motion = MotionDetails{2, Float64}(
        Velocity = 2.8,
        StartTime = 0.0,
        Duration = 3.0,
        Direction = SVector{2, Float64}(1.0, 0.0),
    )
    SavePolygonMotionSequence(
        joinpath(output_dir, "MovingSquare2D_Motion.vtkhdf"),
        (; tank = polygons.tank, square = polygons.square);
        motions = (; square = preview_motion),
        times = 0.0:0.1:3.0,
    )

    return particles
end

# Including this file starts the simulation.
run_moving_square_2d()
