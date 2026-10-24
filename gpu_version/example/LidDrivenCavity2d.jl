# GPU simulation of the Re = 100, 2D lid-driven cavity in the Rocky 2025 R1
# verification manual. Requires an NVIDIA GPU with CUDA.
#
# Particles and ghost nodes are generated in this file, then the GPU solver runs.
# Including or running this file starts the simulation:
#     julia --project=gpu_version gpu_version/example/LidDrivenCavity2d.jl
#
# The top wall has a prescribed horizontal velocity but does not translate out
# of the cavity (`MoveParticles = false`). The lid spans the outside width of
# the tank to overlap both side walls. mDBC, density diffusion and shifting
# keep the fluid coupled to the lid throughout the run. The default duration
# is the 60 s specified by ANSYS. Optional arguments are
# [save_dir] [dx] [duration] [input_dir].
using SPHExampleGPU
using Meshes
using StaticArrays: SVector
using StructArrays


"""
    run_lid_driven_cavity_2d(; FloatType = Float32, dx = 0.01,
        simulation_time = 60.0, output_interval = 0.5, visualize = true,
        open_log_file = true, mdbc = true, shifting = true,
        density_diffusion = true, input_dir, save_location)

Run the Re = 100 cavity with particles and ghost nodes generated in memory. Set `mdbc`, `shifting`, or
`density_diffusion` to false to disable those options. Returns the final
`particles` and `meta`.
"""
function run_lid_driven_cavity_2d(;
    FloatType = Float32,
    dx = 0.01,
    simulation_time = 60.0,
    output_interval = 0.5,
    visualize::Bool = true,
    open_log_file::Bool = true,
    mdbc::Bool = true,
    shifting::Bool = true,
    density_diffusion::Bool = true,
    input_dir = normpath(
        joinpath(@__DIR__, "..", "input", "lid_driven_cavity_2d_generated"),
    ),
    save_location = raw"C:\TestSimulations\LidDrivenCavity2D_GPU",
)
    (isfinite(simulation_time) && simulation_time > 0) ||
        throw(ArgumentError("simulation_time must be finite and positive"))
    (isfinite(output_interval) && output_interval > 0) ||
        throw(ArgumentError("output_interval must be finite and positive"))

    # Generate particles and boundary ghost nodes directly inside the run.
    T = FloatType
    SimConstants = lid_driven_cavity_2d_constants(T; dx)
    shapes = lid_driven_cavity_2d_shapes()
    regions = [
        ParticleRegion("Fixed", shapes.walls, Fixed),
        ParticleRegion("Lid", shapes.lid, Moving),
        ParticleRegion("Fluid", shapes.fluid, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    simulation_geometry = map(enumerate(sampled)) do (marker, region)
        positions = region.positions
        particles = if region.type != Fluid
            ghosts = [
                SVector{2, Float64}(lid_driven_cavity_ghost_node(x, dx)) for x in positions
            ]
            particle_struct_array(
                positions,
                SimConstants.ρ₀;
                GhostPoints = ghosts,
                GhostNormals = ghosts .- positions,
            )
        else
            particle_struct_array(positions, SimConstants.ρ₀)
        end
        motion =
            region.type == Moving ?
            MotionDetails{2, T}(
                Velocity = T(LID_CAVITY_LID_SPEED),
                StartTime = zero(T),
                Duration = T(simulation_time),
                Direction = SVector{2, T}(one(T), zero(T)),
                MoveParticles = false,
            ) : nothing
        SPHGeometry{2, T}(
            Particles = particles,
            GroupMarker = marker,
            # CSVFile = joinpath(input_dir, "LidDrivenCavity2D_Dp$(dx)_$(region.name).csv"),
            Type = region.type,
            Motion = motion,
        )
    end

    mkpath(save_location)
    BMode = mdbc ? SimpleMDBC : NoMDBC
    SMode = shifting ? PlanarShifting : NoShifting
    meta = SimulationMetaData{2, T, SMode, NoKernelOutput, BMode, StoreLog}(
        SimulationName = "LidDrivenCavity2D",
        SaveLocation = save_location,
        SimulationTime = T(simulation_time),
        OutputTimes = T(output_interval),
        VisualizeInParaview = visualize,
        ExportSingleVTKHDF = true,
        OpenLogFile = open_log_file,
        GPUBoundaryForces = false,
        GPUMaxStepsPerSync = 256,
    )
    particles = AllocateDataStructures(simulation_geometry, meta)
    logger = SimulationLogger(save_location)
    kernel = SPHKernelInstance{2, T}(WendlandC2(); h = T(1.2 * sqrt(2) * dx))

    RunSimulation(
        SimGeometry = simulation_geometry,
        SimMetaData = meta,
        SimConstants = SimConstants,
        SimKernel = kernel,
        SimLogger = logger,
        SimParticles = particles,
        SimViscosity = Laminar(),
        SimDensityDiffusion = density_diffusion ? ZeroGravityLinearDensityDiffusion() :
                              ZeroDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        # ParticleNormalsPath = joinpath(input_dir, "LidDrivenCavity2D_Dp$(dx)_GhostNodes.csv"),
    )
    return (; particles, meta)
end

const LID_CAVITY_DENSITY = 10.0
const LID_CAVITY_DYNAMIC_VISCOSITY = 0.1
const LID_CAVITY_SOUND_SPEED = 10.0
const LID_CAVITY_LENGTH = 1.0
const LID_CAVITY_WALL_THICKNESS = 0.05
const LID_CAVITY_LID_SPEED = 1.0

"""
    lid_driven_cavity_2d_shapes(; side_length = 1.0, wall_thickness = 0.05)

Stationary bottom and side walls, a separate top lid spanning the complete
outside width of the tank, and the fluid-filled square cavity. Both the
stationary walls and the lid extend outwards from the cavity, leaving the full
1 m × 1 m square for the fluid.
"""
function lid_driven_cavity_2d_shapes(;
    side_length = LID_CAVITY_LENGTH,
    wall_thickness = LID_CAVITY_WALL_THICKNESS,
)
    (isfinite(side_length) && side_length > 0) ||
        throw(ArgumentError("side_length must be finite and positive"))
    (isfinite(wall_thickness) && 0 < wall_thickness < side_length) || throw(
        ArgumentError("wall_thickness must be finite, positive and less than side_length"),
    )

    walls = polyline(
        [(0.0, side_length), (0.0, 0.0), (side_length, 0.0), (side_length, side_length)];
        thickness = wall_thickness,
        side = :right,
    )
    lid = line(
        (-wall_thickness, side_length),
        (side_length + wall_thickness, side_length);
        thickness = wall_thickness,
        side = :left,
    )
    fluid = rectangle((0.0, 0.0), side_length, side_length)
    return (; walls, lid, fluid)
end

"""
    lid_driven_cavity_2d_constants(::Type{T} = Float64; dx = 0.01)

Rocky verification inputs in SPH form: `ρ₀ = 10 kg/m³`, dynamic viscosity
`μ = 0.1 Pa·s`, hence kinematic viscosity `ν₀ = μ / ρ₀ = 0.01 m²/s`;
`c₀ = 10 m/s`, zero gravity and `CFL = 0.2`. For `L = 1 m` and lid speed
`U = 1 m/s` these give `Re = ρ₀ U L / μ = 100`.
"""
function lid_driven_cavity_2d_constants(
    ::Type{T} = Float64;
    dx = 0.01,
) where {T <: AbstractFloat}
    (isfinite(dx) && dx > 0) || throw(ArgumentError("dx must be finite and positive"))
    ρ₀ = T(LID_CAVITY_DENSITY)
    μ = T(LID_CAVITY_DYNAMIC_VISCOSITY)
    return SimulationConstants{T}(;
        dx = T(dx),
        ρ₀,
        g = zero(T),
        c₀ = T(LID_CAVITY_SOUND_SPEED),
        ν₀ = μ / ρ₀,
        CFL = T(0.2),
    )
end

"""
    lid_driven_cavity_ghost_node(x, dx; side_length = 1.0)

Mirror a wall particle across the nearest wetted face into the cavity. At
corners both coordinates are mirrored, placing the ghost node on the fluid
side of both faces.
"""
function lid_driven_cavity_ghost_node(x, dx; side_length = LID_CAVITY_LENGTH)
    mirror(c) = c < dx / 2 ? dx - c : c > side_length - dx / 2 ? 2side_length - dx - c : c
    return (mirror(x[1]), mirror(x[2]))
end

"""
    write_ghost_nodes_csv(path, positions; first_id = 0, dx)

Write one mDBC ghost node per 2D wall particle in the CSV layout consumed by
`LoadBoundaryNormals`. Points and normals use the solver's `(x, 0, y)` layout.
"""
function write_ghost_nodes_csv(
    path::AbstractString,
    positions;
    first_id::Integer = 0,
    dx::Real,
)
    open(path, "w") do io
        println(
            io,
            "\"Idp\",\"Mk\",\"Normal:0\",\"Normal:1\",\"Normal:2\",\"NormalSize\"," *
            "\"Points:0\",\"Points:1\",\"Points:2\"",
        )
        for (k, x) in enumerate(positions)
            ghost = lid_driven_cavity_ghost_node(x, dx)
            normal = (ghost[1] - x[1], ghost[2] - x[2])
            println(
                io,
                first_id + k - 1,
                ",0,",
                normal[1],
                ",0,",
                normal[2],
                ",",
                hypot(normal...),
                ",",
                x[1],
                ",0,",
                x[2],
            )
        end
    end
    return first_id + length(positions)
end

"""
    generate_lid_driven_cavity_2d(output_dir; dx = 0.01,
        SimConstants = lid_driven_cavity_2d_constants(; dx))

Write `LidDrivenCavity2D_Geometry.vtkhdf`,
`LidDrivenCavity2D_Dp<dx>_Particles.vtkhdf`, `Fixed`, `Lid`, and `Fluid` CSV
particle files, and the wall/lid mDBC ghost-node file to `output_dir`. All
particles start at the reference density. Returns the sampled regions.
"""
function generate_lid_driven_cavity_2d(
    output_dir;
    dx = 0.01,
    SimConstants = lid_driven_cavity_2d_constants(; dx),
)
    (isfinite(dx) && 0 < dx <= LID_CAVITY_LENGTH) ||
        throw(ArgumentError("dx must be finite, positive and no larger than the cavity"))

    shapes = lid_driven_cavity_2d_shapes()
    regions = [
        ParticleRegion("Fixed", shapes.walls, Fixed),
        ParticleRegion("Lid", shapes.lid, Moving),
        ParticleRegion("Fluid", shapes.fluid, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    particles = map(sampled) do region
        density = fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    SavePolygonVTKHDF(joinpath(output_dir, "LidDrivenCavity2D_Geometry.vtkhdf"), shapes)
    prefix = joinpath(output_dir, "LidDrivenCavity2D_Dp$(dx)")
    next_id = 0
    for region in particles
        next_id = write_particle_csv(
            "$(prefix)_$(region.name).csv",
            region.positions;
            density = region.density,
            first_id = next_id,
        )
    end
    wall_positions = SVector{2, Float64}[]
    for region in particles
        region.type != Fluid && append!(wall_positions, region.positions)
    end
    write_ghost_nodes_csv("$(prefix)_GhostNodes.csv", wall_positions; dx)

    (; ρ₀, c₀) = SimConstants
    positions = SVector{3, Float64}[]
    density   = Float64[]
    types     = Int8[]
    markers   = Int[]
    for (k, region) in enumerate(particles)
        append!(positions, to_3d(region.positions))
        append!(density, region.density)
        append!(types, fill(Int8(region.type), length(region.positions)))
        append!(markers, fill(k, length(region.positions)))
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
    return particles
end

# Including this file starts the simulation.
save_location =
    length(ARGS) >= 1 ? ARGS[1] : raw"C:\TestSimulations\LidDrivenCavity2D_GPU"
dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
simulation_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 60.0
input_dir =
    length(ARGS) >= 4 ? ARGS[4] :
    normpath(joinpath(@__DIR__, "..", "input", "lid_driven_cavity_2d_generated"))
run_lid_driven_cavity_2d(; save_location, dx, simulation_time, input_dir)
