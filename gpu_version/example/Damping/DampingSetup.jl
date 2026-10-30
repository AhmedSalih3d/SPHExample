using SPHExampleGPU
using LinearAlgebra: norm
using StaticArrays: SVector

const DAMPING_VARIANTS = (:plane_mdbc, :plane, :angle, :box, :cylinder)
const DAMPING_CASE_NAMES = (
    plane_mdbc = "CaseDampingPlane_mDBC",
    plane = "CaseDampingPlaneNot_mDBC",
    angle = "CaseDampingAngle",
    box = "CaseDampingBox",
    cylinder = "CaseDampingCylinder",
)
const DAMPING_PISTON_MOTION = joinpath(
    @__DIR__, "..", "data", "damping", "CaseDampingPlane_PistonMovement.dat",
)

function damping_piston_trajectory(path = DAMPING_PISTON_MOTION)
    isfile(path) || throw(ArgumentError("piston motion history does not exist: $path"))
    times, rows = SPHExampleGPU.PrescribedMotions.load_motion_table(
        path; time_column = 1, value_columns = (2,), skip_rows = 0,
    )
    displacement = [SVector{3,Float64}(row[1], 0.0, 0.0) for row in rows]
    return SampledTrajectory(times, displacement)
end

function add_wall_normal!(normals, index, normal)
    normals[index] = get(normals, index, zero(normal)) + normal
    return nothing
end

function damping_plane_geometry(dx, constants, ::Type{T}) where T
    tank_length, tank_width, water_depth, tank_height, wall_layers =
        6.5, 0.37, 0.4, 0.55, 3
    paddle_x, paddle_thickness = 1.0, 0.03
    nx, ny, nz, water_yrows, water_rows = round.(Int,
        (tank_length, tank_width, tank_height, 0.3, water_depth) ./ dx)

    wall_normal_sums = Dict{Tuple{Int,Int,Int},SVector{3,Float64}}()
    for k in -wall_layers:0, j in 0:ny, i in 0:nx
        add_wall_normal!(wall_normal_sums, (i, j, k), SVector(0.0, 0.0, 1.0))
    end
    for k in 0:nz, j in 0:ny, i in -wall_layers:0
        add_wall_normal!(wall_normal_sums, (i, j, k), SVector(1.0, 0.0, 0.0))
    end
    for k in 0:nz, j in 0:ny, i in nx:(nx + wall_layers)
        add_wall_normal!(wall_normal_sums, (i, j, k), SVector(-1.0, 0.0, 0.0))
    end
    for k in 0:nz, i in 0:nx, j in -wall_layers:0
        add_wall_normal!(wall_normal_sums, (i, j, k), SVector(0.0, 1.0, 0.0))
    end
    for k in 0:nz, i in 0:nx, j in ny:(ny + wall_layers)
        add_wall_normal!(wall_normal_sums, (i, j, k), SVector(0.0, -1.0, 0.0))
    end

    wall_indices = sort!(collect(keys(wall_normal_sums)); by = ij -> (ij[3], ij[2], ij[1]))
    wall_positions = [SVector(i * dx, j * dx, k * dx) for (i, j, k) in wall_indices]
    wall_normals = [
        normal / norm(normal) for normal in (wall_normal_sums[index] for index in wall_indices)
    ]

    paddle_i0 = round(Int, paddle_x / dx)
    paddle_i1 = max(paddle_i0, floor(Int, (paddle_x + paddle_thickness) / dx))
    paddle_indices = [
        (i, j, k) for k in 0:nz, j in 0:ny, i in paddle_i0:paddle_i1
        if !haskey(wall_normal_sums, (i, j, k))
    ]
    paddle_positions = [SVector(i * dx, j * dx, k * dx) for (i, j, k) in paddle_indices]
    # The reference's norplane for the moving paddle points along +x.
    paddle_normals = fill(SVector(1.0, 0.0, 0.0), length(paddle_positions))

    paddle_grid = Set((i, j, k) for k in 0:nz, j in 0:ny, i in paddle_i0:paddle_i1)
    # GenCase seeds its void fill at x = 3 m, ahead of the paddle.
    water_i0 = ceil(Int, (paddle_x + paddle_thickness) / dx)
    fluid_positions = vec(SVector{3,Float64}[
        SVector(i * dx, j * dx, k * dx)
        for k in 0:water_rows, j in 0:water_yrows, i in water_i0:nx
        if !haskey(wall_normal_sums, (i, j, k)) && !((i, j, k) in paddle_grid)
    ])
    fluid_density = hydrostatic_density(
        fluid_positions, constants; water_level = water_depth,
    )

    function boundary_geometry(positions, normals, marker, type)
        ghost_points = [point + dx * normal for (point, normal) in zip(positions, normals)]
        ghost_normals = [SVector{3,T}(dx * normal) for normal in normals]
        particles = particle_struct_array(positions, constants.ρ₀;
            GhostPoints = ghost_points, GhostNormals = ghost_normals)
        return SPHGeometry{3,T}(; Particles = particles, GroupMarker = marker, Type = type)
    end

    tank = boundary_geometry(wall_positions, wall_normals, 1, Fixed)
    paddle = boundary_geometry(paddle_positions, paddle_normals, 2, Moving)
    water = SPHGeometry{3,T}(fluid_positions;
        Density = fluid_density, GroupMarker = 3, Type = Fluid)
    footprint = rectangle((0.0, 0.0), tank_length, tank_width)
    tank_wall = prism(outline(footprint; thickness = wall_layers * dx, side = :outward),
        -wall_layers * dx, tank_height)
    tank_floor = prism(offset_polygon(footprint, wall_layers * dx),
        -wall_layers * dx, 0.0)
    shapes = (;
        tank_wall,
        tank_floor,
        piston = prism(rectangle((paddle_x, 0.0), paddle_thickness, tank_width),
            0.0, tank_height),
        water = prism(
            rectangle((paddle_x + paddle_thickness, 0.0),
                tank_length - paddle_x - paddle_thickness, 0.3),
            0.0, water_depth,
        ),
    )
    return [tank, paddle, water], shapes
end

function box_tank_parts(x, dx; length, width, height, closed_top = false)
    footprint = rectangle((x, 0.0), length, width)
    wall = prism(outline(footprint; thickness = 3dx, side = :outward), -3dx, height)
    floor = prism(offset_polygon(footprint, 3dx), -3dx, 0.0)
    ceiling = closed_top ? prism(offset_polygon(footprint, 3dx), height, height + 3dx) : nothing
    boundary = closed_top ? (wall, floor, ceiling) : (wall, floor)
    return boundary, (; wall, floor, ceiling)
end

function cylinder_tank_parts(x, dx; radius, height)
    footprint = circle((x, 0.0), radius; segments = 128)
    wall = prism(outline(footprint; thickness = 3dx, side = :outward), -3dx, height)
    floor = prism(offset_polygon(footprint, 3dx), -3dx, 0.0)
    return (wall, floor), (; wall, floor)
end

function sampled_geometry(regions, dx, constants, ::Val{D}, ::Type{T}; water_level) where {D,T}
    sampled = sample_particles(regions, dx)
    return [
        SPHGeometry{D,T}(;
            Particles = particle_struct_array(
                particles.positions,
                constants.ρ₀;
                Density = particles.type == Fluid ?
                    hydrostatic_density(particles.positions, constants; water_level) :
                    fill(constants.ρ₀, length(particles.positions)),
            ),
            GroupMarker = marker,
            Type = particles.type,
        )
        for (marker, particles) in enumerate(sampled)
    ]
end

function box_drop_case(dx, constants, ::Type{T}) where T
    tank1, shapes1 = box_tank_parts(0.0, dx; length = 1.0, width = 1.0, height = 0.15)
    tank2, shapes2 = box_tank_parts(1.15, dx; length = 1.0, width = 1.0, height = 0.15)
    pool1 = prism(rectangle((0.0, 0.0), 1.0, 1.0), 0.0, 0.1)
    drop1 = prism(rectangle((0.45, 0.45), 0.1, 0.1), 0.15, 0.25)
    pool2 = prism(rectangle((1.15, 0.0), 1.0, 1.0), 0.0, 0.1)
    drop2 = prism(rectangle((1.60, 0.45), 0.1, 0.1), 0.15, 0.25)
    regions = [
        ParticleRegion("TankDamped", tank1, Fixed),
        ParticleRegion("TankControl", tank2, Fixed),
        ParticleRegion("PoolDamped", pool1, Fluid; include_surface = true),
        ParticleRegion("DropDamped", drop1, Fluid; include_surface = true),
        ParticleRegion("PoolControl", pool2, Fluid; include_surface = true),
        ParticleRegion("DropControl", drop2, Fluid; include_surface = true),
    ]
    geometry = sampled_geometry(regions, dx, constants, Val(3), T; water_level = 0.25)
    shapes = (
        tank_damped_wall = shapes1.wall,
        tank_damped_floor = shapes1.floor,
        pool_damped = pool1,
        drop_damped = drop1,
        tank_control_wall = shapes2.wall,
        tank_control_floor = shapes2.floor,
        pool_control = pool2,
        drop_control = drop2,
    )
    return geometry, shapes
end

function cylinder_drop_case(dx, constants, ::Type{T}) where T
    tank1, shapes1 = cylinder_tank_parts(0.0, dx; radius = 0.6, height = 0.15)
    tank2, shapes2 = cylinder_tank_parts(1.35, dx; radius = 0.6, height = 0.15)
    pool1 = prism(circle((0.0, 0.0), 0.6; segments = 128), 0.0, 0.1)
    drop1 = prism(circle((0.0, 0.0), 0.06; segments = 64), 0.15, 0.25)
    pool2 = prism(circle((1.35, 0.0), 0.6; segments = 128), 0.0, 0.1)
    drop2 = prism(circle((1.35, 0.0), 0.06; segments = 64), 0.15, 0.25)
    regions = [
        ParticleRegion("TankDamped", tank1, Fixed),
        ParticleRegion("TankControl", tank2, Fixed),
        ParticleRegion("PoolDamped", pool1, Fluid; include_surface = true),
        ParticleRegion("DropDamped", drop1, Fluid; include_surface = true),
        ParticleRegion("PoolControl", pool2, Fluid; include_surface = true),
        ParticleRegion("DropControl", drop2, Fluid; include_surface = true),
    ]
    geometry = sampled_geometry(regions, dx, constants, Val(3), T; water_level = 0.25)
    shapes = (
        tank_damped_wall = shapes1.wall,
        tank_damped_floor = shapes1.floor,
        pool_damped = pool1,
        drop_damped = drop1,
        tank_control_wall = shapes2.wall,
        tank_control_floor = shapes2.floor,
        pool_control = pool2,
        drop_control = drop2,
    )
    return geometry, shapes
end

function damping_angle_case(dx, constants, ::Type{T}) where T
    tank, tank_shapes = box_tank_parts(
        0.0, dx; length = 1.3, width = 1.0, height = 0.4, closed_top = true,
    )
    water = prism(rectangle((0.0, 0.0), 0.3, 1.0), 0.0, 0.3)
    regions = [
        ParticleRegion("Tank", tank, Fixed),
        ParticleRegion("WaterBlock", water, Fluid; include_surface = true),
    ]
    geometry = sampled_geometry(regions, dx, constants, Val(3), T; water_level = 0.3)
    shapes = (; tank_wall = tank_shapes.wall, floor = tank_shapes.floor,
        ceiling = tank_shapes.ceiling, water)
    return geometry, shapes
end

function damping_case20(; variant = :plane_mdbc, dx = nothing, duration = nothing,
        output_interval = nothing, visualize = true, save_location = nothing,
        piston_motion_path = DAMPING_PISTON_MOTION, FloatType = Float32)
    variant in DAMPING_VARIANTS ||
        throw(ArgumentError("variant must be one of $(join(DAMPING_VARIANTS, ", "))"))
    dimension = 3
    default_dx = variant in (:plane_mdbc, :plane) ? 0.0125 : 0.005
    default_duration = variant in (:plane_mdbc, :plane) ? 20.0 :
        variant == :angle ? 3.5 : 2.5
    default_output_interval = variant in (:plane_mdbc, :plane) ? 0.05 : 0.01
    dx === nothing && (dx = default_dx)
    duration === nothing && (duration = default_duration)
    output_interval === nothing && (output_interval = default_output_interval)
    maximum_dx = 0.05
    isfinite(dx) && 0 < dx <= maximum_dx ||
        throw(ArgumentError("dx must be finite and in (0, $maximum_dx] m"))
    isfinite(duration) && duration > 0 ||
        throw(ArgumentError("duration must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("output_interval must be finite and positive"))

    T = FloatType
    water_level = variant in (:plane_mdbc, :plane) ? 0.4 :
        variant == :angle ? 0.3 : 0.25
    viscosity_alpha = variant in (:plane_mdbc, :plane) ? 0.02 : 0.01
    constants = SimulationConstants{T}(dimension; dx,
        c₀ = 20 * sqrt(9.81 * water_level),
        CFL = 0.2, α = viscosity_alpha, δᵩ = 0.1)

    if variant in (:plane_mdbc, :plane)
        geometry, shapes = damping_plane_geometry(dx, constants, T)
        zone = variant == :plane_mdbc ?
            DampingZone((5.0, 0.0, 0.0), (6.4, 0.0, 0.0);
                overlimit = 1.0, rate = 10) : nothing
        motion = PrescribedMotion(2;
            translation = damping_piston_trajectory(piston_motion_path),
            center = (0.0, 0.0, 0.0))
        metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(
            SimulationName = getproperty(DAMPING_CASE_NAMES, variant),
            SaveLocation = something(save_location,
                joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
            SimulationTime = duration, OutputTimes = output_interval,
            VisualizeInParaview = visualize, OpenLogFile = visualize,
            ExportSingleVTKHDF = true, ExportGridCells = true,
        )
    elseif variant == :angle
        geometry, shapes = damping_angle_case(dx, constants, T)
        zone = DampingZone((0.8, 0.5, 0.0), (1.3, 1.0, 0.0);
            overlimit = 0.2, rate = 30)
        motion = nothing
        metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName = DAMPING_CASE_NAMES.angle,
            SaveLocation = something(save_location,
                joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
            SimulationTime = duration, OutputTimes = output_interval,
            VisualizeInParaview = visualize, OpenLogFile = visualize,
            ExportSingleVTKHDF = true, ExportGridCells = false,
        )
    elseif variant == :box
        geometry, shapes = box_drop_case(dx, constants, T)
        zone = DampingBox(
            ((0.3, 0.3, 0.0), (0.7, 0.7, 0.12)),
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.12));
            directions = (:left, :right, :front, :back),
            overlimit = 0.1, rate = 10,
        )
        motion = nothing
        metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName = DAMPING_CASE_NAMES.box,
            SaveLocation = something(save_location,
                joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
            SimulationTime = duration, OutputTimes = output_interval,
            VisualizeInParaview = visualize, OpenLogFile = visualize,
            ExportSingleVTKHDF = true, ExportGridCells = false,
        )
    else
        geometry, shapes = cylinder_drop_case(dx, constants, T)
        zone = DampingCylinder((0.0, 0.0, 0.0), (0.0, 0.0, 0.12),
            0.25, 0.6; overlimit = 0.1, rate = 10)
        motion = nothing
        metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName = DAMPING_CASE_NAMES.cylinder,
            SaveLocation = something(save_location,
                joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
            SimulationTime = duration, OutputTimes = output_interval,
            VisualizeInParaview = visualize, OpenLogFile = visualize,
            ExportSingleVTKHDF = true, ExportGridCells = false,
        )
    end

    coefh = variant in (:plane_mdbc, :plane) ? 1.5 : 0.866025
    kernel = SPHKernelInstance{dimension,T}(WendlandC2();
        h = T(coefh * sqrt(3) * dx))
    measurements = variant in (:plane_mdbc, :plane) ? MeasurementConfig(
        water_column_probes = [
            WaterColumnProbe("WG$(x)", (Float64(x), 0.2, 0.0); radius = 2dx)
            for x in (2, 5)
        ],
    ) : nothing
    return (; geometry, shapes, constants, kernel, damping = zone, metadata,
        measurements, motion, dimension, variant, dx)
end

function run_damping_case20(; kwargs...)
    case = damping_case20(; kwargs...)
    (; metadata) = case
    mkpath(metadata.SaveLocation)
    particles = AllocateDataStructures(case.geometry, metadata)
    logger = SimulationLogger(metadata.SaveLocation)
    CleanUpSimulationFolder(metadata.SaveLocation)
    SavePolygonVTKHDF(
        joinpath(metadata.SaveLocation, metadata.SimulationName * "_Geometry.vtkhdf"),
        case.shapes,
    )
    simulation = RunSimulation(
        SimGeometry = case.geometry,
        SimMetaData = metadata,
        SimConstants = case.constants,
        SimKernel = case.kernel,
        SimLogger = logger,
        SimParticles = particles,
        SimViscosity = ArtificialViscosity(boundary_factor = 0),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        SimDamping = case.damping,
        SimMotion = case.motion,
        SimMeasurements = case.measurements,
    )
    return (; case, simulation, particles)
end
