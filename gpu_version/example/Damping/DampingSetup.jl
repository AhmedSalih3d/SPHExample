using SPHExampleGPU
using StaticArrays: SVector

const DAMPING_VARIANTS = (:plane_mdbc, :plane, :angle, :box, :cylinder)
const DAMPING_CASE_NAMES = (
    plane_mdbc = "CaseDampingPlane_mDBC",
    plane = "CaseDampingPlaneNot_mDBC",
    angle = "CaseDampingAngle",
    box = "CaseDampingBox",
    cylinder = "CaseDampingCylinder",
)

function damping_plane_geometry(dx, constants, ::Type{T}; mdbc) where T
    length, depth, tank_height, wall_layers = 8.0, 0.4, 0.8, 3
    nx, water_rows, tank_rows = round.(Int, (length, depth, tank_height) ./ dx)

    wall_normals = Dict{Tuple{Int,Int},SVector{2,Float64}}()
    for j in -wall_layers:0, i in -wall_layers:(nx + wall_layers)
        wall_normals[(i, j)] = SVector(0.0, 1.0)
    end
    for j in 1:tank_rows, i in -wall_layers:0
        wall_normals[(i, j)] = SVector(1.0, 0.0)
    end
    for j in 1:tank_rows, i in nx:(nx + wall_layers)
        wall_normals[(i, j)] = SVector(-1.0, 0.0)
    end

    wall_indices = collect(keys(wall_normals))
    sort!(wall_indices; by = ij -> (ij[2], ij[1]))
    wall_positions = [SVector(i * dx, j * dx) for (i, j) in wall_indices]
    boundary = if mdbc
        normals = [wall_normals[index] for index in wall_indices]
        ghost_points = [point + dx * normal for (point, normal) in zip(wall_positions, normals)]
        ghost_normals = [SVector{2,T}(dx * normal) for normal in normals]
        particles = particle_struct_array(wall_positions, constants.ρ₀;
            GhostPoints = ghost_points, GhostNormals = ghost_normals)
        SPHGeometry{2,T}(; Particles = particles, GroupMarker = 1, Type = Fixed)
    else
        SPHGeometry{2,T}(wall_positions; Density = constants.ρ₀,
            GroupMarker = 1, Type = Fixed)
    end

    fluid_positions = vec(SVector{2,Float64}[
        SVector(i * dx, j * dx)
        for j in 1:(water_rows - 1), i in 1:(nx - 1)
    ])
    initial_velocity(point) = SVector{2,T}(
        T(0.25 * exp(-((point[1] - 1.2) / 0.45)^2)), zero(T),
    )
    water = SPHGeometry{2,T}(;
        Particles = particle_struct_array(
            fluid_positions, constants.ρ₀;
            Density = hydrostatic_density(fluid_positions, constants; water_level = depth),
            Velocity = initial_velocity.(fluid_positions),
        ),
        GroupMarker = 2,
        Type = Fluid,
    )
    shapes = (
        tank = polyline(
            [(0.0, tank_height), (0.0, 0.0), (length, 0.0), (length, tank_height)];
            thickness = wall_layers * dx, side = :right,
        ),
        water = rectangle((0.0, 0.0), length, depth),
    )
    return [boundary, water], shapes
end

function damping_tank_3d(dx, constants, ::Type{T}; footprint, length, width, radius, water_depth, height, velocity) where T
    interior = footprint === :cylinder ?
        circle((0.0, 0.0), radius; segments = 128) :
        rectangle((0.0, 0.0), length, width)
    wall_outline = outline(interior; thickness = 3dx, side = :outward)
    floor_outline = offset_polygon(interior, 3dx)
    wall_shape = prism(wall_outline, -3dx, height)
    floor_shape = prism(floor_outline, -3dx, 0.0)
    water_shape = prism(interior, 0.0, water_depth)
    regions = [
        ParticleRegion("Tank", (wall_shape, floor_shape), Fixed),
        ParticleRegion("Water", water_shape, Fluid; include_surface = false),
    ]
    boundary_particles, water_particles = sample_particles(regions, dx)
    water_density = hydrostatic_density(
        water_particles.positions, constants; water_level = water_depth,
    )
    water_velocity = velocity.(water_particles.positions)
    geometry = [
        SPHGeometry{3,T}(boundary_particles.positions; Density = constants.ρ₀,
            GroupMarker = 1, Type = Fixed),
        SPHGeometry{3,T}(;
            Particles = particle_struct_array(water_particles.positions, constants.ρ₀;
                Density = water_density, Velocity = SVector{3,T}.(water_velocity)),
            GroupMarker = 2,
            Type = Fluid,
        ),
    ]
    return geometry, (; tank_walls = wall_shape, floor = floor_shape, water = water_shape)
end

function damping_case20(; variant = :plane_mdbc, dx = nothing, duration = nothing,
        output_interval = 0.05, visualize = true,
        save_location = nothing, FloatType = Float32)
    variant in DAMPING_VARIANTS ||
        throw(ArgumentError("variant must be one of $(join(DAMPING_VARIANTS, ", "))"))
    dimension = variant in (:plane_mdbc, :plane) ? 2 : 3
    dx === nothing && (dx = dimension == 2 ? 0.02 : 0.1)
    duration === nothing && (duration = variant in (:plane_mdbc, :plane) ? 12.0 :
        variant == :angle ? 0.66 : variant == :box ? 1.33 : 1.43)
    isfinite(dx) && 0 < dx <= (dimension == 2 ? 0.1 : 0.2) ||
        throw(ArgumentError("dx must be finite and in (0, $(dimension == 2 ? 0.1 : 0.2)] m"))
    isfinite(duration) && duration > 0 ||
        throw(ArgumentError("duration must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("output_interval must be finite and positive"))

    T = FloatType
    constants = SimulationConstants{T}(dimension; dx,
        c₀ = 20 * sqrt(9.81 * (dimension == 2 ? 0.4 : 0.8)),
        CFL = 0.5, α = 0.01, δᵩ = 0.1)

    geometry, shapes, damping, metadata = if variant in (:plane_mdbc, :plane)
        mdbc = variant == :plane_mdbc
        generated_geometry, generated_shapes = damping_plane_geometry(dx, constants, T; mdbc)
        zone = DampingZone((5.0, 0.0), (6.5, 0.0); overlimit = 0.3, rate = 10)
        metadata = if mdbc
            SimulationMetaData{2,T,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(
                SimulationName = DAMPING_CASE_NAMES.plane_mdbc,
                SaveLocation = something(save_location,
                    joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
                SimulationTime = duration, OutputTimes = output_interval,
                VisualizeInParaview = visualize, OpenLogFile = visualize,
                ExportSingleVTKHDF = true, ExportGridCells = true,
            )
        else
            SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
                SimulationName = DAMPING_CASE_NAMES.plane,
                SaveLocation = something(save_location,
                    joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
                SimulationTime = duration, OutputTimes = output_interval,
                VisualizeInParaview = visualize, OpenLogFile = visualize,
                ExportSingleVTKHDF = true, ExportGridCells = true,
            )
        end
        generated_geometry, generated_shapes, zone, metadata
    elseif variant == :angle
        length, width, height, depth = 6.0, 2.0, 1.2, 0.8
        pulse = point -> SVector{3,T}(
            T(0.25 * exp(-((point[1] - 1.0) / 0.5)^2)), zero(T), zero(T),
        )
        generated_geometry, generated_shapes = damping_tank_3d(dx, constants, T;
            footprint = :box, length, width, radius = 0.0, water_depth = depth,
            height, velocity = pulse)
        zone = DampingZone((4.0, 0.0, 0.0), (5.0, 0.4, 0.0);
            overlimit = 0.3, rate = 10)
        metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName = DAMPING_CASE_NAMES.angle,
            SaveLocation = something(save_location,
                joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
            SimulationTime = duration, OutputTimes = output_interval,
            VisualizeInParaview = visualize, OpenLogFile = visualize,
            ExportSingleVTKHDF = true, ExportGridCells = false,
        )
        generated_geometry, generated_shapes, zone, metadata
    elseif variant == :box
        length, width, height, depth = 4.0, 4.0, 1.2, 0.8
        center = (length / 2, width / 2)
        pulse = point -> radial_damping_pulse(point, center, 0.5, 0.3, 0.2, T)
        generated_geometry, generated_shapes = damping_tank_3d(dx, constants, T;
            footprint = :box, length, width, radius = 0.0, water_depth = depth,
            height, velocity = pulse)
        box = DampingBox(
            ((1.0, 1.0, 0.0), (length - 1.0, width - 1.0, depth)),
            ((dx, dx, 0.0), (length - dx, width - dx, depth));
            directions = (:left, :right, :front, :back),
            overlimit = 0.1, rate = 10,
        )
        metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName = DAMPING_CASE_NAMES.box,
            SaveLocation = something(save_location,
                joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
            SimulationTime = duration, OutputTimes = output_interval,
            VisualizeInParaview = visualize, OpenLogFile = visualize,
            ExportSingleVTKHDF = true, ExportGridCells = false,
        )
        generated_geometry, generated_shapes, box, metadata
    else
        radius, height, depth = 2.0, 1.2, 0.8
        pulse = point -> radial_damping_pulse(point, (0.0, 0.0), 0.5, 0.3, 0.2, T)
        generated_geometry, generated_shapes = damping_tank_3d(dx, constants, T;
            footprint = :cylinder, length = 0.0, width = 0.0, radius,
            water_depth = depth, height, velocity = pulse)
        cylinder = DampingCylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            1.2, radius - 2dx; overlimit = 0.1, rate = 10)
        metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName = DAMPING_CASE_NAMES.cylinder,
            SaveLocation = something(save_location,
                joinpath(@__DIR__, "..", "..", "output", "20_Damping", String(variant))),
            SimulationTime = duration, OutputTimes = output_interval,
            VisualizeInParaview = visualize, OpenLogFile = visualize,
            ExportSingleVTKHDF = true, ExportGridCells = false,
        )
        generated_geometry, generated_shapes, cylinder, metadata
    end

    kernel = SPHKernelInstance{dimension,T}(WendlandC2(); dx = T(dx))
    measurements = dimension == 2 ?
        MeasurementConfig(water_column_probes = [
            WaterColumnProbe("WG$(x)", (Float64(x), 0.0); radius = 2dx)
            for x in (2, 6)
        ]) : nothing
    return (; geometry, shapes, constants, kernel, damping, metadata, measurements,
        dimension, variant)
end

function radial_damping_pulse(point, center, radius, width, amplitude, ::Type{T}) where T
    x, y = point[1] - center[1], point[2] - center[2]
    distance = sqrt(x^2 + y^2)
    distance == 0 && return zero(SVector{3,T})
    speed = amplitude * exp(-((distance - radius) / width)^2)
    return SVector{3,T}(T(speed * x / distance), T(speed * y / distance), zero(T))
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
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        SimDamping = case.damping,
        SimMeasurements = case.measurements,
    )
    return (; case, simulation, particles)
end
