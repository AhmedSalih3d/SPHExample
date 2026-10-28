# A video-inspired, motor-driven floating paddle wheel in a periodic 3D trough.
# Run from the repository root:
# julia --project=gpu_version -t 2,0 gpu_version/example/WheelPeriodic3d.jl [output_dir] [dx] [duration]
module PeriodicWheelCase

using SPHExampleGPU
using StaticArrays: SVector
using LinearAlgebra: norm

export WheelPeriodicConfig, wheel_periodic_geometry, prepare_wheel_periodic_3d, run_wheel_periodic_3d

"""
    WheelPeriodicConfig(; kwargs...)

Illustrative SI inputs, chosen for a showcase with the user's permission, not
measurements from `WheelFreecad_anim0001-0151.mp4`. Coordinates are longitudinal
`x`, transverse/axle `y`, vertical `z`. `domain_min/max` bound the trough's wetted
interior up to its side-wall tops; air above it is open. See `WheelPeriodic3d.md`.
Changing `wheel_radius` scales the default geometry and numerical spacing only;
speeds, gravity and time remain independent physical inputs.
"""
Base.@kwdef struct WheelPeriodicConfig
    wheel_radius::Float64 = 0.3
    particle_spacing::Float64 = wheel_radius / 12
    domain_min::SVector{3,Float64} = SVector(-3wheel_radius, -1.5wheel_radius, 0.0)
    domain_max::SVector{3,Float64} = SVector(3wheel_radius, 1.5wheel_radius, 0.8wheel_radius)
    wall_thickness::Float64 = 4particle_spacing
    fluid_level::Float64 = 0.6wheel_radius
    initial_fluid_velocity::SVector{3,Float64} = SVector(0.4, 0.0, 0.0)

    periodic_axis::Int = 1
    periodic_min::Float64 = domain_min[1]
    periodic_max::Float64 = domain_max[1]
    periodic_shift::SVector{3,Float64} = SVector(0.0, 0.0, 0.0)

    wheel_center::SVector{3,Float64} = SVector(0.0, 0.0, 1.25wheel_radius)
    wheel_axis::SVector{3,Float64} = SVector(0.0, 1.0, 0.0)
    wheel_angular_speed::Float64 = -2.0 # rad/s, right-hand rule about wheel_axis
    wheel_width::Float64 = 1.4wheel_radius
    rim_width::Float64 = 0.2wheel_radius
    rim_thickness::Float64 = 0.2wheel_radius
    paddle_count::Int = 12
    paddle_depth::Float64 = 0.3wheel_radius
    paddle_thickness::Float64 = 0.2wheel_radius
    spoke_count::Int = 4
    spoke_thickness::Float64 = 0.2wheel_radius
    hub_radius::Float64 = 0.25wheel_radius
    shaft_radius::Float64 = 0.1wheel_radius
    shaft_length::Float64 = 2.4wheel_radius

    reference_density::Float64 = 1000.0
    gravity::Float64 = 9.81
    sound_speed::Float64 = 20.0
    artificial_viscosity::Float64 = 0.01
    density_diffusion::Float64 = 0.1
    cfl::Float64 = 0.2
    simulation_time::Float64 = 3.2
    output_interval::Float64 = 0.01 #0.08
end

function validate_wheel_config(c::WheelPeriodicConfig)
    for name in (:wheel_radius, :particle_spacing, :wall_thickness, :wheel_width,
                 :rim_width, :rim_thickness, :paddle_depth, :paddle_thickness,
                 :spoke_thickness, :hub_radius, :shaft_radius, :shaft_length,
                 :reference_density, :sound_speed, :cfl, :simulation_time, :output_interval)
        value = getproperty(c, name)
        isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    for name in (:gravity, :artificial_viscosity, :density_diffusion)
        value = getproperty(c, name)
        isfinite(value) && value >= 0 || throw(ArgumentError("$name must be finite and nonnegative"))
    end
    all(isfinite, c.domain_min) && all(isfinite, c.domain_max) && all(c.domain_max .> c.domain_min) ||
        throw(ArgumentError("domain bounds must be finite and strictly ordered"))
    all(isfinite, c.wheel_center) && all(isfinite, c.initial_fluid_velocity) && isfinite(c.wheel_angular_speed) ||
        throw(ArgumentError("wheel_center and velocities must be finite"))
    isfinite(c.fluid_level) && c.domain_min[3] + 2c.particle_spacing < c.fluid_level < c.domain_max[3] ||
        throw(ArgumentError("fluid_level must resolve at least two layers below the side-wall tops"))
    c.periodic_axis == 1 && iszero(c.periodic_shift) ||
        throw(ArgumentError("this straight trough requires x periodicity with zero transverse shift"))
    c.periodic_min == c.domain_min[1] && c.periodic_max == c.domain_max[1] ||
        throw(ArgumentError("periodic range must match the open x ends of the trough"))
    cells = (c.periodic_max - c.periodic_min) / c.particle_spacing
    isapprox(cells, round(cells); atol = 1e-8, rtol = 0) ||
        throw(ArgumentError("periodic length must be an integer multiple of particle_spacing"))
    c.periodic_min <= c.wheel_center[1] < c.periodic_max ||
        throw(ArgumentError("wheel_center must be in the periodic interval"))
    all(isfinite, c.wheel_axis) && iszero(c.wheel_axis[1]) && iszero(c.wheel_axis[3]) && !iszero(c.wheel_axis[2]) ||
        throw(ArgumentError("wheel_axis must be parallel to y, matching the generated shaft"))
    0 < c.rim_width < c.wheel_width / 2 && 0 < c.rim_thickness < c.wheel_radius &&
        c.shaft_radius <= c.hub_radius < c.wheel_radius - c.rim_thickness &&
        c.paddle_depth < c.wheel_radius - c.hub_radius ||
        throw(ArgumentError("rim, hub, shaft and paddle dimensions must fit the wheel"))
    c.paddle_count >= 3 && c.spoke_count >= 3 || throw(ArgumentError("at least three paddles and spokes are required"))
    minimum((c.rim_width, c.rim_thickness, c.paddle_thickness, c.paddle_depth,
             c.spoke_thickness, 2c.shaft_radius)) >= 2c.particle_spacing ||
        throw(ArgumentError("wheel features need at least two particle spacings; reduce particle_spacing"))
    c.wall_thickness >= 4c.particle_spacing || throw(ArgumentError("wall_thickness must cover the kernel support (4dx)"))
    c.paddle_thickness < 2c.wheel_radius || throw(ArgumentError("paddle_thickness must fit inside the wheel diameter"))
    c.wheel_center[3] - c.wheel_radius >= c.domain_min[3] + c.particle_spacing ||
        throw(ArgumentError("the full rotating wheel must clear the floor by at least particle_spacing"))
    c.wheel_center[3] - c.wheel_radius < c.fluid_level < c.wheel_center[3] + c.wheel_radius ||
        throw(ArgumentError("the showcase wheel must be partially submerged"))
    c.shaft_length >= c.wheel_width || throw(ArgumentError("shaft_length must span wheel_width"))
    c.domain_min[2] + c.particle_spacing < c.wheel_center[2] - c.shaft_length / 2 &&
        c.wheel_center[2] + c.shaft_length / 2 < c.domain_max[2] - c.particle_spacing ||
        throw(ArgumentError("shaft must clear both side walls"))
    radius_bound = hypot(c.wheel_radius, c.shaft_length / 2)
    2(radius_bound + 4c.particle_spacing) < c.periodic_max - c.periodic_min ||
        throw(ArgumentError("wheel diameter plus twice the support radius must fit the periodic interval"))
    speed = max(norm(c.initial_fluid_velocity), abs(c.wheel_angular_speed) * c.wheel_radius,
                sqrt(c.gravity * (c.fluid_level - c.domain_min[3])))
    c.sound_speed >= 10speed || throw(ArgumentError("sound_speed must be at least 10 times the initial, tip and gravity-wave speeds"))
    return c
end

# Draw cross sections in (x,z), extrude along y, then permute into physical
# (x,y,z). This reuses the package's 3D prism/lattice ownership implementation.
function wheel_regions(c)
    lo, hi, center = c.domain_min, c.domain_max, c.wheel_center
    r, w, t = c.wheel_radius, c.wheel_width, c.wall_thickness
    xz = (center[1], center[3])
    disk = circle(xz, r; segments = 64)
    rim = outline(disk; thickness = c.rim_thickness, side = :inward)
    sides = ((center[2] - w / 2, center[2] - w / 2 + c.rim_width),
             (center[2] + w / 2 - c.rim_width, center[2] + w / 2))
    wheel = [prism(rim, a, b) for (a, b) in sides]
    for (a, b) in sides, k in 0:(c.spoke_count - 1)
        angle = 2pi * k / c.spoke_count
        endpoint = xz .+ (r - c.rim_thickness / 2) .* (cos(angle), sin(angle))
        push!(wheel, prism(line(xz, endpoint; thickness = c.spoke_thickness), a, b))
    end
    push!(wheel, prism(circle(xz, c.hub_radius; segments = 32), center[2] - w / 2, center[2] + w / 2))
    push!(wheel, prism(circle(xz, c.shaft_radius; segments = 32),
        center[2] - c.shaft_length / 2, center[2] + c.shaft_length / 2))
    for k in 0:(c.paddle_count - 1)
        outer_edge = sqrt(r^2 - (c.paddle_thickness / 2)^2)
        paddle = rectangle((center[1] + r - c.paddle_depth, center[3] - c.paddle_thickness / 2),
            outer_edge - (r - c.paddle_depth), c.paddle_thickness)
        paddle = rotate(paddle, 2pi * k / c.paddle_count; origin = xz)
        push!(wheel, prism(paddle, center[2] - w / 2, center[2] + w / 2))
    end
    floor = prism(rectangle((lo[1], lo[3] - t), hi[1] - lo[1], t), lo[2] - t, hi[2] + t)
    wall_section = rectangle((lo[1], lo[3] - t), hi[1] - lo[1], hi[3] - lo[3] + t)
    walls = (floor, prism(wall_section, lo[2] - t, lo[2]), prism(wall_section, hi[2], hi[2] + t))
    fluid = prism(rectangle((lo[1], lo[3]), hi[1] - lo[1], c.fluid_level - lo[3]), lo[2], hi[2])
    return [ParticleRegion("Trough", walls, Fixed), ParticleRegion("Wheel", wheel, Floating),
        # Include both x faces here, then retain only the half-open interval.
        # Using the usual open-fluid sampling would leave a dry plane at the seam.
        ParticleRegion("Fluid", fluid, Fluid; include_surface = true)]
end

"""
    wheel_periodic_geometry(config = WheelPeriodicConfig(); FloatType = Float32)

Build all particles in memory, a physical `PeriodicBoundary`, and a floating
wheel with `FloatingConstraints(HingePoint, HingeAxis, HingeVelocity)`. No CUDA
hardware or input assets are needed. Every fluid particle carries the configured
velocity in its input `SPHGeometry`, before host allocation or GPU upload.
"""
function wheel_periodic_geometry(c::WheelPeriodicConfig = WheelPeriodicConfig(); FloatType = Float32)
    validate_wheel_config(c)
    T = FloatType
    constants = SimulationConstants{T}(3; dx = c.particle_spacing, ρ₀ = c.reference_density,
        c₀ = c.sound_speed, g = c.gravity, α = c.artificial_viscosity, δᵩ = c.density_diffusion, CFL = c.cfl)
    periodicity = PeriodicBoundary{3}(c.periodic_axis; min = c.periodic_min, max = c.periodic_max, shift = c.periodic_shift)
    kernel = SPHKernelInstance{3,T}(WendlandC2(); dx = constants.dx)
    constraints = FloatingConstraints{3,T}(HingePoint = c.wheel_center,
        HingeAxis = c.wheel_axis, HingeVelocity = c.wheel_angular_speed)
    regions = wheel_regions(c)
    sampled = sample_particles(regions, c.particle_spacing)
    geometry = map(enumerate(sampled)) do (marker, region)
        # Never duplicate an endpoint or wrap wheel particles independently.
        # The floating solver keeps a seam-straddling rigid body whole.
        tolerance = 1e-6c.particle_spacing
        positions = [SVector(region.type != Floating && abs(x[1] - c.periodic_min) <= tolerance ? c.periodic_min : x[1], x[3], x[2])
            for x in region.positions if region.type == Floating ||
                c.periodic_min - tolerance <= x[1] < c.periodic_max - tolerance]
        isempty(positions) && throw(ArgumentError("$(region.name) has no particles; check geometry and spacing"))
        particles = if region.type == Fluid
            particle_struct_array(positions, constants.ρ₀;
                Density = T.(hydrostatic_density(positions, constants; water_level = c.fluid_level)),
                Velocity = fill(SVector{3,T}(c.initial_fluid_velocity), length(positions)))
        else
            particle_struct_array(positions, constants.ρ₀)
        end
        # RelativeWeight is bookkeeping for this fully prescribed hinge; its
        # mass does not determine the imposed motion. This is not free flotation.
        floating = region.type == Floating ? FloatingDetails{T}(RelativeWeight = one(T), Constraints = constraints) : nothing
        SPHGeometry{3,T}(; Particles = particles, GroupMarker = marker, Type = region.type, Floating = floating)
    end
    return (; config = c, geometry, constants, kernel, periodicity, constraints)
end

"""
    prepare_wheel_periodic_3d(config; save_location, FloatType = Float32, ...)

Prepare the exact solver inputs on the host, including initial velocities.
Does not create files, loggers or device buffers; useful for inspecting t = 0.
"""
function prepare_wheel_periodic_3d(c::WheelPeriodicConfig = WheelPeriodicConfig();
        save_location = joinpath(@__DIR__, "..", "output", "wheel_periodic_3d"),
        FloatType = Float32, visualize = false, open_log_file = false, use_graph = true)
    case = wheel_periodic_geometry(c; FloatType)
    meta = SimulationMetaData{3,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "WheelPeriodic3D", SaveLocation = abspath(save_location),
        SimulationTime = c.simulation_time, OutputTimes = c.output_interval,
        VisualizeInParaview = visualize, OpenLogFile = open_log_file,
        Periodicity = case.periodicity, GPUCellSubdivision = 2, GPUDoublePosition = true, GPULanesPerParticle = 1,
        GPUUseGraph = use_graph, GPUClipTimeStepToOutput = true,
        GPUFixedGrid = true, GPUGridExpansion = ((false, false), (true, true), (false, true)),
        ExportSingleVTKHDF = true, ExportGridCells = false)
    particles = AllocateDataStructures(case.geometry, meta)
    return (; case, meta, particles)
end

"""
    run_wheel_periodic_3d(config = WheelPeriodicConfig(); save_location, ...)

Run the motor-driven floating wheel with 3D DBC, Wendland C2, artificial
viscosity, hydrostatic initialization and density diffusion. Requires CUDA.
Reject a nonempty output directory so existing user data cannot be overwritten.
Return the standard `SimulationResult`, including the final floating-body state.
"""
function run_wheel_periodic_3d(c::WheelPeriodicConfig = WheelPeriodicConfig();
        save_location = joinpath(@__DIR__, "..", "output", "wheel_periodic_3d_dp0.0125_LinearDensityDiffusion_GPULanesPerParticle=1"), kwargs...)
    save = abspath(save_location)
    ispath(save) && (!isdir(save) || !isempty(readdir(save))) &&
        throw(ArgumentError("output path must be a new or empty directory: $save"))
    setup = prepare_wheel_periodic_3d(c; save_location = save, kwargs...)
    mkpath(save)
    # Record all assumptions, including the full vector, separately from the
    # solver log and the wheel's angular velocity in the floating-body CSV.
    open(joinpath(save, "WheelPeriodic3D_config.txt"), "w") do io
        println(io, "Video-inspired illustrative inputs, not measured video values. SI units.")
        for name in fieldnames(WheelPeriodicConfig)
            println(io, name, " = ", getproperty(c, name))
        end
    end
    logger = SimulationLogger(save)
    try
        return RunSimulation(SimGeometry = setup.case.geometry, SimMetaData = setup.meta,
            SimParticles = setup.particles, SimConstants = setup.case.constants,
            SimKernel = setup.case.kernel, SimLogger = logger,
            SimTimeStepping = SingleNeighborTimeStepping(), SimViscosity = ArtificialViscosity(),
            SimDensityDiffusion = LinearDensityDiffusion())
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

end # module PeriodicWheelCase

# if abspath(PROGRAM_FILE) == @__FILE__
    using .PeriodicWheelCase
    length(ARGS) <= 3 || error("usage: WheelPeriodic3d.jl [output_dir] [dx] [duration]")
    config = PeriodicWheelCase.WheelPeriodicConfig(
        particle_spacing = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.0125,
        simulation_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 3.2)
    PeriodicWheelCase.run_wheel_periodic_3d(config; save_location = isempty(ARGS) ?
        joinpath(@__DIR__, "..", "output", "wheel_periodic_3d_dp0.0125_LinearDensityDiffusion_GPULanesPerParticle=1_Verlet") : ARGS[1])
# end
