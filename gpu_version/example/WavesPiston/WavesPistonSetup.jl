# Definitions only; example/WavesPiston2d.jl starts the selected simulation.
using SPHExampleGPU
using CSV
using HDF5

const PISTON_WAVE_DATA = joinpath(@__DIR__, "..", "data", "waves_piston")

"Second-order regular piston displacement and exact velocity, with the reference's one-period ramp."
function regular_piston_state(t; depth = 0.66, height = 0.15, period = 2.0, gravity = 9.81)
    omega = 2pi / period
    k = omega^2 / gravity
    for _ in 1:50
        k -= (gravity * k * tanh(k * depth) - omega^2) /
             (gravity * tanh(k * depth) + gravity * k * depth * sech(k * depth)^2)
    end
    transfer = 2sinh(k * depth)^2 / (sinh(k * depth) * cosh(k * depth) + k * depth)
    a1 = height / (2transfer)
    a2 = height^2 / (32depth) * (3cosh(k * depth) / sinh(k * depth)^3 - 2 / transfer)
    s = clamp(t / period, 0.0, 1.0)
    ramp = s^2 * (6 - 8s + 3s^2)
    ramp_velocity = 0 < t < period ? 12s * (1 - s)^2 / period : 0.0
    stroke = a1 * sin(omega * t) + a2 * sin(2omega * t)
    velocity = a1 * omega * cos(omega * t) + 2a2 * omega * cos(2omega * t)
    return ramp * stroke, ramp_velocity * stroke + ramp * velocity
end

"""
    waves_piston_case(; termination = :beach, waves = :regular, dx = 0.02,
        duration = 30, output_interval = 0.05, visualize = true, save_location = ...)

Build one of the four 2D DualSPHysics 5.4 `09_WavesPiston` cases with native
polygon geometry. `termination` is `:beach` or `:damping`; `waves` is `:regular`
or `:irregular`. Construction does not run the solver. Irregular motion uses
the bundled 1024-component second-order JONSWAP reference history (0–30 s).
"""
function waves_piston_case(; termination = :beach, waves = :regular, dx = 0.02,
        duration = 30.0, output_interval = 0.05, visualize = true,
        save_location = joinpath(@__DIR__, "..", "..", "output", "waves_piston_$(termination)_$(waves)"))
    termination in (:beach, :damping) || throw(ArgumentError("termination must be :beach or :damping"))
    waves in (:regular, :irregular) || throw(ArgumentError("waves must be :regular or :irregular"))
    isfinite(dx) && 0 < dx <= 0.1 || throw(ArgumentError("dx must satisfy 0 < dx <= 0.1 m"))
    isfinite(duration) && 0 < duration <= 30 || throw(ArgumentError("duration must satisfy 0 < duration <= 30 s, the bundled motion/theory interval"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))

    depth, top = 0.66, 1.0
    # Four conforming wall layers surround the 2h = 3.39dx support. The wet
    # piston face is x = 0 and slides one spacing above the bed.
    if termination == :beach
        tank = polyline([(-0.5, 0.0), (9.0, 0.0), (19.0, top)]; thickness = 3dx, side = :right)
        water = polygon([(0.0, 0.0), (9.0, 0.0), (15.6, depth), (0.0, depth)])
    else
        tank = polyline([(-0.5, 0.0), (15.5, 0.0), (15.5, top)]; thickness = 3dx, side = :right)
        water = rectangle((0.0, 0.0), 15.5, depth)
    end
    piston = rectangle((-3dx, dx), 3dx, top - dx)
    regions = [ParticleRegion("Tank", tank, Fixed; sampling = :conforming),
               ParticleRegion("Piston", piston, Moving), ParticleRegion("Water", water, Fluid)]
    sampled = sample_particles(regions, dx)
    water_bottom, water_level = extrema(last, sampled[3].positions)
    sound_speed = 20sqrt(9.81 * (water_level - water_bottom))
    constants = SimulationConstants{Float32}(2; dx, c₀ = sound_speed)
    geometry = [SPHGeometry{2,Float32}(particles.positions;
                    Density = hydrostatic_density(particles.positions, constants; water_level),
                    GroupMarker = i, Type = regions[i].type)
                for (i, particles) in enumerate(sampled)]

    if waves == :regular
        times = range(0.0, duration + 0.01; step = 0.01)
        states = regular_piston_state.(times)
        trajectory = SampledTrajectory(times, [(state[1], 0.0) for state in states];
            velocities = [(state[2], 0.0) for state in states])
    else
        rows = collect(CSV.File(joinpath(PISTON_WAVE_DATA, "IrregularMotion.csv")))
        trajectory = SampledTrajectory([row.time for row in rows], [(row.displacement, 0.0) for row in rows])
    end
    motion = PrescribedMotion(2; translation = trajectory, center = (0.0, 0.0))
    damping = termination == :damping ? DampingZone((11.0, 0.0), (15.5, 0.0); overlimit = 1, rate = 10) : nothing
    name = "WavesPiston$(titlecase(string(termination)))$(titlecase(string(waves)))2D"
    metadata = SimulationMetaData{2,Float32,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = name, SaveLocation = abspath(save_location), SimulationTime = duration,
        OutputTimes = output_interval, VisualizeInParaview = visualize, OpenLogFile = visualize,
        ExportSingleVTKHDF = true, ExportGridCells = true, GPUDoublePosition = true,
        GPUUseGraph = false, GPUBoundaryForces = false, GPULanesPerParticle = 1)
    h = 1.2sqrt(2) * dx
    kernel = SPHKernelInstance{2,Float32}(WendlandC2(); h = Float32(h), η² = Float32((0.1h)^2))
    # Match the reference's integration, DDT2 and zero wall-viscosity factor.
    scheme = DualSPHysicsSymplectic(; smoothing_length = 1.2sqrt(2) * dx,
        sound_speed, density_min = 700, density_max = 1300)
    measurements = MeasurementConfig(water_column_probes = [
        WaterColumnProbe("WG$(round(Int, x))", (x, 0.0); radius = 2dx) for x in (2.0, 6.0, 10.0)],
        velocity_probes = [MeasurementProbe("V6", (6.0, 0.4))])
    return (; geometry, constants, metadata, kernel, motion, damping, measurements, scheme,
        shapes = (; tank, piston, water), sampled, termination, waves)
end

"Write wave elevations and submerged velocity alongside the reference theory at x = 6 m."
function write_piston_comparison(case)
    (; metadata) = case
    theory = collect(CSV.File(joinpath(PISTON_WAVE_DATA, "$(titlecase(string(case.waves)))Theory.csv")))
    reference_times = getproperty.(theory, :time)
    output = joinpath(metadata.SaveLocation, metadata.SimulationName * "_Comparison.csv")
    h5open(joinpath(metadata.SaveLocation, metadata.SimulationName * ".vtkhdf"), "r") do file
        column = file["VTKHDF/WaterColumnProbes"]
        times = read(column["Steps/Values"])
        heights = reshape(read(column["PointData/WaterColumnHeight"]), 3, :)
        velocities = reshape(read(file["VTKHDF/VelocityProbes/PointData/Velocity"]), 3, :)
        open(output, "w") do io
            println(io, "time,eta_WG2,eta_WG6,eta_WG10,eta_theory_o1,eta_theory_o2,vx_SPH,vy_SPH,vx_theory_o1,vy_theory_o1,vx_theory_o2,vy_theory_o2")
            for (i, time) in enumerate(times)
                k = clamp(searchsortedlast(reference_times, time), 1, length(theory) - 1)
                fraction = clamp((time - reference_times[k]) / (reference_times[k+1] - reference_times[k]), 0, 1)
                sample(name) = (1 - fraction) * getproperty(theory[k], name) + fraction * getproperty(theory[k+1], name)
                row = (time, (heights[:, i] .- heights[:, 1])..., sample(:elevation_o1), sample(:elevation_o2),
                    velocities[1, i], velocities[2, i], sample(:velocity_x_o1), sample(:velocity_y_o1),
                    sample(:velocity_x_o2), sample(:velocity_y_o2))
                println(io, join(row, ','))
            end
        end
    end
    return output
end

"Run a piston-wave variant and write particles, geometry, wave gauges and theory at x = 6 m."
function run_waves_piston(; kwargs...)
    case = waves_piston_case(; kwargs...)
    (; metadata) = case
    mkpath(metadata.SaveLocation)
    particles = AllocateDataStructures(case.geometry, metadata)
    logger = SimulationLogger(metadata.SaveLocation)
    CleanUpSimulationFolder(metadata.SaveLocation)
    SavePolygonVTKHDF(joinpath(metadata.SaveLocation, metadata.SimulationName * "_Geometry.vtkhdf"), case.shapes)
    theory = joinpath(PISTON_WAVE_DATA, "$(titlecase(string(case.waves)))Theory.csv")
    cp(theory, joinpath(metadata.SaveLocation, metadata.SimulationName * "_Theory.csv"); force = true)
    simulation = RunSimulation(SimGeometry = case.geometry, SimMetaData = metadata, SimConstants = case.constants,
        SimParticles = particles, SimLogger = logger, SimKernel = case.kernel,
        SimViscosity = ArtificialViscosity(boundary_factor = 0), SimDensityDiffusion = FourtakasDensityDiffusion(),
        SimTimeStepping = case.scheme, SimMotion = case.motion, SimDamping = case.damping, SimMeasurements = case.measurements)
    comparison = write_piston_comparison(case)
    return (; case, simulation, particles, comparison)
end
