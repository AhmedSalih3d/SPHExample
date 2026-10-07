# Four independent floating cylinders with different densities in one tank.
# Usage from gpu_version/:
# julia --project=. -t 1,0 example/MultipleFloatingCylinders2d.jl [output_dir] [time] [dx]
using SPHExampleGPU
using Dates

"""
    run_multiple_floating_cylinders(output_dir; simtime = 2.0, dx = 0.05)

Run four radius-0.6 m cylinders in a 10 m wide, 6 m deep tank. Relative densities
are 0.7, 1.0, 1.2 and 1.5 from left to right. Release them after a 0.25 s settling
pause and save particle, grid and rigid-body output, then open the ParaView state.
The output directory must be new or empty to preserve earlier simulations.
"""
function run_multiple_floating_cylinders(output_dir; simtime = 2.0, dx = 0.05)
    isdir(output_dir) && !isempty(readdir(output_dir)) &&
        error("Output directory must be new or empty: $output_dir")
    mkpath(output_dir)
    T = Float32
    depth = 6.0
    radius = 0.6
    relative_weights = [0.7, 1.0, 1.2, 1.5]
    horizontal_positions = [-3.0, -1.0, 1.0, 3.0]
    constants = SimulationConstants{T}(; dx, g = 9.81,
        c₀ = 30 * sqrt(9.81 * 0.8), CFL = 0.2, δᵩ = 0.1, ν₀ = 1e-6)
    regions = [ParticleRegion("Cylinder_$b",
        circle((horizontal_positions[b], depth), radius; segments = 128),
        Floating; sampling = :conforming) for b in eachindex(relative_weights)]
    push!(regions, ParticleRegion("Bound",
        polyline([(-5.0, 8.0), (-5.0, 0.0), (5.0, 0.0), (5.0, 8.0)];
            thickness = 3dx, side = :right), Fixed))
    push!(regions, ParticleRegion("Fluid",
        rectangle((-5.0, 0.0), 10.0, depth + dx / 2), Fluid))
    sampled = sample_particles(regions, constants.dx)
    water_level = maximum(last, sampled[end].positions)
    geometry = [SPHGeometry{2, T}(region.positions;
        Density = hydrostatic_density(region.positions, constants; water_level),
        GroupMarker = b, Type = regions[b].type,
        Floating = b <= length(relative_weights) ?
            FloatingDetails{T}(RelativeWeight = relative_weights[b], PauseTime = 0.25) :
            nothing) for (b, region) in enumerate(sampled)]
    metadata = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(
        SimulationName = "MultipleFloatingCylinders2D", SaveLocation = abspath(output_dir),
        SimulationTime = simtime, OutputTimes = 0.04,
        ExportSingleVTKHDF = true, ExportGridCells = true,
        GPUDoublePosition = true, VisualizeInParaview = true, OpenLogFile = false)
    particles = AllocateDataStructures(geometry, metadata)
    @info "Four floating cylinders" particles = length(particles) relative_weights
    RunSimulation(SimGeometry = geometry, SimMetaData = metadata,
        SimConstants = constants,
        SimKernel = SPHKernelInstance{2, T}(WendlandC2(); h = T(1.2 * sqrt(2) * dx)),
        SimLogger = SimulationLogger(output_dir), SimParticles = particles,
        SimViscosity = LaminarSPS(), SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping())
    return metadata
end

if abspath(PROGRAM_FILE) == @__FILE__
    output = isempty(ARGS) ? joinpath(@__DIR__, "..", "particles",
        "MultipleFloatingCylinders2D_" * Dates.format(now(), "yyyymmdd_HHMMSS")) : ARGS[1]
    simtime = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0
    spacing = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05
    run_multiple_floating_cylinders(output; simtime, dx = spacing)
end
