# GPU simulation of the Re = 100, 2D lid-driven cavity in the Rocky 2025 R1
# verification manual. Requires an NVIDIA GPU with CUDA.
#
# Particles and ghost nodes are generated in this run file. Optional CSV export:
#     julia --project=gpu_version gpu_version/example/GenerateLidDrivenCavity2D.jl
#     julia --project=gpu_version gpu_version/example/LidDrivenCavity2d.jl
#
# The top wall has a prescribed horizontal velocity but does not translate out
# of the cavity (`MoveParticles = false`). The lid spans the outside width of
# the tank to overlap both side walls. mDBC, density diffusion and shifting
# keep the fluid coupled to the lid throughout the run. The default duration
# is the 60 s specified by ANSYS. Optional arguments are
# [save_dir] [dx] [duration] [input_dir].
using SPHExampleGPU
using StaticArrays: SVector
using StructArrays

include(joinpath(@__DIR__, "GenerateLidDrivenCavity2D.jl"))

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
        fields =
            (Position = positions, Density = fill(SimConstants.ρ₀, length(positions)))
        if region.type != Fluid
            ghosts = [
                SVector{2, Float64}(lid_driven_cavity_ghost_node(x, dx)) for x in positions
            ]
            fields = merge(
                fields,
                (GhostPoints = ghosts, GhostNormals = ghosts .- positions),
            )
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
            Particles = StructArray(fields),
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
    logger = SimulationLogger(save_location; to_console = true)
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

# if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    save_location =
        length(ARGS) >= 1 ? ARGS[1] : raw"C:\TestSimulations\LidDrivenCavity2D_GPU"
    dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
    simulation_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 60.0
    input_dir =
        length(ARGS) >= 4 ? ARGS[4] :
        normpath(joinpath(@__DIR__, "..", "input", "lid_driven_cavity_2d_generated"))
    run_lid_driven_cavity_2d(; save_location, dx, simulation_time, input_dir)
# end
