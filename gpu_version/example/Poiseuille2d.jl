# Newtonian plane Poiseuille flow between stationary plates, with periodic x
# boundaries, matching DualSPHysics 5.4's main/15_Poiseuille example.
# Reference x/z coordinates map to simulation x/y. Including or running this
# file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/Poiseuille2d.jl
using SPHExampleGPU
using StaticArrays: SVector

include(joinpath(@__DIR__, "data", "poiseuille", "validation.jl"))

let
    Dimensions = 2
    FloatType  = Float32

    # Channel dimensions and particle spacing in metres. The 5 mm × 1 mm
    # channel is centred on y = 0; 20 µm spacing gives 250 fluid columns.
    channel_length  = 0.005
    channel_height  = 0.001
    dx              = 0.00002
    simulation_time = 0.5
    output_interval = 0.002
    save_location   = joinpath(@__DIR__, "..", "output", "poiseuille_2d")
    visualize       = true
    open_log_file   = true
    use_graph       = true

    isfinite(dx) && 0 < dx < channel_height / 2 || throw(ArgumentError("dx must be finite, positive and smaller than the half-height"))
    all(x -> isfinite(x) && x > 0, (channel_length, channel_height, simulation_time, output_interval)) ||
        throw(ArgumentError("channel dimensions, simulation_time and output_interval must be finite and positive"))
    nx, ny = round(Int, channel_length / dx), round(Int, channel_height / dx)
    isapprox(nx * dx, channel_length; atol = 1e-10, rtol = 0) &&
        isapprox(ny * dx, channel_height; atol = 1e-10, rtol = 0) && iseven(ny) ||
        throw(ArgumentError("dx must divide the channel length and half-height exactly"))

    # Scalar gravity acts vertically, so horizontal acceleration is supplied
    # to the water with ExternalForce below.
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx  = dx,
        ρ₀  = 1000,
        g   = 0,
        c₀  = 1,
        ν₀  = 1e-6,
        CFL = 0.2,
    )

    # GenCase's pointref shifts x by half a spacing. The fixed plates have
    # one particle row each, and fluid fills the 49 rows between them:
    # 500 fixed + 12,250 fluid = 12,750 particles, exactly as in the reference.
    # Across the periodic seam the end columns remain one spacing apart.
    WallPoints = [SVector((i + 0.5) * dx, j * dx)
                  for j in (-ny÷2, ny÷2) for i in 0:nx-1]
    FluidPoints = [SVector((i + 0.5) * dx, j * dx)
                   for j in -ny÷2+1:ny÷2-1 for i in 0:nx-1]

    Walls = SPHGeometry{Dimensions, FloatType}(WallPoints;
        Density     = SimConstants.ρ₀,
        GroupMarker = 1,
        Type        = Fixed,
    )

    Water = SPHGeometry{Dimensions, FloatType}(FluidPoints;
        Density     = SimConstants.ρ₀,
        GroupMarker = 2,
        Type        = Fluid,
    )

    SimulationGeometry = [Walls, Water]

    # XPeriodicIncZ = 0 in DualSPHysics: the period has no transverse shift.
    # Limits lie half a spacing beyond the first and last particle columns.
    Periodicity = PeriodicBoundary{Dimensions}(1;
        min = 0,
        max = channel_length,
    )

    SimExternalForces = ExternalForce{Dimensions, FloatType}(Water.GroupMarker;
        times               = [0.0, 1.0],
        linear_acceleration = [SVector(0.0001, 0.0), SVector(0.0001, 0.0)],
    )

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName      = "Poiseuille2D",
        SaveLocation        = save_location,
        SimulationTime      = simulation_time,
        OutputTimes         = output_interval,
        VisualizeInParaview = visualize,
        ExportSingleVTKHDF  = true,
        OpenLogFile         = open_log_file,
        Periodicity         = Periodicity,
        GPUUseGraph         = use_graph,
        GPUCellSubdivision  = 1,
        GPUBoundaryForces   = false,
        GPUFixedGrid        = true,
        GPUDoublePosition   = true,
    )

    # h = √2 dx, the reference's coefh = 1 in 2D. Keep its laminar + SPS
    # viscosity, without shifting or density diffusion. Julia uses symplectic
    # stepping; the distributed DualSPHysics case selects Verlet (40 steps).
    SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(sqrt(2) * dx))
    SimViscosity        = LaminarSPS()
    SimDensityDiffusion = ZeroDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)

    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    result = try
        RunSimulation(
            SimGeometry         = SimulationGeometry,
            SimMetaData         = SimMetaData,
            SimConstants        = SimConstants,
            SimLogger           = SimLogger,
            SimParticles        = SimParticles,
            SimKernel           = SimKernel,
            SimViscosity        = SimViscosity,
            SimDensityDiffusion = SimDensityDiffusion,
            SimTimeStepping     = SymplecticTimeStepping(),
            SimExternalForces   = SimExternalForces,
        )
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end

    # Compare against the startup solution at the actual final simulation time.
    println(poiseuille_profile(result;
        path = joinpath(SimMetaData.SaveLocation, "Poiseuille2D_Profile.csv")))
    result
end
