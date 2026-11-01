# 2D dam break in a 4 m by 3 m open tank, with a downstream refinement patch.
# Coarse particles use dx = 0.01 m; the patch uses dx / 2. Both domains run on
# the GPU and exchange reconstructed buffer states and accumulated mass flux.
# The geometry follows DualSPHysics 5.4 vresolution/01_DamBreak2D; density
# diffusion uses SPHExample's LinearDensityDiffusion term.
# Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/VResolutionDamBreak2d.jl
using SPHExampleGPU
using StaticArrays: SVector

let
    Dimensions = 2
    FloatType  = Float32

    # The water initially occupies the leftmost 1 m of the tank, up to y = 2 m.
    # The refinement rectangle covers the lower part of the downstream wall.
    dx              = 0.01
    simulation_time = 1.0
    tank_width      = 4.0
    tank_height     = 3.0
    water_width     = 1.0
    water_height    = 2.0
    wall_layers     = 5
    refinement      = VariableResolutionBox((2.5, -0.2), (4.1, 0.6); ratio = 2)
    fine_dx         = dx / refinement.ratio

    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx  = dx,
        c₀  = 10sqrt(water_height * 9.81),
        γ   = 7,
        α   = 0.05,
        δᵩ  = 0.1,
        CFL = 0.2,
    )

    # Each domain has its own kernel and particle mass, with h = 1.5 dx.
    SimKernel  = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(1.5dx), η² = FloatType((0.1 * 1.5dx)^2))
    FineKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(1.5fine_dx), η² = FloatType((0.1 * 1.5fine_dx)^2))

    function domain_geometry(spacing, constants, kernel; fine = false)
        nx, ny = round(Int, tank_width / spacing), round(Int, tank_height / spacing)

        # Cell-centred particles share one lattice, including the wall corners.
        # Five wall layers lie outside the physical tank surfaces.
        wall = [SVector((i + 0.5) * spacing, (j + 0.5) * spacing)
                for j in -wall_layers:(ny - 1) for i in -wall_layers:(nx + wall_layers - 1)
                if j < 0 || i < 0 || i >= nx]
        water = [SVector((i + 0.5) * spacing, (j + 0.5) * spacing)
                 for j in 0:(round(Int, water_height / spacing) - 1) for i in 0:(round(Int, water_width / spacing) - 1)]

        if fine
            # Start with the patch's fluid buffer and enough physical wall
            # support around it. Incoming water creates fine particles later.
            margin = refinement.buffer_h * kernel.h
            lower, upper = refinement.lower .- margin, refinement.upper .+ margin
            filter!(x -> all(lower .- kernel.H .<= x) && all(x .<= upper .+ kernel.H), wall)
            filter!(x -> all(lower .<= x) && all(x .<= upper), water)
        end

        # mDBC ghost points mirror the walls through x = 0, x = 4 and y = 0.
        ghosts = [SVector(x[1] < 0 ? -x[1] : x[1] > tank_width ? 2tank_width - x[1] : x[1],
                          x[2] < 0 ? -x[2] : x[2]) for x in wall]

        Tank = SPHGeometry{Dimensions, FloatType}(;
            Particles   = particle_struct_array(wall, constants.ρ₀;
                GhostPoints = ghosts, GhostNormals = ghosts .- wall),
            GroupMarker = 1,
            Type        = Fixed,
        )

        Water = SPHGeometry{Dimensions, FloatType}(water;
            Density     = hydrostatic_density(water, constants; water_level = water_height),
            GroupMarker = 2,
            Type        = Fluid,
        )

        return [Tank, Water]
    end

    SimulationGeometry = domain_geometry(dx, SimConstants, SimKernel)
    FineGeometry       = domain_geometry(fine_dx, SimConstants, FineKernel; fine = true)

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        SimpleMDBC,
        StoreLog,
    }(
        SimulationName      = "DamBreak2D_VRes",
        SaveLocation        = joinpath(@__DIR__, "..", "output", "vresolution_dambreak2d"),
        SimulationTime      = simulation_time,
        OutputTimes         = 0.0025,
        VisualizeInParaview = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = true,
        OpenLogFile         = true,
        GPUDoublePosition   = true,
        GPUClipTimeStepToOutput = true,
        VariableResolution  = VariableResolutionConfig(refinement;
            fine_geometry  = FineGeometry,
            # Material properties are shared; fine spacing and mass are derived.
            fine_kernel    = FineKernel,
        ),
        OutputVariables = ["Velocity", "Density", "Pressure", "Type", "ID", "Resolution", "Spacing", "Mass", "Owned", "IsBuffer"],
    )

    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)

    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    CleanUpSimulationFolder(SimMetaData.SaveLocation)
    tank  = polyline([(0.0, tank_height), (0.0, 0.0), (tank_width, 0.0), (tank_width, tank_height)]; thickness = wall_layers * dx, side = :right)
    water = rectangle((0.0, 0.0), water_width, water_height)
    patch = rectangle(Tuple(refinement.lower), refinement.upper[1] - refinement.lower[1], refinement.upper[2] - refinement.lower[2])
    SavePolygonVTKHDF(joinpath(SimMetaData.SaveLocation, "DamBreak2D_VRes_Geometry.vtkhdf"), (; tank, water, refinement = patch))

    try
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
        )
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
