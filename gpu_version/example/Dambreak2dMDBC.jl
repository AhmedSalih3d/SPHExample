# GPU version of example/Dambreak2dMDBC.jl. Requires an NVIDIA GPU with CUDA.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/Dambreak2dMDBC.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
using SPHExampleGPU
using StructArrays
using StaticArrays: SVector
include(joinpath(@__DIR__, "GenerateDamBreak2DMDBC.jl"))

let
    Dimensions = 2
    FloatType = Float32

    SimConstantsDambreak = SimulationConstants{FloatType}(
        dx = 0.01,
        c₀ = 88.14487860902641,
        δᵩ = 0.1,
        CFL = 0.5,
        α = 0.01,
    )

    # Create SPHGeometry instances
    # Sample particles here; the generator script supplies shape definitions.
    polygons = dam_break_2d_polygons()
    regions = [
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, SimConstantsDambreak.dx)
    positions(name) = only(r.positions for r in sampled if r.name == name)
    water_level = maximum(last, positions("Fluid"))

    wall_positions = positions("Bound")
    dx = SimConstantsDambreak.dx
    mirror_x(x) = x < dx / 2 ? dx - x : x > 4 - dx / 2 ? 8 - dx - x : x
    ghosts =
        [SVector(mirror_x(x[1]), x[2] < dx / 2 ? dx - x[2] : x[2]) for x in wall_positions]

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        Particles = StructArray((
            Position = positions("Bound"),
            Density = fill(SimConstantsDambreak.ρ₀, length(positions("Bound"))),
            GhostPoints = ghosts,
            GhostNormals = ghosts .- wall_positions,
        )),
        # CSVFile     = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_Bound_ThreeLayers.csv",
        GroupMarker = 1,
        Type = Fixed,   # Using the enum value Fixed
        Motion = nothing,
    )

    Water = SPHGeometry{Dimensions, FloatType}(
        Particles = StructArray((
            Position = positions("Fluid"),
            Density = hydrostatic_density(
                positions("Fluid"),
                SimConstantsDambreak;
                water_level,
            ),
        )),
        # CSVFile     = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_Fluid_ThreeLayers.csv",
        GroupMarker = 2,
        Type = Fluid,   # Using the enum value Fluid
        Motion = nothing,
    )

    # Collect the SPHGeometry instances into a vector
    SimulationGeometry = [FixedBoundary; Water]

    SimMetaDataDambreak = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        SimpleMDBC,
        StoreLog,
    }(
        SimulationName = "DamBreak2D",
        SaveLocation = "C:/TestSimulations/DamBreak2D_MDBC_GPU/",
        SimulationTime = 2,
        OutputTimes = collect(0.01:0.01:2),
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OpenLogFile = true,
    )

    # If save directory is not already made, make it
    if !isdir(SimMetaDataDambreak.SaveLocation)
        mkpath(SimMetaDataDambreak.SaveLocation)
    end

    # How to overload and define your own viscosity model:
    # Artificial viscosity formulation.
    # using Parameters
    # using LinearAlgebra
    # struct MyTurbulenceModel <: SPHViscosity end
    # The densities of the evaluated state and their reciprocals are passed in
    # (ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹); do not read them from SimParticles.
    # @inline function SPHExampleGPU.compute_viscosity(::MyTurbulenceModel, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)
    #     @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    #     @unpack h, η² = SimKernel

    #     invd²η²   =  1.0 / (d² + η²)
    #     ρ̄ = (ρᵢ + ρⱼ) * 0.5
    #     cond = dot(vᵢⱼ, xᵢⱼ)
    #     flag = cond < 0 ? one(eltype(cond)) : zero(eltype(cond))
    #     μ = h * cond * invd²η²
    #     Π = -m₀ * (flag * (-α * c₀ * μ) / ρ̄) * ∇ᵢWᵢⱼ
    #     return 0*Π, -Π*0
    # end

    # Load in particles
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataDambreak)

    SimLogger = SimulationLogger(SimMetaDataDambreak.SaveLocation; to_console = true)

    CleanUpSimulationFolder(SimMetaDataDambreak.SaveLocation)

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataDambreak,
        SimConstants = SimConstantsDambreak,
        SimKernel = SPHKernelInstance{Dimensions, FloatType}(
            WendlandC2();
            dx = SimConstantsDambreak.dx,
        ),
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        # ParticleNormalsPath  = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_GhostNodes_ThreeLayers.csv"
    )
end
