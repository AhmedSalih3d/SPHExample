using SPHExample
using PointNeighbors
using CUDA, Adapt
using FastPow
using Parameters
using LinearAlgebra
using Chairmarks

Dimensions = 2
FloatType  = Float64
SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.5)

# Assuming SimConstantsWedge is defined somewhere else with the field `dx`
FixedBoundary = Geometry{Dimensions, FloatType}(
    CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
    GroupMarker = 1,
    Type        = Fixed,   # Using the enum value Fixed
    Motion      = nothing
)
Water = Geometry{Dimensions, FloatType}(
    CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
    GroupMarker = 2,
    Type        = Fluid,   # Using the enum value Fluid
    Motion      = nothing
)

SimulationGeometry = [FixedBoundary;Water]

# Load in particles
SimParticles = AllocateDataStructures(SimulationGeometry)
SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
    SimulationName="StillWedge", 
    SaveLocation="E:/SecondApproach/StillWedge2D_MDBC",
    SimulationTime=4.0,
    OutputTimes=0.01,
    VisualizeInParaview=true,
    ExportSingleVTKHDF=true,
    ExportGridCells=true,
    OpenLogFile=true,
    FlagOutputKernelValues=false,
    FlagLog=true,
    FlagMDBCSimple=true,
    # OutputVariables = [
    #     # "ChunkID",
    #     # "Kernel",
    #     # "KernelGradient",
    #     "Density",
    #     "Pressure",
    #     "Velocity",
    #     "Acceleration",
    #     # "BoundaryBool",
    #     # "ID",
    #     # "Type",
    #     # "GroupMarker",
    #     # "GhostPoints",
    #     # "GhostNormals",
    # ]
)

# SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation; to_console=true)
# CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)
SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)
SimDensityDiffusion = SPHExample.SPHDensityDiffusionModels.LinearDensityDiffusion()
SimViscosity        = SPHExample.SPHViscosityModels.ArtificialViscosity()

# Generate grid of particles
coordinates = reshape(reinterpret(eltype(eltype(SimParticles.Position)), SimParticles.Position), 2, :)
n_particles = size(coordinates, 2)

# `FullGridCellList` requires a bounding box
min_corner = minimum(coordinates, dims=2)
max_corner = maximum(coordinates, dims=2)
search_radius = SimKernel.H
cell_list = FullGridCellList(; search_radius, min_corner, max_corner)
nhs = GridNeighborhoodSearch{2}(; search_radius, cell_list)

# Initialize the NHS to find neighbors in `coordinates` of particles in `coordinates`
initialize!(nhs, coordinates, coordinates)

dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimParticles.Position)

Accelerationₙ⁺ = zero(Velocityₙ⁺)

# Use a function for performance reasons
function count_neighbors!(SimParticles, coordinates, nhs, SimKernel, SimConstants, dρdtI, Accelerationₙ⁺, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺)
    @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    @unpack h⁻¹, h, η², H², αD = SimKernel

    @unpack Position, Density, Pressure, Velocity, Acceleration, MotionLimiter, Type, GroupMarker, Kernel, KernelGradient, GhostPoints, GhostNormals, GravityFactor = SimParticles

    dt  = Δt(Position, Velocity, Acceleration, SimConstants, SimKernel)
    dt₂ = dt * 0.5

    Kernel         .= 0.0
    KernelGradient .= zero(KernelGradient)
    @. Acceleration   = ConstructGravitySVector(Acceleration, SimConstants.g * GravityFactor)

    @b foreach_point_neighbor(coordinates, coordinates, nhs) do i, j, xᵢⱼ, distance
        q     = distance / SimKernel.H

        Wᵢⱼ   = @fastpow SPHExample.SPHKernels.Wᵢⱼ(SimKernel, q)

        ∇ᵢWᵢⱼ = @fastpow SPHExample.SPHKernels.∇Wᵢⱼ(SimKernel, q, xᵢⱼ)


        Kernel[i]          += Wᵢⱼ
        KernelGradient[i]  += ∇ᵢWᵢⱼ

        
        ρᵢ        = Density[i]
        ρⱼ        = Density[j]
    
        vᵢ        = Velocity[i]
        vⱼ        = Velocity[j]
        vᵢⱼ       = vᵢ - vⱼ
        density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
        dρdt⁺  = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
        Dᵢ, _  = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, i, j, MotionLimiter)

        dρdtI[i] += dρdt⁺ + Dᵢ

        Pᵢ      =  Pressure[i]
        Pⱼ      =  Pressure[j]
        Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
        f_ab    = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
        dvdt⁺   = - m₀ * (Pfac + f_ab) *  ∇ᵢWᵢⱼ
        visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)
        uₘ = dvdt⁺ + visc_term

        

    end
end

CUDA.@profile count_neighbors!(SimParticles, coordinates, nhs, SimKernel, SimConstantsWedge, dρdtI, Accelerationₙ⁺, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺)


