using SPHExample
using FastPow
using LinearAlgebra
using Parameters


 Dimensions = 2
 FloatType  = Float64
 SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.5)
#  SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=43.4, δᵩ = 0.1, CFL=0.2)

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
        SimulationTime=4,
        OutputEach=0.01,
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

SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation; to_console=true)

CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

# This example demonstrates building a neighbor list using a compressed
# representation inspired by the "Compressed Neighbor Lists" approach.
# The implementation is self contained and does not rely on other files
# in this repository.

using StaticArrays, Random
using Base.Threads

"""
    build_neighbor_list(positions, cutoff)

Construct a neighbor list for the given particle positions.
Particles within `cutoff` distance are considered neighbors.
Returns a vector of integer vectors containing the neighbor indices
for each particle.
"""
function build_neighbor_list(positions::Vector{SVector{D,Float64}}, cutoff) where D
    inv_cutoff = 1 / cutoff
    grid = Dict{NTuple{D,Int}, Vector{Int}}()
    for (i, p) in pairs(positions)
        cell = ntuple(j -> floor(Int, p[j] * inv_cutoff), D)
        push!(get!(grid, cell, Int[]), i)
    end
    stencil = collect(CartesianIndices(ntuple(_ -> -1:1, D)))
    neighbor_lists = [Int[] for _ in positions]
    cutoff2 = cutoff^2
    for (i, p) in pairs(positions)
        cell = ntuple(j -> floor(Int, p[j] * inv_cutoff), D)
        for offset in stencil
            neigh_cell = ntuple(j -> cell[j] + offset[j], D)
            for j in get(grid, neigh_cell, Int[])
                j == i && continue
                dist2 = sum((positions[j][k] - p[k])^2 for k in 1:D)
                dist2 <= cutoff2 && push!(neighbor_lists[i], j)
            end
        end
    end
    return neighbor_lists
end

"""
    compress_neighbors(neighbor_lists)

Compress a vector of neighbor lists using delta encoding.
The neighbors for each particle are sorted and stored as differences
between consecutive indices.
"""
function compress_neighbors(neighbor_lists::Vector{Vector{Int}})
    compressed = Vector{Vector{Int}}(undef, length(neighbor_lists))
    for (i, nbrs) in pairs(neighbor_lists)
        sort!(nbrs)
        diffs = Vector{Int}(undef, length(nbrs))
        prev = 0
        for (k, idx) in enumerate(nbrs)
            diffs[k] = idx - prev
            prev = idx
        end
        compressed[i] = diffs
    end
    return compressed
end

"""
    decompress_neighbors(compressed)

Reconstruct the neighbor lists from a delta encoded representation.
"""
function decompress_neighbors(compressed::Vector{Vector{Int}})
    neighbor_lists = Vector{Vector{Int}}(undef, length(compressed))
    for (i, diffs) in pairs(compressed)
        nbrs = Vector{Int}(undef, length(diffs))
        prev = 0
        for (k, diff) in enumerate(diffs)
            nbrs[k] = prev + diff
            prev = nbrs[k]
        end
        neighbor_lists[i] = nbrs
    end
    return neighbor_lists
end

function main()
    positions = SimParticles.Position
    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)
    SimParticles.Kernel .= 0.0  # Initialize kernel values to zero
    nbrs = build_neighbor_list(positions, SimKernel.H)
    # compressed = compress_neighbors(copy.(nbrs))
    # recovered = decompress_neighbors(compressed)
    # @assert recovered == sort.(nbrs)
    # println("constructed $(n) particles with compressed neighbor lists")
    

    # println(recovered[1:5])  # Print first 5 neighbor lists for verification

    Position       = SimParticles.Position
    Density        = SimParticles.Density
    Pressure       = SimParticles.Pressure
    Velocity       = SimParticles.Velocity
    Acceleration   = SimParticles.Acceleration
    MotionLimiter  = SimParticles.MotionLimiter
    ParticleType   = SimParticles.Type
    ParticleMarker = SimParticles.GroupMarker
    Kernel         = SimParticles.Kernel
    KernelGradient = SimParticles.KernelGradient
    GhostPoints    = SimParticles.GhostPoints
    GhostNormals   = SimParticles.GhostNormals
    GravityFactor  = SimParticles.GravityFactor

    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    @unpack FlagOutputKernelValues = SimMetaDataWedge
    @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstantsWedge
    @unpack h⁻¹, h, η², H², αD = SimKernel 

    @b begin 
        dt  = Δt(Position, Velocity, Acceleration, SimConstantsWedge, SimKernel)
        dt₂ = dt * 0.5

        @batch per=thread for i in eachindex(nbrs)
            Accelerationₙ⁺   =  ConstructGravitySVector(Acceleration[i], SimConstantsWedge.g * GravityFactor[i])
            Positionₙ⁺       =  Position[i]  + Velocity[i]   * dt₂  * MotionLimiter[i]
            Velocityₙ⁺       =  Velocity[i]
            ρₙ⁺              =  Density[i]

            for j in nbrs[i]
                xᵢⱼ  = SimParticles.Position[i] - SimParticles.Position[j]
                xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
                #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
                dᵢⱼ  = sqrt(abs(xᵢⱼ²))

                # clamp seems faster than min, no util
                q    = clamp(dᵢⱼ * SimKernel.h⁻¹, 0.0, 2.0) #min(dᵢⱼ * h⁻¹, 2.0) - 8% util no DDT
                # Wᵢⱼ  = @fastpow SPHExample.SPHKernels.Wᵢⱼ(SimKernel, q)
                # SimParticles.Kernel[i] += Wᵢⱼ

                ∇ᵢWᵢⱼ     = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                ρᵢ        = Density[i]
                ρⱼ        = Density[j]
            
                vᵢ        = Velocity[i]
                vⱼ        = Velocity[j]
                vᵢⱼ       = vᵢ - vⱼ
                density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
                dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

                Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstantsWedge, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, i, j, MotionLimiter)

                DensityDerivativeᵢ = dρdt⁺ + Dᵢ

                Pᵢ      =  Pressure[i]
                Pⱼ      =  Pressure[j]
                Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
                f_ab    = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
                dvdt⁺   = - m₀ * (Pfac + f_ab) *  ∇ᵢWᵢⱼ

                visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstantsWedge, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)

                uₘ = dvdt⁺ + visc_term

                Accelerationₙ⁺ += uₘ
                # Positionₙ⁺     +=
                Velocityₙ⁺     += uₘ * dt₂ * MotionLimiter[i]
                ρₙ⁺            += DensityDerivativeᵢ
            end

            for j in nbrs[i]
                xᵢⱼ  = SimParticles.Position[i] - SimParticles.Position[j]
                xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
                #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
                dᵢⱼ  = sqrt(abs(xᵢⱼ²))

                # clamp seems faster than min, no util
                q    = clamp(dᵢⱼ * SimKernel.h⁻¹, 0.0, 2.0) #min(dᵢⱼ * h⁻¹, 2.0) - 8% util no DDT
                # Wᵢⱼ  = @fastpow SPHExample.SPHKernels.Wᵢⱼ(SimKernel, q)
                # SimParticles.Kernel[i] += Wᵢⱼ

                ∇ᵢWᵢⱼ     = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                ρᵢ        = Density[i]
                ρⱼ        = Density[j]
            
                vᵢ        = Velocity[i]
                vⱼ        = Velocity[j]
                vᵢⱼ       = vᵢ - vⱼ
                density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
                dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

                Dᵢ, Dⱼ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstantsWedge, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, i, j, MotionLimiter)

                # SimThreadedArrays.dρdtIThreaded[ichunk][i] += dρdt⁺ + Dᵢ
                # SimThreadedArrays.dρdtIThreaded[ichunk][j] += dρdt⁻ + Dⱼ

                # DensityDerivativeᵢ += dρdt⁺ + Dᵢ

                Pᵢ      =  Pressure[i]
                Pⱼ      =  Pressure[j]
                Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
                f_ab    = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
                dvdt⁺   = - m₀ * (Pfac + f_ab) *  ∇ᵢWᵢⱼ

                visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstantsWedge, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)

                uₘ = dvdt⁺ + visc_term
                # MomentumDerivativeᵢ += uₘ
                # SimThreadedArrays.AccelerationThreaded[ichunk][i] += uₘ
                # SimThreadedArrays.AccelerationThreaded[ichunk][j] -= uₘ 
            end
            # SimParticles.Density[i] = DensityDerivativeᵢ
            # SimParticles.Acceleration[i] = MomentumDerivativeᵢ
        end
    end

    # return nbrs
end

nbrs = main()

# using Plots

# positions = SimParticles.Position
# x = [p[1] for p in positions]
# y = [p[2] for p in positions]
# kernel_vals = SimParticles.Kernel

# scatter(x, y; marker_z=kernel_vals, color=:viridis, xlabel="x", ylabel="y",
#     title="Particle Positions Colored by Kernel Value", legend=false,
#     aspect_ratio=:equal, size=(600, 500))

# @b @main()