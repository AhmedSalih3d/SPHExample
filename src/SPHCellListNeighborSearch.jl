module SPHCellListNeighborSearch

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!,
       NeighborLoopMDBC!

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
using ..SimulationEquations
using ..SimulationGeometry
using ..AuxiliaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..OpenExternalPrograms
using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SPHCellListComputation

import StructArrays: StructArray, foreachfield
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using UnicodePlots
using LinearAlgebra
using Bumper

function ConstructStencil(v::Val{d}) where d
    n_ = CartesianIndices(ntuple(_->-1:1,v))
    half_length = length(n_) ÷ 2
    n  = n_[1:half_length]

    return n
end

"""
Extracts the cells for each particle based on their positions and the inverse cutoff value.

# Arguments
- `Particles`: The particles whose cells are to be extracted.
- `::Val{InverseCutOff}`: The inverse cutoff value used for cell extraction.

# Returns
- `nothing`: This function modifies the `Particles` in place.
"""
# Replace unsafe_trunc with trunc if this ever errors
@inline function map_floor(x, InverseCutOff)
    # This is different than just doing muladd(x,InverseCutOff,0.5) because it rounds towards zero.
    # Consider -1.7 + 0.5, this would give -1.2 and then trunced 1, but we want -2, therefore absolute addition before hand
    # We add 0.5 instead of 1, to ensure proper rounding behavior when restoring the sign for negative numbers.
    Int(sign(x)) * unsafe_trunc(Int, muladd(abs(x),InverseCutOff,0.5))
end
   
@inline function ExtractCells!(Particles, InverseCutOff)
    @inbounds @simd ivdep for i ∈ eachindex(Particles.Cells)
        # t = map(map_floor, Tuple(Particles.Position[i]))
        t = CartesianIndex(map(x -> map_floor(x, InverseCutOff), Tuple(Particles.Position[i])))
        Particles.Cells[i] = CartesianIndex(t)
    end

    return nothing
end

"""
Updates the neighbor list and sorts particles by their cell indices.

# Arguments
- `Particles`: The particles whose neighbors are to be updated.
- `CutOff`: The cutoff value used for cell extraction.
- `SortingScratchSpace`: Scratch space for sorting.
- `ParticleRanges`: Array to store the ranges of particles in each cell.
- `UniqueCells`: Array to store the unique cells.

# Returns
- `IndexCounter`: The number of unique cells identified.
"""
function UpdateNeighbors!(Particles, InverseCutOff, SortingScratchSpace, ParticleRanges, UniqueCells)
    ExtractCells!(Particles, InverseCutOff)

    sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)
    Cells = @views Particles.Cells
    @. ParticleRanges             = zero(eltype(ParticleRanges))
    IndexCounter                  = 1
    ParticleRanges[IndexCounter]  = 1
    UniqueCells[IndexCounter]     = Cells[1]

    @inbounds @simd ivdep for i in eachindex(Cells)[2:end]
        if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
            IndexCounter                 += 1
            ParticleRanges[IndexCounter]  = i
            UniqueCells[IndexCounter]     = Cells[i]
        end
    end
    ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

    return IndexCounter 
end


# Neither Polyester.@batch per core or thread is faster
###=== Function to process each cell and its neighbors
function NeighborLoop!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, ParticleRanges, Stencil, Position, Density, Pressure, Velocity, MotionLimiter, UniqueCells, EnumeratedIndices)
    @sync begin
        for (ichunk, inds) ∈ EnumeratedIndices 
            @spawn for iter ∈ inds

                CellIndex = UniqueCells[iter]
                SimParticles.ChunkID[iter] = ichunk

                StartIndex = ParticleRanges[iter]
                EndIndex   = ParticleRanges[iter+1] - 1

                @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
                    @inline ComputeInteractions!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
                end

                @inbounds for S ∈ Stencil
                    SCellIndex = CellIndex + S

                    # Returns a range, x:x for exact match and x:(x-1) for no match
                    # utilizes that it is a sorted array and requires no isequal constructor,
                    # so I prefer this for now
                    NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

                    if length(NeighborCellIndex) != 0
                        StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                        EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                        @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
                            @inline ComputeInteractions!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

function NeighborLoopMDBC!(SimKernel, SimMetaData::SimulationMetaData{Dimensions, _}, SimConstants, ParticleRanges, Position, Density, UniqueCells, GhostPoints, GhostNormals,  ParticleType, bᵧ, Aᵧ) where {Dimensions, _}
    
    FullStencil = CartesianIndices(ntuple(_->-1:1, Dimensions))

    @inbounds @threads for iter in eachindex(GhostPoints)
        GhostPoint = GhostPoints[iter]

        if !iszero(GhostPoint)
            # zero‐initialize per‐ghost accumulators
            b_acc = zero(bᵧ[iter])            # an SVector{D+1,FloatType}
            A_acc = zero(Aᵧ[iter])            # an SMatrix{D+1,D+1,FloatType}
        
            # compute and accumulate into the locals
            GhostCellIndex = CartesianIndex(map(x->map_floor(x,SimKernel.H⁻¹), Tuple(GhostPoints[iter])))
            @inbounds for S ∈ FullStencil
                SCellIndex = GhostCellIndex + S
                # Returns a range, x:x for exact match and x:(x-1) for no match
                # utilizes that it is a sorted array and requires no isequal constructor,
                # so I prefer this for now
                NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

                if length(NeighborCellIndex) != 0
                    StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                    EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                    for j in StartIndex_:EndIndex_
                        # change ComputeInteractions to take & return contributions, e.g.:
                        bΔ, AΔ = ComputeInteractionsMDBC!(SimKernel, SimMetaData, SimConstants,
                                                        Position, Density, ParticleType,
                                                        GhostPoints, iter, j)
                        b_acc += bΔ
                        A_acc += AΔ
                    end
                end
            end
        
            # write out once
            bᵧ[iter] = b_acc
            Aᵧ[iter] = A_acc
        end    
    end

    return nothing
end
end
