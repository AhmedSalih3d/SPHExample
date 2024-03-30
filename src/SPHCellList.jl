module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters, Polyester
import LinearAlgebra: dot

using ..SimulationEquations
using ..AuxillaryFunctions

    function ConstructStencil(v::Val{d}) where d
        n_ = CartesianIndices(ntuple(_->-1:1,v))
        half_length = length(n_) ÷ 2
        n  = n_[1:half_length]

        return n
    end

    @inline function ExtractCells!(Particles, CutOff)
        Cells  = @views Particles.Cells
        Points = @views Particles.Position
        # Now it seems more feasible to do @batch, since floating division being used now
        @batch per=thread for i ∈ eachindex(Particles)
            # Cells[i]  =  CartesianIndex(@. Int(fld(Points[i] + 2, CutOff)) ...)
            # Cells[i] +=  2 * one(Cells[i])  # + CartesianIndex(1,1) + CartesianIndex(1,1) #+ ZeroOffset + HalfPad
            Cells[i]   = CartesianIndex(@. floor(Int, 4 + Points[i] / CutOff)...)
        end
        return nothing
    end

    ###=== Function to update ordering
    function UpdateNeighbors!(Particles, CutOff, SortingScratchSpace, ParticleRanges, UniqueCells)
        ExtractCells!(Particles, CutOff)

        sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)

        Cells = @views Particles.Cells
        @. ParticleRanges             = zero(eltype(ParticleRanges))
        IndexCounter                  = 1
        ParticleRanges[IndexCounter]  = 1
        UniqueCells[IndexCounter]     = Cells[1]

        for i in 2:length(Cells)
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                IndexCounter                += 1
                ParticleRanges[IndexCounter] = i
                UniqueCells[IndexCounter]    = Cells[i]
            end
        end
        ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

        return IndexCounter 
    end

# Neither Polyester.@batch per core or thread is faster
###=== Function to process each cell and its neighbors
    function NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI,  MotionLimiter, UniqueCells, IndexCounter)
        UniqueCells = view(UniqueCells, 1:IndexCounter)
        @threads for (ichunk, inds) in enumerate(chunks(UniqueCells; n=nthreads()))
            for iter in inds
                CellIndex = UniqueCells[iter]

                StartIndex = ParticleRanges[iter] 
                EndIndex   = ParticleRanges[iter+1] - 1

                @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
                    @inline ComputeInteractions!(SimMetaData, SimConstants, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
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
                            @inline ComputeInteractions!(SimMetaData, SimConstants, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
                        end
                    end
                end
            end
        end

        return nothing
    end

end
