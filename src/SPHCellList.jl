module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!, RunSimulation

using Parameters, FastPow, StaticArrays, Base.Threads
import LinearAlgebra: dot

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

using StaticArrays
import StructArrays: StructArray, foreachfield
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using Base.Threads
using UnicodePlots
using LinearAlgebra
    using Bumper

    function ConstructFullStencil(v::Val{d}) where d
        return CartesianIndices(ntuple(_->-1:1, v))
    end

    const MAX_DENSE_CELLS = 2_000_000

    mutable struct CellIndexLookup{D}
        DictMap::Dict{CartesianIndex{D}, Int}
        DenseMap::Vector{Int}
        Offset::NTuple{D, Int}
        Dims::NTuple{D, Int}
        UseDense::Bool
    end

    function CellIndexLookup{D}() where D
        return CellIndexLookup{D}(Dict{CartesianIndex{D}, Int}(),
                                  Int[],
                                  ntuple(_ -> 0, D),
                                  ntuple(_ -> 0, D),
                                  false)
    end

    @inline function CellToLinearIndex(Cell::CartesianIndex{D},
                                       Offset::NTuple{D, Int},
                                       Dims::NTuple{D, Int}) where D
        Idx = 1
        Stride = 1
        @inbounds for Dim in 1:D
            LocalIdx = Cell[Dim] - Offset[Dim] + 1
            if LocalIdx < 1 || LocalIdx > Dims[Dim]
                return 0
            end
            Idx += (LocalIdx - 1) * Stride
            Stride *= Dims[Dim]
        end
        return Idx
    end

    @inline function LookupCellIndex(CellLookup::CellIndexLookup{D},
                                     Cell::CartesianIndex{D}) where D
        if CellLookup.UseDense
            LinearIndex = CellToLinearIndex(Cell, CellLookup.Offset, CellLookup.Dims)
            if LinearIndex == 0
                return 1
            end
            return CellLookup.DenseMap[LinearIndex]
        end
        return get(CellLookup.DictMap, Cell, 1)
    end

    function UpdateCellIndexLookup!(CellLookup::CellIndexLookup{D},
                                    UniqueCellsView,
                                    CellCount::Int) where D
        RealCells = max(CellCount - 1, 0)
        if RealCells == 0
            CellLookup.UseDense = false
            empty!(CellLookup.DenseMap)
            return nothing
        end

        MinIdx = fill(typemax(Int), D)
        MaxIdx = fill(typemin(Int), D)
        @inbounds for CellIndex in 2:CellCount
            Cell = UniqueCellsView[CellIndex]
            for Dim in 1:D
                Value = Cell[Dim]
                if Value < MinIdx[Dim]
                    MinIdx[Dim] = Value
                end
                if Value > MaxIdx[Dim]
                    MaxIdx[Dim] = Value
                end
            end
        end

        Dims = ntuple(Dim -> MaxIdx[Dim] - MinIdx[Dim] + 1, D)
        TotalCells = prod(Dims)
        UseDenseMap = TotalCells <= MAX_DENSE_CELLS && TotalCells <= 8 * RealCells

        if UseDenseMap
            CellLookup.DenseMap = fill(1, TotalCells)
            CellLookup.Offset = ntuple(Dim -> MinIdx[Dim], D)
            CellLookup.Dims = Dims
            CellLookup.UseDense = true
            @inbounds for (Cell, CellIndex) in CellLookup.DictMap
                LinearIndex = CellToLinearIndex(Cell, CellLookup.Offset, CellLookup.Dims)
                if LinearIndex != 0
                    CellLookup.DenseMap[LinearIndex] = CellIndex
                end
            end
        else
            CellLookup.UseDense = false
            empty!(CellLookup.DenseMap)
        end

        return nothing
    end

    function BuildNeighborCellLists!(NeighborCellOffsets, NeighborCellIndices,
                                     FullStencil, UniqueCellsView,
                                     ParticleRanges, CellLookup)
        CellCount = length(UniqueCellsView)
        resize!(NeighborCellOffsets, CellCount + 1)
        NeighborCellOffsets[1] = 1
        if CellCount >= 1
            NeighborCellOffsets[2] = 1
        end

        TotalNeighbors = 0
        @inbounds for CellIndex in 2:CellCount
            Cell = UniqueCellsView[CellIndex]
            for Offset in FullStencil
                NeighborCell = Cell + Offset
                NeighborIndex = LookupCellIndex(CellLookup, NeighborCell)
                StartIndex = ParticleRanges[NeighborIndex]
                EndIndex = ParticleRanges[NeighborIndex + 1] - 1
                if StartIndex <= EndIndex && NeighborIndex != CellIndex
                    TotalNeighbors += 1
                end
            end
            NeighborCellOffsets[CellIndex + 1] = TotalNeighbors + 1
        end

        resize!(NeighborCellIndices, TotalNeighbors)
        Current = 1
        @inbounds for CellIndex in 2:CellCount
            Cell = UniqueCellsView[CellIndex]
            for Offset in FullStencil
                NeighborCell = Cell + Offset
                NeighborIndex = LookupCellIndex(CellLookup, NeighborCell)
                StartIndex = ParticleRanges[NeighborIndex]
                EndIndex = ParticleRanges[NeighborIndex + 1] - 1
                if StartIndex <= EndIndex && NeighborIndex != CellIndex
                    NeighborCellIndices[Current] = NeighborIndex
                    Current += 1
                end
            end
        end

        return nothing
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



    @inline function KernelOutputLocal!(::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                        kernel_acc, kernel_grad_acc, SimKernel,
                                        q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                        B<:MDBCMode,
                                                        L<:LogMode}
        return kernel_acc, kernel_grad_acc
    end

    @inline function KernelOutputLocal!(::SimulationMetaData{D,T,S,StoreKernelOutput,B,L},
                                        kernel_acc, kernel_grad_acc, SimKernel,
                                        q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                        B<:MDBCMode,
                                                        L<:LogMode}
        Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
        return kernel_acc + Wᵢⱼ, kernel_grad_acc + ∇ᵢWᵢⱼ
    end
   
    @inline function CellFromPosition(Position::SVector{D, T}, InverseCutOff) where {D, T}
        return CartesianIndex(ntuple(Dimension -> map_floor(Position[Dimension], InverseCutOff), Val(D)))
    end

    @inline function ExtractCells!(Particles, InverseCutOff)
        @inbounds @simd ivdep for i ∈ eachindex(Particles.Cells)
            Particles.Cells[i] = CellFromPosition(Particles.Position[i], InverseCutOff)
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
    function UpdateNeighbors!(Particles, InverseCutOff, SortingScratchSpace,
                              ParticleRanges, UniqueCells, CellLookup,
                              CellListIndex)
        ExtractCells!(Particles, InverseCutOff)

        sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)
        Cells = @views Particles.Cells
        @. ParticleRanges             = zero(eltype(ParticleRanges))
        ParticleRanges[1] = 1
        IndexCounter                  = 2
        ParticleRanges[IndexCounter]  = 1
        UniqueCells[IndexCounter]     = Cells[1]
        CellDict = CellLookup.DictMap
        empty!(CellDict)
        CellDict[Cells[1]] = IndexCounter

        @inbounds @simd ivdep for i in eachindex(Cells)[2:end]
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                IndexCounter                 += 1
                ParticleRanges[IndexCounter]  = i
                UniqueCells[IndexCounter]     = Cells[i]
                CellDict[Cells[i]]           = IndexCounter
            end
        end
        ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

        UpdateCellIndexLookup!(CellLookup, view(UniqueCells, 1:IndexCounter), IndexCounter)
        FillCellListIndex!(CellListIndex, ParticleRanges, IndexCounter)

        return IndexCounter 
    end

    function FillCellListIndex!(CellListIndex, ParticleRanges, CellCount)
        @inbounds for CellIndex in 2:CellCount
            StartIndex = ParticleRanges[CellIndex]
            EndIndex = ParticleRanges[CellIndex + 1] - 1
            for i in StartIndex:EndIndex
                CellListIndex[i] = CellIndex
            end
        end
        return nothing
    end

    function compute_cell_particle_counts(particle_ranges, n_cells)
        counts = Vector{Int}(undef, n_cells)
        @inbounds for i in 1:n_cells
            counts[i] = particle_ranges[i + 1] - particle_ranges[i]
        end
        return counts
    end

    function compute_cell_neighbor_counts(particle_ranges, neighbor_cell_offsets,
                                          neighbor_cell_indices, n_cells)
        counts = compute_cell_particle_counts(particle_ranges, n_cells)
        neighbors = Vector{Int}(undef, n_cells)
        @inbounds for i in 1:n_cells
            NeighborTotal = 0
            StartIndex = neighbor_cell_offsets[i]
            EndIndex = neighbor_cell_offsets[i + 1] - 1
            for NeighborOffsetIndex in StartIndex:EndIndex
                NeighborIndex = neighbor_cell_indices[NeighborOffsetIndex]
                NeighborTotal += counts[NeighborIndex]
            end
            neighbors[i] = max(counts[i] - 1, 0) + NeighborTotal
        end
        return neighbors
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndex, NeighborCellOffsets,
                                      NeighborCellIndices, Position, Density, Pressure,
                                      Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ, max_visc = nothing,
                                      min_dt_force = nothing) where {D,T,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            CellListIndexLocal = CellListIndex[i]
            SameCellStart = ParticleRanges[CellListIndexLocal]
            SameCellEnd = ParticleRanges[CellListIndexLocal + 1] - 1
            NeighborStart = NeighborCellOffsets[CellListIndexLocal]
            NeighborEnd = NeighborCellOffsets[CellListIndexLocal + 1] - 1

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, i, j,
                )
            end
            for NeighborOffsetIndex in NeighborStart:NeighborEnd
                NeighborIndex = NeighborCellIndices[NeighborOffsetIndex]
                StartIndex_ = ParticleRanges[NeighborIndex]
                EndIndex_ = ParticleRanges[NeighborIndex + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            UpdateTimeStepBuffers!(max_visc, min_dt_force, i, Position[i], Velocity[i], acc_acc, SimKernel)
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndex, NeighborCellOffsets,
                                      NeighborCellIndices, Position, Density, Pressure,
                                      Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ, max_visc = nothing,
                                      min_dt_force = nothing) where {D,T,
                                                  K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            CellListIndexLocal = CellListIndex[i]
            SameCellStart = ParticleRanges[CellListIndexLocal]
            SameCellEnd = ParticleRanges[CellListIndexLocal + 1] - 1
            NeighborStart = NeighborCellOffsets[CellListIndexLocal]
            NeighborEnd = NeighborCellOffsets[CellListIndexLocal + 1] - 1

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, i, j,
                )
            end
            for NeighborOffsetIndex in NeighborStart:NeighborEnd
                NeighborIndex = NeighborCellIndices[NeighborOffsetIndex]
                StartIndex_ = ParticleRanges[NeighborIndex]
                EndIndex_ = ParticleRanges[NeighborIndex + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                        ComputeInteractionsPerParticle!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, Pressure,
                            Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                            kernel_grad_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
            UpdateTimeStepBuffers!(max_visc, min_dt_force, i, Position[i], Velocity[i], acc_acc, SimKernel)
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndex, NeighborCellOffsets,
                                      NeighborCellIndices, Position, Density, Pressure,
                                      Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ, max_visc = nothing,
                                      min_dt_force = nothing) where {D,T,
                                                  S<:ShiftingMode,B<:MDBCMode,
                                                  L<:LogMode,SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndexLocal = CellListIndex[i]
            SameCellStart = ParticleRanges[CellListIndexLocal]
            SameCellEnd = ParticleRanges[CellListIndexLocal + 1] - 1
            NeighborStart = NeighborCellOffsets[CellListIndexLocal]
            NeighborEnd = NeighborCellOffsets[CellListIndexLocal + 1] - 1

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, shift_c_acc,
                        shift_r_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, shift_c_acc,
                    shift_r_acc, i, j,
                )
            end
            for NeighborOffsetIndex in NeighborStart:NeighborEnd
                NeighborIndex = NeighborCellIndices[NeighborOffsetIndex]
                StartIndex_ = ParticleRanges[NeighborIndex]
                EndIndex_ = ParticleRanges[NeighborIndex + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                        ComputeInteractionsPerParticleNoKernel!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, Pressure,
                            Velocity, MotionLimiter, dρdt_acc, acc_acc, shift_c_acc,
                            shift_r_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            ∇Cᵢ[i] = shift_c_acc
            ∇◌rᵢ[i] = shift_r_acc
            UpdateTimeStepBuffers!(max_visc, min_dt_force, i, Position[i], Velocity[i], acc_acc, SimKernel)
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndex, NeighborCellOffsets,
                                      NeighborCellIndices, Position, Density, Pressure,
                                      Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ, max_visc = nothing,
                                      min_dt_force = nothing) where {D,T,
                                                  S<:ShiftingMode,
                                                  K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndexLocal = CellListIndex[i]
            SameCellStart = ParticleRanges[CellListIndexLocal]
            SameCellEnd = ParticleRanges[CellListIndexLocal + 1] - 1
            NeighborStart = NeighborCellOffsets[CellListIndexLocal]
            NeighborEnd = NeighborCellOffsets[CellListIndexLocal + 1] - 1

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                )
            end
            for NeighborOffsetIndex in NeighborStart:NeighborEnd
                NeighborIndex = NeighborCellIndices[NeighborOffsetIndex]
                StartIndex_ = ParticleRanges[NeighborIndex]
                EndIndex_ = ParticleRanges[NeighborIndex + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                    shift_r_acc = ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
            ∇Cᵢ[i] = shift_c_acc
            ∇◌rᵢ[i] = shift_r_acc
            UpdateTimeStepBuffers!(max_visc, min_dt_force, i, Position[i],
                                      Velocity[i], acc_acc, SimKernel)
        end

        return nothing
    end

    f(SimKernel, GhostPoint) = CartesianIndex(map(x->map_floor(x,SimKernel.H⁻¹), Tuple(GhostPoint)))
    function NeighborLoopMDBC!(SimKernel,
                               SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                               SimConstants, ParticleRanges, CellLookup, Position,
                               Density, GhostPoints, GhostNormals, ParticleType,
                               bᵧ, Aᵧ) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
        
        FullStencil = ConstructFullStencil(Val(Dimensions))

        @inbounds @threads for iter in eachindex(GhostPoints)
            GhostPoint = GhostPoints[iter]

            if !iszero(GhostPoint)
                # zero‐initialize per‐ghost accumulators
                b_acc = zero(bᵧ[iter])            # an SVector{D+1,FloatType}
                A_acc = zero(Aᵧ[iter])            # an SMatrix{D+1,D+1,FloatType}
            
                # compute and accumulate into the locals
                GhostCellIndex = f(SimKernel, GhostPoints[iter])
                @inbounds for offset ∈ FullStencil
                    SCellIndex = GhostCellIndex + offset

                    # Returns a range, x>:x for exact match and x=:x for no match
                    # utilizes that it is a sorted array and requires no isequal constructor,
                    # so I prefer this for now
                    NeighborIdx = LookupCellIndex(CellLookup, SCellIndex)

                    StartIndex_       = ParticleRanges[NeighborIdx] 
                    EndIndex_         = ParticleRanges[NeighborIdx + 1] - 1

                    for j in StartIndex_:EndIndex_
                        # change ComputeInteractions to take & return contributions, e.g.:
                        bΔ, AΔ = ComputeInteractionsMDBC!(SimKernel, SimMetaData, SimConstants,
                                                        Position, Density, ParticleType,
                                                        GhostPoints, iter, j)
                        b_acc += bΔ
                        A_acc += AΔ
                    end
                end
            
                # write out once
                bᵧ[iter] = b_acc
                Aᵧ[iter] = A_acc
            end    
        end

        return nothing
    end

    # The previous generic `ComputeInteractions!` implementation was unused
    # in favour of the per-particle variants (ComputeInteractionsPerParticle! etc.).
    # It has been removed to reduce code size and avoid dead code.

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, i, j) where {D,T,
                                                                     K<:KernelOutputMode,
                                                                     B<:MDBCMode,
                                                                     L<:LogMode,
                                                                     SDD<:SPHDensityDiffusion,
                                                                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(xᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc =
                KernelOutputLocal!(SimMetaData, kernel_acc, kernel_grad_acc,
                                   SimKernel, q, ∇ᵢWᵢⱼ)
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticleNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, i, j) where {D,T,B<:MDBCMode,L<:LogMode,
                                        SDD<:SPHDensityDiffusion,
                                        SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(xᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term
        end

        return dρdt_acc, acc_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
        shift_r_acc, i, j) where {D,T,S<:ShiftingMode,
                                  K<:KernelOutputMode,
                                  B<:MDBCMode,
                                  L<:LogMode,
                                  SDD<:SPHDensityDiffusion,
                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(xᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc =
                KernelOutputLocal!(SimMetaData, kernel_acc, kernel_grad_acc,
                                   SimKernel, q, ∇ᵢWᵢⱼ)

            MLcond = MotionLimiter[i] * MotionLimiter[j]
            shift_c_acc += (m₀ / ρᵢ) * ∇ᵢWᵢⱼ
            shift_r_acc += (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticleNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, shift_c_acc, shift_r_acc, i, j) where {D,T,
                                                                  S<:ShiftingMode,
                                                                  B<:MDBCMode,
                                                                  L<:LogMode,
                                                                  SDD<:SPHDensityDiffusion,
                                                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(xᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            MLcond = MotionLimiter[i] * MotionLimiter[j]
            shift_c_acc += (m₀ / ρᵢ) * ∇ᵢWᵢⱼ
            shift_r_acc += (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
        end

        return dρdt_acc, acc_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsMDBC!(SimKernel, SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}, SimConstants, Position, Density, ParticleType, GhostPoints, i, j) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
        @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
        
        @unpack h⁻¹, h, η², H², αD = SimKernel 

        DimensionsPlus = Dimensions + 1
        # always zero‐initialize
        bΔ = zero(SVector{DimensionsPlus,FloatType})
        AΔ = zero(SMatrix{DimensionsPlus, DimensionsPlus,FloatType})

        # ᵢ is ghost node! ⱼ is fluid node

        if ParticleType[j] == Fluid

            xᵢⱼ  = GhostPoints[i] - Position[j]

            xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
            if xᵢⱼ² <= H²
                dᵢⱼ = sqrt(xᵢⱼ²)
                q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
        
                ρⱼ = Density[j]

        
                Wᵢⱼ = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)

                ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                Vⱼ = m₀ / ρⱼ
        
                VⱼWᵢⱼ = Vⱼ * Wᵢⱼ
        
                bΔ  = SVector{DimensionsPlus, FloatType}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)

                # Filling the Aᵧ matrix is done in column-major order
                xⱼᵢ = -xᵢⱼ
                first_column = [VⱼWᵢⱼ; Vⱼ * ∇ᵢWᵢⱼ]
                AΔ = SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus*DimensionsPlus}(
                    first_column...,
                    ((xⱼᵢ * first_column')')...
                )
            end
        end
        
    
        return bΔ, AΔ
    end

    function ApplyMDBCBeforeHalf!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, _args...) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
        return nothing
    end

    function ApplyMDBCBeforeHalf!(SimMetaData::SimulationMetaData{D,T,S,K,SimpleMDBC,L},
                                  SimKernel, SimConstants, SimParticles,
                                  ParticleRanges, CellLookup, Position, Density,
                                  GhostPoints, GhostNormals, ParticleType
                                 ) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        @no_escape begin
            DimensionsPlus = D + 1
            bᵧ = @alloc(SVector{DimensionsPlus, T}, length(Position))
            Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, T, DimensionsPlus*DimensionsPlus}, length(Position))
            NeighborLoopMDBC!(SimKernel, SimMetaData, SimConstants, ParticleRanges, CellLookup, Position, Density, GhostPoints, GhostNormals, ParticleType, bᵧ, Aᵧ)
            ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)
        end

        return nothing
    end

    function LoadMDBCNormals!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, SimParticles, path) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
        return nothing
    end
    function LoadMDBCNormals!(::SimulationMetaData{D,T,S,K,SimpleMDBC,L}, SimParticles, path) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
        if isnothing(path)
            return nothing
        end
        _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(D), T, path)
        for gi ∈ eachindex(GhostPoints)
            SimParticles.GhostPoints[gi]  = GhostPoints[gi]
            SimParticles.GhostNormals[gi] = GhostNormals[gi]
        end
        return nothing
    end

    function InitializeLog!(::SimulationMetaData{D,T,S,K,B,NoLog}, _args...) where {D,T,S<:ShiftingMode,
                                                                                      K<:KernelOutputMode,
                                                                                      B<:MDBCMode}
        return nothing
    end
    function InitializeLog!(SimMetaData::SimulationMetaData{D,T,S,K,B,StoreLog}, SimLogger,
                             SimConstants, SimKernel, SimViscosity, SimDensityDiffusion,
                             SimGeometry, SimParticles) where {D,T,S<:ShiftingMode,
                                                              K<:KernelOutputMode,
                                                              B<:MDBCMode}
        InitializeLogger(SimLogger, SimConstants, SimMetaData, SimKernel,
                         SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        LogStep(SimLogger, SimMetaData, SimMetaData.HourGlass)
        SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        return nothing
    end

    function LogStep!(::SimulationMetaData{D,T,S,K,B,NoLog}, _...) where {D,T,S<:ShiftingMode,
                                                                           K<:KernelOutputMode,
                                                                           B<:MDBCMode}
        return nothing
    end

    function LogStep!(SimMetaData::SimulationMetaData{D,T,S,K,B,StoreLog}, SimLogger) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, B<:MDBCMode}
        LogStep(SimLogger, SimMetaData, SimMetaData.HourGlass)
        SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        return nothing
    end

    function FinalizeLog!(::SimulationMetaData{D,T,S,K,B,NoLog}, _...) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, B<:MDBCMode}
        return nothing
    end
    
    function FinalizeLog!(SimMetaData::SimulationMetaData{D,T,S,K,B,StoreLog}, SimLogger) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, B<:MDBCMode}
        LogFinal(SimLogger, SimMetaData.HourGlass)
        
        # Time steps line plot
        UnicodeTimeStepsGraph = lineplot(
            1:length(SimMetaData.TimeSteps),
            SimMetaData.TimeSteps,
            title="Time Steps [s] as a function of iteration",
            name="Time Steps",
            xlabel="Iterations [-]",
            ylabel="Time Step Size [s]",
        )

        with_logger(SimLogger.Logger) do
            @info ""
            show(SimLogger.LoggerIo, UnicodeTimeStepsGraph)
        end
        
        close(SimLogger.LoggerIo)
        
        AutoOpenLogFile(SimLogger, SimMetaData)
        return nothing
    end

    function ProgressMotion(_SimParticles, _dt₂, ::Nothing, _SimMetaData)
        return nothing
    end

    function ProgressMotion(SimParticles, dt₂, MotionsDefinition, SimMetaData)
        @unpack Position, Velocity = SimParticles
        ParticleMarker  = SimParticles.GroupMarker
        ParticleType    = SimParticles.Type
        @inbounds @simd ivdep for i in eachindex(Position)
            if ParticleType[i] == Moving
                motion = MotionsDefinition[ParticleMarker[i]]
    
                if motion !== nothing
                    ShouldMove = (motion.StartTime <= SimMetaData.TotalTime) &&
                                 (SimMetaData.TotalTime <= (motion.StartTime + motion.Duration))
    
                    # Retrieve motion parameters
                    MotionVel = motion.Velocity
                    MotionDir = motion.Direction
    
                    # Update Velocity and Position
                    Velocity[i] = MotionVel * MotionDir * ShouldMove
                    Position[i] += Velocity[i] * dt₂
                end
            end
        end

        return nothing
    end

    function ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)

        Position    = SimParticles.Position
        Density     = SimParticles.Density
        GhostPoints = SimParticles.GhostPoints

        ρ₀ = SimConstants.ρ₀
        #https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
        @inbounds @simd ivdep for i in eachindex(Position)
            A = Aᵧ[i]

            # Since Aᵧ is not reset anymore, we need to check if it is zero
            if !iszero(GhostPoints[i])
                if abs(det(A)) >= 1e-3
                        GhostPointDensity = A \ bᵧ[i]
                        diff = Position[i] - GhostPoints[i]
                        v1   = first(GhostPointDensity) + sum(GhostPointDensity[j+1] * diff[j] for j in eachindex(diff))
                        Density[i] = isnan(v1) ? ρ₀ : v1
                elseif first(A) > 0.0
                        v = first(bᵧ[i]) / first(A)
                        Density[i] = isnan(v) ? ρ₀ : v
                end
            end
        end
    end
    
    function HalfTimeStep(::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                          SimConstants, SimParticles, Positionₙ⁺,
                          Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
        @unpack Position, Density, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles

        @inbounds @simd ivdep for i in eachindex(Position)
            Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
            Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
            ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
        end


        return nothing
    end

    function FullTimeStep(::SimulationMetaData{D,T,NoShifting,K,B,L}, SimKernel,
                          SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,
                                                                             K<:KernelOutputMode,
                                                                             B<:MDBCMode,
                                                                             L<:LogMode}
        @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles
        @inbounds @simd ivdep for i in eachindex(Position)
            Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
            Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
        end
        return nothing
    end

    function FullTimeStep(::SimulationMetaData{D,T,S,K,B,L}, SimKernel, SimConstants,
                          SimParticles, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,S<:ShiftingMode,
                                                             K<:KernelOutputMode,
                                                             B<:MDBCMode,
                                                             L<:LogMode}
        @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles
        A     = 2# Value between 1 to 6 advised
        A_FST = 0; # zero for internal flows
        A_FSM = length(first(Position)); #2d, 3d val different
        @inbounds @simd ivdep for i in eachindex(Position)
            Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]

            A_FSC                  = (∇◌rᵢ[i] - A_FST)/(A_FSM - A_FST)
            if A_FSC < 0
                δxᵢ = zero(eltype(Position))
            else
                δxᵢ = -A_FSC * A * SimKernel.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
            end

            Position[i]           += (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt + δxᵢ) * MotionLimiter[i]
        end
        return nothing
    end

    function UpdateMetaData!(SimMetaData, dt)
        SimMetaData.Iteration      += 1
        SimMetaData.CurrentTimeStep = dt
        SimMetaData.TotalTime      += dt

        return nothing
    end

    @inline next_output_time(SimMetaData) = next_output_time(SimMetaData.OutputTimes, SimMetaData)

    @inline next_output_time(interval::Real, SimMetaData) = interval * SimMetaData.OutputIterationCounter

    @inline function next_output_time(times::AbstractVector, SimMetaData)
        idx = SimMetaData.OutputIterationCounter
        if idx < length(times)
            return times[idx]
        else
            return SimMetaData.SimulationTime
        end
    end

    function GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)
        # Assuming group markers are sequential
        MotionDefinition = Vector{Union{Nothing, MotionDetails{Dimensions, FloatType}}}(undef, maximum(SimParticles.GroupMarker))

        for geom in SimGeometry
            group_marker = geom.GroupMarker
            if geom.Motion !== nothing
                MotionDefinition[group_marker] = geom.Motion
            else
                MotionDefinition[group_marker] = nothing
            end
        end
        if !any(!isnothing, MotionDefinition)
            MotionDefinition = nothing
        end
        return MotionDefinition
    end


    """
        UpdateΔx!(Δx, posₙ⁺, pos)

    Increment Δx by twice the maximum ‖posₙ⁺[i] – pos[i]‖, without ever allocating.
    Returns the new Δx.
    """
    @inline function UpdateΔx!(Δx::T,
                                    posₙ⁺::AbstractVector{SVector{D, T}},
                                    pos   ::AbstractVector{SVector{D, T}}) where {D, T<:Real}
        maxd = zero(T)
        @inbounds for i in eachindex(posₙ⁺, pos)
            # compute squared norm manually
            sumsq = zero(T)
            @inbounds for j in 1:D
                d = posₙ⁺[i][j] - pos[i][j]
                sumsq += d*d
            end
            # sqrt/T is allocation-free on scalars
            nrm = sqrt(sumsq)
            if nrm > maxd
                maxd = nrm
            end
        end
        return Δx + 4*maxd
    end

    # Per-particle local Δx removed: use single scalar `SimMetaData.Δx`.

    
    @inbounds function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                      SimConstants, SimParticles, FullStencil,
                                      ParticleRanges, UniqueCells, CellLookup,
                                      SortingScratchSpace, CellListIndex,
                                      NeighborCellOffsets, NeighborCellIndices,
                                      dρdtI, Velocityₙ⁺,
                                      Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ,
                                      MotionDefinition::Union{
                                          Nothing,
                                          AbstractVector{
                                              Union{
                                                  Nothing,
                                                  MotionDetails{Dimensions, FloatType},
                                              },
                                          },
                                      }) where {
                                                Dimensions, FloatType, SMode, KMode,
                                                BMode, LMode,
                                                SDD<:SPHDensityDiffusion,
                                                SV<:SPHViscosity}
        @unpack Position, Density, Pressure, Velocity, Acceleration, MotionLimiter,
                GroupMarker, Kernel, KernelGradient, GhostPoints,
                GhostNormals = SimParticles
        ParticleType   = SimParticles.Type
        ParticleMarker = GroupMarker

        ###
        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
        # This code here is to initialize the first time step for each simulation loop
        dt = Δt(Position, Velocity, Acceleration, SimConstants, SimKernel)

        @no_escape begin
            max_visc = @alloc(FloatType, length(Position))
            min_dt_force = @alloc(FloatType, length(Position))

            dt₂ = dt * 0.5

            while SimMetaData.TotalTime <= next_output_time(SimMetaData)
                @timeit SimMetaData.HourGlass "01 Calculate IndexCounter"  begin

                    SimMetaData.Δx = UpdateΔx!(SimMetaData.Δx, Positionₙ⁺, SimParticles.Position)
                    ShouldRebuild = SimMetaData.Δx >= SimKernel.h

                    # println("Δx: ", Δx, "h: ", SimKernel.h," dt: ", SimMetaData.CurrentTimeStep, " Iteration: ", SimMetaData.Iteration, " TotalTime: ", SimMetaData.TotalTime, " OutputIterationCounter: ", SimMetaData.OutputIterationCounter)

                    # Note: If particles are not inside of the neighbor list visualiation, try setting this if statement to always true, since UniqueCells will be updated always then
                    # In theory, the maximal speed is the speed of sound, this should give a safe guard
                    # and ensure it is always updated in a reasonable manner. This only works well, assuming that
                    # c₀ >= maximum(norm.(Velocity))
                    # Remove if statement logic if you want to update each iteration
                    # if mod(SimMetaData.Iteration, ceil(Int, SimKernel.H / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
                    if ShouldRebuild
                        @timeit SimMetaData.HourGlass "01a Actual Calculate IndexCounter" SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace,  ParticleRanges, UniqueCells, CellLookup, CellListIndex)
                        SimMetaData.Δx    = zero(eltype(dρdtI))
                        UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                        BuildNeighborCellLists!(NeighborCellOffsets, NeighborCellIndices, FullStencil, UniqueCellsView, ParticleRanges, CellLookup)
                    end
                end

                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)
            
                @timeit SimMetaData.HourGlass "02 Pressure"                              Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
                @timeit SimMetaData.HourGlass "03 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, CellLookup, Position, Density, GhostPoints, GhostNormals, ParticleType)

                @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, ParticleRanges, CellListIndex,
                    NeighborCellOffsets, NeighborCellIndices, Position, Density,
                    Pressure, Velocity,
                    MotionLimiter, dρdtI, Acceleration, Kernel,
                    KernelGradient, ∇Cᵢ, ∇◌rᵢ,
                )


                @timeit SimMetaData.HourGlass "05 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)


                @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)
            
                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)
            
                @timeit SimMetaData.HourGlass "07 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
                @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, ParticleRanges, CellListIndex,
                    NeighborCellOffsets, NeighborCellIndices, Positionₙ⁺, ρₙ⁺,
                    Pressure, Velocityₙ⁺,
                    MotionLimiter, dρdtI, Acceleration, Kernel,
                    KernelGradient, ∇Cᵢ, ∇◌rᵢ, max_visc, min_dt_force,
                )

                @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary"          LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)
            
                @timeit SimMetaData.HourGlass "10 Final Density"                         DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)
            
                @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"              FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
            
                @timeit SimMetaData.HourGlass "12 Update MetaData"                       UpdateMetaData!(SimMetaData, dt)

                @timeit SimMetaData.HourGlass "13 Update TimeStep" dt = FinalizeTimeStep(max_visc, min_dt_force, SimConstants, SimKernel)
            end
        end
        
        return nothing
    end
    
    ###===
    function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}}, #Don't further specify type for now
        SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
        SimConstants::SimulationConstants,
        SimKernel::SPHKernelInstance,
        SimLogger::SimulationLogger,
        SimParticles::StructArray,
        SimViscosity::SV,
        SimDensityDiffusion::SDD,
        ParticleNormalsPath::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType,SMode,KMode,BMode,LMode,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}

        NumberOfPoints = length(SimParticles)
        
        dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimMetaData, SimParticles.Position)

        LoadMDBCNormals!(SimMetaData, SimParticles, ParticleNormalsPath)

        InitializeLog!(SimMetaData, SimLogger, SimConstants, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        
        Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    
        # Produce sorting related variables
        ParticleRanges         = zeros(Int, NumberOfPoints + 1 + 1) # +1 for the last particle, +1 for dummy entry
        UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
        CellLookup             = CellIndexLookup{Dimensions}()
        CellListIndex          = zeros(Int, NumberOfPoints)
        FullStencil            = ConstructFullStencil(Val(Dimensions))
        NeighborCellOffsets    = Int[]
        NeighborCellIndices    = Int[]
        _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)

        output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

        # Save initial state, use 1 else this cannot be used to index fid vector
        SimMetaData.OutputIterationCounter = 1
        output.enqueue_particles(SimMetaData.OutputIterationCounter)
        if SimMetaData.IndexCounter > 0
            unique_cells_view = view(UniqueCells, 1:SimMetaData.IndexCounter)
            cell_particle_counts = nothing
            cell_neighbor_counts = nothing
            if SimMetaData.ExportGridCellParticleCounts
                cell_particle_counts = compute_cell_particle_counts(
                    ParticleRanges,
                    SimMetaData.IndexCounter,
                )
                cell_neighbor_counts = compute_cell_neighbor_counts(
                    ParticleRanges,
                    NeighborCellOffsets,
                    NeighborCellIndices,
                    SimMetaData.IndexCounter,
                )
            end
            output.enqueue_grid(
                SimMetaData.OutputIterationCounter,
                unique_cells_view,
                cell_particle_counts=cell_particle_counts,
                cell_neighbor_counts=cell_neighbor_counts,
            )
        end


        MotionDefinition = GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)

        @inbounds while true

            @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoop(
                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                SimConstants, SimParticles, FullStencil, ParticleRanges,
                UniqueCells, CellLookup, SortingScratchSpace, CellListIndex,
                NeighborCellOffsets, NeighborCellIndices, dρdtI, Velocityₙ⁺,
                Positionₙ⁺, ρₙ⁺,
                ∇Cᵢ, ∇◌rᵢ, MotionDefinition,
            )
            push!(SimMetaData.TimeSteps, SimMetaData.CurrentTimeStep)

            LogStep!(SimMetaData, SimLogger)

            SimMetaData.OutputIterationCounter += 1

            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            cell_particle_counts = nothing
            cell_neighbor_counts = nothing
            if SimMetaData.ExportGridCellParticleCounts
                cell_particle_counts = compute_cell_particle_counts(
                    ParticleRanges,
                    length(UniqueCellsView),
                )
                cell_neighbor_counts = compute_cell_neighbor_counts(
                    ParticleRanges,
                    NeighborCellOffsets,
                    NeighborCellIndices,
                    length(UniqueCellsView),
                )
            end
            @timeit SimMetaData.HourGlass "13 Save Particle Data"  begin
                output.enqueue_particles(SimMetaData.OutputIterationCounter)
                output.enqueue_grid(SimMetaData.OutputIterationCounter, UniqueCellsView, cell_particle_counts=cell_particle_counts, cell_neighbor_counts=cell_neighbor_counts)
            end

            if SimMetaData.TotalTime > SimMetaData.SimulationTime

                # At end of simulation
                @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

                show(SimMetaData.HourGlass,sortby=:name)
                show(SimMetaData.HourGlass)

                AutoOpenParaview(SimMetaData, output.variable_names)

                FinalizeLog!(SimMetaData, SimLogger)

                break
            end
        end
    end
    

end
