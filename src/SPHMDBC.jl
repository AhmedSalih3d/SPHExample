module SPHMDBC

export MDBCGhostData, ResetGhostData!, InitializeGhostData!, EnsureGhostNeighborCellListsSize!,
       InitializeGhostDataRuntime, UpdateGhostIndices!, UpdateGhostNeighborCellLists!, ApplyMDBCBeforeHalf!

using StaticArrays
using FastPow: @fastpow
using Parameters: @unpack
using LinearAlgebra: dot, det
using Base.Threads
using Bumper

using ..SimulationGeometry
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SPHKernels
using ..SPHNeighborList: FindCellIndex, MapFloor

"""
    MDBCGhostData

Stores active ghost-particle indices and their neighboring cell lists used by MDBC.
"""
struct MDBCGhostData
    Indices::Vector{Int}
    NeighborCellLists::Vector{Vector{Int}}
end

MDBCGhostData() = MDBCGhostData(Int[], Vector{Vector{Int}}())

function ResetGhostData!(GhostData::MDBCGhostData)
    empty!(GhostData.Indices)
    empty!(GhostData.NeighborCellLists)
    return nothing
end

function InitializeGhostData!(GhostData::MDBCGhostData, GhostIndices::AbstractVector{Int})
    indices = GhostData.Indices
    empty!(indices)
    sizehint!(indices, length(GhostIndices))
    append!(indices, GhostIndices)
    EnsureGhostNeighborCellListsSize!(GhostData)
    return nothing
end

function EnsureGhostNeighborCellListsSize!(GhostData::MDBCGhostData)
    target_len = length(GhostData.Indices)
    neighbor_cell_lists = GhostData.NeighborCellLists
    original_len = length(neighbor_cell_lists)
    resize!(neighbor_cell_lists, target_len)

    if target_len > original_len
        @inbounds for idx in (original_len + 1):target_len
            neighbor_cell_lists[idx] = Int[]
        end
    end

    return nothing
end

@inline function InitializeGhostDataRuntime(::SimulationMetaData{D,T,S,K,NoMDBC,L}) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
    return nothing
end

@inline function InitializeGhostDataRuntime(::SimulationMetaData{D,T,S,K,SimpleMDBC,L}) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
    return MDBCGhostData()
end

@inline f(SimKernel, GhostPoint) = CartesianIndex(map(x -> MapFloor(x, SimKernel.H⁻¹), Tuple(GhostPoint)))

function NeighborLoopMDBC!(SimKernel,
                           SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, SimpleMDBC, LMode},
                           GhostData::MDBCGhostData,
                           SimConstants, ParticleRanges,
                           SimParticles, bᵧ, Aᵧ) where {Dimensions, FloatType, SMode, KMode, LMode}

    @unpack Position, Density, GhostPoints = SimParticles
    ParticleType = SimParticles.Type
    GhostIndices = GhostData.Indices
    GhostNeighborCellLists = GhostData.NeighborCellLists

    @inbounds @threads for gpos in eachindex(GhostIndices)
        iter = GhostIndices[gpos]

        b_acc = zero(bᵧ[iter])
        A_acc = zero(Aᵧ[iter])

        NeighborCellIndices = GhostNeighborCellLists[gpos]
        @inbounds for NeighborIdx in NeighborCellIndices
            StartIndex_ = ParticleRanges[NeighborIdx]
            EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1

            for j in StartIndex_:EndIndex_
                bΔ, AΔ = ComputeInteractionsMDBC!(SimKernel, SimMetaData, SimConstants,
                                                  Position, Density, ParticleType,
                                                  GhostPoints, iter, j)
                b_acc += bΔ
                A_acc += AΔ
            end
        end

        bᵧ[iter] = b_acc
        Aᵧ[iter] = A_acc
    end

    return nothing
end

@inline function UpdateGhostNeighborCellLists!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, ::Nothing, _args...) where {D,T,S<:ShiftingMode,
                                                                                                                 K<:KernelOutputMode,
                                                                                                                 L<:LogMode}
    return nothing
end

@inline function UpdateGhostIndices!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, ::Nothing, _args...) where {D,T,S<:ShiftingMode,
                                                                                                        K<:KernelOutputMode,
                                                                                                        L<:LogMode}
    return nothing
end

function UpdateGhostIndices!(::SimulationMetaData{D,T,S,K,SimpleMDBC,L}, GhostData::MDBCGhostData, SimParticles) where {D,T,S<:ShiftingMode,
                                                                                                                     K<:KernelOutputMode,
                                                                                                                     L<:LogMode}
    GhostIndices = GhostData.Indices
    empty!(GhostIndices)
    @inbounds for i in eachindex(SimParticles.GhostPoints)
        if !iszero(SimParticles.GhostPoints[i])
            push!(GhostIndices, i)
        end
    end

    EnsureGhostNeighborCellListsSize!(GhostData)

    return nothing
end

function UpdateGhostNeighborCellLists!(::SimulationMetaData{Dimensions, FloatType, SMode, KMode, SimpleMDBC, LMode},
                                       GhostData::MDBCGhostData,
                                       SimKernel,
                                       SimParticles,
                                       ParticleRanges,
                                       UniqueCellsView,
                                       FullStencil) where {Dimensions, FloatType, SMode, KMode, LMode}
    GhostIndices = GhostData.Indices

    EnsureGhostNeighborCellListsSize!(GhostData)
    GhostNeighborCellLists = GhostData.NeighborCellLists

    @inbounds for (gpos, iter) in enumerate(GhostIndices)
        NeighborCellIndices = GhostNeighborCellLists[gpos]
        empty!(NeighborCellIndices)
        GhostPoint = SimParticles.GhostPoints[iter]

        GhostCellIndex = f(SimKernel, GhostPoint)
        @inbounds for offset ∈ FullStencil
            SCellIndex = GhostCellIndex + offset
            NeighborIdx = FindCellIndex(UniqueCellsView, SCellIndex)
            StartIndex_ = ParticleRanges[NeighborIdx]
            EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
            if StartIndex_ <= EndIndex_
                push!(NeighborCellIndices, NeighborIdx)
            end
        end
    end

    return nothing
end

Base.@propagate_inbounds function ComputeInteractionsMDBC!(SimKernel,
                                                           SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                                           SimConstants, Position, Density, ParticleType, GhostPoints, i, j) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
    @unpack m₀ = SimConstants
    @unpack h⁻¹, H² = SimKernel

    DimensionsPlus = Dimensions + 1
    bΔ = zero(SVector{DimensionsPlus,FloatType})
    AΔ = zero(SMatrix{DimensionsPlus, DimensionsPlus,FloatType})

    if ParticleType[j] == Fluid
        xᵢⱼ  = GhostPoints[i] - Position[j]

        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(xᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)

            ρⱼ = Density[j]

            Wᵢⱼ = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            ∇ᵢWᵢⱼ = @fastpow SPHKernels.∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            Vⱼ = m₀ / ρⱼ
            VⱼWᵢⱼ = Vⱼ * Wᵢⱼ

            bΔ  = SVector{DimensionsPlus, FloatType}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)

            xⱼᵢ = -xᵢⱼ
            first_column = SVector{DimensionsPlus, FloatType}(VⱼWᵢⱼ, (Vⱼ * ∇ᵢWᵢⱼ)...)
            column_scalars = SVector{DimensionsPlus, FloatType}(one(FloatType), xⱼᵢ...)
            AΔ = first_column * transpose(column_scalars)
        end
    end

    return bΔ, AΔ
end

function ApplyMDBCBeforeHalf!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, ::Nothing, _args...) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
    return nothing
end

function ApplyMDBCBeforeHalf!(SimMetaData::SimulationMetaData{D,T,S,K,SimpleMDBC,L},
                              GhostData::MDBCGhostData,
                              SimKernel, SimConstants, SimParticles,
                              ParticleRanges
                             ) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
    GhostIndices = GhostData.Indices
    @no_escape begin
        DimensionsPlus = D + 1
        bᵧ = @alloc(SVector{DimensionsPlus, T}, length(SimParticles.Position))
        Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, T, DimensionsPlus*DimensionsPlus}, length(SimParticles.Position))
        NeighborLoopMDBC!(SimKernel, SimMetaData, GhostData, SimConstants, ParticleRanges, SimParticles, bᵧ, Aᵧ)
        ApplyMDBCCorrection(SimConstants, SimParticles, GhostIndices, bᵧ, Aᵧ)
    end

    return nothing
end

function ApplyMDBCCorrection(SimConstants, SimParticles, GhostIndices, bᵧ, Aᵧ)
    Position    = SimParticles.Position
    Density     = SimParticles.Density
    GhostPoints = SimParticles.GhostPoints

    ρ₀ = SimConstants.ρ₀
    @inbounds @simd ivdep for gpos in eachindex(GhostIndices)
        i = GhostIndices[gpos]
        A = Aᵧ[i]

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

    return nothing
end

end
