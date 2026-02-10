module SPHMDBC

export MDBCGhostData, ResetGhostData!, InitializeGhostData!, EnsureGhostNeighborCellListsSize!,
       InitializeGhostDataRuntime, UpdateGhostIndices!, UpdateGhostNeighborCellLists!, ApplyMDBCBeforeHalf!

using StaticArrays
using FastPow: @fastpow
using Parameters: @unpack
using LinearAlgebra: dot, det, norm
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

@inline function InitializeGhostDataRuntime(::SimulationMetaData{D,T,S,K,UpdatedMDBC,L}) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
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

function UpdateGhostIndices!(::SimulationMetaData{D,T,S,K,B,L}, GhostData::MDBCGhostData, SimParticles) where {D,T,S<:ShiftingMode,
                                                                                                             K<:KernelOutputMode,
                                                                                                             B<:Union{SimpleMDBC,UpdatedMDBC},
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

function UpdateGhostNeighborCellLists!(::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                       GhostData::MDBCGhostData,
                                       SimKernel,
                                       SimParticles,
                                       ParticleRanges,
                                       UniqueCellsView,
                                       FullStencil) where {Dimensions, FloatType, SMode, KMode, BMode<:Union{SimpleMDBC,UpdatedMDBC}, LMode}
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

Base.@propagate_inbounds function ComputeInteractionsUpdatedMDBC!(SimKernel,
                                                                  SimConstants,
                                                                  Position::AbstractVector{SVector{Dimensions, FloatType}},
                                                                  Density,
                                                                  Velocity::AbstractVector{SVector{Dimensions, FloatType}},
                                                                  ParticleType,
                                                                  GhostPoints::AbstractVector{SVector{Dimensions, FloatType}},
                                                                  i,
                                                                  j) where {Dimensions, FloatType}
    @unpack m₀ = SimConstants
    @unpack h⁻¹, H² = SimKernel

    DimensionsPlus = Dimensions + 1
    bΔ = zero(SVector{DimensionsPlus, FloatType})
    AΔ = zero(SMatrix{DimensionsPlus, DimensionsPlus, FloatType})
    GhostVelocityΔ = zero(SVector{Dimensions, FloatType})
    KernelSumΔ = zero(FloatType)
    SubmergedΔ = zero(FloatType)

    if ParticleType[j] == Fluid
        xᵢⱼ = GhostPoints[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(xᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)

            ρⱼ = Density[j]

            Wᵢⱼ = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            ∇ᵢWᵢⱼ = @fastpow SPHKernels.∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            Vⱼ = m₀ / ρⱼ
            VⱼWᵢⱼ = Vⱼ * Wᵢⱼ

            bΔ = SVector{DimensionsPlus, FloatType}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)

            xⱼᵢ = -xᵢⱼ
            first_column = SVector{DimensionsPlus, FloatType}(VⱼWᵢⱼ, (Vⱼ * ∇ᵢWᵢⱼ)...)
            column_scalars = SVector{DimensionsPlus, FloatType}(one(FloatType), xⱼᵢ...)
            AΔ = first_column * transpose(column_scalars)

            GhostVelocityΔ = VⱼWᵢⱼ * Velocity[j]
            KernelSumΔ = VⱼWᵢⱼ
            SubmergedΔ = -Vⱼ * dot(xᵢⱼ, ∇ᵢWᵢⱼ)
        end
    end

    return bΔ, AΔ, GhostVelocityΔ, KernelSumΔ, SubmergedΔ
end

function NeighborLoopUpdatedMDBC!(SimKernel,
                                  GhostData::MDBCGhostData,
                                  SimConstants,
                                  ParticleRanges,
                                  SimParticles,
                                  bᵧ,
                                  Aᵧ,
                                  GhostVelocityNumerator,
                                  KernelSum,
                                  Submerged)
    @unpack Position, Density, Velocity, GhostPoints = SimParticles
    ParticleType = SimParticles.Type
    GhostIndices = GhostData.Indices
    GhostNeighborCellLists = GhostData.NeighborCellLists

    @inbounds @threads for gpos in eachindex(GhostIndices)
        i = GhostIndices[gpos]

        b_acc = zero(bᵧ[i])
        A_acc = zero(Aᵧ[i])
        GhostVelocity_acc = zero(GhostVelocityNumerator[i])
        KernelSum_acc = zero(KernelSum[i])
        Submerged_acc = zero(Submerged[i])

        NeighborCellIndices = GhostNeighborCellLists[gpos]
        @inbounds for NeighborIdx in NeighborCellIndices
            StartIndex_ = ParticleRanges[NeighborIdx]
            EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
            for j in StartIndex_:EndIndex_
                bΔ, AΔ, GhostVelocityΔ, KernelSumΔ, SubmergedΔ = ComputeInteractionsUpdatedMDBC!(
                    SimKernel, SimConstants, Position, Density, Velocity, ParticleType, GhostPoints, i, j,
                )
                b_acc += bΔ
                A_acc += AΔ
                GhostVelocity_acc += GhostVelocityΔ
                KernelSum_acc += KernelSumΔ
                Submerged_acc += SubmergedΔ
            end
        end

        bᵧ[i] = b_acc
        Aᵧ[i] = A_acc
        GhostVelocityNumerator[i] = GhostVelocity_acc
        KernelSum[i] = KernelSum_acc
        Submerged[i] = Submerged_acc
    end

    return nothing
end

@inline function MatrixInfNorm(A)
    RowNorm = zero(eltype(A))
    @inbounds for i in axes(A, 1)
        CurrentRowNorm = zero(eltype(A))
        for j in axes(A, 2)
            CurrentRowNorm += abs(A[i, j])
        end
        RowNorm = max(RowNorm, CurrentRowNorm)
    end
    return RowNorm
end

@inline function NormalizeBoundaryNormal(Normal::SVector{D, T}) where {D, T}
    NormalNorm = norm(Normal)
    if NormalNorm > eps(T)
        return Normal / NormalNorm, NormalNorm
    end
    return zero(Normal), zero(T)
end

@inline function GravityVectorForMDBC(PositionVector, GravityMagnitude)
    GravityVector = zero(PositionVector)
    return setindex(GravityVector, -GravityMagnitude, length(PositionVector))
end

function ApplyUpdatedMDBCCorrection(SimKernel,
                                    SimConstants,
                                    SimParticles,
                                    GhostIndices,
                                    bᵧ,
                                    Aᵧ,
                                    GhostVelocityNumerator,
                                    KernelSum,
                                    Submerged)
    Position = SimParticles.Position
    Velocity = SimParticles.Velocity
    Density = SimParticles.Density
    GhostPoints = SimParticles.GhostPoints
    GhostNormals = SimParticles.GhostNormals
    ParticleType = SimParticles.Type
    MDBCMotionVelocity = SimParticles.MDBCMotionVelocity
    MDBCTangentVelocity = SimParticles.MDBCTangentVelocity
    MDBCBoundaryFactor = SimParticles.MDBCBoundaryFactor

    ρ₀ = SimConstants.ρ₀
    c₀ = SimConstants.c₀
    Dx² = SimConstants.dx * SimConstants.dx
    Gravity = GravityVectorForMDBC(first(Position), SimConstants.g)
    DensityFloor = ρ₀ * eltype(Density)(1e-3)

    @inbounds for i in eachindex(MDBCBoundaryFactor)
        CurrentType = ParticleType[i]
        if CurrentType == Fixed
            MotionVelocity = zero(MDBCMotionVelocity[i])
            MDBCMotionVelocity[i] = MotionVelocity
            MDBCTangentVelocity[i] = MotionVelocity
        else
            CurrentVelocity = Velocity[i]
            MDBCMotionVelocity[i] = CurrentVelocity
            MDBCTangentVelocity[i] = CurrentVelocity
        end
        MDBCBoundaryFactor[i] = CurrentType == Fluid ? one(eltype(MDBCBoundaryFactor)) : zero(eltype(MDBCBoundaryFactor))
    end

    @inbounds @simd ivdep for gpos in eachindex(GhostIndices)
        i = GhostIndices[gpos]
        MotionVelocity = MDBCMotionVelocity[i]
        BoundaryNormal = GhostNormals[i]
        NormalUnit, NormalScale = NormalizeBoundaryNormal(BoundaryNormal)
        Support = KernelSum[i]
        IsSubmerged = Submerged[i] > zero(Support)

        if IsSubmerged && Support > eps(eltype(Support))
            A = Aᵧ[i]
            b = bᵧ[i]

            GhostDensity = ρ₀
            if Support < eltype(Support)(0.1)
                ShepherdDensity = first(b) / Support
                GhostDensity = max(ρ₀, ShepherdDensity)
            else
                DeterminantA = det(A)
                if abs(DeterminantA) >= eltype(DeterminantA)(1e-3)
                    InverseA = inv(A)
                    ConditionInf = Dx² * MatrixInfNorm(A) * MatrixInfNorm(InverseA)
                    if ConditionInf <= eltype(ConditionInf)(50)
                        GhostDensityState = InverseA * b
                        GhostDensity = first(GhostDensityState)
                    else
                        GhostDensity = first(b) / Support
                    end
                else
                    GhostDensity = first(b) / Support
                end
            end
            GhostDensity = isfinite(GhostDensity) ? GhostDensity : ρ₀

            GhostPressure = c₀ * c₀ * (GhostDensity - ρ₀)
            dpos = Position[i] - GhostPoints[i]
            BoundaryAcceleration = zero(Gravity)
            PressureCloneTerm = ρ₀ * dot(Gravity - BoundaryAcceleration, NormalUnit) * dot(dpos, NormalUnit)
            BoundaryPressure = GhostPressure + PressureCloneTerm
            BoundaryDensity = ρ₀ + BoundaryPressure / (c₀ * c₀)
            if isfinite(BoundaryDensity)
                Density[i] = max(BoundaryDensity, DensityFloor)
            else
                Density[i] = ρ₀
            end

            GhostVelocity = GhostVelocityNumerator[i] / Support
            BoundaryVelocity = (MotionVelocity + MotionVelocity) - GhostVelocity
            Velocity[i] = BoundaryVelocity
            MDBCTangentVelocity[i] = BoundaryVelocity - dot(BoundaryVelocity, NormalUnit) * NormalUnit
            MDBCBoundaryFactor[i] = one(eltype(MDBCBoundaryFactor))
        else
            Density[i] = ρ₀
            Velocity[i] = MotionVelocity
            if NormalScale > zero(NormalScale)
                MDBCTangentVelocity[i] = MotionVelocity - dot(MotionVelocity, NormalUnit) * NormalUnit
            else
                MDBCTangentVelocity[i] = MotionVelocity
            end
            MDBCBoundaryFactor[i] = zero(eltype(MDBCBoundaryFactor))
        end
    end

    return nothing
end

function ApplyUpdatedNoPenetration!(SimConstants,
                                    GhostData::MDBCGhostData,
                                    SimParticles,
                                    ParticleRanges,
                                    NoPenShift,
                                    NoPenCount)
    Position = SimParticles.Position
    Velocity = SimParticles.Velocity
    ParticleType = SimParticles.Type
    GhostNormals = SimParticles.GhostNormals
    MDBCMotionVelocity = SimParticles.MDBCMotionVelocity
    MDBCBoundaryFactor = SimParticles.MDBCBoundaryFactor
    GhostIndices = GhostData.Indices
    GhostNeighborCellLists = GhostData.NeighborCellLists

    fill!(NoPenShift, zero(eltype(NoPenShift)))
    fill!(NoPenCount, zero(eltype(NoPenCount)))

    Dx = SimConstants.dx
    DistLimit = eltype(Dx)(1.25) * Dx
    NormalDistanceLimit = eltype(Dx)(0.75)
    NormalScaleLimit = eltype(Dx)(1.75) * Dx
    RatioFloor = eltype(Dx)(0.25)
    SupportCountLimit = eltype(Dx)(5.0)

    @inbounds for (gpos, i) in enumerate(GhostIndices)
        if MDBCBoundaryFactor[i] <= zero(eltype(MDBCBoundaryFactor))
            continue
        end

        BoundaryNormal = GhostNormals[i]
        NormalUnit, NormalScale = NormalizeBoundaryNormal(BoundaryNormal)
        if NormalScale <= eps(eltype(NormalScale))
            continue
        end
        if NormalScale >= NormalScaleLimit
            continue
        end

        BoundaryPosition = Position[i]
        MotionVelocity = MDBCMotionVelocity[i]
        NeighborCellIndices = GhostNeighborCellLists[gpos]

        for NeighborIdx in NeighborCellIndices
            StartIndex_ = ParticleRanges[NeighborIdx]
            EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
            for j in StartIndex_:EndIndex_
                if ParticleType[j] != Fluid
                    continue
                end

                RelativePosition = Position[j] - BoundaryPosition
                if norm(RelativePosition) > DistLimit
                    continue
                end

                NormalDistance = dot(RelativePosition, NormalUnit)
                if NormalDistance >= NormalDistanceLimit * NormalScale
                    continue
                end

                RelativeVelocity = Velocity[j] - MotionVelocity
                VelocityToBoundary = dot(RelativeVelocity, NormalUnit)
                if VelocityToBoundary < zero(VelocityToBoundary)
                    Ratio = max(abs(NormalDistance / NormalScale), RatioFloor)
                    Factor = -eltype(Ratio)(4.0) * Ratio + eltype(Ratio)(3.0)
                    VelocityCorrection = -(Factor * VelocityToBoundary) * NormalUnit
                    NoPenShift[j] += VelocityCorrection
                    NoPenCount[j] += one(eltype(NoPenCount))
                end
            end
        end
    end

    @inbounds for j in eachindex(Velocity)
        if ParticleType[j] == Fluid && NoPenCount[j] > SupportCountLimit
            Velocity[j] += NoPenShift[j] / NoPenCount[j]
        end
    end

    return nothing
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

function ApplyMDBCBeforeHalf!(::SimulationMetaData{D,T,S,K,UpdatedMDBC,L},
                              GhostData::MDBCGhostData,
                              SimKernel,
                              SimConstants,
                              SimParticles,
                              ParticleRanges
                             ) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
    GhostIndices = GhostData.Indices
    @no_escape begin
        DimensionsPlus = D + 1
        NumberOfParticles = length(SimParticles.Position)
        bᵧ = @alloc(SVector{DimensionsPlus, T}, NumberOfParticles)
        Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, T, DimensionsPlus * DimensionsPlus}, NumberOfParticles)
        GhostVelocityNumerator = @alloc(SVector{D, T}, NumberOfParticles)
        KernelSum = @alloc(T, NumberOfParticles)
        Submerged = @alloc(T, NumberOfParticles)
        NoPenShift = @alloc(SVector{D, T}, NumberOfParticles)
        NoPenCount = @alloc(T, NumberOfParticles)

        NeighborLoopUpdatedMDBC!(
            SimKernel,
            GhostData,
            SimConstants,
            ParticleRanges,
            SimParticles,
            bᵧ,
            Aᵧ,
            GhostVelocityNumerator,
            KernelSum,
            Submerged,
        )
        ApplyUpdatedMDBCCorrection(
            SimKernel,
            SimConstants,
            SimParticles,
            GhostIndices,
            bᵧ,
            Aᵧ,
            GhostVelocityNumerator,
            KernelSum,
            Submerged,
        )
        ApplyUpdatedNoPenetration!(
            SimConstants,
            GhostData,
            SimParticles,
            ParticleRanges,
            NoPenShift,
            NoPenCount,
        )
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
