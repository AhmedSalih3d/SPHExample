module TimeStepping

export Δt, next_output_time, ProgressMotion, HalfTimeStep, FullTimeStep, UpdateTimeStep,
       FluidAccelerationSeries, FluidAccelerationByGroup, LoadFluidAccelerationSeriesCSV, LoadFluidAccelerationByGroupCSV,
       FluidAccelerationInputState, FluidAccelerationInputSeries, FluidAccelerationInputByGroup,
       LoadFluidAccelerationInputSeriesCSV, LoadFluidAccelerationInputByGroupCSV, EvaluateFluidAcceleration,
       RigidRotationMotionSeries, EvaluateRotationState, ApplyRigidRotationMotion!

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper
using StaticArrays: SVector
using ..SimulationEquations
using ..SimulationGeometry
using ..SimulationMetaDataConfiguration
using ..FluidAcceleration: FluidAccelerationSeries, FluidAccelerationByGroup,
                           FluidAccelerationInputState, FluidAccelerationInputSeries, FluidAccelerationInputByGroup,
                           LoadFluidAccelerationSeriesCSV, LoadFluidAccelerationByGroupCSV,
                           LoadFluidAccelerationInputSeriesCSV, LoadFluidAccelerationInputByGroupCSV,
                           EvaluateFluidAcceleration, FluidAccelerationForGroup,
                           LocateTimeInterval, Rotate2D, RotationalVelocity2D

mutable struct RigidRotationMotionSeries{D,T<:AbstractFloat}
    Times::Vector{T}
    Angles::Vector{T}
    ParticleKeys::Vector{Tuple{Int,Int}}
    ParticleIndexByKey::Dict{Tuple{Int,Int},Int}
    InitialPositions::Vector{SVector{D,T}}
    Pivot::SVector{D,T}
    CurrentParticleIndices::Vector{Int}
    IndexCacheReady::Bool
    FollowGhostNormals::Bool
    ReferenceGhostNormals::Union{Nothing,Vector{SVector{D,T}}}
    GhostReferenceReady::Bool
    Cursor::Int
end

function RigidRotationMotionSeries(
    Times::AbstractVector{T},
    Angles::AbstractVector{T},
    ParticleKeys::AbstractVector{Tuple{Int,Int}},
    InitialPositions::AbstractVector{SVector{D,T}},
    Pivot::SVector{D,T},
    ;
    FollowGhostNormals::Bool=false,
) where {D,T<:AbstractFloat}
    @assert !isempty(Times) "Rigid rotation timeline cannot be empty."
    @assert length(Times) == length(Angles) "Rotation times and angles must have the same length."
    @assert length(ParticleKeys) == length(InitialPositions) "Rigid rotation particle keys and initial positions must match."
    times_vector = collect(Times)
    angles_vector = collect(Angles)
    particle_keys_vector = collect(ParticleKeys)
    @assert length(unique(particle_keys_vector)) == length(particle_keys_vector) "Rigid rotation particle keys must be unique."
    particle_index_by_key = Dict{Tuple{Int,Int},Int}()
    @inbounds for i in eachindex(particle_keys_vector)
        particle_index_by_key[particle_keys_vector[i]] = i
    end
    @assert issorted(times_vector) "Rigid rotation times must be sorted in ascending order."
    return RigidRotationMotionSeries{D,T}(
        times_vector,
        angles_vector,
        particle_keys_vector,
        particle_index_by_key,
        collect(InitialPositions),
        Pivot,
        Int[],
        false,
        FollowGhostNormals,
        nothing,
        false,
        1,
    )
end

@inline function RefreshRigidRotationParticleIndices!(::Nothing, _SimParticles)
    return nothing
end

function RefreshRigidRotationParticleIndices!(model::RigidRotationMotionSeries{D,T}, SimParticles) where {D,T<:AbstractFloat}
    cached_indices = model.CurrentParticleIndices
    target_len = length(model.ParticleKeys)
    if length(cached_indices) != target_len
        resize!(cached_indices, target_len)
    end
    fill!(cached_indices, 0)

    @inbounds for particle_index in eachindex(SimParticles.Position)
        key = (Int(SimParticles.GroupMarker[particle_index]), SimParticles.ID[particle_index])
        initial_index = get(model.ParticleIndexByKey, key, 0)
        if initial_index != 0
            cached_indices[initial_index] = particle_index
        end
    end

    @assert all(!iszero, cached_indices) "Could not map all rigid-body particles after neighbor sorting."
    model.IndexCacheReady = true
    return nothing
end

function InitializeGhostNormalReferences!(model::RigidRotationMotionSeries{2,T}, SimParticles) where {T<:AbstractFloat}
    if !model.FollowGhostNormals || model.GhostReferenceReady
        return nothing
    end

    @assert hasproperty(SimParticles, :GhostNormals) "FollowGhostNormals=true requires SimParticles.GhostNormals."
    @assert hasproperty(SimParticles, :GhostPoints) "FollowGhostNormals=true requires SimParticles.GhostPoints."

    if !model.IndexCacheReady
        RefreshRigidRotationParticleIndices!(model, SimParticles)
    end

    references = fill(zero(SVector{2,T}), length(model.ParticleKeys))

    @inbounds for initial_index in eachindex(model.CurrentParticleIndices)
        particle_index = model.CurrentParticleIndices[initial_index]
        @assert particle_index != 0 "Could not initialize rigid-body ghost normal references."
        references[initial_index] = SimParticles.GhostNormals[particle_index]
    end

    model.ReferenceGhostNormals = references
    model.GhostReferenceReady = true
    return nothing
end

@inline function EvaluateRotationState(::Nothing, ::Type{T}, _time) where {T<:AbstractFloat}
    return zero(T), zero(T)
end

function EvaluateRotationState(model::RigidRotationMotionSeries{D,T}, ::Type{T}, time::T) where {D,T<:AbstractFloat}
    TimeSamples = model.Times
    AngleSamples = model.Angles
    LastIndex = length(TimeSamples)

    if LastIndex == 1
        model.Cursor = 1
        return AngleSamples[1], zero(T)
    end

    i = LocateTimeInterval(TimeSamples, model.Cursor, time)
    model.Cursor = i

    @inbounds begin
        t0 = TimeSamples[i]
        t1 = TimeSamples[i + 1]
        a0 = AngleSamples[i]
        a1 = AngleSamples[i + 1]
        dt = t1 - t0
        if iszero(dt)
            return a1, zero(T)
        end

        AngularSpeed = (a1 - a0) / dt
        alpha = clamp((time - t0) / dt, zero(T), one(T))
        Angle = a0 + (a1 - a0) * alpha

        return Angle, AngularSpeed
    end
end

@inline function ApplyRigidRotationMotion!(_SimParticles, ::Nothing, _FloatType, _Time)
    return nothing
end

function ApplyRigidRotationMotion!(SimParticles, model::RigidRotationMotionSeries{2,T}, ::Type{T}, time::T) where {T<:AbstractFloat}
    Angle, AngularSpeed = EvaluateRotationState(model, T, time)
    CosTheta = cos(Angle)
    SinTheta = sin(Angle)
    if !model.IndexCacheReady
        RefreshRigidRotationParticleIndices!(model, SimParticles)
    end
    InitializeGhostNormalReferences!(model, SimParticles)
    ReferenceGhostNormals = model.ReferenceGhostNormals
    ShouldRotateGhostNormals = model.FollowGhostNormals && model.GhostReferenceReady && ReferenceGhostNormals !== nothing

    @inbounds for initial_index in eachindex(model.CurrentParticleIndices)
        ParticleIndex = model.CurrentParticleIndices[initial_index]
        if ParticleIndex == 0
            continue
        end

        InitialRelativePosition = model.InitialPositions[initial_index] - model.Pivot
        RelativePosition = Rotate2D(InitialRelativePosition, CosTheta, SinTheta)
        CurrentPosition = model.Pivot + RelativePosition

        SimParticles.Position[ParticleIndex] = CurrentPosition
        SimParticles.Velocity[ParticleIndex] = RotationalVelocity2D(RelativePosition, AngularSpeed)

        if ShouldRotateGhostNormals
            InitialNormal = (ReferenceGhostNormals::Vector{SVector{2,T}})[initial_index]
            RotatedNormal = Rotate2D(InitialNormal, CosTheta, SinTheta)
            SimParticles.GhostNormals[ParticleIndex] = RotatedNormal
            SimParticles.GhostPoints[ParticleIndex] = CurrentPosition + RotatedNormal
        end
    end

    return nothing
end

function ApplyRigidRotationMotion!(_SimParticles, ::RigidRotationMotionSeries{D,T}, ::Type{T}, _Time) where {D,T<:AbstractFloat}
    throw(ArgumentError("RigidRotationMotionSeries currently supports D=2 only."))
end

"""
    Δt(max_acceleration, SimulationConstants, SPHKernel)

Calculates the adaptive time step for the simulation based on Courant-Friedrichs-Lewy (CFL)
and force-based criteria.

# Arguments
- `max_acceleration`: Maximum acceleration magnitude across particles.
- `SimulationConstants`: Struct containing simulation parameters like `c₀` (speed of sound) and `CFL` number.
- `SPHKernel`: Struct containing kernel parameters like `h` (smoothing length) and `η²`.

# Returns
- The calculated time step `dt`.
"""
function Δt(max_acceleration, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h = SPHKernel

    dt_speed = h / c₀
    dt_force = sqrt(h / max_acceleration)
    return CFL * min(dt_speed, dt_force)
end

"""
    UpdateTimeStep(AccelerationMax, SimConstants, SimKernel)

Computes and returns the updated time step based on maximum acceleration across all particles.

# Arguments
- `AccelerationMax`: Array of acceleration magnitudes for each particle.
- `SimConstants`: Struct containing simulation parameters.
- `SimKernel`: Struct containing kernel parameters.

# Returns
- The calculated adaptive time step `dt`.
"""
function UpdateTimeStep(AccelerationMax, SimConstants, SimKernel)
    max_acceleration = maximum(AccelerationMax)
    return Δt(max_acceleration, SimConstants, SimKernel)
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

function ProgressMotion(_SimParticles, _dt₂, ::Nothing, _SimMetaData)
    return nothing
end

function ProgressMotion(SimParticles, dt₂, MotionsDefinition, SimMetaData)
    @unpack Position, Velocity = SimParticles
    ParticleMarker = SimParticles.GroupMarker
    ParticleType = SimParticles.Type
    @inbounds @simd ivdep for i in eachindex(Position)
        CurrentType = ParticleType[i]
        IsDrivenType = (CurrentType == Moving) || (CurrentType == FixedMoving)
        MotionSpec = IsDrivenType ? MotionsDefinition[ParticleMarker[i]] : nothing
        if MotionSpec !== nothing
            IsWithinMotionWindow = (MotionSpec.StartTime <= SimMetaData.TotalTime) &&
                                   (SimMetaData.TotalTime <= (MotionSpec.StartTime + MotionSpec.Duration))

            PrescribedSpeed = MotionSpec.Velocity
            DirectionVector = MotionSpec.Direction
            MotionScale = IsWithinMotionWindow ? one(PrescribedSpeed) : zero(PrescribedSpeed)
            PositionScale = MotionPositionFactorValue(typeof(PrescribedSpeed), CurrentType)

            Velocity[i] = MotionScale * PrescribedSpeed * DirectionVector
            if MotionSpec.MovePosition && IsWithinMotionWindow && PositionScale != zero(PositionScale)
                Position[i] += Velocity[i] * dt₂ * PositionScale
            end
        end
    end

    return nothing
end

function HalfTimeStep(::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                      SimConstants, SimParticles, Positionₙ⁺,
                      Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂, FluidAcceleration) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
    @unpack Position, Density, Velocity, Acceleration = SimParticles
    ParticleType = SimParticles.Type
    ParticleMarker = SimParticles.GroupMarker
    AccelerationScalarType = eltype(eltype(Acceleration))

    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        gravity_vector = ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        Acceleration[i] += gravity_vector
        if ParticleType[i] == Fluid
            Acceleration[i] += FluidAccelerationForGroup(FluidAcceleration, ParticleMarker[i], Position[i], Velocity[i], gravity_vector)
        end
        Positionₙ⁺[i] = Position[i] + Velocity[i] * dt₂ * MotionLimiterFactor
        Velocityₙ⁺[i] = Velocity[i] + Acceleration[i] * dt₂ * MotionLimiterFactor
        ρₙ⁺[i] = Density[i] + dρdtI[i] * dt₂
    end

    return nothing
end

function FullTimeStep(::SimulationMetaData{D,T,NoShifting,K,B,L}, SimKernel,
                      SimConstants, SimParticles, Velocityₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt, FluidAcceleration) where {D,T,
                                                                                                         K<:KernelOutputMode,
                                                                                                         B<:MDBCMode,
                                                                                                         L<:LogMode}
    @unpack Position, Velocity, Acceleration = SimParticles
    ParticleType = SimParticles.Type
    ParticleMarker = SimParticles.GroupMarker
    AccelerationScalarType = eltype(eltype(Acceleration))
    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        gravity_vector = ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        Acceleration[i] += gravity_vector
        if ParticleType[i] == Fluid
            Acceleration[i] += FluidAccelerationForGroup(FluidAcceleration, ParticleMarker[i], Position[i], Velocity[i], gravity_vector)
        end
        Velocity[i] += Acceleration[i] * dt * MotionLimiterFactor
        Position[i] += (Velocityₙ⁺[i] * dt) * MotionLimiterFactor
    end
    return nothing
end

@inline free_surface_threshold(::Val{2}, ::Type{T}) where {T} = T(1.5)
@inline free_surface_threshold(::Val{3}, ::Type{T}) where {T} = T(2.5)
@inline free_surface_threshold(::Val{D}, ::Type{T}) where {D,T} = T(D) - T(0.5)

function FullTimeStep(::SimulationMetaData{D,T,S,K,B,L}, SimKernel, SimConstants,
                      SimParticles, Velocityₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt, FluidAcceleration) where {D,T,S<:ShiftingMode,
                                                                                         K<:KernelOutputMode,
                                                                                         B<:MDBCMode,
                                                                                         L<:LogMode}
    @unpack Position, Velocity, Acceleration = SimParticles
    ParticleType = SimParticles.Type
    ParticleMarker = SimParticles.GroupMarker
    AccelerationScalarType = eltype(eltype(Acceleration))
    A = SimConstants.A
    A_FST = free_surface_threshold(Val(D), T)
    A_FSM = T(length(first(Position)))
    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        gravity_vector = ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        Acceleration[i] += gravity_vector
        if ParticleType[i] == Fluid
            Acceleration[i] += FluidAccelerationForGroup(FluidAcceleration, ParticleMarker[i], Position[i], Velocity[i], gravity_vector)
        end
        Velocity[i] += Acceleration[i] * dt * MotionLimiterFactor

        δxᵢ = zero(Acceleration[i])
        A_FSC = clamp((∇◌rᵢ[i] - A_FST) / (A_FSM - A_FST), zero(T), one(T))
        δxᵢ = -A_FSC * A * SimKernel.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]

        Position[i] += (Velocityₙ⁺[i] * dt + δxᵢ) * MotionLimiterFactor
    end
    return nothing
end

end
