module TimeStepping

export Δt, next_output_time, ProgressMotion, HalfTimeStep, FullTimeStep, UpdateTimeStep,
       FluidAccelerationSeries, EvaluateFluidAcceleration,
       RigidRotationMotionSeries, EvaluateRotationState, ApplyRigidRotationMotion!

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper
using StaticArrays: SVector
using ..SimulationEquations
using ..SimulationGeometry
using ..SimulationMetaDataConfiguration

mutable struct FluidAccelerationSeries{D,T<:AbstractFloat}
    Times::Vector{T}
    Values::Vector{SVector{D,T}}
    Cursor::Int
end

function FluidAccelerationSeries(Times::AbstractVector{T}, Values::AbstractVector{SVector{D,T}}) where {D,T<:AbstractFloat}
    @assert !isempty(Times) "Fluid acceleration timeline cannot be empty."
    @assert length(Times) == length(Values) "Fluid acceleration times and values must have the same length."
    times_vector = collect(Times)
    values_vector = collect(Values)
    @assert issorted(times_vector) "Fluid acceleration times must be sorted in ascending order."
    return FluidAccelerationSeries{D,T}(times_vector, values_vector, 1)
end

@inline function EvaluateFluidAcceleration(::Nothing, ::Type{T}, ::Val{D}, _time) where {D,T}
    return zero(SVector{D,T})
end

function EvaluateFluidAcceleration(model::FluidAccelerationSeries{D,T}, ::Type{T}, ::Val{D}, time::T) where {D,T<:AbstractFloat}
    times = model.Times
    values = model.Values
    last_index = length(times)

    if time <= times[1]
        model.Cursor = 1
        return values[1]
    elseif time >= times[last_index]
        model.Cursor = last_index
        return values[last_index]
    end

    idx = clamp(model.Cursor, 1, last_index - 1)
    @inbounds begin
        while idx < (last_index - 1) && time > times[idx + 1]
            idx += 1
        end
        while idx > 1 && time < times[idx]
            idx -= 1
        end

        t0 = times[idx]
        t1 = times[idx + 1]
        a0 = values[idx]
        a1 = values[idx + 1]
        model.Cursor = idx

        if t1 == t0
            return a1
        end

        alpha = (time - t0) / (t1 - t0)
        return a0 + (a1 - a0) * alpha
    end
end

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
    times = model.Times
    angles = model.Angles
    last_index = length(times)

    if last_index == 1
        model.Cursor = 1
        return angles[1], zero(T)
    end

    if time <= times[1]
        model.Cursor = 1
        dt = times[2] - times[1]
        omega = dt == zero(T) ? zero(T) : (angles[2] - angles[1]) / dt
        return angles[1], omega
    elseif time >= times[last_index]
        model.Cursor = last_index - 1
        dt = times[last_index] - times[last_index - 1]
        omega = dt == zero(T) ? zero(T) : (angles[last_index] - angles[last_index - 1]) / dt
        return angles[last_index], omega
    end

    idx = clamp(model.Cursor, 1, last_index - 1)
    @inbounds begin
        while idx < (last_index - 1) && time > times[idx + 1]
            idx += 1
        end
        while idx > 1 && time < times[idx]
            idx -= 1
        end

        t0 = times[idx]
        t1 = times[idx + 1]
        a0 = angles[idx]
        a1 = angles[idx + 1]
        model.Cursor = idx

        if t1 == t0
            return a1, zero(T)
        end

        alpha = (time - t0) / (t1 - t0)
        theta = a0 + (a1 - a0) * alpha
        omega = (a1 - a0) / (t1 - t0)

        return theta, omega
    end
end

@inline function ApplyRigidRotationMotion!(_SimParticles, ::Nothing, _FloatType, _Time)
    return nothing
end

function ApplyRigidRotationMotion!(SimParticles, model::RigidRotationMotionSeries{2,T}, ::Type{T}, time::T) where {T<:AbstractFloat}
    theta, omega = EvaluateRotationState(model, T, time)
    ctheta = cos(theta)
    stheta = sin(theta)
    if !model.IndexCacheReady
        RefreshRigidRotationParticleIndices!(model, SimParticles)
    end
    InitializeGhostNormalReferences!(model, SimParticles)
    ghost_refs = model.ReferenceGhostNormals

    @inbounds for initial_index in eachindex(model.CurrentParticleIndices)
        particle_index = model.CurrentParticleIndices[initial_index]
        if particle_index == 0
            continue
        end

        rel0 = model.InitialPositions[initial_index] - model.Pivot
        rel = SVector{2,T}(
            ctheta * rel0[1] - stheta * rel0[2],
            stheta * rel0[1] + ctheta * rel0[2],
        )

        SimParticles.Position[particle_index] = model.Pivot + rel
        SimParticles.Velocity[particle_index] = SVector{2,T}(-omega * rel[2], omega * rel[1])

        if model.FollowGhostNormals && model.GhostReferenceReady && ghost_refs !== nothing
            normal0 = ghost_refs[initial_index]
            normal = SVector{2,T}(
                ctheta * normal0[1] - stheta * normal0[2],
                stheta * normal0[1] + ctheta * normal0[2],
            )
            SimParticles.GhostNormals[particle_index] = normal
            SimParticles.GhostPoints[particle_index] = SimParticles.Position[particle_index] + normal
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
        if ParticleType[i] == Moving || ParticleType[i] == FixedMoving
            motion = MotionsDefinition[ParticleMarker[i]]

            if motion !== nothing
                ShouldMove = (motion.StartTime <= SimMetaData.TotalTime) &&
                             (SimMetaData.TotalTime <= (motion.StartTime + motion.Duration))

                MotionVel = motion.Velocity
                MotionDir = motion.Direction
                MotionFactor = ShouldMove ? one(MotionVel) : zero(MotionVel)
                PositionFactor = MotionPositionFactorValue(typeof(MotionVel), ParticleType[i])

                Velocity[i] = MotionFactor * MotionVel * MotionDir
                if motion.MovePosition && ShouldMove && PositionFactor != zero(PositionFactor)
                    Position[i] += Velocity[i] * dt₂ * PositionFactor
                end
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
    AccelerationScalarType = eltype(eltype(Acceleration))

    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        Acceleration[i] += ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        if ParticleType[i] == Fluid
            Acceleration[i] += FluidAcceleration
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
    AccelerationScalarType = eltype(eltype(Acceleration))
    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        Acceleration[i] += ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        if ParticleType[i] == Fluid
            Acceleration[i] += FluidAcceleration
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
    AccelerationScalarType = eltype(eltype(Acceleration))
    A = SimConstants.A
    A_FST = free_surface_threshold(Val(D), T)
    A_FSM = T(length(first(Position)))
    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        Acceleration[i] += ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        if ParticleType[i] == Fluid
            Acceleration[i] += FluidAcceleration
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
