"""
Floating rigid bodies (the DualSPHysics "floating" objects) on the GPU.

The particles of a body (`Type == Floating`) take part in the neighbour loops
like boundary particles, but their momentum terms are always evaluated. After
each neighbour loop the accelerations of a body's particles are summed into
the force and the torque about its centre (`launch_floating_forces!`), the
body is advanced by one stage of the predictor/corrector scheme
(`launch_floating_update!`) and its particles are placed rigidly
(`launch_floating_particles!`):

* predictor: `V½ = Vₙ + (F/M + g) dt/2`, `ω½ = ωₙ + τ/I dt/2`,
  `C½ = Cₙ + Vₙ dt/2`, particles turned by `ωₙ dt/2` about the centre;
* corrector: `Vₙ₊₁ = Vₙ + (F½/M + g) dt`, `ωₙ₊₁ = ωₙ + τ½/I dt`,
  `Cₙ₊₁ = Cₙ + V½ dt`, particles turned by `ω½ dt`.

With `SingleNeighborTimeStepping`, the predictor reuses the linear and
angular acceleration cached at the previous corrector's half state. Startup
and cell-list rebuilds initialize that cache at the accepted full state.
This includes the 3D gyroscopic acceleration at the evaluated orientation,
so the carried torque is never reduced about an updated body centre.

The force on a body is `m₀ Σ aₖ` over its particles, `m₀` being the particle
mass the neighbour loops use for every neighbour, so the fluid and the body
exchange equal and opposite forces. The mass of the body is
`RelativeWeight m₀ N`. In 2D the moment of inertia is scalar; in 3D the full
inertia tensor and quaternion orientation are used. The body state lives on the
device, so the step stays capturable as a CUDA graph.
"""
module GPUFloating

export FloatingArrays, initialize_floating_velocities!, launch_floating_forces!, launch_floating_update!,
       launch_floating_particles!, launch_floating_derivative!, floating_state, update_floating_indices!, freeze_body_state!

using CUDA
using LinearAlgebra: cross, det, dot
using StaticArrays

using ..SimulationGeometry
using ..GPUCellGrid: thread_index, compact_nonzero!
using ..GPUStepState
using ..GPUKernels: lanes_sum, store_pos_cell!, ELEMENTWISE_THREADS

include("GPUFloatingConstraints.jl")

"""
    FloatingArrays(SimGeometry, SimParticles, SimConstants; position_type, time = 0) -> NamedTuple

Device state of the floating bodies, one entry per `SPHGeometry` of type
`Floating`, in the order of `SimGeometry`. Supports both 2D and 3D particle
vectors. `active` is false when the case has none, in which case nothing is
launched. `body` maps a group marker to its body (0 for other groups).
`time` determines whether prescribed initial velocities are paused. After
construction, `initialize_floating_velocities!` applies them to device particles.
"""
@inline floating_inertia_type(::Val{2}, ::Type{T}) where {T} = T
@inline floating_inertia_type(::Val{3}, ::Type{T}) where {T} = SMatrix{3, 3, T, 9}
@inline floating_dimension(::AbstractVector{SVector{D, T}}) where {D, T} = D

function FloatingArrays(SimGeometry::Vector{SPHGeometry{D, T}}, SimParticles, SimConstants;
                        position_type::Type{TP} = T, time = zero(T)) where {D, T, TP}
    D in (2, 3) || throw(ArgumentError("floating bodies require 2D or 3D particles"))
    floating = filter(geom -> geom.Type == Floating || geom.Floating !== nothing, SimGeometry)
    for geom in floating
        (geom.Type == Floating && geom.Floating !== nothing) ||
            error("Floating bodies need `Type = Floating` together with `Floating = FloatingDetails(...)`" *
                  " (group marker $(geom.GroupMarker)).")
    end
    nbodies = length(floating)

    ngroups = max(1, Int(maximum(SimParticles.GroupMarker; init = 0)))
    nstate = max(nbodies, 1)
    InertiaType = floating_inertia_type(Val(D), T)
    body     = zeros(Int32, ngroups)
    mass     = ones(T, nstate)
    inertia  = fill(one(InertiaType), nstate)
    inertia_inv = fill(one(InertiaType), nstate)
    pause    = zeros(T, nstate)
    center   = zeros(SVector{D, TP}, nstate)
    constraints = fill(FloatingConstraints{D,T}(), nstate)
    velocity = zeros(SVector{D,T}, nstate)
    omega = zeros(D == 2 ? T : SVector{3,T}, nstate)
    count    = zeros(Int, max(nbodies, 1))
    for (b, geom) in enumerate(floating)
        g = geom.GroupMarker
        body[g] != 0 && error("Two floating bodies share the group marker $g.")
        body[g] = b
        members = findall(k -> SimParticles.GroupMarker[k] == g && SimParticles.Type[k] == Floating,
                          eachindex(SimParticles.GroupMarker))
        isempty(members) && error("The floating body with group marker $g has no particles.")
        mₖ = T(geom.Floating.RelativeWeight) * SimConstants.m₀
        positions  = SimParticles.Position[members]
        c          = sum(SVector{D, TP}.(positions)) / length(members)
        count[b]   = length(members)
        center[b]  = c
        mass[b]    = mₖ * length(members)
        if D == 2
            inertia[b] = mₖ * sum(x -> T(sum(abs2, SVector{D, TP}(x) - c)), positions)
            (isfinite(inertia[b]) && inertia[b] > zero(T)) ||
                throw(ArgumentError("Floating body $g has zero or invalid moment of inertia."))
            inertia_inv[b] = inv(inertia[b])
        else
            Ibody = zero(InertiaType)
            identity = one(InertiaType)
            for x in positions
                r = SVector{3, T}(SVector{D, TP}(x) - c)
                Ibody += mₖ * (dot(r, r) * identity - r * transpose(r))
            end
            determinant = det(Ibody)
            (isfinite(determinant) && determinant > zero(T)) ||
                throw(ArgumentError("Floating body $g has a singular or invalid 3D inertia tensor."))
            inertia[b] = Ibody
            inertia_inv[b] = inv(Ibody)
        end
        pause[b]   = T(geom.Floating.PauseTime)
        config = geom.Floating.Constraints
        if config !== nothing
            config isa FloatingConstraints{D,T} ||
                throw(ArgumentError("Floating constraints for group $g must use dimension $D and physics type $T."))
            constraints[b] = config
            if time >= pause[b]
                omega[b] = constrained_omega(zero(eltype(omega)), config, Val(D))
                velocity[b] = config.hinge ? angular_cross(omega[b], SVector{D,T}(c - typeof(c)(config.point))) : config.velocity
            end
        end
    end

    zv(E) = CUDA.zeros(E, nstate)
    common = (active = nbodies > 0, nbodies = Int32(nbodies), count = count,
              constrained = any(c -> c.active, constraints), constraints = CuArray(constraints),
              frozen = CUDA.zeros(Int32, nstate),
              body = CuArray(body), mass = CuArray(mass), inertia = CuArray(inertia),
              inertia_inv = CuArray(inertia_inv), pause = CuArray(pause),
              center = CuArray(center), center_n = CuArray(center),
              center_half = CuArray(center), velocity = CuArray(velocity),
              velocity_half = CuArray(velocity),
              acceleration = zv(SVector{D, T}),
              angular_acceleration = zv(D == 2 ? T : SVector{3, T}),
              indices = CuArray(Int32.(findall(==(Floating), SimParticles.Type))),
              force = CUDA.zeros(T, (D == 2 ? 3 : 6) * nstate))
    if D == 2
        return merge(common, (omega = CuArray(omega), omega_half = CuArray(omega),
                              angle = zv(T), turn = zv(T),
                              rotation = CuArray(fill(SVector{2, T}(0, 1), nstate))))
    end

    identity_orientation = SVector{4, T}(one(T), zero(T), zero(T), zero(T))
    orientation = fill(identity_orientation, nstate)
    return merge(common, (omega = CuArray(omega),
                          omega_half = CuArray(omega),
                          orientation = CuArray(orientation),
                          orientation_half = CuArray(orientation),
                          turn = zv(SVector{3, T}),
                          rotation = CuArray(fill(SVector{5, promote_type(T, TP)}(
                              0, 0, 0, 1, 0.5), nstate))))
end

function floating_membership_kernel!(flags, indices, n::Int32)
    j = thread_index()
    j > n && return nothing
    @inbounds flags[indices[j]] = Int32(1)
    return nothing
end

function reorder_membership_kernel!(permutation, flags, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds permutation[i] = flags[permutation[i]]
    return nothing
end

"""
    update_floating_indices!(floating, particle_types, cell_workspace; preserve_membership = false)

Refresh the compact floating-particle list after cell sorting, preserving its
device pointer for CUDA graphs. With `preserve_membership = true`, map previous
members through the current rebuild permutation, including frozen members.
Consume that permutation before any other compaction reuses workspace scratch.
"""
function update_floating_indices!(floating, particle_types, cell_workspace; preserve_membership = false)
    floating.active || return nothing
    if preserve_membership
        # Preserve identity even when other particle types share a group marker.
        # The old member flags and permutation are idle reorder scratch here.
        ws = cell_workspace
        fill!(ws.CellIDScratch, Int32(0))
        @cuda threads=256 blocks=cld(length(floating.indices), 256) floating_membership_kernel!(
            ws.CellIDScratch, floating.indices, Int32(length(floating.indices)))
        @cuda threads=256 blocks=cld(length(ws.Perm), 256) reorder_membership_kernel!(
            ws.Perm, ws.CellIDScratch, Int32(length(ws.Perm)))
        compact_nonzero!(ws, ws.Perm, floating.indices)
    else
        compact_nonzero!(cell_workspace, particle_types, floating.indices; predicate = ==(Floating))
    end
    return nothing
end

function freeze_body_state_kernel!(f)
    b = thread_index()
    b > f.nbodies && return nothing
    @inbounds if f.frozen[b] != 0
        f.velocity[b] = zero(eltype(f.velocity))
        f.velocity_half[b] = zero(eltype(f.velocity_half))
        f.acceleration[b] = zero(eltype(f.acceleration))
        f.omega[b] = zero(eltype(f.omega))
        f.omega_half[b] = zero(eltype(f.omega_half))
        f.angular_acceleration[b] = zero(eltype(f.angular_acceleration))
        f.center_half[b] = f.center[b]
        f.center_n[b] = f.center[b]
        stride = length(f.force) ÷ max(Int(f.nbodies), 1)
        for d in 1:stride
            f.force[stride * (b - 1) + d] = zero(eltype(f.force))
        end
    end
    return nothing
end

function freeze_body_state!(f)
    f === nothing && return nothing
    f.active && hasproperty(f, :frozen) || return nothing
    device = (; f.nbodies, f.frozen, f.force, f.velocity, f.velocity_half, f.acceleration,
        f.omega, f.omega_half, f.angular_acceleration, f.center, f.center_half, f.center_n)
    @cuda threads=128 blocks=cld(Int(f.nbodies), 128) freeze_body_state_kernel!(device)
    return nothing
end

"""
    floating_state(floating) -> NamedTuple of host vectors

Centre and velocity of every body (synchronizes). In 2D, also returns the
scalar `angle` and `omega`; in 3D, returns the scalar-first quaternion
`orientation` and vector `omega`.
"""
floating_state(f) = floating_state(f, Val(floating_dimension(f.center)))
floating_state(f, ::Val{2}) = (center = Array(f.center), velocity = Array(f.velocity),
                               angle = Array(f.angle), omega = Array(f.omega))
floating_state(f, ::Val{3}) = (center = Array(f.center), velocity = Array(f.velocity),
                               orientation = Array(f.orientation), omega = Array(f.omega))

#---------------------------------------------------------------
# Force and torque on every body
#---------------------------------------------------------------

@inline floating_torque(r::SVector{2, T}, a::SVector{2, T}) where {T} =
    r[1] * a[2] - r[2] * a[1]
@inline floating_torque(r::SVector{3, T}, a::SVector{3, T}) where {T} = cross(r, a)

@inline floating_torque_component(τ::T, d) where {T} = τ
@inline floating_torque_component(τ::SVector{3, T}, d) where {T} = τ[d]

function floating_force_kernel!(force::AbstractVector{T},
                                Acceleration::AbstractVector{SVector{D, T}},
                                Position::AbstractVector{SVector{D, TP}},
                                indices, ParticleType, GroupMarker, body,
                                center::AbstractVector{SVector{D, TP}},
                                m₀::T, step, nbodies::Int32, n::Int32) where {D, T, TP}
    step_active(step) || return nothing
    j = thread_index()
    a = zero(SVector{D, T})
    r = zero(SVector{D, T})
    τ = zero(D == 2 ? T : SVector{3, T})
    b = Int32(0)
    @inbounds if j <= n
        i = indices[j]
        if ParticleType[i] != Frozen
            b = body[GroupMarker[i]]
            a = Acceleration[i]
            r = SVector{D, T}(Position[i] - center[b])
            τ = floating_torque(r, a)
        end
    end
    # Every lane takes part, including padding lanes in the last warp.
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    stride = Int32(D == 2 ? 3 : 6)
    torque_components = D == 2 ? 1 : 3
    # Reduce only bodies represented in this warp, not every body in the case.
    remaining = CUDA.vote_ballot_sync(0xffffffff, b != Int32(0))
    while remaining != UInt32(0)
        k = CUDA.shfl_sync(0xffffffff, b, trailing_zeros(remaining) + 1)
        mine = b == k
        base = (k - Int32(1)) * stride
        for d in 1:D
            fd = lanes_sum(mine ? m₀ * a[d] : zero(T), Val(32))
            if lane == Int32(0) && fd != zero(T)
                @inbounds CUDA.@atomic force[base + Int32(d)] += fd
            end
        end
        for d in 1:torque_components
            τd = lanes_sum(mine ? m₀ * floating_torque_component(τ, d) : zero(T), Val(32))
            if lane == Int32(0) && τd != zero(T)
                @inbounds CUDA.@atomic force[base + Int32(D + d)] += τd
            end
        end
        remaining &= ~CUDA.vote_ballot_sync(0xffffffff, mine)
    end
    return nothing
end

"""
    launch_floating_forces!(floating, Acceleration, Position, ParticleType, GroupMarker, center, m₀, step)

Add the force and the torque about `center` (a vector of body centres) of the
accelerations of every body's particles to `floating.force`.
"""
function launch_floating_forces!(floating, Acceleration, Position, ParticleType, GroupMarker, center, m₀, step)
    n = length(floating.indices)
    n == 0 && return nothing
    T = eltype(floating.force)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) floating_force_kernel!(
        floating.force, Acceleration, Position, floating.indices, ParticleType, GroupMarker, floating.body, center,
        T(m₀), step, floating.nbodies, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Rigid body update (one thread per body)
#---------------------------------------------------------------

function floating_update_kernel!(f, step, g::T, ::Val{2}, ::Val{Final}, ::Val{Carried}) where {T, Final, Carried}
    step_active(step) || return nothing
    b = thread_index()
    b > f.nbodies && return nothing
    dt   = step_dt(step)
    time = step_time(step)
    gvec = SVector{2, T}(zero(T), -g)
    @inbounds begin
        f.frozen[b] != 0 && return nothing
        F = SVector{2, T}(f.force[3b - 2], f.force[3b - 1])
        τ = f.force[3b]
        f.force[3b - 2] = zero(T)
        f.force[3b - 1] = zero(T)
        f.force[3b]     = zero(T)
        if f.constraints[b].active
            constrained_floating_update!(f, b, F, τ, gvec, step, Val(Final), Val(Carried), Val(2))
            return nothing
        end
        held = time < f.pause[b]
        Vₙ = f.velocity[b]
        ωₙ = f.omega[b]
        a  = Carried && !Final ? f.acceleration[b] : F / f.mass[b] + gvec
        α  = Carried && !Final ? f.angular_acceleration[b] : τ / f.inertia[b]
        if Final
            f.acceleration[b] = a
            f.angular_acceleration[b] = α
        end
        if !Final
            Cₙ = f.center[b]
            f.center_n[b] = Cₙ
            if held
                f.velocity_half[b] = zero(Vₙ)
                f.omega_half[b]    = zero(T)
                f.center_half[b]   = Cₙ
                f.turn[b]          = zero(T)
            else
                f.velocity_half[b] = Vₙ + a * (dt / 2)
                f.omega_half[b]    = ωₙ + α * (dt / 2)
                f.center_half[b]   = Cₙ + Vₙ * (dt / 2)
                f.turn[b]          = ωₙ * (dt / 2)
            end
        else
            Cₙ = f.center_n[b]
            if held
                f.velocity[b] = zero(Vₙ)
                f.omega[b]    = zero(T)
                f.center[b]   = Cₙ
                f.turn[b]     = zero(T)
            else
                f.velocity[b] = Vₙ + a * dt
                f.omega[b]    = ωₙ + α * dt
                f.center[b]   = Cₙ + f.velocity_half[b] * dt
                f.turn[b]     = f.omega_half[b] * dt
                f.angle[b]   += f.turn[b]
            end
        end
        s, c = sincos(f.turn[b])
        f.rotation[b] = SVector{2, T}(s, c)
    end
    return nothing
end

@inline quaternion_conjugate(q::SVector{4, T}) where {T} =
    SVector{4, T}(q[1], -q[2], -q[3], -q[4])

@inline function quaternion_multiply(a::SVector{4, T}, b::SVector{4, T}) where {T}
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return SVector{4, T}(aw * bw - ax * bx - ay * by - az * bz,
                         aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw)
end

@inline function rotate_vector(q::SVector{4, T}, v::SVector{3, T}) where {T}
    qv = SVector{3, T}(q[2], q[3], q[4])
    return v + (T(2) * q[1]) * cross(qv, v) + T(2) * cross(qv, cross(qv, v))
end

@inline function rotation_quaternion(turn::SVector{3, T}) where {T}
    turn² = dot(turn, turn)
    if turn² < sqrt(eps(T))
        scale = T(0.5) - turn² / T(48) + turn² * turn² / T(3840)
        scalar = one(T) - turn² / T(8) + turn² * turn² / T(384)
    else
        angle = sqrt(turn²)
        s, c = sincos(angle / T(2))
        scale = s / angle
        scalar = c
    end
    return SVector{4, T}(scalar, scale * turn[1], scale * turn[2], scale * turn[3])
end

@inline function advance_orientation(q::SVector{4, T}, turn::SVector{3, T}) where {T}
    updated = quaternion_multiply(rotation_quaternion(turn), q)
    return updated / sqrt(dot(updated, updated))
end

@inline function rotation_coefficients(turnᵤ::SVector{3, U}) where {U}
    turn² = dot(turnᵤ, turnᵤ)
    if turn² < sqrt(eps(U))
        sine_scale = one(U) - turn² / U(6) + turn² * turn² / U(120)
        cosine_scale = U(0.5) - turn² / U(24) + turn² * turn² / U(720)
    else
        angle = sqrt(turn²)
        s, c = sincos(angle)
        sine_scale = s / angle
        cosine_scale = (one(U) - c) / turn²
    end
    return SVector{5, U}(turnᵤ..., sine_scale, cosine_scale)
end

@inline function rotate_offset(r::SVector{3, TP}, rotation::SVector{5, U}) where {TP, U}
    rᵤ = SVector{3, U}(r)
    turnᵤ = SVector{3, U}(rotation[1], rotation[2], rotation[3])
    return rᵤ + rotation[4] * cross(turnᵤ, rᵤ) +
           rotation[5] * cross(turnᵤ, cross(turnᵤ, rᵤ))
end

@inline function rotate_offset(r::SVector{3, TP}, turn::SVector{3, T}) where {TP, T}
    U = promote_type(TP, T)
    return rotate_offset(r, rotation_coefficients(SVector{3, U}(turn)))
end

@inline function angular_acceleration(q::SVector{4, T}, omega::SVector{3, T},
                                      torque::SVector{3, T},
                                      inertia::SMatrix{3, 3, T, 9},
                                      inertia_inv::SMatrix{3, 3, T, 9}) where {T}
    qinv = quaternion_conjugate(q)
    omega_body = rotate_vector(qinv, omega)
    torque_body = rotate_vector(qinv, torque)
    alpha_body = inertia_inv * (torque_body - cross(omega_body, inertia * omega_body))
    return rotate_vector(q, alpha_body)
end

function floating_update_kernel!(f, step, g::T, ::Val{3}, ::Val{Final}, ::Val{Carried}) where {T, Final, Carried}
    step_active(step) || return nothing
    b = thread_index()
    b > f.nbodies && return nothing
    dt   = step_dt(step)
    time = step_time(step)
    gvec = SVector{3, T}(zero(T), zero(T), -g)
    U = eltype(eltype(f.rotation))
    @inbounds begin
        f.frozen[b] != 0 && return nothing
        base = 6 * (b - 1)
        F = SVector{3, T}(f.force[base + 1], f.force[base + 2], f.force[base + 3])
        τ = SVector{3, T}(f.force[base + 4], f.force[base + 5], f.force[base + 6])
        for d in 1:6
            f.force[base + d] = zero(T)
        end
        if f.constraints[b].active
            constrained_floating_update!(f, b, F, τ, gvec, step, Val(Final), Val(Carried), Val(3))
            return nothing
        end
        held = time < f.pause[b]
        Vₙ = f.velocity[b]
        ωₙ = f.omega[b]
        qₙ = f.orientation[b]
        a = Carried && !Final ? f.acceleration[b] : F / f.mass[b] + gvec
        α = if Carried && !Final
            f.angular_acceleration[b]
        elseif Final
            angular_acceleration(f.orientation_half[b], f.omega_half[b], τ, f.inertia[b], f.inertia_inv[b])
        else
            angular_acceleration(qₙ, ωₙ, τ, f.inertia[b], f.inertia_inv[b])
        end
        if Final
            f.acceleration[b] = a
            f.angular_acceleration[b] = α
        end
        if !Final
            Cₙ = f.center[b]
            f.center_n[b] = Cₙ
            if held
                f.velocity_half[b] = zero(Vₙ)
                f.omega_half[b]    = zero(ωₙ)
                f.center_half[b]   = Cₙ
                f.orientation_half[b] = qₙ
                f.turn[b]          = zero(ωₙ)
            else
                f.velocity_half[b] = Vₙ + a * (dt / 2)
                f.omega_half[b]    = ωₙ + α * (dt / 2)
                f.center_half[b]   = Cₙ + Vₙ * (dt / 2)
                f.turn[b]          = ωₙ * (dt / 2)
                f.orientation_half[b] = advance_orientation(qₙ, f.turn[b])
            end
        else
            Cₙ = f.center_n[b]
            if held
                f.velocity[b] = zero(Vₙ)
                f.omega[b]    = zero(ωₙ)
                f.center[b]   = Cₙ
                f.turn[b]     = zero(ωₙ)
            else
                f.velocity[b] = Vₙ + a * dt
                f.omega[b]    = ωₙ + α * dt
                f.center[b]   = Cₙ + f.velocity_half[b] * dt
                f.turn[b]     = f.omega_half[b] * dt
                f.orientation[b] = advance_orientation(qₙ, f.turn[b])
            end
        end
        f.rotation[b] = rotation_coefficients(SVector{3, U}(f.turn[b]))
    end
    return nothing
end

"""
    launch_floating_update!(floating, step, g, final::Bool; carried = false)

Advance every body by the predictor (`final = false`) or the corrector stage
from the force accumulated since the last update, and clear the force. With
`carried = true`, the predictor uses the cached linear/angular acceleration;
the corrector still evaluates and caches its half-state derivative.
"""
function launch_floating_update!(floating, step, g, final::Bool; carried::Bool = false)
    return launch_floating_update!(floating, step, g, final,
                                   Val(floating_dimension(floating.center)); carried)
end

function launch_floating_update!(floating, step, g, final::Bool, ::Val{2}; carried::Bool = false)
    floating.nbodies == 0 && return nothing
    T = eltype(floating.force)
    f = floating
    device = (; f.nbodies, f.frozen, f.constraints, f.force, f.pause, f.mass, f.inertia, f.center, f.center_n, f.center_half,
                f.velocity, f.velocity_half, f.omega, f.omega_half, f.angle, f.turn, f.rotation,
                f.acceleration, f.angular_acceleration)
    @cuda threads=128 blocks=cld(Int(f.nbodies), 128) floating_update_kernel!(device, step, T(g), Val(2), Val(final), Val(carried))
    return nothing
end

function launch_floating_update!(floating, step, g, final::Bool, ::Val{3}; carried::Bool = false)
    floating.nbodies == 0 && return nothing
    T = eltype(floating.force)
    f = floating
    device = (; f.nbodies, f.frozen, f.constraints, f.force, f.pause, f.mass, f.inertia, f.inertia_inv,
                f.center, f.center_n, f.center_half, f.velocity, f.velocity_half,
                f.omega, f.omega_half, f.orientation, f.orientation_half, f.turn, f.rotation,
                f.acceleration, f.angular_acceleration)
    @cuda threads=128 blocks=cld(Int(f.nbodies), 128) floating_update_kernel!(device, step, T(g), Val(3), Val(final), Val(carried))
    return nothing
end

# Bootstrap/rebuild derivatives are evaluated at the accepted full state.
# Ordinary single-neighbor predictors reuse the previous half-state derivative.
function floating_derivative_kernel!(f, step, g::T, ::Val{D}) where {T, D}
    step_active(step) || return nothing
    b = thread_index()
    b > f.nbodies && return nothing
    gvec = SVector{D, T}(ntuple(d -> d == D ? -g : zero(T), Val(D)))
    stride = D == 2 ? 3 : 6
    @inbounds begin
        f.frozen[b] != 0 && return nothing
        base = stride * (b - 1)
        force = SVector{D, T}(ntuple(d -> f.force[base + d], Val(D)))
        if f.constraints[b].active
            torque = if D == 2
                f.force[base + 3]
            else
                SVector{3,T}(f.force[base + 4], f.force[base + 5], f.force[base + 6])
            end
            q = body_orientation(f, b, Val(D))
            a, alpha = constrained_derivative(f, b, force, torque, gvec, f.center[b], q, f.omega[b], Val(D))
            f.acceleration[b] = a
            f.angular_acceleration[b] = alpha
        elseif D == 2
            f.acceleration[b] = force / f.mass[b] + gvec
            f.angular_acceleration[b] = f.force[base + 3] / f.inertia[b]
        else
            f.acceleration[b] = force / f.mass[b] + gvec
            torque = SVector{3, T}(f.force[base + 4], f.force[base + 5], f.force[base + 6])
            f.angular_acceleration[b] = angular_acceleration(f.orientation[b], f.omega[b], torque,
                                                           f.inertia[b], f.inertia_inv[b])
        end
        for d in 1:stride
            f.force[base + d] = zero(T)
        end
    end
    return nothing
end

"""
    launch_floating_derivative!(floating, step, g)

Cache the linear and angular acceleration from the accumulated full-state
force/torque, and clear the force. Use after startup or rebuild evaluation in
the single-neighbor scheme. Corrector updates refresh the cache at the half state.
"""
function launch_floating_derivative!(floating, step, g)
    f = floating
    f.nbodies == 0 && return nothing
    D = floating_dimension(f.center)
    device = if D == 2
        (; f.nbodies, f.frozen, f.constraints, f.center, f.omega, f.force, f.mass, f.inertia, f.acceleration, f.angular_acceleration)
    else
        (; f.nbodies, f.frozen, f.constraints, f.center, f.force, f.mass, f.inertia, f.inertia_inv, f.orientation, f.omega,
           f.acceleration, f.angular_acceleration)
    end
    @cuda threads=128 blocks=cld(Int(f.nbodies), 128) floating_derivative_kernel!(device, step, eltype(f.force)(g), Val(D))
    return nothing
end

#---------------------------------------------------------------
# Rigid placement of the body particles
#---------------------------------------------------------------

function floating_particles_kernel!(PosOut, VelOut, PosIn, indices, ParticleType, GroupMarker, body, origin,
                                    center, velocity, omega, rotation, step, PosCellsOut, CellID, gridarg, H,
                                    ::Val{2}, n::Int32)
    step_active(step) || return nothing
    j = thread_index()
    j > n && return nothing
    @inbounds begin
        i = indices[j]
        ParticleType[i] == Frozen && return nothing
        b = body[GroupMarker[i]]
        r = PosIn[i] - origin[b]
        s, c = rotation[b]
        r = typeof(r)(c * r[1] - s * r[2], s * r[1] + c * r[2])
        x = center[b] + r
        PosOut[i] = x
        V = velocity[b]
        VelOut[i] = V + omega[b] * typeof(V)(-r[2], r[1])
        store_pos_cell!(PosCellsOut, i, x, CellID, gridarg, H)
    end
    return nothing
end

function floating_particles_kernel!(PosOut::AbstractVector{SVector{3, TP}},
                                    VelOut::AbstractVector{SVector{3, T}},
                                    PosIn::AbstractVector{SVector{3, TP}},
                                    indices, ParticleType, GroupMarker, body, origin,
                                    center::AbstractVector{SVector{3, TP}},
                                    velocity::AbstractVector{SVector{3, T}},
                                    omega::AbstractVector{SVector{3, T}},
                                    rotation::AbstractVector{SVector{5, U}},
                                    step, PosCellsOut, CellID, gridarg, H,
                                    ::Val{3}, n::Int32) where {T, TP, U}
    step_active(step) || return nothing
    j = thread_index()
    j > n && return nothing
    @inbounds begin
        i = indices[j]
        ParticleType[i] == Frozen && return nothing
        b = body[GroupMarker[i]]
        r = SVector{3, TP}(PosIn[i] - origin[b])
        rotated_r = rotate_offset(r, rotation[b])
        x = center[b] + rotated_r
        PosOut[i] = x
        VelOut[i] = velocity[b] + cross(omega[b], SVector{3, T}(rotated_r))
        store_pos_cell!(PosCellsOut, i, x, CellID, gridarg, H)
    end
    return nothing
end

"""
    launch_floating_particles!(floating, PosOut, VelOut, PosIn, ParticleType, GroupMarker, step, final::Bool;
                               pos_cells = nothing, CellID = nothing, grid = nothing, H = 0)

Move the body particles rigidly from their start-of-step positions `PosIn`:
turned by the stage's rotation about the start-of-step centre and carried to
the stage's centre, with the rigid velocity `V + ω × r`. The predictor writes
the half step arrays (and their cell relative form with `pos_cells`), the
corrector the start-of-step arrays of the next step.
"""
function launch_floating_particles!(floating, PosOut, VelOut, PosIn, ParticleType, GroupMarker, step,
                                    final::Bool; pos_cells = nothing, CellID = nothing, grid = nothing,
                                    H = zero(eltype(floating.force)))
    n = length(floating.indices)
    n == 0 && return nothing
    center, velocity, omega = final ? (floating.center, floating.velocity, floating.omega) :
                                      (floating.center_half, floating.velocity_half, floating.omega_half)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) floating_particles_kernel!(
        PosOut, VelOut, PosIn, floating.indices, ParticleType, GroupMarker, floating.body, floating.center_n,
        center, velocity, omega, floating.rotation, step, pos_cells, CellID, grid, H,
        Val(floating_dimension(floating.center)), Int32(n))
    return nothing
end

end # module GPUFloating
