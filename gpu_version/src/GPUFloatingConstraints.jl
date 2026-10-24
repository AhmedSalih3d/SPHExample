# Included inside GPUFloating. All constraint evaluation stays device resident.

@inline angular_cross(w::T, r::SVector{2,T}) where {T} = w * SVector{2,T}(-r[2], r[1])
@inline angular_cross(w::SVector{3,T}, r::SVector{3,T}) where {T} = cross(w, r)
@inline body_orientation(f, b, ::Val{2}) = nothing
@inline body_orientation(f, b, ::Val{3}) = @inbounds f.orientation[b]
@inline half_orientation(f, b, ::Val{2}) = nothing
@inline half_orientation(f, b, ::Val{3}) = @inbounds f.orientation_half[b]

@inline constrained_velocity(v::SVector{D,T}, c) where {D,T} =
    SVector{D,T}(ntuple(d -> c.translation_free[d] ? v[d] : c.velocity[d], Val(D)))

@inline function constrained_omega(w::T, c, ::Val{2}) where {T}
    return c.hinge ? (c.hinge_prescribed ? c.hinge_velocity : w) :
                     (c.rotation_free[1] ? w : c.omega[1])
end

@inline function constrained_omega(w::SVector{3,T}, c, ::Val{3}) where {T}
    if c.hinge
        return c.axis * (c.hinge_prescribed ? c.hinge_velocity : dot(w, c.axis) / dot(c.axis, c.axis))
    end
    return SVector{3,T}(ntuple(d -> c.rotation_free[d] ? w[d] : c.omega[d], Val(3)))
end

@inline function world_inertia(q::SVector{4,T}, inertia) where {T}
    x = rotate_vector(q, SVector{3,T}(1, 0, 0))
    y = rotate_vector(q, SVector{3,T}(0, 1, 0))
    z = rotate_vector(q, SVector{3,T}(0, 0, 1))
    rotation = SMatrix{3,3,T,9}(x..., y..., z...)
    return rotation * inertia * transpose(rotation)
end

@inline function constrained_alpha(f, b, torque::T, q, w::T, c, ::Val{2}) where {T}
    return c.rotation_free[1] ? torque / f.inertia[b] : zero(T)
end

@inline function constrained_alpha(f, b, torque::SVector{3,T}, q, w, c, ::Val{3}) where {T}
    if all(c.rotation_free)
        return angular_acceleration(q, w, torque, f.inertia[b], f.inertia_inv[b])
    end
    inertia = world_inertia(q, f.inertia[b])
    rhs = torque - cross(w, inertia * w)
    # Solve in the allowed subspace. Masking the unconstrained acceleration
    # gives the wrong answer when the world inertia has off-diagonal entries.
    matrix = SMatrix{3,3,T,9}(ntuple(Val(9)) do i
        row = (i - 1) % 3 + 1
        col = (i - 1) ÷ 3 + 1
        c.rotation_free[row] && c.rotation_free[col] ? inertia[row,col] :
            (row == col && !c.rotation_free[row] ? one(T) : zero(T))
    end)
    load = SVector{3,T}(ntuple(d -> c.rotation_free[d] ? rhs[d] : zero(T), Val(3)))
    return matrix \ load
end

@inline function hinge_alpha(f, b, torque::T, force, gvec, r, q, c, ::Val{2}) where {T}
    c.hinge_prescribed && return zero(T)
    moment = torque + floating_torque(r, force + f.mass[b] * gvec)
    return moment / (f.inertia[b] + f.mass[b] * dot(r, r))
end

@inline function hinge_alpha(f, b, torque::SVector{3,T}, force, gvec, r, q, c, ::Val{3}) where {T}
    c.hinge_prescribed && return zero(torque)
    inertia = world_inertia(q, f.inertia[b])
    arm = cross(r, c.axis)
    moment = dot(c.axis, torque + cross(r, force + f.mass[b] * gvec))
    pivot_inertia = dot(c.axis, inertia * c.axis) + f.mass[b] * dot(arm, arm)
    return c.axis * (moment / pivot_inertia)
end

@inline function constrained_derivative(f, b, force::SVector{D,T}, torque, gvec, center, q, omega, dim) where {D,T}
    c = f.constraints[b]
    w = constrained_omega(omega, c, dim)
    if c.hinge
        r = SVector{D,T}(center - typeof(center)(c.point))
        alpha = hinge_alpha(f, b, torque, force, gvec, r, q, c, dim)
        return angular_cross(alpha, r) + angular_cross(w, angular_cross(w, r)), alpha
    end
    free_acceleration = force / f.mass[b] + gvec
    a = SVector{D,T}(ntuple(d -> c.translation_free[d] ? free_acceleration[d] : zero(T), Val(D)))
    return a, constrained_alpha(f, b, torque, q, w, c, dim)
end

@inline function rotate_hinge_offset(r::SVector{2,TP}, turn) where {TP}
    s, c = sincos(turn)
    return SVector{2,TP}(c * r[1] - s * r[2], s * r[1] + c * r[2])
end
@inline rotate_hinge_offset(r::SVector{3,TP}, turn) where {TP} = rotate_offset(r, turn)

@inline function store_constrained_rotation!(f, b, q, turn::T, ::Val{Final}, ::Val{2}) where {T,Final}
    Final && (f.angle[b] += turn)
    s, c = sincos(turn)
    f.rotation[b] = SVector{2,T}(s, c)
    return nothing
end

@inline function store_constrained_rotation!(f, b, q, turn::SVector{3,T}, ::Val{Final}, ::Val{3}) where {T,Final}
    if Final
        f.orientation[b] = advance_orientation(q, turn)
    else
        f.orientation_half[b] = advance_orientation(q, turn)
    end
    U = eltype(eltype(f.rotation))
    f.rotation[b] = rotation_coefficients(SVector{3,U}(turn))
    return nothing
end

@inline function constrained_floating_update!(f, b, force, torque, gvec::SVector{D,T}, step,
                                             final::Val{Final}, ::Val{Carried}, dim) where {D,T,Final,Carried}
    c = f.constraints[b]
    dt = step_dt(step)
    cn = Final ? f.center_n[b] : f.center[b]
    qn = body_orientation(f, b, dim)
    vn = constrained_velocity(f.velocity[b], c)
    wn = constrained_omega(f.omega[b], c, dim)
    if Carried && !Final
        a, alpha = f.acceleration[b], f.angular_acceleration[b]
    else
        ceval = Final ? f.center_half[b] : cn
        qeval = Final ? half_orientation(f, b, dim) : qn
        weval = Final ? f.omega_half[b] : wn
        a, alpha = constrained_derivative(f, b, force, torque, gvec, ceval, qeval, weval, dim)
    end
    if Final
        f.acceleration[b] = a
        f.angular_acceleration[b] = alpha
    else
        f.center_n[b] = cn
    end
    if step_time(step) < f.pause[b]
        v, w, center, turn = zero(vn), zero(wn), cn, zero(wn)
    else
        h = Final ? dt : dt / 2
        v = constrained_velocity(vn + a * h, c)
        w = constrained_omega(wn + alpha * h, c, dim)
        vmove = Final ? constrained_velocity(f.velocity_half[b], c) : vn
        wmove = Final ? constrained_omega(f.omega_half[b], c, dim) : wn
        turn = wmove * h
        center = cn + vmove * h
        if c.hinge
            pivot = typeof(cn)(c.point)
            center = pivot + rotate_hinge_offset(cn - pivot, turn)
            v = angular_cross(w, SVector{D,T}(center - pivot))
        end
    end
    f.turn[b] = turn
    if Final
        f.center[b], f.velocity[b], f.omega[b] = center, v, w
    else
        f.center_half[b], f.velocity_half[b], f.omega_half[b] = center, v, w
    end
    store_constrained_rotation!(f, b, qn, turn, final, dim)
    return nothing
end

function initial_floating_velocity_kernel!(velocities, positions, indices, groups, body, constraints, center, velocity, omega, n::Int32)
    j = thread_index()
    j > n && return nothing
    @inbounds begin
        i = indices[j]
        b = body[groups[i]]
        constraints[b].active || return nothing
        r = eltype(velocities)(positions[i] - center[b])
        velocities[i] = velocity[b] + angular_cross(omega[b], r)
    end
    return nothing
end

"""
    initialize_floating_velocities!(floating, gpu)

Set initial rigid particle velocities from the body state without moving any
particles. Call after constructing constrained bodies and before interactions
or the initial output frame. `FloatingArrays(...; time)` applies pause rules.
"""
function initialize_floating_velocities!(f, gpu)
    f.active && f.constrained || return nothing
    n = length(f.indices)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) initial_floating_velocity_kernel!(
        gpu.Velocity, gpu.Position, f.indices, gpu.GroupMarker, f.body, f.constraints, f.center, f.velocity, f.omega, Int32(n))
    return nothing
end
