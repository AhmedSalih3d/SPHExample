# Included inside GPUKernels. The original path keeps its vector RHS and
# arithmetic; the updated path also accumulates a Shepard velocity numerator.
@inline function mdbc_accumulate(b::SVector{N, T}, mass, W, grad, VW, velocity, j) where {N, T}
    return b + SVector{N, T}(mass * W, (mass * grad)...)
end

@inline function mdbc_accumulate(b::SMatrix{N, 2, T}, mass, W, grad, VW, velocity, j) where {N, T}
    density = SVector{N, T}(mass * W, (mass * grad)...)
    @inbounds speed = SVector{N, T}(zero(T), (VW * velocity[j])...)
    return b + hcat(density, speed)
end

@inline boundary_slot(::Nothing, types, i) = Int32(0)
@inline function boundary_slot(data, types, i)
    @inbounds return is_wall(types[i]) ? (data.slots === nothing ? i : data.slots[i]) : Int32(0)
end
@inline boundary_enabled(::Nothing, slot) = true
@inline boundary_enabled(data, slot) = @inbounds slot == 0 || data.active[slot]
@inline boundary_velocity(::Nothing, velocity, types, i, slot) = @inbounds velocity[i]
@inline boundary_velocity(data, velocity, types, i, slot) = @inbounds is_wall(types[i]) ?
    (slot == 0 ? zero(eltype(velocity)) : data.velocity[slot]) : velocity[i]
@inline boundary_viscous_velocity(::Nothing, velocity, types, i, slot) = @inbounds velocity[i]
@inline boundary_viscous_velocity(data, velocity, types, i, slot) = @inbounds is_wall(types[i]) ?
    (slot == 0 ? zero(eltype(velocity)) : data.tangent_velocity[slot]) : velocity[i]

# An infinity-norm condition estimate uses only fixed-size arithmetic, including
# on CUDA (no LAPACK/SVD). DualSPHysics uses dx^2 * norm(A, Inf) * norm(inv(A), Inf).
@inline function mdbc_matrix_norm(A::SMatrix{N, N, T}) where {N, T}
    result = zero(T)
    for i in 1:N
        row = zero(T)
        for j in 1:N
            row += abs(A[i, j])
        end
        result = max(result, row)
    end
    return result
end

@inline function updated_ghost_density(A::SMatrix{N, N, T}, b, reference_density, dx = one(T)) where {N, T}
    weight = A[1, 1]
    if !(isfinite(weight) && weight > zero(T))
        return reference_density
    end
    density = b[1] / weight
    if weight < T(0.1)
        density = max(reference_density, density)
    elseif abs(det(A)) >= T(0.001)
        inverse = inv(A)
        condition = dx^2 * mdbc_matrix_norm(A) * mdbc_matrix_norm(inverse)
        if isfinite(condition) && condition <= T(50)
            density = (inverse * b)[1]
        end
    end
    return isfinite(density) && density > zero(T) ? density : reference_density
end

@inline function cloned_boundary_state(density, normal, displacement, gravity, acceleration, c₀, ρ₀)
    # Momentum balance gives grad(P) = rho0 * (gravity - wall acceleration).
    # displacement is xb - xg: a submerged bottom wall has greater pressure.
    pressure = c₀^2 * (density - ρ₀) + ρ₀ * dot(gravity - acceleration, normal) * dot(displacement, normal)
    return ρ₀ + pressure / c₀^2, pressure
end

@inline wall_acceleration(::Nothing, groups, i, step, velocity) = zero(velocity)
@inline function wall_acceleration(motion, groups, i, step, velocity)
    @inbounds begin
        group = Int(groups[i])
        motion.has[group] && motion.move_particles[group] || return zero(velocity)
        dt = step_dt(step)
        dt > zero(dt) || return zero(velocity)
        previous_time = step_time(step) - dt
        start = motion.start[group]
        stop = start + motion.duration[group]
        previous = motion.velocity[group] * motion.direction[group] * (start <= previous_time <= stop)
        return (velocity - previous) / dt
    end
end

@inline prescribed_wall_acceleration(motion, group, coefficient_slot, zero_acc) = zero_acc
@inline prepare_prescribed_mdbc!(motion, step, fraction) = nothing

function updated_mdbc_kernel!(Density::AbstractVector{T}, Position::AbstractVector{SVector{D, TP}},
                              Pairs, Velocity, GhostPoints, GhostNormals, GhostIndex, ParticleType,
                              CellStart, gridarg, step, SimKernel, SimConstants, data, motion, groups,
                              prescribed_motion, coefficient_slot::Int32,
                              InvDensity, Pressure, ::Val{K}, ::Val{Compact}, nghost::Int32, periodicity, periodic_bounds,
                              ::Val{MirrorViscosity}) where {T, D, TP, K, Compact, MirrorViscosity}
    step_active(step) || return nothing
    grid = load_grid(gridarg)
    t = thread_index()
    owner = (t - Int32(1)) ÷ Int32(K) + Int32(1)
    lane = (t - Int32(1)) % Int32(K)
    valid = owner <= nghost
    N = D + 1
    A = zero(SMatrix{N, N, T, N * N})
    b = zero(SMatrix{N, 2, T, N * 2})
    i = Int32(1)
    slot = Int32(1)
    gp = zero(SVector{D, TP})
    @inbounds if valid
        i = GhostIndex[owner]
        slot = Compact ? owner : i
        gp = GhostPoints[slot]
        valid = ParticleType[i] != Frozen
    end
    @inbounds if valid
        # Normals in this package store the boundary-to-ghost displacement.
        # Reconstruct translating ghosts from the evaluated particle position.
        if ParticleType[i] == Moving
            gp = Position[i] + SVector{D, TP}(GhostNormals[slot])
        end
        s = cell_size(grid, SimKernel.H)
        b, A = mdbc_neighborhood(b, A, gp, Pairs, Position, grid, CellStart, s,
            Density, ParticleType, SimKernel, SimConstants.m₀, lane, Val(K), Velocity, InvDensity, periodicity, periodic_bounds)
    end
    if K > 1
        b = lanes_sum(b, Val(K))
        A = lanes_sum(A, Val(K))
    end
    @inbounds if valid && lane == Int32(0)
        (; ρ₀, c₀, g) = SimConstants
        # Sum Vj (xj-xg) dot grad(Wgj), the divergence of position.
        divergence = zero(T)
        for k in 2:N
            divergence += A[k, k]
        end
        active = isfinite(divergence) && divergence > zero(T) && A[1, 1] > zero(T)
        state_slot = data.slots === nothing ? i : owner
        data.active[state_slot] = active
        density = updated_ghost_density(A, b[:, 1], ρ₀, SimConstants.dx)
        normal = GhostNormals[slot]
        normal /= norm(normal)
        gravity = ConstructGravitySVector(normal, -g)
        # Interaction acceleration is not the prescribed wall acceleration.
        acceleration = if ParticleType[i] == Moving
            wall_acceleration(motion, groups, i, step, Velocity[i]) +
                prescribed_wall_acceleration(prescribed_motion, groups === nothing ? Int32(0) : Int32(groups[i]), coefficient_slot, zero(normal))
        else
            zero(normal)
        end
        ρ, pressure = cloned_boundary_state(density, normal, SVector{D, T}(Position[i] - gp),
                                            gravity, acceleration, c₀, ρ₀)
        Density[i] = active ? ρ : ρ₀
        store_inverse_density!(InvDensity, i, Density[i])
        p_wall = active ? pressure : zero(T)
        store_pressure!(Pressure, i, iszero(SimConstants.BackgroundPressure) ? p_wall : p_wall + SimConstants.BackgroundPressure)
        ghost_velocity = let rhs = b, weight = A[1, 1]
            active ? SVector{D, T}(ntuple(k -> rhs[k + 1, 2] / weight, Val(D))) : zero(normal)
        end
        wall_velocity = Velocity[i]
        mirrored = active ? 2 * wall_velocity - ghost_velocity : wall_velocity
        # Continuity uses the physical wall velocity. Mirroring is the no-slip
        # viscous construction, not a second normal wall velocity for density.
        data.velocity[state_slot] = wall_velocity
        # Project out only the mirrored relative normal velocity so co-moving
        # normal wall translation stays Galilean invariant in viscosity.
        data.tangent_velocity[state_slot] = active && MirrorViscosity ?
            mirrored - dot(mirrored - wall_velocity, normal) * normal : wall_velocity
        ParticleType[i] == Moving && (GhostPoints[slot] = gp)
    end
    return nothing
end

function launch_updated_mdbc!(Density, Position, Velocity, GhostPoints, GhostNormals,
                               GhostIndex, ParticleType, CellStart, grid, step, SimKernel, SimConstants,
                               data; threads = 128, lanes = Val(1), pos_cells = nothing, motion = nothing, groups = nothing,
                               prescribed_motion = nothing, prepared::Bool = false, coefficient_slot::Integer = 1,
                               fraction = 0.0,
                               compact_ghosts::Val = Val(false), inverse_density = nothing, pressure = nothing,
                               periodicity = nothing, periodic_bounds = nothing, mirror_viscosity::Bool = true)
    n = length(GhostIndex)
    n == 0 && return nothing
    if !prepared && prescribed_motion !== nothing
        prepare_prescribed_mdbc!(prescribed_motion, step, Float64(fraction))
    end
    slot = Int32(prepared ? coefficient_slot : 1)
    K = typeof(lanes).parameters[1]
    pairs = pos_cells === nothing ? Position : pos_cells
    @cuda threads=threads blocks=cld(n * K, threads) updated_mdbc_kernel!(
        Density, Position, pairs, Velocity, GhostPoints, GhostNormals, GhostIndex, ParticleType,
        CellStart, grid, step, SimKernel, SimConstants, data, motion, groups, prescribed_motion, slot,
        inverse_density, pressure, lanes, compact_ghosts, Int32(n), periodicity, periodic_bounds, Val(mirror_viscosity))
    return nothing
end
