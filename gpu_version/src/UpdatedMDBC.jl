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

@inline boundary_enabled(::Nothing, types, i) = true
@inline boundary_enabled(data, types, i) = @inbounds !is_wall(types[i]) || data.active[i]
@inline boundary_velocity(::Nothing, velocity, types, i) = @inbounds velocity[i]
@inline boundary_velocity(data, velocity, types, i) = @inbounds is_wall(types[i]) ? data.velocity[i] : velocity[i]
@inline boundary_viscous_velocity(::Nothing, velocity, types, i) = @inbounds velocity[i]
@inline boundary_viscous_velocity(data, velocity, types, i) = @inbounds is_wall(types[i]) ? data.tangent_velocity[i] : velocity[i]

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

function updated_mdbc_kernel!(Density::AbstractVector{T}, Pressure, Position::AbstractVector{SVector{D, TP}},
                              Pairs, Velocity, GhostPoints, GhostNormals, GhostIndex, ParticleType,
                              CellStart, gridarg, step, SimKernel, SimConstants, data, motion, groups,
                              ::Val{K}, nghost::Int32) where {T, D, TP, K}
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
    gp = zero(SVector{D, TP})
    @inbounds if valid
        i = GhostIndex[owner]
        gp = GhostPoints[i]
        # Normals in this package store the boundary-to-ghost displacement.
        # Reconstruct translating ghosts from the evaluated particle position.
        if ParticleType[i] == Moving
            gp = Position[i] + SVector{D, TP}(GhostNormals[i])
        end
        cg = cell_coords(gp, bin_scale(grid, SimKernel.H⁻¹))
        lg = ntuple(d -> cg[d] - grid.origin[d], Val(D))
        s = cell_size(grid, SimKernel.H)
        ref = mdbc_reference(Pairs, gp, cg, s)
        b, A = mdbc_rows(b, A, ref, lg, grid, CellStart, Pairs, s, Density, ParticleType,
                         SimKernel, SimConstants.m₀, lane, Val(K), Velocity)
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
        data.active[i] = active
        density = updated_ghost_density(A, b[:, 1], ρ₀, SimConstants.dx)
        normal = GhostNormals[i]
        normal /= norm(normal)
        gravity = ConstructGravitySVector(normal, -g)
        # Interaction acceleration is not the prescribed wall acceleration.
        acceleration = ParticleType[i] == Moving ? wall_acceleration(motion, groups, i, step, Velocity[i]) : zero(normal)
        ρ, pressure = cloned_boundary_state(density, normal, SVector{D, T}(Position[i] - gp),
                                            gravity, acceleration, c₀, ρ₀)
        Density[i] = active ? ρ : ρ₀
        Pressure[i] = active ? pressure : zero(T)
        ghost_velocity = let rhs = b, weight = A[1, 1]
            active ? SVector{D, T}(ntuple(k -> rhs[k + 1, 2] / weight, Val(D))) : zero(normal)
        end
        mirrored = active ? 2 * Velocity[i] - ghost_velocity : Velocity[i]
        data.velocity[i] = mirrored
        data.tangent_velocity[i] = mirrored - dot(mirrored, normal) * normal
        ParticleType[i] == Moving && (GhostPoints[i] = gp)
    end
    return nothing
end

function launch_updated_mdbc!(Density, Pressure, Position, Velocity, GhostPoints, GhostNormals,
                               GhostIndex, ParticleType, CellStart, grid, step, SimKernel, SimConstants,
                               data; threads = 128, lanes = Val(1), pos_cells = nothing, motion = nothing, groups = nothing)
    n = length(GhostIndex)
    n == 0 && return nothing
    K = typeof(lanes).parameters[1]
    pairs = pos_cells === nothing ? Position : pos_cells
    @cuda threads=threads blocks=cld(n * K, threads) updated_mdbc_kernel!(
        Density, Pressure, Position, pairs, Velocity, GhostPoints, GhostNormals, GhostIndex, ParticleType,
        CellStart, grid, step, SimKernel, SimConstants, data, motion, groups, lanes, Int32(n))
    return nothing
end
