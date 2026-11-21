# Included in VariableResolution. Each domain keeps dense, uniform-mass
# interaction arrays; only interface reconstruction reads the other domain.
@inline is_dormant(kind) = kind == Frozen

function copy_predictor_kernel!(x,v,rho,types,owners,groups,ghosts,normals,
        xp,vp,rhop,accepted_types,accepted_groups,accepted_ghosts,accepted_normals,n)
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        x[i],v[i],rho[i] = xp[i],vp[i],rhop[i]
        types[i],groups[i] = accepted_types[i],accepted_groups[i]
        ghosts[i],normals[i] = accepted_ghosts[i],accepted_normals[i]
        owners[i] = i
    end
    return nothing
end

function scatter_predictor_kernel!(acceleration,rate,rhop,sorted_acceleration,sorted_rate,sorted_density,owners,n)
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        original = owners[i]
        acceleration[original] = sorted_acceleration[i]
        rate[original] = sorted_rate[i]
        rhop[original] = sorted_density[i]
    end
    return nothing
end

"First-order MLS in coordinates scaled by h, with a normalized SPH fallback."
@inline function reconstruct_state(query::SVector{D}, position, velocity, density, types, starts, gridarg,
        kernel, constants, limits, fine) where D
    grid = load_grid(gridarg)
    coordinates = cell_coords(query, kernel.H⁻¹)
    N = D + 1
    empty_value = SVector{N,Float64}(constants.ρ₀, Tuple(zero(query))...)
    in_grid(grid, coordinates) || return (empty_value, false, 0.0, Inf)
    cell = linear_cell(grid, coordinates)
    matrix = zero(SMatrix{N,N,Float64,N*N})
    rhs = zero(SMatrix{N,N,Float64,N*N})
    saturation, nearest = 0.0, Inf
    for row in cell_rows(grid)
        first, last = row_range(grid, starts, cell, row)
        @inbounds for j in first:last
            kind = types[j]
            kind == Frozen && continue
            displacement = query - position[j]
            distance_squared = dot(displacement, displacement)
            distance_squared <= kernel.H² || continue
            q = eltype(density)(sqrt(distance_squared) * kernel.h⁻¹)
            gradient = ∇Wᵢⱼ(kernel, q, SVector{D,eltype(density)}(displacement))
            if is_wall(kind) || kind == Floating
                saturation -= Float64(constants.m₀ / constants.ρ₀) * dot(displacement, gradient)
            elseif kind == Fluid && !buffer_particle(position[j], limits, fine)
                volume = Float64(constants.m₀) / Float64(density[j])
                saturation -= volume * dot(displacement, gradient)
                nearest = min(nearest, distance_squared)
                basis = SVector{N,Float64}(1.0, Tuple(-displacement / kernel.h)...)
                value = SVector{N,Float64}(density[j], Tuple(velocity[j])...)
                weight = volume * Float64(Wᵢⱼ(kernel,q))
                matrix += weight * (basis * transpose(basis))
                rhs += weight * (basis * transpose(value))
            end
        end
    end
    shepard = matrix[1,1]
    shepard > 0.05 || return (empty_value, false, saturation, nearest)
    value = rhs[1,:] / shepard
    # h scaling makes this rank check independent of particle spacing. A thin
    # free-surface support can be singular while its Shepard value is usable.
    if abs(det(matrix)) > 1e-8 * shepard^N
        weights = D == 2 ? matrix \ SVector(1.0, 0.0, 0.0) : inv(matrix)[:,1]
        reconstructed = transpose(rhs) * weights
        if all(isfinite,reconstructed) && 0.7constants.ρ₀ < reconstructed[1] < 1.3constants.ρ₀
            value = reconstructed
        end
    end
    return (value, true, saturation, nearest)
end

@inline function reconstructed_velocity(value, ::Type{SVector{D,T}}) where {D,T}
    return SVector{D,T}(ntuple(d -> value[d + 1], Val(D)))
end

function reconstruct_buffers_kernel!(position, velocity, density, types,
        source_position, source_velocity, source_density, source_types, starts, grid,
        kernel, constants, limits, fine, source_fine, n, source_limits=limits)
    i = thread_index()
    i > n && return nothing
    @inbounds if types[i] == Fluid && buffer_particle(position[i],limits,fine)
        value, valid, _, _ = reconstruct_state(position[i],source_position,source_velocity,
            source_density,source_types,starts,grid,kernel,constants,source_limits,source_fine)
        if valid
            density[i] = value[1]
            velocity[i] = reconstructed_velocity(value, eltype(velocity))
        end
    end
    return nothing
end

@inline function blocked_interface(query, normal, position, types, starts, gridarg, kernel, dx)
    grid = load_grid(gridarg)
    coordinates = cell_coords(query,kernel.H⁻¹)
    in_grid(grid,coordinates) || return false
    cell = linear_cell(grid,coordinates)
    for row in cell_rows(grid)
        first, last = row_range(grid,starts,cell,row)
        @inbounds for j in first:last
            (is_wall(types[j]) || types[j] == Floating) || continue
            displacement = position[j] - query
            distance_squared = dot(displacement,displacement)
            if distance_squared < dx^2 && (iszero(distance_squared) ||
                    dot(normal,displacement) < sqrt(0.5distance_squared))
                return true
            end
        end
    end
    return false
end

function interface_flux_kernel!(points, normals, widths, accumulated_mass,
        position, velocity, density, types, ids, groups, free_slots, allocation, next_id, events,
        target_starts, target_grid, target_kernel,
        source_position, source_velocity, source_density, source_types, starts, grid,
        kernel, constants, target_constants, fluid_marker, limits, source_fine, dt, n, pool = nothing,
        source_limits=limits, id_stride=Int64(2))
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        point,normal,interface_velocity = resolution_face(limits,points[i],normals[i],dt)
        if blocked_interface(point,normal,position,types,target_starts,target_grid,target_kernel,target_constants.dx)
            accumulated_mass[i] = 0
            return nothing
        end
        value, valid, saturation, nearest = reconstruct_state(point,source_position,source_velocity,
            source_density,source_types,starts,grid,kernel,constants,source_limits,source_fine)
        wet = saturation > 1.5 || nearest < min(constants.dx,target_constants.dx)^2
        valid && wet || return nothing
        incoming_mass = -value[1] * dot(reconstructed_velocity(value, typeof(point)) - interface_velocity,
            normal) * widths[i] * dt
        m0 = Float64(target_constants.m₀)
        dx = Float64(target_constants.dx)
        denom = value[1] * widths[i]
        budget = max(accumulated_mass[i] + incoming_mass, -0.25m0)
        while budget >= m0
            request = CUDA.atomic_add!(pointer(allocation,1),Int32(1)) + Int32(1)
            if request > (pool === nothing ? length(free_slots) : length(types) - pool[1])
                CUDA.atomic_xchg!(pointer(allocation,2),Int32(1))
                break
            end
            slot = pool === nothing ? free_slots[request] : pool[1] + request
            if slot == 0
                CUDA.atomic_xchg!(pointer(allocation,2),Int32(1))
                break
            end
            inward = 0.5dx + (denom > 0 ? clamp((budget - m0) / denom, 0.0, dx) : 0.0)
            position[slot] = point - inward * normal
            velocity[slot] = reconstructed_velocity(value, eltype(velocity))
            density[slot] = value[1]
            ids[slot] = CUDA.atomic_add!(pointer(next_id,1),id_stride)
            groups[slot] = fluid_marker
            types[slot] = Fluid
            budget -= m0
            CUDA.atomic_add!(pointer(events,1),Int64(1))
        end
        accumulated_mass[i] = budget
    end
    return nothing
end

function retire_particles_kernel!(position, velocity, density, types, events, limits, fine, reference_density, n)
    i = thread_index()
    i > n && return nothing
    @inbounds if types[i] == Fluid
        nonfinite = !all(isfinite,position[i]) || !all(isfinite,velocity[i]) || !isfinite(density[i])
        density_out = !nonfinite && (density[i] < 0.7reference_density || density[i] > 1.3reference_density)
        if nonfinite || density_out || outside_domain(position[i],limits,fine)
            types[i] = Frozen
            position[i] = zero(eltype(position))
            velocity[i] = zero(eltype(velocity))
            CUDA.atomic_add!(pointer(events,nonfinite ? 4 : density_out ? 3 : 2),Int64(1))
        end
    end
    return nothing
end
