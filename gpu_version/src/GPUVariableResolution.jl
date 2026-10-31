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
@inline function reconstruct_state(query, position, velocity, density, types, starts, gridarg,
        kernel, constants, limits, fine)
    grid = load_grid(gridarg)
    coordinates = cell_coords(query, kernel.H⁻¹)
    empty_value = SVector{3,Float64}(constants.ρ₀, 0, 0)
    in_grid(grid, coordinates) || return (empty_value, false, 0.0, Inf)
    cell = linear_cell(grid, coordinates)
    matrix = zero(SMatrix{3,3,Float64,9})
    rhs = zero(SMatrix{3,3,Float64,9})
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
            gradient = ∇Wᵢⱼ(kernel, q, SVector{2,eltype(density)}(displacement))
            if kind == Fixed
                saturation -= Float64(constants.m₀ / constants.ρ₀) * dot(displacement, gradient)
            elseif kind == Fluid && !buffer_particle(position[j], limits, fine)
                volume = Float64(constants.m₀) / Float64(density[j])
                saturation -= volume * dot(displacement, gradient)
                nearest = min(nearest, distance_squared)
                basis = SVector(1.0, -displacement[1] / kernel.h, -displacement[2] / kernel.h)
                value = SVector{3,Float64}(density[j], velocity[j][1], velocity[j][2])
                weight = volume * Float64(Wᵢⱼ(kernel,q))
                matrix += weight * (basis * transpose(basis))
                rhs += weight * (basis * transpose(value))
            end
        end
    end
    shepard = matrix[1,1]
    shepard > 0.05 || return (empty_value, false, saturation, nearest)
    value = SVector(rhs[1,1],rhs[1,2],rhs[1,3]) / shepard
    # h scaling makes this rank check independent of particle spacing. A thin
    # free-surface support can be singular while its Shepard value is usable.
    if abs(det(matrix)) > 1e-8 * shepard^3
        weights = matrix \ SVector(1.0,0.0,0.0)
        reconstructed = transpose(rhs) * weights
        if all(isfinite,reconstructed) && 0.7constants.ρ₀ < reconstructed[1] < 1.3constants.ρ₀
            value = reconstructed
        end
    end
    return (value, true, saturation, nearest)
end

function reconstruct_buffers_kernel!(position, velocity, density, types,
        source_position, source_velocity, source_density, source_types, starts, grid,
        kernel, constants, limits, fine, source_fine, n)
    i = thread_index()
    i > n && return nothing
    @inbounds if types[i] == Fluid && buffer_particle(position[i],limits,fine)
        value, valid, _, _ = reconstruct_state(position[i],source_position,source_velocity,
            source_density,source_types,starts,grid,kernel,constants,limits,source_fine)
        if valid
            density[i] = value[1]
            velocity[i] = SVector{2,eltype(density)}(value[2],value[3])
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
            types[j] == Fixed || continue
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
        kernel, constants, target_constants, fluid_marker, limits, source_fine, dt, n, pool = nothing)
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        point, normal = points[i], normals[i]
        if blocked_interface(point,normal,position,types,target_starts,target_grid,target_kernel,target_constants.dx)
            accumulated_mass[i] = 0
            return nothing
        end
        value, valid, saturation, nearest = reconstruct_state(point,source_position,source_velocity,
            source_density,source_types,starts,grid,kernel,constants,limits,source_fine)
        wet = saturation > 1.5 || nearest < min(constants.dx,target_constants.dx)^2
        valid && wet || return nothing
        incoming_mass = -value[1] * (value[2]*normal[1] + value[3]*normal[2]) * widths[i] * dt
        budget = accumulated_mass[i]
        budget += budget < 0 ? max(0.0,incoming_mass) : incoming_mass
        while budget >= target_constants.m₀
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
            position[slot] = point - 0.5Float64(target_constants.dx) * normal
            velocity[slot] = SVector{2,eltype(density)}(value[2],value[3])
            density[slot] = value[1]
            ids[slot] = CUDA.atomic_add!(pointer(next_id,1),Int64(2))
            groups[slot] = fluid_marker
            types[slot] = Fluid
            budget -= target_constants.m₀
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
