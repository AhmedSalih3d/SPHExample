# Included in GPUFloating. Native 3D joint/contact projection; no host step solve.
using ..Constraints: RigidBodyDynamics, RigidContacts
using ..GPUCellGrid: load_grid, cell_rows, row_range
using Adapt

struct DeviceRigidJoint{T,TP}
    kind::UInt8
    a::Int32
    b::Int32
    ra::SVector{3,TP}
    rb::SVector{3,TP}
    axis_a::SVector{3,T}
    axis_b::SVector{3,T}
end

# Greedy edge colouring: a colour never writes the same dynamic body twice.
function rigid_edge_colours(edges; indices = eachindex(edges))
    occupied = Set{Int}[]
    groups = Vector{Int32}[]
    for (i, (a, b)) in enumerate(edges)
        colour = findfirst(s -> !(a in s) && (b == 0 || !(b in s)), occupied)
        if colour === nothing
            push!(occupied, Set{Int}())
            push!(groups, Int32[])
            colour = length(groups)
        end
        push!(occupied[colour], a)
        b == 0 || push!(occupied[colour], b)
        push!(groups[colour], Int32(indices[i]))
    end
    order, starts = Int32[], Int32[1]
    for group in groups
        append!(order, group)
        push!(starts, Int32(length(order) + 1))
    end
    return CuArray(order), CuArray(starts), Int32(length(groups))
end

function attach_rigid_dynamics(f, specification::RigidBodyDynamics, particles, kernel; gravity = 0)
    floating_dimension(f.center) == 2 && return attach_planar_rigid_dynamics(f,specification,particles,kernel;gravity)
    floating_dimension(f.center) == 3 || throw(ArgumentError("native multibody dynamics currently requires 3D"))
    f.active || throw(ArgumentError("native multibody dynamics needs at least one floating body"))
    f.constrained && throw(ArgumentError("use native joints instead of per-body FloatingConstraints in multibody mode"))
    T, TP = eltype(f.mass), eltype(eltype(f.center))
    solid_only = all(!=(Fluid), particles.Type)
    specification.max_timestep === nothing || solid_only || throw(ArgumentError("native MaxTimeStep requires a case with no fluid; coupled fluid retains the SPH timestep"))
    specification.max_timestep === nothing || (isfinite(T(specification.max_timestep)) && T(specification.max_timestep) > 0) ||
        throw(ArgumentError("MaxTimeStep must be positive and representable in $T"))
    centres, body = Array(f.center), Array(f.body)
    sort(Int.(particles.ID)) == collect(1:length(particles)) || throw(ArgumentError("native rigid dynamics requires dense IDs 1:N"))
    joints = DeviceRigidJoint{T,TP}[]
    for joint in specification.joints
        a = joint.body_a <= length(body) ? body[joint.body_a] : Int32(0)
        b = joint.body_b == 0 ? Int32(0) : joint.body_b <= length(body) ? body[joint.body_b] : Int32(0)
        a != 0 && (joint.body_b == 0 || b != 0) || throw(ArgumentError("joint markers must identify floating groups (or zero for world)"))
        push!(joints, DeviceRigidJoint(joint.kind, a, b,
            SVector{3,TP}(joint.attachment) - centres[a],
            b == 0 ? SVector{3,TP}(joint.anchor) : SVector{3,TP}(joint.anchor) - centres[b],
            SVector{3,T}(joint.axis), SVector{3,T}(joint.axis)))
    end
    joint_order, joint_starts, joint_colours = rigid_edge_colours([(Int(j.a), Int(j.b)) for j in joints])
    contact = specification.contacts
    references = zeros(SVector{3,TP}, length(particles))
    for i in eachindex(particles.ID)
        b = body[particles.GroupMarker[i]]
        b == 0 || (references[particles.ID[i]] = SVector{3,TP}(particles.Position[i]) - centres[b])
    end
    pairs = SVector{2,Int32}[]
    fixed_enabled = ones(UInt8, length(body))
    floating_enabled = contact === nothing ? UInt8[] : ones(UInt8, Int(f.nbodies))
    friction, restitution = T[], T[]
    if contact !== nothing
        any(==(Moving), particles.Type) && throw(ArgumentError("rigid contacts currently support static Fixed surfaces, not Moving particles"))
        if contact.fixed_groups !== nothing
            fill!(fixed_enabled, 0)
            for group in contact.fixed_groups
                group <= length(body) || throw(ArgumentError("unknown fixed contact group $group"))
                members = findall(==(group), particles.GroupMarker)
                !isempty(members) && all(==(Fixed), particles.Type[members]) || throw(ArgumentError("fixed contact group $group must be exclusively Fixed"))
                fixed_enabled[group] = 1
            end
        end
        if contact.floating_groups !== nothing
            fill!(floating_enabled, 0)
            for group in contact.floating_groups
                0 < group <= length(body) && body[group] != 0 || throw(ArgumentError("unknown floating contact group $group"))
                floating_enabled[body[group]] = 1
            end
        end
        body_materials = fill((contact.friction,contact.restitution), Int(f.nbodies))
        for (group,material) in contact.materials
            group <= length(body) && body[group] != 0 || throw(ArgumentError("material group $group must identify a floating body"))
            body_materials[body[group]] = (material.friction,material.restitution)
        end
        2(contact.diameter + contact.margin) <= Float64(kernel.H) * (1 + 8eps(T)) ||
            throw(ArgumentError("contact diameter + margin must not exceed half the SPH search support"))
        for a in 1:Int(f.nbodies), b in 0:a-1
            push!(pairs, SVector{2,Int32}(a, b))
            other = b == 0 ? (contact.fixed_friction,contact.fixed_restitution) : body_materials[b]
            push!(friction,T(min(body_materials[a][1],other[1])))
            push!(restitution,T(min(body_materials[a][2],other[2])))
        end
    end
    all(isfinite,friction) && all(isfinite,restitution) || throw(ArgumentError("material coefficients must be representable in $T"))
    active_pairs = findall(p -> floating_enabled[p[1]] != 0 && (p[2] == 0 || floating_enabled[p[2]] != 0),pairs)
    pair_order, pair_starts, pair_colours = rigid_edge_colours([(Int(pairs[i][1]),Int(pairs[i][2])) for i in active_pairs]; indices = active_pairs)
    slots = 24length(pairs)
    parameters = contact === nothing ? (; distance = zero(T), search = zero(T), friction = zero(T), fixed_friction = zero(T),
        restitution = zero(T), fixed_restitution = zero(T), bounce_threshold = zero(T)) :
        (; distance = T(contact.diameter - contact.slop), search = T(contact.diameter + contact.margin),
            friction = T(contact.friction), fixed_friction = T(contact.fixed_friction), restitution = T(contact.restitution),
            fixed_restitution = T(contact.fixed_restitution), bounce_threshold = T(contact.bounce_threshold))
    all(isfinite, values(parameters)) || throw(ArgumentError("contact parameters must be representable in $T"))
    contact === nothing || parameters.distance > 0 || throw(ArgumentError("contact distance must remain positive in $T"))
    radii = zeros(T, Int(f.nbodies))
    for i in eachindex(particles.Position)
        b = body[particles.GroupMarker[i]]
        b == 0 || (radii[b] = max(radii[b], T(sqrt(sum(abs2, SVector{3,TP}(particles.Position[i]) - centres[b])))))
    end
    mechanics = (; joints = CuArray(joints), joint_order, joint_starts, joint_colours,
        pairs = CuArray(pairs), pair_order, pair_starts, pair_colours,
        references = CuArray(references), fixed_enabled = CuArray(fixed_enabled), floating_enabled = CuArray(floating_enabled),
        contact_friction = CuArray(friction), contact_restitution = CuArray(restitution), old_orientation = similar(f.orientation),
        free_velocity = similar(f.velocity), free_omega = similar(f.omega),
        half_impulse = CUDA.zeros(SVector{3,T}, Int(f.nbodies)),
        half_angular_impulse = CUDA.zeros(SVector{3,T}, Int(f.nbodies)),
        reaction_force = CUDA.zeros(SVector{3,T}, Int(f.nbodies)),
        reaction_torque = CUDA.zeros(SVector{3,T}, Int(f.nbodies)),
        velocity_inertia = similar(f.inertia_inv),
        winners = CUDA.fill(typemax(UInt64), slots), partner = CUDA.zeros(Int32, slots),
        contact_order = CUDA.zeros(Int32, slots), contact_count = CUDA.zeros(Int32, length(pairs)),
        normal_impulse = CUDA.zeros(T, slots), tangent_impulse = CUDA.zeros(SVector{3,T}, slots),
        bounce = CUDA.zeros(T, slots), iterations = Int32(specification.iterations),
        has_contacts = contact !== nothing, solid_only, parameters, radii = CuArray(radii),
        maximum_timestep = specification.max_timestep === nothing ? 0.0 : specification.max_timestep)
    configuration = (; backend = :native_cuda, joints = length(joints), bodies = Int(f.nbodies),
        iterations = specification.iterations, contacts = contact !== nothing,
        contact_samples_per_pair = 24, contact_parameters = contact, precision = nameof(T), position_precision = nameof(TP),
        solid_interactions = :native_constraints_and_contacts, max_timestep = specification.max_timestep,
        cache_velocity_inertia = true)
    gravity_force = CuArray([SVector{3,T}(0, 0, -T(gravity) * mass) for mass in Array(f.mass)])
    return attach_rigid_torsions(merge(f, (; mechanics, rigid_configuration = configuration,
        constraint_force = mechanics.reaction_force, constraint_torque = mechanics.reaction_torque, gravity_force)),specification)
end

struct NativeRigidTimeStep{B,R,T}
    body::B
    radius::R
    maximum::T
    travel::T
    gravity::T
end
Adapt.@adapt_structure NativeRigidTimeStep

function rigid_timestep_controller(f, constants)
    hasproperty(f, :mechanics) && f.mechanics.maximum_timestep > 0 || return nothing
    d = f.mechanics
    distance = d.has_contacts ? d.parameters.distance : constants.dx
    body = (; f.velocity, f.omega, f.acceleration, f.angular_acceleration)
    T = eltype(f.mass)
    return NativeRigidTimeStep(body, d.radii, T(d.maximum_timestep), T(constants.CFL * distance), T(constants.g))
end

@inline function GPUStepState.controlled_timestep(controller::NativeRigidTimeStep, acoustic_dt)
    dt = controller.maximum
    for b in eachindex(controller.radius)
        @inbounds begin
            radius = controller.radius[b]
            v, w = controller.body.velocity[b], controller.body.omega[b]
            speed = sqrt(dot(v, v)) + radius * sqrt(dot(w, w))
            a, alpha = controller.body.acceleration[b], controller.body.angular_acceleration[b]
            acceleration = max(controller.gravity, sqrt(dot(a, a))) + radius * (sqrt(dot(alpha, alpha)) + dot(w, w))
            speed > 0 && (dt = min(dt, controller.travel / speed))
            acceleration > 0 && (dt = min(dt, sqrt(controller.travel / acceleration)))
        end
    end
    return dt
end

function rigid_configuration(configuration, floating)
    hasproperty(floating, :rigid_configuration) || return configuration
    return merge(configuration, (; effective = merge(configuration.effective,
        (; rigid_body_dynamics = floating.rigid_configuration))))
end

@inline function rigid_mobility(f, b, step)
    b == 0 && return zero(eltype(f.mass))
    hasproperty(f, :inverse_mass) && return @inbounds f.inverse_mass[b]
    @inbounds return f.frozen[b] != 0 || step_time(step) < f.pause[b] ? zero(eltype(f.mass)) : inv(f.mass[b])
end
@inline function rigid_inverse_inertia(f, b, q, vector, step)
    b == 0 && return zero(vector)
    hasproperty(f, :world_inverse_inertia) && return @inbounds f.world_inverse_inertia[b] * vector
    rigid_mobility(f, b, step) == 0 && return zero(vector)
    @inbounds orientation = q[b]
    local_vector = rotate_vector(quaternion_conjugate(orientation), vector)
    @inbounds return rotate_vector(orientation, f.inertia_inv[b] * local_vector)
end

@inline function rigid_basis(axis::SVector{3,T}) where T
    seed = abs(axis[1]) < T(0.8) ? SVector{3,T}(1, 0, 0) : SVector{3,T}(0, 1, 0)
    first = cross(axis, seed)
    first /= sqrt(dot(first, first))
    return first, cross(axis, first)
end

@inline function rigid_arm(orientation, b, offset::SVector{3,TP}) where TP
    # CoM/anchor subtraction retains position precision. Rotations and the
    # local lever arm use the selected physics precision, as do inertia and
    # impulses; promoting every rotation to TP is costly on GeForce FP64.
    T = eltype(eltype(orientation))
    b == 0 && return zero(SVector{3,T})
    @inbounds return rotate_vector(orientation[b], SVector{3,T}(offset))
end

@inline rigid_pair_index(a, b) = (a * (a - Int32(1))) ÷ Int32(2) + b + Int32(1)

@inline function rigid_particle_point(f, d, positions, ids, groups, center, q, i)
    @inbounds b = f.body[groups[i]]
    @inbounds return b == 0 ? positions[i] : rigid_point(center, q, b, d.references[ids[i]])
end

@inline function rigid_contact_bin(delta::SVector{3,T}, relative) where T
    axis = abs(delta[1]) >= abs(delta[2]) ? (abs(delta[1]) >= abs(delta[3]) ? 1 : 3) : (abs(delta[2]) >= abs(delta[3]) ? 2 : 3)
    u, v = mod1(axis + 1, 3), mod1(axis + 2, 3)
    face = 2(axis - 1) + (delta[axis] < 0 ? 1 : 0)
    return Int32(4face + (relative[u] >= 0 ? 2 : 0) + (relative[v] >= 0 ? 1 : 0) + 1)
end

function rigid_reset_kernel!(f, d, step)
    step_active(step) || return nothing
    i = thread_index()
    @inbounds if i <= f.nbodies
        d.old_orientation[i] = f.orientation[i]
    end
    @inbounds if i <= length(d.winners)
        d.winners[i] = typemax(UInt64)
        d.partner[i] = 0
        d.normal_impulse[i] = 0
        d.tangent_impulse[i] = zero(eltype(d.tangent_impulse))
        d.bounce[i] = 0
    end
    return nothing
end

function rigid_candidates_kernel!(f, d, positions, ids, groups, types, indices, cells, starts, gridarg, center, q, step)
    step_active(step) || return nothing
    t = thread_index()
    t > length(indices) && return nothing
    @inbounds i = indices[t]
    @inbounds types[i] == Frozen && return nothing
    @inbounds a = f.body[groups[i]]
    @inbounds d.floating_enabled[a] == 0 && return nothing
    grid = load_grid(gridarg)
    xi = rigid_particle_point(f, d, positions, ids, groups, center, q, i)
    @inbounds cell = cells[i]
    for row in cell_rows(grid)
        lo, hi = row_range(grid, starts, cell, row)
        for j in lo:hi
            @inbounds kind = types[j]
            @inbounds if kind == Floating || (kind == Fixed && d.fixed_enabled[groups[j]] != 0)
                @inbounds b = f.body[groups[j]]
                if a > b && (b == 0 || d.floating_enabled[b] != 0)
                    xj = rigid_particle_point(f, d, positions, ids, groups, center, q, j)
                    delta = SVector{3,eltype(f.mass)}(xi - xj)
                    distance2 = dot(delta, delta)
                    if distance2 <= d.parameters.search^2
                        bin = rigid_contact_bin(delta, xi - center[a])
                        slot = 24(rigid_pair_index(a, b) - 1) + bin
                        # Positive Float32 distance bits sort monotonically. The
                        # particle index resolves equal-distance ties deterministically.
                        token = (UInt64(reinterpret(UInt32, Float32(distance2))) << 32) | UInt64(i)
                        CUDA.@atomic d.winners[slot] = min(d.winners[slot], token)
                    end
                end
            end
        end
    end
    return nothing
end

@inline function rigid_contact_geometry(f, d, positions, ids, groups, center, q, a, b, i, j)
    T = eltype(f.mass)
    xi = rigid_particle_point(f, d, positions, ids, groups, center, q, i)
    xj = rigid_particle_point(f, d, positions, ids, groups, center, q, j)
    delta = SVector{3,T}(xi - xj)
    distance = sqrt(dot(delta, delta))
    n = distance > 0 ? delta / distance : SVector{3,T}(0, 0, 1)
    # A common world-space impulse point conserves angular momentum as well
    # as linear momentum, including tangential impulses on particle surfaces.
    point = (xi + xj) / 2
    ra = SVector{3,T}(point - center[a])
    rb = b == 0 ? zero(ra) : SVector{3,T}(point - center[b])
    return n, distance - d.parameters.distance, ra, rb
end

function rigid_partners_kernel!(f, d, positions, ids, groups, types, cells, starts, gridarg, center, q, velocity, omega, step)
    step_active(step) || return nothing
    slot = thread_index()
    slot > length(d.winners) && return nothing
    @inbounds token = d.winners[slot]
    token == typemax(UInt64) && return nothing
    i = Int32(token & UInt64(0xffffffff))
    pair = (slot - 1) ÷ 24 + 1
    bin = Int32((slot - 1) % 24 + 1)
    @inbounds a, b = d.pairs[pair]
    grid = load_grid(gridarg)
    xi = rigid_particle_point(f, d, positions, ids, groups, center, q, i)
    best = d.parameters.search^2
    partner = Int32(0)
    @inbounds cell = cells[i]
    for row in cell_rows(grid)
        lo, hi = row_range(grid, starts, cell, row)
        for j in lo:hi
            @inbounds kind = types[j]
            @inbounds if (kind == Floating || (kind == Fixed && d.fixed_enabled[groups[j]] != 0)) && f.body[groups[j]] == b
                xj = rigid_particle_point(f, d, positions, ids, groups, center, q, j)
                delta = SVector{3,eltype(f.mass)}(xi - xj)
                distance2 = dot(delta, delta)
                same_bin = rigid_contact_bin(delta, xi - center[a]) == bin
                if same_bin && (distance2 < best || (distance2 == best && (partner == 0 || j < partner)))
                    best, partner = distance2, j
                end
            end
        end
    end
    @inbounds d.partner[slot] = partner
    if partner != 0
        n, gap, ra, rb = rigid_contact_geometry(f, d, positions, ids, groups, center, q, a, b, i, partner)
        vn = dot(rigid_point_velocity(velocity, omega, a, ra) - rigid_point_velocity(velocity, omega, b, rb), n)
        @inbounds restitution = d.contact_restitution[pair]
        tolerance = eltype(f.mass)(1e-5) * d.parameters.distance
        imminent = gap <= max(tolerance, -vn * eltype(f.mass)(step_dt(step)))
        @inbounds d.bounce[slot] = imminent && vn < -d.parameters.bounce_threshold ? -restitution * vn : zero(vn)
    end
    return nothing
end

function rigid_pack_contacts_kernel!(d, step)
    step_active(step) || return nothing
    pair = thread_index()
    pair > length(d.pairs) && return nothing
    count = Int32(0)
    for sample in Int32(1):Int32(24)
        slot = Int32(24) * (pair - Int32(1)) + sample
        @inbounds if d.partner[slot] != 0
            count += Int32(1)
            d.contact_order[Int32(24) * (pair - Int32(1)) + count] = slot
        end
    end
    @inbounds d.contact_count[pair] = count
    return nothing
end

@inline function rigid_tangent_mass(f, q, a, b, ra, rb, n1, n2, step)
    return (rigid_mobility(f, a, step) + rigid_mobility(f, b, step)) * dot(n1, n2) +
        dot(cross(ra, n1), rigid_inverse_inertia(f, a, q, cross(ra, n2), step)) +
        dot(cross(rb, n1), rigid_inverse_inertia(f, b, q, cross(rb, n2), step))
end

@inline function solve_rigid_contact!(f, d, positions, ids, groups, center, q, velocity, omega, a, b, slot, step, position)
    @inbounds j = d.partner[slot]
    j == 0 && return nothing
    @inbounds i = Int32(d.winners[slot] & UInt64(0xffffffff))
    n, gap, ra, rb = rigid_contact_geometry(f, d, positions, ids, groups, center, q, a, b, i, j)
    if position
        if gap < 0
            mobility = rigid_effective_mass(f, q, a, b, ra, rb, n, step)
            mobility > 0 || return nothing
            impulse = (-gap / mobility) * n
            rigid_apply_position!(f, center, q, a, ra, impulse, step)
            rigid_apply_position!(f, center, q, b, rb, -impulse, step)
        end
    else
        T = eltype(f.mass)
        relative = rigid_point_velocity(velocity, omega, a, ra) - rigid_point_velocity(velocity, omega, b, rb)
        # Speculative contacts permit closure of a positive gap within dt;
        # restitution is applied only to actual impacts, using incident speed.
        @inbounds target = d.bounce[slot] > 0 ? d.bounce[slot] : (gap > 0 ? -gap / T(step_dt(step)) : zero(T))
        @inbounds previous = d.normal_impulse[slot]
        normal_speed = dot(relative, n)
        @inbounds old_tangent = d.tangent_impulse[slot]
        previous == 0 && normal_speed >= target && iszero(old_tangent) && return nothing
        mobility = rigid_effective_mass(f, q, a, b, ra, rb, n, step)
        mobility > 0 || return nothing
        total = max(zero(T), previous + (target - normal_speed) / mobility)
        impulse = (total - previous) * n
        @inbounds d.normal_impulse[slot] = total
        rigid_apply_velocity!(f, velocity, omega, q, a, ra, impulse, step)
        rigid_apply_velocity!(f, velocity, omega, q, b, rb, -impulse, step)
        total == 0 && iszero(old_tangent) && return nothing
        relative = rigid_point_velocity(velocity, omega, a, ra) - rigid_point_velocity(velocity, omega, b, rb)
        t1, t2 = rigid_basis(n)
        k11 = rigid_tangent_mass(f, q, a, b, ra, rb, t1, t1, step)
        k12 = rigid_tangent_mass(f, q, a, b, ra, rb, t1, t2, step)
        k22 = rigid_tangent_mass(f, q, a, b, ra, rb, t2, t2, step)
        determinant = k11 * k22 - k12^2
        if determinant > 0
            v1, v2 = dot(relative, t1), dot(relative, t2)
            @inbounds old = d.tangent_impulse[slot]
            candidate = old - dot(old, n) * n - ((k22 * v1 - k12 * v2) * t1 + (k11 * v2 - k12 * v1) * t2) / determinant
            @inbounds friction = d.contact_friction[rigid_pair_index(a,b)]
            limit = friction * total
            magnitude = sqrt(dot(candidate, candidate))
            magnitude > limit && (candidate *= limit / magnitude)
            @inbounds d.tangent_impulse[slot] = candidate
            impulse = candidate - old
            rigid_apply_velocity!(f, velocity, omega, q, a, ra, impulse, step)
            rigid_apply_velocity!(f, velocity, omega, q, b, rb, -impulse, step)
        end
    end
    return nothing
end

rigid_shared_bytes(f) = Int(f.nbodies) * (sizeof(eltype(f.center)) + sizeof(eltype(f.orientation)) +
    2sizeof(eltype(f.velocity)) + 2sizeof(eltype(f.inertia_inv)) + 2sizeof(eltype(f.mass)))

@inline function rigid_shared_state(f, center, q, velocity, omega, step)
    n = f.nbodies
    offset = 0
    local_center = CuDynamicSharedArray(eltype(center), n, offset)
    offset += sizeof(eltype(center)) * n
    local_q = CuDynamicSharedArray(eltype(q), n, offset)
    offset += sizeof(eltype(q)) * n
    local_velocity = CuDynamicSharedArray(eltype(velocity), n, offset)
    offset += sizeof(eltype(velocity)) * n
    local_omega = CuDynamicSharedArray(eltype(omega), n, offset)
    offset += sizeof(eltype(omega)) * n
    inertia_inv = CuDynamicSharedArray(eltype(f.inertia_inv), n, offset)
    offset += sizeof(eltype(f.inertia_inv)) * n
    mass = CuDynamicSharedArray(eltype(f.mass), n, offset)
    offset += sizeof(eltype(f.mass)) * n
    inverse_mass = CuDynamicSharedArray(eltype(f.mass), n, offset)
    offset += sizeof(eltype(f.mass)) * n
    velocity_inertia = CuDynamicSharedArray(eltype(f.inertia_inv), n, offset)
    b = threadIdx().x
    while b <= n
        @inbounds begin
            local_center[b], local_q[b] = center[b], q[b]
            local_velocity[b], local_omega[b] = velocity[b], omega[b]
            inertia_inv[b], mass[b] = f.inertia_inv[b], f.mass[b]
            inverse_mass[b] = rigid_mobility(f, b, step)
        end
        b += blockDim().x
    end
    sync_threads()
    return merge(f, (; inertia_inv, mass, inverse_mass)), local_center, local_q, local_velocity, local_omega, velocity_inertia
end

@inline function rigid_project_iterations!(f, d, positions, ids, groups, center, q, velocity, omega, step, ::Val{Position}) where Position
    thread = threadIdx().x
    for iteration in 1:d.iterations
        for colour in 1:d.joint_colours
            @inbounds first, stop = d.joint_starts[colour], d.joint_starts[colour + 1]
            index = first + thread - 1
            while index < stop
                @inbounds solve_rigid_joint!(f, center, q, velocity, omega, d.joints[d.joint_order[index]], step, Position)
                index += blockDim().x
            end
            sync_threads()
        end
        for colour in 1:d.pair_colours
            @inbounds first, stop = d.pair_starts[colour], d.pair_starts[colour + 1]
            index = first + thread - 1
            while index < stop
                @inbounds pair = d.pair_order[index]
                @inbounds a, b = d.pairs[pair]
                @inbounds count = d.contact_count[pair]
                for sample in Int32(1):count
                    @inbounds slot = d.contact_order[Int32(24) * (pair - Int32(1)) + sample]
                    solve_rigid_contact!(f, d, positions, ids, groups, center, q, velocity, omega,
                        a, b, slot, step, Position)
                end
                index += blockDim().x
            end
            sync_threads()
        end
    end
    return nothing
end

function rigid_project_kernel!(f, d, positions, ids, groups, center, q, velocity, omega, step, final,
        ::Val{Shared} = Val(false), ::Val{CacheVelocity} = Val(true)) where {Shared,CacheVelocity}
    step_active(step) || return nothing
    thread = threadIdx().x
    b = thread
    while b <= f.nbodies
        @inbounds d.free_velocity[b], d.free_omega[b] = velocity[b], omega[b]
        b += blockDim().x
    end
    sync_threads()
    working_f, working_center, working_q, working_velocity, working_omega, velocity_inertia = Shared ?
        rigid_shared_state(f, center, q, velocity, omega, step) : (f, center, q, velocity, omega, d.velocity_inertia)
    rigid_project_iterations!(working_f, d, positions, ids, groups, working_center, working_q, working_velocity, working_omega, step, Val(true))
    sync_threads()
    if CacheVelocity
        # Poses remain fixed throughout velocity projection. Cache the world
        # inverse inertia once, rather than rotating every impulse into and
        # out of the body frame on every iteration.
        T = eltype(f.mass)
        b = thread
        while b <= f.nbodies
            e1, e2, e3 = SVector{3,T}(1,0,0), SVector{3,T}(0,1,0), SVector{3,T}(0,0,1)
            @inbounds velocity_inertia[b] = SMatrix{3,3,T}(hcat(
                rigid_inverse_inertia(working_f, b, working_q, e1, step),
                rigid_inverse_inertia(working_f, b, working_q, e2, step),
                rigid_inverse_inertia(working_f, b, working_q, e3, step)))
            b += blockDim().x
        end
        sync_threads()
        velocity_f = merge(working_f, (; world_inverse_inertia = velocity_inertia))
        rigid_project_iterations!(velocity_f, d, positions, ids, groups, working_center, working_q, working_velocity, working_omega, step, Val(false))
    else
        rigid_project_iterations!(working_f, d, positions, ids, groups, working_center, working_q, working_velocity, working_omega, step, Val(false))
    end
    # A bilateral predictor impulse is a half-step constraint acceleration:
    # the midpoint corrector needs twice that impulse over the full step.
    # Carrying it only once introduces first-order damping in pendulums.
    # Contact-enabled solves retain the existing single-application impact treatment.
    predictor_scale = d.has_contacts ? one(eltype(f.mass)) : eltype(f.mass)(2)
    b = thread
    while b <= f.nbodies
        @inbounds begin
            if Shared
                center[b], q[b] = working_center[b], working_q[b]
                velocity[b], omega[b] = working_velocity[b], working_omega[b]
            end
            impulse = f.mass[b] * (velocity[b] - d.free_velocity[b])
            local_delta = rotate_vector(quaternion_conjugate(q[b]), omega[b] - d.free_omega[b])
            angular = rotate_vector(q[b], f.inertia[b] * local_delta)
            if final
                d.reaction_force[b] = (impulse + d.half_impulse[b]) / eltype(f.mass)(step_dt(step))
                d.reaction_torque[b] = (angular + d.half_angular_impulse[b]) / eltype(f.mass)(step_dt(step))
            else
                d.half_impulse[b], d.half_angular_impulse[b] = predictor_scale * impulse, predictor_scale * angular
                f.velocity[b] += predictor_scale * (velocity[b] - d.free_velocity[b])
                f.omega[b] += predictor_scale * (omega[b] - d.free_omega[b])
            end
        end
        b += blockDim().x
    end
    return nothing
end

function rigid_rotation_kernel!(f, d, q, step)
    step_active(step) || return nothing
    b = thread_index()
    b > f.nbodies && return nothing
    @inbounds begin
        delta = quaternion_multiply(q[b], quaternion_conjugate(d.old_orientation[b]))
        delta[1] < 0 && (delta = -delta)
        U = eltype(eltype(f.rotation))
        f.rotation[b] = SVector{5,U}(delta[2], delta[3], delta[4], 2delta[1], 2)
        vector = SVector(delta[2], delta[3], delta[4])
        magnitude = sqrt(dot(vector, vector))
        f.turn[b] = magnitude > eps(eltype(delta)) ? vector * (2atan(magnitude, delta[1]) / magnitude) : 2vector
    end
    return nothing
end

function native_rigid_particles_kernel!(positions, velocities, ids, types, groups, indices, body,
        references, center, q, velocity, omega, step, pos_cells, cells, grid, support)
    step_active(step) || return nothing
    t = thread_index()
    t > length(indices) && return nothing
    @inbounds begin
        i = indices[t]
        types[i] == Frozen && return nothing
        b = body[groups[i]]
        arm = rigid_arm(q, b, references[ids[i]])
        x = center[b] + arm
        positions[i] = x
        velocities[i] = velocity[b] + cross(omega[b], arm)
        store_pos_cell!(pos_cells, i, x, cells, grid, support)
    end
    return nothing
end

function place_native_rigid_particles!(floating, positions, velocities, ids, types, groups, step, final;
        pos_cells, cells, grid, support)
    floating_dimension(floating.center) == 2 && return place_planar_rigid_particles!(floating,positions,velocities,ids,types,groups,step,final;
        pos_cells,cells,grid,support)
    ids === nothing && throw(ArgumentError("native rigid particle placement requires stable particle IDs"))
    f = floating
    center, q, velocity, omega = final ? (f.center, f.orientation, f.velocity, f.omega) :
        (f.center_half, f.orientation_half, f.velocity_half, f.omega_half)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(length(f.indices), ELEMENTWISE_THREADS) native_rigid_particles_kernel!(
        positions, velocities, ids, types, groups, f.indices, f.body, f.mechanics.references,
        center, q, velocity, omega, step, pos_cells, cells, grid, support)
    return nothing
end

function prepare_rigid_stage!(f, step)
    hasproperty(f, :mechanics) || return nothing
    floating_dimension(f.center) == 2 && return prepare_planar_rigid_stage!(f,step)
    d = f.mechanics
    count = max(Int(f.nbodies), length(d.winners))
    @cuda threads=128 blocks=cld(count, 128) rigid_reset_kernel!(rigid_device(f), d, step)
    return nothing
end

function solve_rigid_stage!(f, gpu, cl, kernel, step, final::Bool)
    hasproperty(f, :mechanics) || return nothing
    floating_dimension(f.center) == 2 && return solve_planar_rigid_stage!(f,step,final)
    d = f.mechanics
    device = rigid_device(f)
    center, q, velocity, omega = final ? (f.center, f.orientation, f.velocity, f.omega) :
        (f.center_half, f.orientation_half, f.velocity_half, f.omega_half)
    if d.has_contacts
        @cuda threads=128 blocks=cld(length(f.indices), 128) rigid_candidates_kernel!(device, d,
            gpu.Position, gpu.ID, gpu.GroupMarker, gpu.Type, f.indices, gpu.CellID, cl.CellStart, cl.grid_dev, center, q, step)
        @cuda threads=128 blocks=cld(length(d.winners), 128) rigid_partners_kernel!(device, d,
            gpu.Position, gpu.ID, gpu.GroupMarker, gpu.Type, gpu.CellID, cl.CellStart, cl.grid_dev, center, q, velocity, omega, step)
        @cuda threads=128 blocks=cld(length(d.pairs), 128) rigid_pack_contacts_kernel!(d, step)
    end
    # Small articulated systems fit comfortably in shared memory. Larger
    # systems retain the same solver with global storage instead of exceeding
    # a device's shared-memory launch limit.
    shared_bytes = rigid_shared_bytes(f)
    shared = shared_bytes <= 32 * 1024
    @cuda threads=128 blocks=1 shmem=(shared ? shared_bytes : 0) rigid_project_kernel!(device, d,
        gpu.Position, gpu.ID, gpu.GroupMarker, center, q, velocity, omega, step, final, Val(shared))
    @cuda threads=128 blocks=cld(Int(f.nbodies), 128) rigid_rotation_kernel!(device, d, q, step)
    return nothing
end

rigid_device(f) = (; f.nbodies, f.mass, f.inertia, f.inertia_inv, f.pause, f.frozen, f.body,
    f.orientation, f.rotation, f.turn, f.velocity, f.omega)

function zero_rigid_interactions_kernel!(acceleration, rate, step, n)
    step_active(step) || return nothing
    i = thread_index()
    @inbounds if i <= n
        acceleration[i] = zero(eltype(acceleration))
        rate[i] = zero(eltype(rate))
    end
    return nothing
end

function zero_rigid_interactions!(acceleration, rate, step)
    @cuda threads=256 blocks=cld(length(rate), 256) zero_rigid_interactions_kernel!(acceleration, rate, step, Int32(length(rate)))
    return nothing
end

function initial_rigid_velocity_kernel!(f, d, center, q, step)
    thread = threadIdx().x
    for iteration in 1:d.iterations, colour in 1:d.joint_colours
        @inbounds index, stop = d.joint_starts[colour] + thread - 1, d.joint_starts[colour + 1]
        while index < stop
            @inbounds solve_rigid_joint!(f, center, q, f.velocity, f.omega, d.joints[d.joint_order[index]], step, false)
            index += blockDim().x
        end
        sync_threads()
    end
    return nothing
end

function initialize_rigid_velocities!(f, time)
    hasproperty(f, :mechanics) || return nothing
    floating_dimension(f.center) == 2 && return initialize_planar_rigid_velocities!(f,time)
    @cuda threads=128 blocks=1 initial_rigid_velocity_kernel!(rigid_device(f), f.mechanics, f.center,
        f.orientation, HostStep(zero(eltype(f.mass)), Float64(time)))
    return nothing
end

"Download native constraint/contact reactions at output time (N and N·m)."
rigid_dynamics_state(f) = (; reaction_force = Array(f.mechanics.reaction_force),
    reaction_torque = Array(f.mechanics.reaction_torque))

"""
    rigid_joint_errors(specification, geometry, floating_state)

Evaluate native joint position and hinge-axis residuals from a host body-state
snapshot, such as `result.final_state.floating`. This output-time check performs
no per-step downloads. Errors are metres and radians, in joint declaration order.
"""
function rigid_joint_errors(specification::RigidBodyDynamics, geometry, state)
    isempty(specification.joints) && return NamedTuple[]
    length(first(state.center)) == 2 && return planar_rigid_joint_errors(specification,geometry,state)
    bodies = filter(g -> g.Type == Floating, geometry)
    mapping = Dict(g.GroupMarker => b for (b, g) in enumerate(bodies))
    initial = [sum(SVector{3,Float64}.(g.Particles.Position)) / length(g.Particles) for g in bodies]
    results = NamedTuple[]
    for (index, joint) in enumerate(specification.joints)
        a = mapping[joint.body_a]
        b = joint.body_b == 0 ? 0 : mapping[joint.body_b]
        qa = SVector{4,Float64}(state.orientation[a])
        pa = state.center[a] + rotate_vector(qa, joint.attachment - initial[a])
        pb = b == 0 ? joint.anchor : state.center[b] + rotate_vector(SVector{4,Float64}(state.orientation[b]), joint.anchor - initial[b])
        delta = pa - pb
        translation = joint.kind == 2 ? sqrt(sum(abs2, delta - dot(delta, joint.axis) * joint.axis)) : sqrt(sum(abs2, delta))
        angle = 0.0
        if joint.kind == 1
            aa = rotate_vector(qa, joint.axis)
            ab = b == 0 ? joint.axis : rotate_vector(SVector{4,Float64}(state.orientation[b]), joint.axis)
            angle = atan(sqrt(sum(abs2, cross(aa, ab))), dot(aa, ab))
        end
        push!(results, (; joint = index, translation_error = translation, rotation_error = angle))
    end
    return results
end

@inline function rigid_point(center, orientation, b, offset)
    b == 0 && return offset
    @inbounds return center[b] + rigid_arm(orientation, b, offset)
end

@inline function rigid_effective_mass(f, q, a, b, ra, rb, n, step)
    ca, cb = cross(ra, n), cross(rb, n)
    return rigid_mobility(f, a, step) + rigid_mobility(f, b, step) +
        dot(ca, rigid_inverse_inertia(f, a, q, ca, step)) + dot(cb, rigid_inverse_inertia(f, b, q, cb, step))
end

@inline function rigid_point_response(f, q, a, b, ra, rb, impulse, step)
    return (rigid_mobility(f, a, step) + rigid_mobility(f, b, step)) * impulse -
        cross(ra, rigid_inverse_inertia(f, a, q, cross(ra, impulse), step)) -
        cross(rb, rigid_inverse_inertia(f, b, q, cross(rb, impulse), step))
end

@inline function rigid_apply_position!(f, center, q, b, arm, impulse, step)
    inverse_mass = rigid_mobility(f, b, step)
    inverse_mass == 0 && return nothing
    @inbounds begin
        turn = rigid_inverse_inertia(f, b, q, cross(arm, impulse), step)
        center[b] += eltype(center)(inverse_mass * impulse)
        q[b] = advance_orientation(q[b], turn)
    end
    return nothing
end

@inline function rigid_apply_velocity!(f, velocity, omega, q, b, arm, impulse, step)
    inverse_mass = rigid_mobility(f, b, step)
    inverse_mass == 0 && return nothing
    @inbounds begin
        velocity[b] += inverse_mass * impulse
        omega[b] += rigid_inverse_inertia(f, b, q, cross(arm, impulse), step)
    end
    return nothing
end

@inline function rigid_point_velocity(velocity, omega, b, arm)
    b == 0 && return zero(eltype(velocity))
    @inbounds return velocity[b] + cross(omega[b], arm)
end

@inline function solve_rigid_joint!(f, center, q, velocity, omega, joint, step, position::Bool)
    T = eltype(f.mass)
    a, b = joint.a, joint.b
    ra = rigid_arm(q, a, joint.ra)
    rb = rigid_arm(q, b, joint.rb)
    error = position ? SVector{3,T}(rigid_point(center, q, a, joint.ra) - rigid_point(center, q, b, joint.rb)) :
        rigid_point_velocity(velocity, omega, a, ra) - rigid_point_velocity(velocity, omega, b, rb)
    if !iszero(error)
        impulse = zero(error)
        if joint.kind == 2
            n1, n2 = rigid_basis(joint.axis_b)
            response1 = rigid_point_response(f, q, a, b, ra, rb, n1, step)
            response2 = rigid_point_response(f, q, a, b, ra, rb, n2, step)
            k11, k12, k22 = dot(n1, response1), dot(n1, response2), dot(n2, response2)
            determinant = k11 * k22 - k12^2
            if determinant > 0
                e1, e2 = dot(error, n1), dot(error, n2)
                impulse = -((k22 * e1 - k12 * e2) * n1 + (k11 * e2 - k12 * e1) * n2) / determinant
            end
        else
            e1, e2, e3 = SVector{3,T}(1,0,0), SVector{3,T}(0,1,0), SVector{3,T}(0,0,1)
            mobility = SMatrix{3,3,T}(hcat(rigid_point_response(f, q, a, b, ra, rb, e1, step),
                rigid_point_response(f, q, a, b, ra, rb, e2, step), rigid_point_response(f, q, a, b, ra, rb, e3, step)))
            det(mobility) > 0 && (impulse = -(mobility \ error))
        end
        if !iszero(impulse)
            if position
                rigid_apply_position!(f, center, q, a, ra, impulse, step)
                rigid_apply_position!(f, center, q, b, rb, -impulse, step)
            else
                rigid_apply_velocity!(f, velocity, omega, q, a, ra, impulse, step)
                rigid_apply_velocity!(f, velocity, omega, q, b, rb, -impulse, step)
            end
        end
    end
    if joint.kind == 1
        for component in 1:2
            axis_a = rotate_vector(q[a], joint.axis_a)
            axis_b = b == 0 ? joint.axis_b : rotate_vector(q[b], joint.axis_b)
            n1, n2 = rigid_basis(axis_b)
            n = component == 1 ? n1 : n2
            ia, ib = rigid_inverse_inertia(f, a, q, n, step), rigid_inverse_inertia(f, b, q, n, step)
            mobility = dot(n, ia + ib)
            if mobility > 0
                error = position ? dot(cross(axis_a, axis_b), n) :
                    -dot(omega[a] - (b == 0 ? zero(omega[a]) : omega[b]), n)
                impulse = error / mobility
                if position
                    rigid_mobility(f, a, step) == 0 || (q[a] = advance_orientation(q[a], ia * impulse))
                    rigid_mobility(f, b, step) == 0 || (q[b] = advance_orientation(q[b], -ib * impulse))
                else
                    rigid_mobility(f, a, step) == 0 || (omega[a] += ia * impulse)
                    rigid_mobility(f, b, step) == 0 || (omega[b] -= ib * impulse)
                end
            end
        end
    end
    return nothing
end
