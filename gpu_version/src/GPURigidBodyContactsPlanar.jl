# Planar contacts share the native 3D broad phase, pair colouring and packing.
# Eight of the 24 directional slots are used in a plane.
function attach_planar_contacts(f,mechanics,specification,particles,kernel)
    contact = specification.contacts
    contact === nothing && return mechanics
    T = eltype(f.mass)
    body = Array(f.body)
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

    return merge(mechanics,(;pairs = CuArray(pairs),pair_order,pair_starts,pair_colours,
        fixed_enabled = CuArray(fixed_enabled),floating_enabled = CuArray(floating_enabled),
        contact_friction = CuArray(friction),contact_restitution = CuArray(restitution),
        winners = CUDA.fill(typemax(UInt64),slots),partner = CUDA.zeros(Int32,slots),
        contact_order = CUDA.zeros(Int32,slots),contact_count = CUDA.zeros(Int32,length(pairs)),
        normal_impulse = CUDA.zeros(T,slots),tangent_impulse = CUDA.zeros(SVector{2,T},slots),bounce = CUDA.zeros(T,slots),
        old_orientation = mechanics.old_angle,has_contacts = true,parameters))
end
@inline function rigid_contact_bin(delta::SVector{2,T},relative) where T
    axis = abs(delta[1]) >= abs(delta[2]) ? 1 : 2
    face = 2(axis-1)+(delta[axis] < 0 ? 1 : 0)
    return Int32(2face+(relative[3-axis] >= 0 ? 1 : 0)+1)
end
@inline function rigid_arm(angle::AbstractVector{T},b,offset::SVector{2,TP}) where {T<:Real,TP}
    return b == 0 ? zero(SVector{2,T}) : planar_rotate(@inbounds(angle[b]),SVector{2,T}(offset))
end
@inline function rigid_point_velocity(velocity,omega,b,arm::SVector{2})
    return planar_velocity(velocity,omega,b,arm)
end
@inline function rigid_contact_geometry(f,d,positions::AbstractVector{SVector{2,TP}},ids,groups,center,angle,a,b,i,j) where TP
    T = eltype(f.mass)
    xi = rigid_particle_point(f,d,positions,ids,groups,center,angle,i)
    xj = rigid_particle_point(f,d,positions,ids,groups,center,angle,j)
    delta = SVector{2,T}(xi-xj)
    distance = norm(delta)
    n = distance > 0 ? delta/distance : SVector{2,T}(0,1)
    point = (xi+xj)/2
    return n,distance-d.parameters.distance,SVector{2,T}(point-center[a]),b == 0 ? zero(n) : SVector{2,T}(point-center[b])
end
@inline function planar_contact_mobility(f,a,b,ra,rb,n,step)
    return rigid_mobility(f,a,step)+rigid_mobility(f,b,step)+
        planar_inverse_inertia(f,a,step)*planar_cross(ra,n)^2+planar_inverse_inertia(f,b,step)*planar_cross(rb,n)^2
end
@inline function solve_rigid_contact!(f,d,positions::AbstractVector{SVector{2,TP}},ids,groups,center,angle,velocity,omega,a,b,slot,step,position) where TP
    @inbounds j = d.partner[slot]
    j == 0 && return nothing
    @inbounds i = Int32(d.winners[slot] & UInt64(0xffffffff))
    n,gap,ra,rb = rigid_contact_geometry(f,d,positions,ids,groups,center,angle,a,b,i,j)
    mobility = planar_contact_mobility(f,a,b,ra,rb,n,step)
    mobility > 0 || return nothing
    if position
        if gap < 0
            impulse = (-gap/mobility)*n
            apply_planar_impulse!(f,center,angle,velocity,omega,a,ra,impulse,step,true)
            apply_planar_impulse!(f,center,angle,velocity,omega,b,rb,-impulse,step,true)
        end
    else
        T = eltype(f.mass)
        relative = planar_velocity(velocity,omega,a,ra)-planar_velocity(velocity,omega,b,rb)
        @inbounds target = d.bounce[slot] > 0 ? d.bounce[slot] : gap > 0 ? -gap/T(step_dt(step)) : zero(T)
        @inbounds previous,old_tangent = d.normal_impulse[slot],d.tangent_impulse[slot]
        total = max(zero(T),previous+(target-dot(relative,n))/mobility)
        impulse = (total-previous)*n
        @inbounds d.normal_impulse[slot] = total
        apply_planar_impulse!(f,center,angle,velocity,omega,a,ra,impulse,step,false)
        apply_planar_impulse!(f,center,angle,velocity,omega,b,rb,-impulse,step,false)
        relative = planar_velocity(velocity,omega,a,ra)-planar_velocity(velocity,omega,b,rb)
        tangent = planar_perpendicular(n)
        kt = planar_contact_mobility(f,a,b,ra,rb,tangent,step)
        if kt > 0
            @inbounds limit = d.contact_friction[rigid_pair_index(a,b)]*total
            candidate = clamp(dot(old_tangent,tangent)-dot(relative,tangent)/kt,-limit,limit)*tangent
            @inbounds d.tangent_impulse[slot] = candidate
            impulse = candidate-old_tangent
            apply_planar_impulse!(f,center,angle,velocity,omega,a,ra,impulse,step,false)
            apply_planar_impulse!(f,center,angle,velocity,omega,b,rb,-impulse,step,false)
        end
    end
    return nothing
end
