# Each level owns the space inside its parent and outside its child. All boxes
# share the followed body's pose, while each directed interface has its own
# flux budget. Interaction inputs remain dense and uniform-mass on every level.
struct ResolutionLevelLimits{P,C}
    parent::P
    child::C
end
Adapt.@adapt_structure ResolutionLevelLimits

@inline buffer_particle(x,l::ResolutionLevelLimits,fine) =
    (l.parent !== nothing && buffer_particle(x,l.parent,true)) ||
    (l.child !== nothing && buffer_particle(x,l.child,false))
@inline outside_domain(x,l::ResolutionLevelLimits,fine) =
    (l.parent !== nothing && outside_domain(x,l.parent,true)) ||
    (l.child !== nothing && outside_domain(x,l.child,false))
@inline resolution_owned(x,l::ResolutionLevelLimits,fine) =
    (l.parent === nothing || resolution_owned(x,l.parent,true)) &&
    (l.child === nothing || resolution_owned(x,l.child,false))
resolution_host_limits(l::ResolutionLevelLimits) = ResolutionLevelLimits(
    l.parent === nothing ? nothing : resolution_host_limits(l.parent),
    l.child === nothing ? nothing : resolution_host_limits(l.child))

struct ResolutionHierarchy{P,L,E}
    pairs::P
    levels::L
    edges::E
end

function hierarchy_interface(limits,fine,domain)
    faces = interface_points(limits,fine,Float64(domain.constants.dx))
    return (;points=CuArray(faces.points),normals=CuArray(faces.normals),
        widths=CuArray(faces.widths),mass=CUDA.zeros(Float64,length(faces.points)),
        fluid_marker=domain.interface.fluid_marker)
end

function resolution_hierarchy(pairs,domains)
    n = length(domains)
    levels = ntuple(i -> ResolutionLevelLimits(i == 1 ? nothing : pairs[i-1],i == n ? nothing : pairs[i]),n)
    edges = Tuple((;target=t,source=s,pair=i,face=hierarchy_interface(pairs[i],t> s,domains[t]))
        for i in eachindex(pairs) for (t,s) in ((i,i+1),(i+1,i)))
    return ResolutionHierarchy(pairs,levels,edges)
end

function hierarchy_dt_kernel!(partials,kernels,constants,step)
    GPUStepState.step_active(step) || return nothing
    dt = Inf
    for i in eachindex(partials)
        dt = min(dt,resolution_standard_dt(partials[i],kernels[i],constants[i]))
    end
    @inbounds begin
        step.f[1] = step.i[GPUStepState.I_CLIP_OUTPUT] != 0 ? min(dt,step.f[4]-step.f[2]) : dt
        isfinite(step.f[1]) && step.f[1] > 0 || (step.i[1] = RESOLUTION_STATE_ERROR)
    end
    return nothing
end

function hierarchy_transfer_kernel!(step,source_limits,stride,args...)
    GPUStepState.step_active(step) || return nothing
    interface_flux_kernel!(args...,GPUStepState.step_dt(step),Int32(length(args[1])),step.count,source_limits,stride)
    return nothing
end
function hierarchy_exchange_kernel!(step,n,source_limits,args...)
    GPUStepState.step_active(step) || return nothing
    reconstruct_buffers_kernel!(args...,GPUStepState.step_particle_count(step,n),source_limits)
    return nothing
end

function exchange_hierarchy!(domains,h,state,phase)
    for edge in h.edges
        d,source = domains[edge.target],domains[edge.source]
        p,s,c = d.particles,source.particles,source.cells
        @cuda threads=128 blocks=cld(length(p),128) hierarchy_exchange_kernel!(ResolutionStep(state,d,phase),Int32(length(p)),
            h.levels[edge.source],p.Position,p.Velocity,p.Density,p.Type,s.Position,s.Velocity,s.Density,s.Type,
            c.CellStart,c.grid_dev,source.kernel,source.constants,h.levels[edge.target],d.fine,source.fine)
    end
    return nothing
end

function enqueue_resolution_step!(domains,h::ResolutionHierarchy,state;
        density_diffusion=LinearDensityDiffusion(),viscosity=ArtificialViscosity(),
        scheme=SymplecticTimeStepping(),damping=(),interaction=(;))
    n = length(domains)
    root,leaf = first(domains),last(domains)
    corrector,flux,commit = n+1,2n+2,3n+3
    @cuda threads=1 hierarchy_dt_kernel!(map(d -> d.pool.reduction.partial,domains),
        map(d -> d.kernel,domains),map(d -> d.constants,domains),ResolutionStep(state,root,0))
    if leaf.attachment !== nothing && leaf.attachment.floating !== nothing
        @cuda threads=1 resolution_timestep_limit_kernel!(ResolutionStep(state,leaf,0),leaf.attachment.floating.spring_dt)
    end
    for d in domains
        resident_force_stage!(d,ResolutionStep(state,d,0);density_diffusion,viscosity,scheme,interaction)
    end
    leaf.attachment === nothing || update_resolution_floating!(leaf,ResolutionStep(state,leaf,0),false;limits=first(h.pairs))
    for d in domains
        p,s,q = d.particles,d.support,d.predictor.particles
        step = ResolutionStep(state,d,0)
        @cuda threads=128 blocks=cld(length(p),128) resolution_standard_predictor_kernel!(q.Position,q.Velocity,q.Density,
            p.Position,p.Velocity,p.Density,p.Acceleration,s.dρdtI,p.Type,p.GroupMarker,
            ResolutionArithmeticStep{eltype(p.Density)}(step),d.constants,d.boundary_mode,Int32(length(p)))
        @cuda threads=128 blocks=cld(length(p),128) resolution_predictor_metadata_kernel!(q.Type,q.ID,q.GroupMarker,
            q.GhostPoints,q.GhostNormals,d.predictor.pool.owners,d.predictor.pool.count,p.Type,p.GroupMarker,
            p.GhostPoints,p.GhostNormals,d.pool.owners,step,Int32(length(p)))
        if d.attachment !== nothing
            a,qa = d.attachment,d.predictor.attachment
            @cuda threads=128 blocks=cld(length(p),128) resolution_body_predictor_indices_kernel!(qa.slots,qa.indices,a.slots,step,Int32(length(p)))
            sample_resolution_body!(a,step,0.5)
            place_resolution_body!(d.predictor,step)
        end
    end
    @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,root,0),Int32(1))
    for (i,d) in enumerate(domains)
        enqueue_resolution_rebuild!(d.predictor,ResolutionStep(state,d.predictor,i))
        @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,d.predictor,i),Int32(i+1))
    end
    for d in domains
        resident_force_stage!(d.predictor,ResolutionStep(state,d.predictor,corrector);
            density_diffusion,viscosity,scheme,correct_boundary=false,interaction)
    end
    leaf.attachment === nothing || update_resolution_floating!(leaf.predictor,ResolutionStep(state,leaf.predictor,corrector),true)
    for d in domains
        p,s,q,qs = d.particles,d.support,d.predictor.particles,d.predictor.support
        step = ResolutionStep(state,d,corrector)
        @cuda threads=128 blocks=cld(length(p),128) resolution_scatter_predictor_kernel!(p.Acceleration,s.dρdtI,s.ρₙ⁺,
            s.Positionₙ⁺,s.Velocityₙ⁺,q.Acceleration,qs.dρdtI,q.Density,q.Position,q.Velocity,q.ID,
            ResolutionStep(state,d.predictor,corrector),Int32(length(p)))
        red = d.pool.reduction
        @cuda threads=REDUCE_THREADS blocks=red.nblocks resolution_standard_corrector_kernel!(p.Position,p.Velocity,
            p.Acceleration,p.Density,s.dρdtI,s.ρₙ⁺,s.Positionₙ⁺,s.Velocityₙ⁺,p.Type,s.∇Cᵢ,s.∇◌rᵢ,
            ResolutionArithmeticStep{eltype(p.Density)}(step),d.kernel,d.constants,d.boundary_mode,red.partial,damping,Int32(length(p)))
    end
    if leaf.attachment !== nothing
        step = ResolutionStep(state,leaf,corrector)
        sample_resolution_body!(leaf.attachment,step,1.0)
        place_resolution_body!(leaf,step)
        pair = first(h.pairs)
        @cuda threads=1 resolution_frame_kernel!(pair.frames,leaf.attachment.pose,pair.reference,step,0.5Float64(leaf.kernel.h),false)
        p,red = leaf.particles,leaf.pool.reduction
        @cuda threads=REDUCE_THREADS blocks=red.nblocks resolution_standard_reduction_kernel!(
            red.partial,p.Position,p.Velocity,p.Acceleration,p.Type,step,leaf.kernel,Int32(length(p)))
    end
    for (i,d) in enumerate(domains)
        p = d.particles
        @cuda threads=128 blocks=cld(length(p),128) resolution_retire_kernel!(p.Position,p.Velocity,p.Density,p.Type,
            d.events,h.levels[i],d.fine,d.constants.ρ₀,ResolutionStep(state,d,corrector),Int32(length(p)))
    end
    @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,root,corrector),Int32(corrector+1))
    for (i,d) in enumerate(domains)
        phase = corrector+i
        enqueue_resolution_rebuild!(d,ResolutionStep(state,d,phase))
        @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,d,phase),Int32(phase+1))
    end
    for d in domains
        @cuda threads=2 resolution_clear_kernel!(d.allocation,ResolutionStep(state,d,flux))
    end
    for edge in h.edges
        d,source,face = domains[edge.target],domains[edge.source],edge.face
        p,s,c = d.particles,source.particles,source.cells
        @cuda threads=128 blocks=cld(length(face.points),128) hierarchy_transfer_kernel!(ResolutionStep(state,d,flux),
            h.levels[edge.source],d.id_stride,face.points,face.normals,face.widths,face.mass,
            p.Position,p.Velocity,p.Density,p.Type,p.ID,p.GroupMarker,nothing,d.allocation,d.next_id,d.events,
            d.cells.CellStart,d.cells.grid_dev,d.kernel,s.Position,s.Velocity,s.Density,s.Type,c.CellStart,c.grid_dev,
            source.kernel,source.constants,d.constants,face.fluid_marker,h.pairs[edge.pair],source.fine)
    end
    for d in domains
        @cuda threads=1 resolution_allocation_finish_kernel!(d.allocation,d.events,length(d.particles),ResolutionStep(state,d,flux))
    end
    exchange_hierarchy!(domains,h,state,flux)
    @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,root,flux),Int32(flux+1))
    for (i,d) in enumerate(domains)
        phase = flux+i
        enqueue_resolution_rebuild!(d,ResolutionStep(state,d,phase;created_only=true))
        @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,d,phase),Int32(phase+1))
    end
    @cuda threads=1 resolution_commit_kernel!(ResolutionStep(state,root,commit))
    return nothing
end
