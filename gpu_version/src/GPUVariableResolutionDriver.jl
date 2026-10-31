# Included in VariableResolution. A restartable device phase machine keeps
# capacity growth on the host without repeating an integration or mass flux.
# StepState slots retain dt/time/deadline/iteration/stop; F_DX is reused for
# the previous predictor's timestep limit. Phases: 0 accepted forces/predict;
# 1–2 predictor rebuilds; 3 predictor forces/correct/retire; 4–5 accepted
# rebuilds; 6 flux/exchange; 7–8 conditional creation rebuilds; 9 commit.
struct ResolutionPool
    count::CuVector{Int32} # live prefix (may contain newly retired holes), spare, rebuilds
    owners::CuVector{Int32}
    owner_scratch::CuVector{Int32}
end

function ResolutionPool(host, initial_count)
    owners = zeros(Int32, length(host))
    indices = findall(!iszero, host.GhostPoints)
    owners[indices] = Int32.(eachindex(indices))
    return ResolutionPool(CuArray(Int32[initial_count, 0, 0]), CuArray(owners), CuVector{Int32}(undef,length(owners)))
end

struct ResolutionStep{F,I,P,A}
    f::F
    i::I
    count::P
    allocation::A
    phase::Int32
    created_only::Bool
end
ResolutionStep(state, domain, phase; created_only = false) = ResolutionStep(
    state.f, state.i, domain.pool.count, hasproperty(domain, :allocation) ? domain.allocation : nothing,
    Int32(phase), created_only)
Adapt.@adapt_structure ResolutionStep

@inline GPUStepState.step_active(s::ResolutionStep) = @inbounds s.i[1] == 0 && s.i[2] == s.phase &&
    (s.allocation === nothing || !s.created_only || s.allocation[1] > 0)
@inline GPUStepState.step_dt(s::ResolutionStep) = @inbounds s.f[1]
@inline GPUStepState.step_time(s::ResolutionStep) = @inbounds s.f[2]
@inline GPUStepState.step_particle_count(s::ResolutionStep, n) = @inbounds min(n, s.count[1])
@inline GPUCellGrid.grid_build_needed(s::ResolutionStep) = GPUStepState.step_active(s)
@inline GPUCellGrid.publish_grid_state!(::ResolutionStep, state) = nothing

const RESOLUTION_CAPACITY = Int32(3)
const RESOLUTION_GRID_ERROR = Int32(4)
const RESOLUTION_RESERVE_ERROR = Int32(5)
const RESOLUTION_STATE_ERROR = Int32(6)

function resolution_phase_kernel!(step, next)
    GPUStepState.step_active(step) || return nothing
    @inbounds step.i[2] = next
    return nothing
end

function resolution_clear_kernel!(values, step)
    GPUStepState.step_active(step) || return nothing
    i = thread_index()
    @inbounds i <= length(values) && (values[i] = 0)
    return nothing
end

function resolution_scan_block_kernel!(out, input, sums, n, inclusive, step)
    GPUStepState.step_active(step) || return nothing
    GPUCellGrid.scan_block_kernel!(out, input, sums, n, inclusive)
    return nothing
end
function resolution_scan_offsets_kernel!(out, offsets, n, step)
    GPUStepState.step_active(step) || return nothing
    GPUCellGrid.scan_add_offsets_kernel!(out, offsets, n)
    return nothing
end
function resolution_scan!(ws, out, input, n, step, level = 1; inclusive = true)
    blocks = cld(n, 256)
    sums, offsets = ws.sums[level], ws.offsets[level]
    @cuda threads=256 blocks=blocks resolution_scan_block_kernel!(out,input,sums,Int32(n),inclusive,step)
    if blocks > 1
        resolution_scan!(ws, offsets, sums, blocks, step, level + 1; inclusive = false)
        @cuda threads=256 blocks=blocks resolution_scan_offsets_kernel!(out,offsets,Int32(n),step)
    end
end

function resolution_bbox_kernel!(partial, invalid, positions, types, n, step)
    GPUStepState.step_active(step) || return nothing
    GPUCellGrid.bbox_reduce_scalar_kernel!(partial,invalid,positions,types,
        GPUStepState.step_particle_count(step,n),Inf,-Inf,step,Val(2))
    return nothing
end
function resolution_grid_status_kernel!(status, step)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        code = status[1].status
        if code != GRID_OK
            step.i[1] = code == GRID_CAPACITY ? RESOLUTION_CAPACITY : RESOLUTION_GRID_ERROR
        end
    end
    return nothing
end

function resolution_grid_metadata_kernel!(step,args...)
    GPUStepState.step_active(step) || return nothing
    GPUCellGrid.finish_grid_metadata_kernel!(args...,step,Val(2),Val(1))
    return nothing
end

function resolution_histogram_kernel!(ids, counts, positions, types, inverse, gridarg, step, n)
    GPUStepState.step_active(step) || return nothing
    i = thread_index()
    i > GPUStepState.step_particle_count(step,n) && return nothing
    @inbounds begin
        c = types[i] == Frozen ? Int32(0) : linear_cell(load_grid(gridarg),cell_coords(positions[i],inverse))
        ids[i] = c
        c == 0 || CUDA.atomic_add!(pointer(counts,c),Int32(1))
    end
    return nothing
end
function resolution_scatter_kernel!(perm, counts, starts, ids, step, n)
    GPUStepState.step_active(step) || return nothing
    i = thread_index()
    i > GPUStepState.step_particle_count(step,n) && return nothing
    @inbounds begin
        c = ids[i]
        if c != 0
            slot = starts[c] + CUDA.atomic_add!(pointer(counts,c),Int32(1)) + Int32(1)
            perm[slot] = i
        end
    end
    return nothing
end
function resolution_sort_kernel!(perm, starts, grid, step)
    GPUStepState.step_active(step) || return nothing
    GPUCellGrid.cell_sort_kernel!(perm,starts,grid)
    return nothing
end
function resolution_gather_kernel!(perm, srcs, dsts, starts, step)
    GPUStepState.step_active(step) || return nothing
    @inbounds GPUCellGrid.gather_kernel!(perm,srcs,dsts,starts[end])
    return nothing
end
function resolution_copyback_kernel!(srcs, dsts, cellids, sorted_ids, owners, sorted_owners,
        ghost_index, types, ghosts, normals, starts, step)
    GPUStepState.step_active(step) || return nothing
    i = thread_index()
    @inbounds begin
        if i <= starts[end]
            GPUCellGrid.gather_fields!(i,i,srcs,dsts)
            cellids[i] = sorted_ids[i]
            slot = sorted_owners[i]
            owners[i] = slot
            slot == 0 || (ghost_index[slot] = i)
        elseif i <= step.count[1]
            types[i] = Frozen
            ghosts[i] = zero(eltype(ghosts))
            normals[i] = zero(eltype(normals))
            owners[i] = 0
        end
    end
    return nothing
end
function resolution_rebuild_finish_kernel!(starts, step)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        step.count[1] = starts[end]
        step.count[3] += 1
    end
    return nothing
end

const RESOLUTION_FIELDS = (:Position, :Velocity, :Density, :Type, :ID, :GroupMarker, :GhostPoints, :GhostNormals)

function enqueue_resolution_rebuild!(domain, step)
    p,c,pool = domain.particles,domain.cells,domain.pool
    n,capacity = length(p),length(c.Counts)
    blocks = cld(n,256)
    @cuda threads=256 blocks=c.bbox_ws.nblocks resolution_bbox_kernel!(c.bbox_ws.partial,
        c.bbox_ws.invalid,reinterpret(Float64,p.Position),p.Type,Int32(n),step)
    # Retain allocated bounds and expand on the device. An empty pool retains
    # the last valid grid, so all-fluid domains can empty and refill safely.
    @cuda threads=256 blocks=1 resolution_grid_metadata_kernel!(step,c.grid_dev,c.grid_state_dev,
        c.bbox_ws.partial,c.bbox_ws.invalid,Int32(c.bbox_ws.nblocks),Inf,-Inf,domain.kernel.H⁻¹,
        Int64(c.max_cells),Int64(capacity),true,((true,true),(true,true)),true,nothing)
    @cuda threads=1 resolution_grid_status_kernel!(c.grid_state_dev,step)
    @cuda threads=256 blocks=cld(capacity,256) resolution_clear_kernel!(c.Counts,step)
    @cuda threads=1 resolution_clear_kernel!(view(c.CellStart,1:1),step)
    @cuda threads=256 blocks=blocks resolution_histogram_kernel!(c.CellIDScratch,c.Counts,p.Position,p.Type,
        domain.kernel.H⁻¹,c.grid_dev,step,Int32(n))
    resolution_scan!(c.scan_ws,view(c.CellStart,2:capacity+1),c.Counts,capacity,step)
    @cuda threads=256 blocks=cld(capacity,256) resolution_clear_kernel!(c.Counts,step)
    @cuda threads=256 blocks=blocks resolution_scatter_kernel!(c.Perm,c.Counts,c.CellStart,c.CellIDScratch,step,Int32(n))
    @cuda threads=256 blocks=cld(capacity,256) resolution_sort_kernel!(c.Perm,c.CellStart,c.grid_dev,step)
    srcs = (map(f -> getproperty(p,f),RESOLUTION_FIELDS)...,c.CellIDScratch,pool.owners)
    dsts = (map(f -> getproperty(p.scratch,f),RESOLUTION_FIELDS)...,p.CellID,pool.owner_scratch)
    @cuda threads=256 blocks=blocks resolution_gather_kernel!(c.Perm,srcs,dsts,c.CellStart,step)
    # Stable pointers let one graph survive every reorder. Only the live prefix
    # is copied; capacity tails do not enter either sorting pass.
    @cuda threads=256 blocks=blocks resolution_copyback_kernel!(
        map(f -> getproperty(p.scratch,f),RESOLUTION_FIELDS),map(f -> getproperty(p,f),RESOLUTION_FIELDS),
        p.CellID,p.CellID,pool.owners,pool.owner_scratch,p.GhostIndex,p.Type,p.GhostPoints,p.GhostNormals,c.CellStart,step)
    @cuda threads=1 resolution_rebuild_finish_kernel!(c.CellStart,step)
    return nothing
end

function resolution_criteria_kernel!(partial, velocity, acceleration, types, viscous, step, n)
    GPUStepState.step_active(step) || return nothing
    GPUKernels.dual_criteria_kernel!(partial,velocity,acceleration,types,viscous,
        GPUStepState.step_particle_count(step,n),nothing)
    return nothing
end
function resident_force_stage!(domain, step)
    p,s,c,k = domain.particles,domain.support,domain.cells,domain.kernel
    launch_pos_cells!(s.PosCells,p.Position,p.CellID,c.grid_dev,step,k)
    launch_mdbc!(p.Density,p.Position,p.GhostPoints,p.GhostIndex,p.Type,c.CellStart,c.grid_dev,step,k,domain.constants;
        pos_cells=s.PosCells)
    launch_interactions!(s.dρdtI,p.Acceleration,p.Kernel,p.KernelGradient,s.∇Cᵢ,s.∇◌rᵢ,
        p.Position,p.Density,p.Velocity,p.Type,c.CellStart,p.CellID,c.grid_dev,step,
        FourtakasDensityDiffusion(full=true),ArtificialViscosity(),k,domain.constants,Val(false),Val(false);
        boundary_forces=Val(false),pos_cells=s.PosCells,viscous_dt=s.dual_state.viscous,fastmath=false)
    r = s.dual_state.reduction
    @cuda threads=REDUCE_THREADS blocks=r.nblocks resolution_criteria_kernel!(r.partial,p.Velocity,p.Acceleration,p.Type,
        s.dual_state.viscous,step,Int32(length(p)))
end

@inline function resolution_dt(partial, kernel, constants)
    value = zero(eltype(partial))
    @inbounds for i in eachindex(partial)
        value = GPUKernels.dual_max(value,partial[i])
    end
    acceleration, speed = sqrt(Float64(value[1])),sqrt(Float64(value[2]))
    h, sound = Float64(kernel.h),Float64(constants.c₀)
    force = iszero(acceleration) ? Inf : sqrt(h / acceleration)
    acoustic = h / (max(sound,10speed) + h * Float64(value[3]))
    return max(0.05h / sound,Float64(constants.CFL) * min(force,acoustic))
end
function resolution_dt_kernel!(a,b,ka,kb,ca,cb,step,predictor)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        dt = min(resolution_dt(a,ka,ca),resolution_dt(b,kb,cb))
        if predictor
            step.f[3] = dt
        else
            step.f[1] = min(dt,step.f[3],step.f[4]-step.f[2])
            dt = step.f[1]
        end
        isfinite(dt) && dt > 0 || (step.i[1] = RESOLUTION_STATE_ERROR)
    end
    return nothing
end

function resolution_predictor_kernel!(xp,vp,rp,x,v,rho,a,rate,types,step,constants,n)
    GPUStepState.step_active(step) || return nothing
    GPUKernels.dual_predictor_kernel!(xp,vp,rp,x,v,rho,a,rate,types,GPUStepState.step_dt(step),constants,
        nothing,nothing,DualSPHysicsSymplectic(),GPUStepState.step_particle_count(step,n))
    return nothing
end
function resolution_predictor_metadata_kernel!(types,ids,groups,ghosts,normals,owners,predictor_count,
        source_types,source_groups,source_ghosts,source_normals,source_owners,step,n)
    GPUStepState.step_active(step) || return nothing
    i = thread_index()
    @inbounds begin
        i == 1 && (predictor_count[1] = step.count[1])
        if i <= GPUStepState.step_particle_count(step,n)
            types[i],ids[i],groups[i] = source_types[i],i,source_groups[i]
            ghosts[i],normals[i],owners[i] = source_ghosts[i],source_normals[i],source_owners[i]
        end
    end
    return nothing
end
function resolution_scatter_predictor_kernel!(a,rate,rho,qa,qrate,qrho,ids,step,n)
    GPUStepState.step_active(step) || return nothing
    scatter_predictor_kernel!(a,rate,rho,qa,qrate,qrho,ids,GPUStepState.step_particle_count(step,n))
    return nothing
end
function resolution_corrector_kernel!(x,v,rho,xp,rp,a,rate,types,step,constants,n)
    GPUStepState.step_active(step) || return nothing
    GPUKernels.dual_corrector_kernel!(x,v,rho,xp,rp,a,rate,types,GPUStepState.step_dt(step),constants,
        GPUStepState.step_particle_count(step,n))
    return nothing
end
function resolution_retire_kernel!(x,v,rho,types,events,limits,fine,reference,step,n)
    GPUStepState.step_active(step) || return nothing
    retire_particles_kernel!(x,v,rho,types,events,limits,fine,reference,GPUStepState.step_particle_count(step,n))
    return nothing
end
function resolution_transfer_kernel!(step,args...)
    GPUStepState.step_active(step) || return nothing
    interface_flux_kernel!(args...,GPUStepState.step_dt(step),Int32(length(args[1])),step.count)
    return nothing
end
function resolution_allocation_finish_kernel!(allocation, events, capacity, step)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        step.count[1] = min(Int32(capacity),step.count[1]+allocation[1])
        if allocation[2] != 0
            step.i[1] = RESOLUTION_RESERVE_ERROR
        elseif events[4] != 0
            step.i[1] = RESOLUTION_STATE_ERROR
        end
    end
    return nothing
end
function resolution_exchange_kernel!(step,n,args...)
    GPUStepState.step_active(step) || return nothing
    reconstruct_buffers_kernel!(args...,GPUStepState.step_particle_count(step,n))
    return nothing
end
function resolution_commit_kernel!(step)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        dt,time,deadline = step.f[1],step.f[2],step.f[4]
        step.f[2] = dt == deadline-time ? deadline : time+dt
        step.i[3] += 1
        step.i[2] = 0
        step.f[2] >= deadline && (step.i[1] = GPUStepState.STOP_OUTPUT)
    end
    return nothing
end

function enqueue_resolution_step!(domains,limits,state)
    coarse,fine = domains
    for d in domains
        resident_force_stage!(d,ResolutionStep(state,d,0))
    end
    @cuda threads=1 resolution_dt_kernel!(coarse.support.dual_state.reduction.partial,fine.support.dual_state.reduction.partial,
        coarse.kernel,fine.kernel,coarse.constants,fine.constants,ResolutionStep(state,coarse,0),false)
    for d in domains
        p,s,q = d.particles,d.support,d.predictor.particles
        step = ResolutionStep(state,d,0)
        @cuda threads=128 blocks=cld(length(p),128) resolution_predictor_kernel!(q.Position,q.Velocity,q.Density,
            p.Position,p.Velocity,p.Density,p.Acceleration,s.dρdtI,p.Type,step,d.constants,Int32(length(p)))
        @cuda threads=128 blocks=cld(length(p),128) resolution_predictor_metadata_kernel!(q.Type,q.ID,q.GroupMarker,
            q.GhostPoints,q.GhostNormals,d.predictor.pool.owners,d.predictor.pool.count,p.Type,p.GroupMarker,
            p.GhostPoints,p.GhostNormals,d.pool.owners,step,Int32(length(p)))
    end
    @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,coarse,0),Int32(1))
    for (phase,d) in ((1,coarse.predictor),(2,fine.predictor))
        enqueue_resolution_rebuild!(d,ResolutionStep(state,d,phase))
        @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,d,phase),Int32(phase+1))
    end
    for d in domains
        resident_force_stage!(d.predictor,ResolutionStep(state,d.predictor,3))
    end
    @cuda threads=1 resolution_dt_kernel!(coarse.predictor.support.dual_state.reduction.partial,fine.predictor.support.dual_state.reduction.partial,
        coarse.kernel,fine.kernel,coarse.constants,fine.constants,ResolutionStep(state,coarse,3),true)
    for d in domains
        p,s,q,qs = d.particles,d.support,d.predictor.particles,d.predictor.support
        step = ResolutionStep(state,d,3)
        @cuda threads=128 blocks=cld(length(p),128) resolution_scatter_predictor_kernel!(p.Acceleration,s.dρdtI,s.ρₙ⁺,
            q.Acceleration,qs.dρdtI,q.Density,q.ID,ResolutionStep(state,d.predictor,3),Int32(length(p)))
        @cuda threads=128 blocks=cld(length(p),128) resolution_corrector_kernel!(p.Position,p.Velocity,p.Density,
            s.Positionₙ⁺,s.ρₙ⁺,p.Acceleration,s.dρdtI,p.Type,step,d.constants,Int32(length(p)))
        @cuda threads=128 blocks=cld(length(p),128) resolution_retire_kernel!(p.Position,p.Velocity,p.Density,p.Type,
            d.events,limits,d.fine,d.constants.ρ₀,step,Int32(length(p)))
    end
    @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,coarse,3),Int32(4))
    for (phase,d) in ((4,coarse),(5,fine))
        enqueue_resolution_rebuild!(d,ResolutionStep(state,d,phase))
        @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,d,phase),Int32(phase+1))
    end
    for (d,source) in ((coarse,fine),(fine,coarse))
        p,s,c,face = d.particles,source.particles,source.cells,d.interface
        step = ResolutionStep(state,d,6)
        @cuda threads=2 resolution_clear_kernel!(d.allocation,step)
        @cuda threads=128 blocks=cld(length(face.points),128) resolution_transfer_kernel!(step,
            face.points,face.normals,face.widths,face.mass,p.Position,p.Velocity,p.Density,p.Type,p.ID,p.GroupMarker,
            nothing,d.allocation,d.next_id,d.events,d.cells.CellStart,d.cells.grid_dev,d.kernel,
            s.Position,s.Velocity,s.Density,s.Type,c.CellStart,c.grid_dev,source.kernel,source.constants,
            d.constants,face.fluid_marker,limits,source.fine)
    end
    # Neither source grid includes particles born in this transfer, matching
    # the reference ordering of the two flux evaluations and exchanges.
    for d in domains
        @cuda threads=1 resolution_allocation_finish_kernel!(d.allocation,d.events,length(d.particles),ResolutionStep(state,d,6))
    end
    for (d,source) in ((coarse,fine),(fine,coarse))
        p,s,c = d.particles,source.particles,source.cells
        @cuda threads=128 blocks=cld(length(p),128) resolution_exchange_kernel!(ResolutionStep(state,d,6),Int32(length(p)),
            p.Position,p.Velocity,p.Density,p.Type,s.Position,s.Velocity,s.Density,s.Type,c.CellStart,c.grid_dev,
            source.kernel,source.constants,limits,d.fine,source.fine)
    end
    @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,coarse,6),Int32(7))
    for (phase,d) in ((7,coarse),(8,fine))
        enqueue_resolution_rebuild!(d,ResolutionStep(state,d,phase; created_only=true))
        @cuda threads=1 resolution_phase_kernel!(ResolutionStep(state,d,phase),Int32(phase+1))
    end
    @cuda threads=1 resolution_commit_kernel!(ResolutionStep(state,coarse,9))
    return nothing
end

function initialize_resident_domains!(domains)
    state = GPUStepState.StepState{Float64}()
    state.fh[3] = Inf
    copyto!(state.f,state.fh)
    for d in (domains...,map(d -> d.predictor,domains)...)
        c,p = d.cells,d.particles
        # Initialization can allocate and synchronize. Steady-state rebuilds
        # use the same validated metadata kernel without its host-return API.
        prepare_cell_grid!(c,p.Position,d.kernel.H⁻¹; freeze_state=(; types=p.Type))
        sync_cell_grid!(c)
        hasproperty(d,:fine) && enqueue_resolution_rebuild!(d,ResolutionStep(state,d,0))
    end
    CUDA.synchronize()
    GPUStepState.readback!(state)
    state.ih[1] == 0 || error("Variable-resolution grid initialization failed")
    return state
end

function prepare_resolution_graph!(domains,limits,state; batch_steps,use_graphs)
    !use_graphs && return nothing
    key = (UInt(0),0,Int(batch_steps))
    haskey(state.graphs,key) && return nothing
    # Compile with stopped state; even scans are gated, so warmup/capture never
    # destroys the accepted grid or mutates the flux budget.
    saved_stop = state.ih[1]
    GPUStepState.set_slot!(state.i,state.ih,1,GPUStepState.STOP_OUTPUT)
    enqueue_resolution_step!(domains,limits,state)
    CUDA.synchronize()
    graph = CUDA.capture(; flags=CUDA.STREAM_CAPTURE_MODE_RELAXED) do
        for _ in 1:batch_steps
            enqueue_resolution_step!(domains,limits,state)
        end
    end
    state.graphs[key] = CUDA.instantiate(graph)
    GPUStepState.set_slot!(state.i,state.ih,1,saved_stop)
    return nothing
end

function resolution_pending_domain(domains,phase)
    phase == 1 && return domains[1].predictor
    phase == 2 && return domains[2].predictor
    phase in (4,7) && return domains[1]
    phase in (5,8) && return domains[2]
    error("Invalid variable-resolution rebuild phase: $phase")
end

function advance_resolution!(domains,limits,state,deadline; batch_steps=32,use_graphs=true)
    GPUStepState.set_output_time!(state,deadline; clip_timestep=true)
    while state.fh[2] < deadline
        prepare_resolution_graph!(domains,limits,state; batch_steps,use_graphs)
        if use_graphs
            CUDA.launch(state.graphs[(UInt(0),0,Int(batch_steps))])
            state.graph_batches += 1
        else
            for _ in 1:batch_steps
                enqueue_resolution_step!(domains,limits,state)
            end
            state.direct_batches += 1
        end
        GPUStepState.readback!(state)
        stop = state.ih[1]
        if stop in (RESOLUTION_CAPACITY,RESOLUTION_GRID_ERROR)
            d = resolution_pending_domain(domains,state.ih[2])
            sync_cell_grid!(d.cells) # exceptional growth/error only
            empty!(state.graphs)
            GPUStepState.set_slot!(state.i,state.ih,1,0)
        elseif stop == RESOLUTION_RESERVE_ERROR
            error("Variable-resolution particle reserve exhausted; increase particle_reserve (no buffer particles were silently dropped)")
        elseif stop == RESOLUTION_STATE_ERROR
            error("Non-finite variable-resolution particle state or nonpositive/nonfinite VRes timestep")
        elseif !(stop in (0,GPUStepState.STOP_OUTPUT))
            error("Unexpected variable-resolution device status: $stop")
        end
    end
    return nothing
end

function sync_resolution_counters!(domains)
    for d in (domains...,map(d -> d.predictor,domains)...)
        count = Array(d.pool.count)
        d.cells.nrebuilds = count[3]
        d.cells.frozen_total = length(d.particles)-count[1]
        d.cells.grid = only(Array(d.cells.grid_dev))
    end
    return nothing
end
