function nested_resolution_plan(config,constants,kernel)
    configs,cs,ks = VariableResolutionConfig[config],[constants],[kernel]
    current = config
    while true
        c,k = fine_resolution_parameters(current,last(cs),last(ks))
        push!(cs,c); push!(ks,k)
        current.child === nothing && break
        current = current.child
        push!(configs,current)
    end
    length(cs) <= 128 || throw(ArgumentError("Resolution output supports at most 128 nested levels"))
    references = Tuple(resolution_limits(c.region,Float64(ks[i].h),Float64(ks[i+1].h)) for (i,c) in enumerate(configs))
    for i in 2:length(references)
        parent,child = references[i-1],references[i]
        all(child.fine_outer_lo .- ks[i].H .> parent.fine_inner_lo) &&
            all(child.fine_outer_hi .+ ks[i].H .< parent.fine_inner_hi) ||
            throw(ArgumentError("nested refinement buffers need one parent-level support radius of clearance inside the parent owned rectangle"))
    end
    return (;configs=Tuple(configs),constants=Tuple(cs),kernels=Tuple(ks),references)
end

function hierarchy_snapshots(domains,h,config,scheme)
    return map(eachindex(domains)) do i
        snapshot = solver_resolution_snapshot(domains[i],config.region,h.levels[i];include_buffers=config.include_buffers,scheme)
        snapshot.Resolution .= Int8(i-1)
        snapshot
    end |> Tuple
end

function run_nested_variable_resolution(;SimGeometry,SimMetaData,SimConstants,SimKernel,SimLogger,SimParticles,
        SimViscosity,SimDensityDiffusion,SimTimeStepping,ParticleNormalsPath,SimMeasurements,SimExternalForces,
        SimDamping,SimRotation,SimMotion,ParaviewPressureRange)
    meta,config = SimMetaData,SimMetaData.VariableResolution
    validate_resolution_solver(meta,SimGeometry,SimParticles,SimViscosity,SimDensityDiffusion,SimTimeStepping,
        SimMeasurements,SimExternalForces,SimRotation,SimMotion)
    SimTimeStepping isa Union{SymplecticTimeStepping,SingleNeighborTimeStepping} ||
        throw(ArgumentError("nested resolution supports ordinary symplectic or single-neighbor stepping"))
    meta.ExportSingleVTKHDF || throw(ArgumentError("nested resolution requires ExportSingleVTKHDF=true"))
    ParticleNormalsPath === nothing || throw(ArgumentError("supply nested boundary ghost fields in the geometry instead of ParticleNormalsPath"))
    SimMotion === nothing && SimRotation === nothing || throw(ArgumentError("nested tracking currently supports a force-responsive floating body"))
    plan = nested_resolution_plan(config,SimConstants,SimKernel)
    n = length(plan.constants)
    leaf_config = last(plan.configs)
    attachment = resolution_attachment_config(leaf_config,SimGeometry,nothing,nothing,plan.kernels[end-1])
    attachment === nothing || attachment.motion === nothing && attachment.boundary_motion === nothing ||
        throw(ArgumentError("nested tracking supports one floating body without additional prescribed walls"))
    geometries = (SimGeometry,map(c -> c.fine_geometry,plan.configs)...)
    for i in 1:n-1
        all(g -> g.Type in (Fixed,Fluid) && g.Floating === nothing && g.Motion === nothing,geometries[i]) ||
            throw(ArgumentError("only the finest nested level may contain the followed floating body"))
    end
    deadlines = resolution_output_deadlines(meta)
    resolve_output_variables!(meta)
    meta.TimeSteppingMode = SimTimeStepping
    damping = prepare_damping(SimDamping,Val(2))
    scheme,density_diffusion,viscosity = SimTimeStepping,SimDensityDiffusion,SimViscosity
    boundary_mode = typeof(meta).parameters[5]()
    interaction = (;threads=meta.GPUInteractionThreads,always_inline=meta.GPUInteractionAlwaysInline)
    batch_steps = meta.GPUSyncTimers ? 1 : clamp(meta.GPUMaxStepsPerSync,1,256)
    use_graphs = meta.GPUUseGraph && !meta.GPUSyncTimers
    initial_time,initial_steps = meta.TotalTime,meta.Iteration
    LOGGER.start_runtime!(SimLogger,initial_time)
    timer = meta.HourGlass
    store_log = typeof(meta).parameters[6] === StoreLog
    output,history,floating_history,spring_history = nothing,nothing,nothing,nothing
    status,frames,write_seconds = "failed",0,0.0
    snapshots = nothing
    local domains,h,state,frame
    history_path = joinpath(meta.SaveLocation,meta.SimulationName*"_History.csv")
    try
        @timeit timer "00 Initialization" begin
            poses = config.follow === nothing ? nothing : CuArray(fill(identity_resolution_pose(),2))
            pairs = poses === nothing ? plan.references : map(l -> AttachedResolutionLimits(l,poses),plan.references)
            domains = ntuple(n) do i
                ci = plan.configs[max(1,i-1)]
                pair = pairs[max(1,i-1)]
                area = prod(resolution_base(pair).fine_outer_hi-resolution_base(pair).fine_outer_lo)
                reserve = ci.particle_reserve === nothing ? max(1024,ceil(Int,2area/plan.constants[i].dx^2)) : ci.particle_reserve
                make_resolution_domain(geometries[i],plan.constants[i],plan.kernels[i],pair,i>1,reserve;
                    resident=true,standard=true,boundary_mode,attachment_config=i == n ? attachment : nothing,
                    time=initial_time,id_stride=n,id_offset=i)
            end
            h = resolution_hierarchy(pairs,domains)
            for d in (domains...,map(d -> d.predictor,domains)...)
                d.cells.max_cells = meta.GPUMaxCells
                d.cells.deterministic = meta.GPUDeterministicSort
            end
            state = GPUStepState.StepState{Float64}(;time=initial_time,iteration=initial_steps)
            initialize_resolution_attachment!(last(domains),first(pairs),state)
            for (i,d) in enumerate(domains)
                retire_particles!(d,h.levels[i])
            end
            initialize_resident_domains!(domains;state)
            exchange_hierarchy!(domains,h,state,0)
            initialize_resolution_numerics!(domains,state,scheme;density_diffusion,viscosity)
            state.configuration = effective_configuration(meta,SimKernel,density_diffusion,viscosity;particle_count=sum(length(d.particles) for d in domains))
            refresh_resolution_output!(domains,state,interaction;density_diffusion,viscosity,scheme)
            snapshots = hierarchy_snapshots(domains,h,config,scheme)
            frame = vcat(snapshots...)
            replace_resolution_particles!(SimParticles,frame)
            output = SetupVTKOutput(meta,frame,SimKernel,2;constants=SimConstants,damping,variable_count=true)
            history = open(history_path,"w")
            println(history,join(vcat(["time","iteration"],["level$(i-1)_fluid" for i in 1:n],["represented_fluid_mass"],
                ["level$(i-1)_$(event)" for event in ("created","retired","excluded","nonfinite") for i in 1:n]),','))
            if last(domains).attachment !== nothing
                floating_history = open(joinpath(meta.SaveLocation,meta.SimulationName*"_Floating.csv"),"w")
                println(floating_history,"Time,Body,GroupMarker,Center:0,Center:1,Velocity:0,Velocity:1,Angle,Omega")
                if !isempty(last(domains).attachment.floating.springs)
                    spring_history = open(joinpath(meta.SaveLocation,meta.SimulationName*"_Springs.csv"),"w")
                    println(spring_history,"Time,Body,Spring,Anchor:0,Anchor:1,Attachment:0,Attachment:1,Length,Extension,Force:0,Force:1,Torque")
                end
            end
            if store_log
                log_effective_configuration(SimLogger,state.configuration;phase="nested-resolution startup")
                LOGGER.InitializeLogger(SimLogger,SimConstants,meta,SimKernel,viscosity,density_diffusion,
                    vcat(geometries...),SimParticles;
                    gpu_buffers=(;particles=map(d -> (d.particles,d.predictor.particles),domains),
                        support=map(d -> (d.support,d.predictor.support),domains),
                        cells=map(d -> (d.cells,d.predictor.cells),domains),auxiliary=(domains,h,state.f,state.i)))
                log_line(SimLogger,"Nested resolution: $n levels; spacings=$(map(c -> c.dx,plan.constants)); followed group=$(config.follow)")
            end
            CUDA.synchronize()
        end
        function write_frame!(initial=false)
            started = time_ns()
            if !initial
                refresh_resolution_output!(domains,state,interaction;density_diffusion,viscosity,scheme)
                snapshots = hierarchy_snapshots(domains,h,config,scheme)
                replace_resolution_particles!(frame,vcat(snapshots...))
            end
            events = map(d -> Array(d.events),domains)
            all(e -> e[4] == 0,events) || error("Non-finite nested-resolution state at t=$(meta.TotalTime)")
            meta.OutputIterationCounter = frames+1
            output.save_particles(frames+1,meta.TotalTime)
            if meta.ExportGridCells
                cells = occupied_cells_host(frame.Position,SimKernel.H⁻¹)
                output.save_grid(frames+1,cells,frame,meta.TotalTime)
            end
            counts = map(s -> count(==(Fluid),s.Type),snapshots)
            mass = sum(frame.Mass[i] for i in eachindex(frame.Type) if frame.Type[i] == Fluid && frame.Owned[i] == 1;init=0.0)
            println(history,join((meta.TotalTime,meta.Iteration,counts...,mass,(events[i][event] for event in 1:4 for i in 1:n)...),','))
            flush(history)
            if floating_history !== nothing
                body = floating_state(last(domains).attachment.floating)
                println(floating_history,join((meta.TotalTime,1,config.follow,body.center[1]...,body.velocity[1]...,body.angle[1],body.omega[1]),','))
                flush(floating_history)
            end
            if spring_history !== nothing
                for spring in floating_spring_state(last(domains).attachment.floating)
                    println(spring_history,join((meta.TotalTime,spring.body,spring.spring,spring.anchor...,
                        spring.attachment...,spring.length,spring.extension,spring.force...,spring.torque),','))
                end
                flush(spring_history)
            end
            store_log && LogStep(SimLogger,meta,timer)
            meta.StepsTakenForLastOutput = meta.Iteration
            frames += 1
            write_seconds += (time_ns()-started)/1e9
            return nothing
        end
        write_frame!(true)
        for deadline in deadlines
            advance_resolution!(domains,h,state,deadline;batch_steps,use_graphs,clip_timestep=meta.GPUClipTimeStepToOutput,
                density_diffusion,viscosity,scheme,damping,interaction)
            meta.TotalTime,meta.Iteration,meta.CurrentTimeStep = state.fh[2],Int(state.ih[3]),typeof(SimConstants.dx)(state.fh[1])
            write_frame!()
        end
        sync_resolution_counters!(domains)
        replace_resolution_particles!(SimParticles,frame)
        status = "finished"
    catch failure
        if failure isa InterruptException && snapshots !== nothing
            GPUStepState.readback!(state)
            meta.TotalTime,meta.Iteration = state.fh[2],Int(state.ih[3])
            snapshots = hierarchy_snapshots(domains,h,config,scheme)
            replace_resolution_particles!(SimParticles,vcat(snapshots...))
            sync_resolution_counters!(domains)
            status = "interrupted"
        else
            rethrow()
        end
    finally
        try
            try
                output === nothing || output.close_files()
            finally
                try
                    history === nothing || close(history)
                finally
                    try
                        floating_history === nothing || close(floating_history)
                    finally
                        spring_history === nothing || close(spring_history)
                    end
                end
            end
        catch
            status = "failed"
            rethrow()
        finally
            if store_log
                try
                    LogFinal(SimLogger,timer;status)
                finally
                    isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
                end
            end
        end
    end
    elapsed = LOGGER.elapsed_wall_time(SimLogger)
    steps,simulated = meta.Iteration-initial_steps,meta.TotalTime-initial_time
    leaf = last(domains)
    body = leaf.attachment === nothing ? nothing : floating_state(leaf.attachment.floating)
    pose = leaf.attachment === nothing ? identity_resolution_pose() : first(Array(first(h.pairs).frames))
    final_state = (;particles=SimParticles,floating=body,time=meta.TotalTime,iteration=meta.Iteration,
        timestep=meta.CurrentTimeStep,complete=status == "finished",resolution=(;levels=snapshots,coarse=first(snapshots),fine=last(snapshots)),
        resolution_pose=pose,springs=leaf.attachment === nothing ? NamedTuple[] : floating_spring_state(leaf.attachment.floating))
    performance = (;steps,simulated_time=simulated,wall_time_seconds=elapsed,mean_timestep=steps > 0 ? simulated/steps : NaN,
        milliseconds_per_step=steps > 0 ? 1e3elapsed/steps : NaN,mparticle_steps_per_second=NaN,
        output_write_seconds=write_seconds,phase_timings=copy(timer),phase_timings_synchronized=meta.GPUSyncTimers)
    cells = (map(d -> d.cells,domains)...,map(d -> d.predictor.cells,domains)...)
    diagnostics = (;particle_count=length(SimParticles),output_frames=frames,cell_list_rebuilds=sum(c.nrebuilds for c in cells),
        grid_dims=map(d -> Tuple(d.cells.grid.dims),domains),periodic_source_rebuilds=0,periodic_source_count=0,frozen_particles=0,
        step_state_readbacks=state.readbacks,grid_status_readbacks=sum(c.grid_status_readbacks for c in cells),
        capacity_growths=sum(c.capacity_growths for c in cells),bbox_host_readbacks=sum(c.bbox_host_readbacks for c in cells),
        captured_graphs=length(state.graphs),resolution=(;level_particles=map(length,snapshots),capacities=map(d -> length(d.particles),domains),
            dormant_slots=map(d -> d.cells.frozen_total,domains),events=map(d -> Array(d.events),domains),history_path))
    AutoOpenParaview(meta,SimConstants,meta.OutputVariables;pressure_range=ParaviewPressureRange,show_damping=!isempty(damping))
    return SimulationResult(final_state,performance,diagnostics,status == "finished" ? :completed : :interrupted,configuration_snapshot(state))
end
