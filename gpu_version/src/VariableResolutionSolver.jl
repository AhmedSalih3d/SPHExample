# Normal RunSimulation integration; the standalone validation driver remains
# available, while both entry points share the same GPU phase machine.
using StructArrays: StructArray, components
import ..PreProcess
using ..SimulationMetaDataConfiguration: SimulationMetaData, NoShifting, PlanarShifting, NoKernelOutput,
    SimpleMDBC, StoreLog, resolve_output_variables!
using ..SPHCellList: run_variable_resolution
import ..SPHCellList
using ..SimulationResults: SimulationResult
using ..GPUConfiguration: effective_configuration, configuration_snapshot, log_effective_configuration, mdbc_launch_shape
using ..DampingZones: prepare_damping
using ..ProduceHDFVTK: SetupVTKOutput
using ..SPHMeasurements: has_measurements
using ..OpenExternalPrograms: AutoOpenParaview, AutoOpenLogFile

function fine_resolution_parameters(config,constants::SimulationConstants{T},kernel::SPHKernelInstance{K,D}) where {T,K,D}
    ratio = config.region.ratio
    fine_constants = if config.fine_constants === nothing
        fields = NamedTuple{fieldnames(typeof(constants))}(Tuple(getfield(constants,f) for f in fieldnames(typeof(constants))))
        SimulationConstants{T}(; merge(fields,(; dx=constants.dx/ratio,m₀=constants.m₀/ratio^D))...)
    else
        config.fine_constants
    end
    fine_kernel = config.fine_kernel === nothing ? SPHKernelInstance{D,T}(kernel.kernel;
        h=T(kernel.h/ratio),k=kernel.k,η²=T(kernel.η²/ratio^2),αD=T(kernel.αD*ratio^D)) : config.fine_kernel
    typeof(fine_constants) === typeof(constants) || throw(ArgumentError("fine and coarse constants must use the same precision"))
    validate_resolution_physics(constants,fine_constants,kernel,fine_kernel,config.region)
    return fine_constants,fine_kernel
end

function fixed_resolution_attachment_config(config,coarse_geometry,limits,coarse_kernel)
    bodies = filter(g -> g.Type == Floating,config.fine_geometry)
    isempty(bodies) && return nothing
    length(bodies) == 1 || throw(ArgumentError("a fixed 3D refinement box supports one floating body"))
    body = only(bodies)
    marker = body.GroupMarker
    body.Floating !== nothing && body.Floating.InteractionMass === nothing && body.Motion === nothing &&
        !isempty(body.Particles) && marker > 0 ||
        throw(ArgumentError("the fine floating body needs particles, a positive marker and FloatingDetails with standard interaction mass"))
    all(g -> g.GroupMarker != marker,coarse_geometry) &&
        count(g -> g.GroupMarker == marker,config.fine_geometry) == 1 ||
        throw(ArgumentError("the fine floating body's marker must be unique across both domains"))
    clearance = (;lower=limits.coarse_inner_lo .+ coarse_kernel.H,upper=limits.coarse_inner_hi .- coarse_kernel.H)
    all(x -> inside_box(x,clearance.lower,clearance.upper),body.Particles.Position) ||
        throw(ArgumentError("the fine floating body needs one coarse support radius of clearance inside the coarse exclusion box"))
    return (;marker,motion=nothing,boundary_motion=nothing,clearance)
end

function validate_resolution_solver(meta::SimulationMetaData{D,T,S,K,B},geometry,particles,
        viscosity,diffusion,scheme,measurements,external,rotation,motion) where {D,T,S,K,B}
    config = meta.VariableResolution
    config.region isa VariableResolutionBox{D} && K === NoKernelOutput ||
        throw(ArgumentError("variable resolution requires a box matching the simulation dimension and NoKernelOutput"))
    if D == 3
        S === NoShifting && B === UpdatedMDBC && scheme isa SymplecticTimeStepping &&
            config.follow === nothing && config.child === nothing ||
            throw(ArgumentError("3D variable resolution requires one fixed box, NoShifting, UpdatedMDBC and SymplecticTimeStepping"))
    else
        D == 2 && S in (NoShifting,PlanarShifting) && B in (NoMDBC,SimpleMDBC) ||
            throw(ArgumentError("2D variable resolution requires NoShifting or PlanarShifting and NoMDBC or SimpleMDBC"))
    end
    scheme isa Union{SymplecticTimeStepping,SingleNeighborTimeStepping,DualSPHysicsSymplectic} ||
        throw(ArgumentError("unsupported variable-resolution timestep scheme: $(typeof(scheme))"))
    S === PlanarShifting && scheme isa DualSPHysicsSymplectic &&
        throw(ArgumentError("variable-resolution PlanarShifting requires SymplecticTimeStepping or SingleNeighborTimeStepping"))
    if scheme isa DualSPHysicsSymplectic
        viscosity isa ArtificialViscosity || throw(ArgumentError("reference variable-resolution stepping requires ArtificialViscosity(); select a normal timestep scheme for other viscosity models"))
        all(!isfinite,scheme.domain_min) && all(!isfinite,scheme.domain_max) &&
            !isfinite(scheme.density_min) && !isfinite(scheme.density_max) &&
            !isfinite(scheme.smoothing_length) && !isfinite(scheme.sound_speed) && !scheme.cache_periodic_queries ||
            throw(ArgumentError("variable resolution derives per-domain timestep lengths and uses its own domain/density retirement policy; custom reference bounds are unsupported"))
    end
    meta.Periodicity === nothing && !meta.GPUFixedGrid && meta.GPUCellSubdivision == 1 && (meta.GPUXCellSubdivision in (1, 4)) ||
        throw(ArgumentError("variable resolution requires nonperiodic expanding grids and GPUCellSubdivision=1"))
    meta.GPUTimestepDiagnosticsEvery == 0 || throw(ArgumentError("variable-resolution timestep-ID diagnostics are not supported"))
    measurements === nothing || !has_measurements(measurements) ||
        throw(ArgumentError("variable-resolution measurements require per-particle mass weighting and are not supported yet"))
    external === nothing || throw(ArgumentError("variable resolution does not support SimExternalForces"))
    if meta.VariableResolution.follow === nothing
        rotation === nothing && motion === nothing &&
            all(g -> g.Type in (Fixed,Fluid) && g.Motion === nothing && g.Floating === nothing,geometry) ||
            throw(ArgumentError("moving/floating refinement requires VariableResolutionConfig(...; follow=GroupMarker)"))
        all(t -> t in (Fixed,Fluid),particles.Type) || throw(ArgumentError("variable resolution requires initially active fluid/fixed particles"))
    else
        scheme isa DualSPHysicsSymplectic && throw(ArgumentError("attached refinement requires SymplecticTimeStepping or SingleNeighborTimeStepping"))
        all(t -> t in (Fixed,Fluid,Moving,Floating),particles.Type) || throw(ArgumentError("attached refinement requires initially active particles"))
    end
    eltype(eltype(particles.Position)) === Float64 ||
        throw(ArgumentError("allocate particles with AllocateDataStructures(geometry, metadata) after enabling VariableResolution so host positions retain Float64 precision"))
    isfinite(meta.TotalTime) && isfinite(meta.SimulationTime) && meta.SimulationTime > meta.TotalTime ||
        throw(ArgumentError("SimulationTime must be finite and greater than TotalTime"))
    meta.Iteration >= 0 && meta.GPUMaxStepsPerSync > 0 && meta.GPUMaxCells > 0 ||
        throw(ArgumentError("iteration must be nonnegative and GPU batch/cell limits must be positive"))
    meta.GPUInteractionThreads in (32,64,128,256,512,1024) ||
        throw(ArgumentError("GPUInteractionThreads must be a supported whole-warp block size"))
    return nothing
end

function resolution_output_deadlines(meta)
    initial,finish = Float64(meta.TotalTime),Float64(meta.SimulationTime)
    times = if meta.OutputTimes isa AbstractVector
        all(isfinite,meta.OutputTimes) && issorted(meta.OutputTimes) && all(>(0),diff(meta.OutputTimes)) ||
            throw(ArgumentError("OutputTimes must be finite and strictly increasing"))
        filter(t -> initial < t < finish,meta.OutputTimes)
    else
        interval = Float64(meta.OutputTimes)
        isfinite(interval) && interval > 0 || throw(ArgumentError("OutputTimes must be a positive finite interval"))
        count = floor(Int,finish/interval)
        [i*interval for i in 1:count if initial < i*interval < finish]
    end
    return [times;finish]
end

function resolution_geometry(particles,::Type{T}; normals_path=nothing,spacing) where T
    D = length(eltype(particles.Position))
    if normals_path !== nothing
        points,ghosts,normals = PreProcess.LoadBoundaryNormals(Val(D),Float64,normals_path)
        tolerance = Float64(spacing)*1e-3
        key(x) = ntuple(d -> round(Int,x[d]/tolerance),D)
        lookup = Dict(key(x) => i for (i,x) in enumerate(points))
        for i in eachindex(particles.Type)
            particles.Type[i] == Fixed || continue
            j = get(lookup,key(particles.Position[i]),0)
            j > 0 || throw(ArgumentError("Boundary normals do not match generated particles"))
            particles.GhostPoints[i],particles.GhostNormals[i] = ghosts[j],normals[j]
        end
    end
    fields = filter(f -> hasproperty(particles,f),(:Position,:Density,:Velocity,:ID,:GhostPoints,:GhostNormals))
    geometry = SPHGeometry{D,T}[]
    for (kind,marker) in unique(collect(zip(particles.Type,particles.GroupMarker)))
        indices = findall(i -> particles.Type[i] == kind && particles.GroupMarker[i] == marker,eachindex(particles.Type))
        data = StructArray(NamedTuple{fields}(map(f -> getproperty(particles,f)[indices],fields)))
        push!(geometry,SPHGeometry{D,T}(;Particles=data,Type=kind,GroupMarker=Int(marker)))
    end
    return geometry
end

function solver_resolution_snapshot(domain,region,limits; include_buffers=false,scheme=DualSPHysicsSymplectic())
    p,c = domain.particles,domain.constants
    x,types = Array(p.Position),Array(p.Type)
    limits = resolution_host_limits(limits)
    owned = [resolution_owned(position,limits,domain.fine) for position in x]
    keep = findall(i -> types[i] != Frozen && (include_buffers || owned[i]),eachindex(types))
    rho = Array(p.Density)[keep]
    return StructArray((;
        Position=x[keep],Velocity=Array(p.Velocity)[keep],Density=rho,
        Pressure=[scheme isa DualSPHysicsSymplectic ? eltype(rho)(Float64(c.Cb)*((Float64(r)/c.ρ₀)^c.γ-1)+Float64(c.BackgroundPressure)) :
            GPUKernels.particle_pressure(r,types[i],c.c₀,c.ρ₀,domain.boundary_mode,inv(c.ρ₀),c.BackgroundPressure)
            for (i,r) in zip(keep,rho)],
        Acceleration=Array(p.Acceleration)[keep],Type=types[keep],ID=Array(p.ID)[keep],
        GroupMarker=Array(p.GroupMarker)[keep],GhostPoints=Array(p.GhostPoints)[keep],GhostNormals=Array(p.GhostNormals)[keep],
        Resolution=fill(Int8(domain.fine),length(keep)),Spacing=fill(c.dx,length(keep)),Mass=fill(c.m₀,length(keep)),
        Owned=Int8.(owned[keep]),IsBuffer=Int8[types[i] == Fluid && buffer_particle(x[i],limits,domain.fine) for i in keep],
        Cells=[CartesianIndex(Int.(cell_coords(position,domain.kernel.H⁻¹))) for position in x[keep]]))
end

function replace_resolution_particles!(destination,source)
    for (name,values) in pairs(components(destination))
        resize!(values,length(source))
        hasproperty(source,name) ? copyto!(values,getproperty(source,name)) : fill!(values,zero(eltype(values)))
    end
    return destination
end

function refresh_resolution_output!(domains,state,interaction; density_diffusion=LinearDensityDiffusion(),viscosity=ArtificialViscosity(),scheme=DualSPHysicsSymplectic())
    # Rebuilds discard derivatives. Evaluate the accepted state once at a frame
    # boundary so acceleration and mDBC wall density are current in host output.
    stop = state.ih[GPUStepState.I_STOP]
    GPUStepState.set_slot!(state.i,state.ih,GPUStepState.I_STOP,0)
    for d in domains
        resident_force_stage!(d,ResolutionStep(state,d,0); density_diffusion,viscosity,scheme,
            include_gravity=Val(!(scheme isa DualSPHysicsSymplectic)),interaction)
    end
    GPUStepState.set_slot!(state.i,state.ih,GPUStepState.I_STOP,stop)
    return nothing
end

function SPHCellList.run_variable_resolution(; SimGeometry,SimMetaData,SimConstants,SimKernel,SimLogger,SimParticles,
        SimViscosity,SimDensityDiffusion,SimTimeStepping,ParticleNormalsPath,SimMeasurements,SimExternalForces,
        SimDamping,SimRotation,SimMotion,ParaviewPressureRange,SimStateSampling=nothing)
    meta,config = SimMetaData,SimMetaData.VariableResolution
    D = length(eltype(SimParticles.Position))
    T = eltype(SimParticles.Density)
    validate_resolution_solver(meta,SimGeometry,SimParticles,SimViscosity,SimDensityDiffusion,SimTimeStepping,
        SimMeasurements,SimExternalForces,SimRotation,SimMotion)
    config.child === nothing || return run_nested_variable_resolution(;SimGeometry,SimMetaData,SimConstants,SimKernel,SimLogger,SimParticles,
        SimViscosity,SimDensityDiffusion,SimTimeStepping,ParticleNormalsPath,SimMeasurements,SimExternalForces,
        SimDamping,SimRotation,SimMotion,ParaviewPressureRange)
    fine_constants,fine_kernel = fine_resolution_parameters(config,SimConstants,SimKernel)
    limits = resolution_limits(config.region,Float64(SimKernel.h),Float64(fine_kernel.h))
    attachment_config = D == 3 ? fixed_resolution_attachment_config(config,SimGeometry,limits,SimKernel) :
        resolution_attachment_config(config,SimGeometry,SimRotation,SimMotion,SimKernel)
    body_marker = attachment_config === nothing ? nothing : attachment_config.marker
    deadlines = resolution_output_deadlines(meta)
    sample_times = SimStateSampling === nothing ? Float64[] : Float64.(SimStateSampling.times)
    isempty(sample_times) || body_marker !== nothing ||
        throw(ArgumentError("independent variable-resolution body sampling requires a fine floating body"))
    resolve_output_variables!(meta)
    damping = prepare_damping(SimDamping,Val(D))
    meta.TimeSteppingMode = SimTimeStepping
    initial_time,initial_steps = meta.TotalTime,meta.Iteration
    batch_steps = meta.GPUSyncTimers ? 1 : clamp(meta.GPUMaxStepsPerSync,1,256)
    use_graphs = meta.GPUUseGraph && !meta.GPUSyncTimers
    mdbc = mdbc_launch_shape(meta, 1)
    interaction = (; force = (; threads = meta.GPUInteractionThreads, always_inline = meta.GPUInteractionAlwaysInline),
        mdbc = (; threads = mdbc.threads, lanes = Val(mdbc.lanes)))
    density_diffusion = SimDensityDiffusion
    viscosity = SimViscosity
    scheme = SimTimeStepping
    boundary_mode = typeof(meta).parameters[5]()
    LOGGER.start_runtime!(SimLogger,initial_time)
    timer = meta.HourGlass
    output,history,floating_history,spring_history = nothing,nothing,nothing,nothing
    status = "failed"
    frames,write_seconds = 0,0.0
    times,files = Float64[],String[]
    snapshots = nothing
    local state,coarse,fine,domains,frame
    history_path = joinpath(meta.SaveLocation,meta.SimulationName*"_History.csv")
    store_log = typeof(meta).parameters[6] === StoreLog
    try
        @timeit timer "00 Initialization" begin
            coarse_geometry = resolution_geometry(SimParticles,T; normals_path=ParticleNormalsPath,spacing=SimConstants.dx)
            if config.follow !== nothing
                filter!(g -> g.GroupMarker != config.follow,coarse_geometry)
            end
            reserves = map((SimConstants,fine_constants),(false,true)) do c,is_fine
                config.particle_reserve === nothing || return config.particle_reserve
                if D == 2
                    return max(1024,ceil(Int,2prod(limits.fine_outer_hi-limits.fine_outer_lo)/c.dx^2))
                end
                return max(1024,2length(interface_points(limits,is_fine,Float64(c.dx)).points))
            end
            standard = !(scheme isa DualSPHysicsSymplectic)
            shifting = typeof(meta).parameters[3] === PlanarShifting
            prescribed_system = attachment_config === nothing ? nothing : attachment_config.boundary_motion
            coarse = make_resolution_domain(coarse_geometry,SimConstants,SimKernel,limits,false,reserves[1];resident=true,standard,pairwise_timestep=standard && !meta.GPULegacyTimestep,shifting,boundary_mode,prescribed_system)
            fine = make_resolution_domain(config.fine_geometry,fine_constants,fine_kernel,limits,true,reserves[2];
                resident=true,standard,pairwise_timestep=standard && !meta.GPULegacyTimestep,shifting,boundary_mode,attachment_config,prescribed_system,time=initial_time)
            if config.follow !== nothing
                limits = AttachedResolutionLimits(limits,CuArray(fill(identity_resolution_pose(),2)))
            end
            domains = (coarse,fine)
            for d in (domains...,coarse.predictor,fine.predictor)
                d.cells.max_cells = meta.GPUMaxCells
                d.cells.deterministic = meta.GPUDeterministicSort
            end
            state = GPUStepState.StepState{Float64}(;time=initial_time,iteration=initial_steps)
            initialize_resolution_attachment!(fine,limits,state)
            for d in domains
                place_resolution_prescribed!(d,ResolutionStep(state,d,0),0.0)
            end
            foreach(d -> retire_particles!(d,limits),domains)
            initialize_resident_domains!(domains;state)
            exchange_buffers!(coarse,fine,limits)
            exchange_buffers!(fine,coarse,limits)
            initialize_resolution_numerics!(domains,state,scheme;density_diffusion,viscosity)
            prepare_resolution_graph!(domains,limits,state;batch_steps,use_graphs,density_diffusion,viscosity,scheme,damping,interaction)
            state.configuration = effective_configuration(meta,SimKernel,SimDensityDiffusion,SimViscosity;
                particle_count=length(SimParticles))
            refresh_resolution_output!(domains,state,interaction; density_diffusion,viscosity,scheme)
            snapshots = map(d -> solver_resolution_snapshot(d,config.region,limits;include_buffers=config.include_buffers,scheme),domains)
            frame = vcat(snapshots...)
            replace_resolution_particles!(SimParticles,frame)
            body = fine.attachment === nothing || fine.attachment.floating === nothing ? nothing : floating_state(fine.attachment.floating)
            output = SetupVTKOutput(meta,frame,SimKernel,D;constants=SimConstants,damping,variable_count=true,
                floating_state=body,floating_markers=body === nothing ? Int[] : [body_marker])
            history = open(history_path,"w")
            println(history,"time,iteration,coarse_fluid,fine_fluid,represented_fluid_mass,coarse_created,fine_created,coarse_retired,fine_retired,coarse_excluded,fine_excluded,coarse_nonfinite,fine_nonfinite")
            if fine.attachment !== nothing && fine.attachment.floating !== nothing
                floating_history = open(joinpath(meta.SaveLocation,meta.SimulationName*"_Floating.csv"),"w")
                header = D == 2 ? "Time,Body,GroupMarker,Center:0,Center:1,Velocity:0,Velocity:1,Angle,Omega,Force:0,Force:1,Torque" :
                    "Time,Body,GroupMarker,Center:0,Center:1,Center:2,Velocity:0,Velocity:1,Velocity:2,Orientation:0,Orientation:1,Orientation:2,Orientation:3,Omega:0,Omega:1,Omega:2,Force:0,Force:1,Force:2,Torque:0,Torque:1,Torque:2"
                println(floating_history,header)
                if D == 2 && !isempty(fine.attachment.floating.springs)
                    spring_history = open(joinpath(meta.SaveLocation,meta.SimulationName*"_Springs.csv"),"w")
                    println(spring_history,"Time,Body,Spring,Anchor:0,Anchor:1,Attachment:0,Attachment:1,Length,Extension,Force:0,Force:1,Torque")
                end
            end
            if store_log
                log_effective_configuration(SimLogger,state.configuration;phase="variable-resolution startup")
                LOGGER.InitializeLogger(SimLogger,SimConstants,meta,SimKernel,SimViscosity,SimDensityDiffusion,
                    [coarse_geometry;config.fine_geometry],SimParticles;
                    gpu_buffers=(;particles=(coarse.particles,fine.particles,coarse.predictor.particles,fine.predictor.particles),
                        support=(coarse.support,fine.support,coarse.predictor.support,fine.predictor.support),
                        cells=(coarse.cells,fine.cells,coarse.predictor.cells,fine.predictor.cells),
                        auxiliary=(coarse,fine,state.f,state.i)))
                log_line(SimLogger,"Variable resolution: ratio=$(config.region.ratio), coarse/fine spacing=$(SimConstants.dx)/$(fine_constants.dx)")
                config.follow === nothing || log_line(SimLogger,"Refinement follows body GroupMarker=$(config.follow); fine-domain body forces and GPU rigid transforms")
                log_line(SimLogger,"Resolution pool capacities (coarse/fine): $(length(coarse.particles))/$(length(fine.particles)); includes dormant creation reserves")
            end
            CUDA.synchronize()
        end
        function write_body!(body)
            orientation = D == 2 ? (body.angle[1],body.omega[1]) : (body.orientation[1]...,body.omega[1]...)
            torque = D == 2 ? (body.load_torque[1],) : Tuple(body.load_torque[1])
            println(floating_history,join((meta.TotalTime,1,body_marker,body.center[1]...,body.velocity[1]...,
                orientation...,body.load_force[1]...,torque...),','))
            flush(floating_history)
        end
        function write_frame!(; initial=false)
            started = time_ns()
            @timeit timer "13 Output Frame" begin
                events = map(d -> Array(d.events),domains)
                all(e -> e[4] == 0,events) || error("Non-finite variable-resolution particle state at t=$(meta.TotalTime)")
                if !initial
                    refresh_resolution_output!(domains,state,interaction; density_diffusion,viscosity,scheme)
                    snapshots = map(d -> solver_resolution_snapshot(d,config.region,limits;include_buffers=config.include_buffers,scheme),domains)
                    replace_resolution_particles!(frame,vcat(snapshots...))
                end
                meta.OutputIterationCounter = frames+1
                body = fine.attachment === nothing || fine.attachment.floating === nothing ? nothing : floating_state(fine.attachment.floating)
                output.save_particles(meta.OutputIterationCounter,meta.TotalTime;body_state=body)
                if meta.ExportGridCells
                    cells = occupied_cells_host(frame.Position,SimKernel.H⁻¹)
                    output.save_grid(meta.OutputIterationCounter,cells,frame,meta.TotalTime)
                    meta.IndexCounter = length(cells)
                end
                counts = map(s -> count(==(Fluid),s.Type),snapshots)
                mass = sum(frame.Mass[i] for i in eachindex(frame.Type) if frame.Type[i] == Fluid && frame.Owned[i] == 1;init=0.0)
                println(history,join((meta.TotalTime,meta.Iteration,counts...,mass,events[1][1],events[2][1],events[1][2],events[2][2],events[1][3],events[2][3],events[1][4],events[2][4]),','))
                flush(history)
                if floating_history !== nothing && (isempty(sample_times) || initial || meta.TotalTime in sample_times)
                    write_body!(body)
                end
                if spring_history !== nothing
                    for spring in floating_spring_state(fine.attachment.floating)
                        println(spring_history,join((meta.TotalTime,spring.body,spring.spring,spring.anchor...,
                            spring.attachment...,spring.length,spring.extension,spring.force...,spring.torque),','))
                    end
                    flush(spring_history)
                end
                push!(times,meta.TotalTime)
                if !meta.ExportSingleVTKHDF
                    push!(files,meta.SimulationName*"_"*lpad(string(frames+1),6,'0')*".vtkhdf")
                    open(joinpath(meta.SaveLocation,meta.SimulationName*".vtkhdf.series"),"w") do io
                        println(io,"{\"file-series-version\": \"1.0\", \"files\": [")
                        for i in eachindex(files)
                            println(io,"{\"name\": \"",files[i],"\", \"time\": ",times[i],"}",i == length(files) ? "" : ",")
                        end
                        println(io,"]}")
                    end
                end
                store_log && LogStep(SimLogger,meta,timer)
                meta.StepsTakenForLastOutput = meta.Iteration
                frames += 1
            end
            write_seconds += (time_ns()-started)/1e9
            return nothing
        end
        write_frame!(;initial=true)
        for deadline in sort!(union(deadlines,sample_times))
            GPUStepState.output_time_reached(meta.TotalTime,deadline,meta.GPUClipTimeStepToOutput) && continue
            @timeit timer "03 Launch Steps" advance_resolution!(domains,limits,state,deadline;
                batch_steps,use_graphs,clip_timestep=meta.GPUClipTimeStepToOutput,density_diffusion,viscosity,scheme,damping,interaction)
            meta.TotalTime,meta.Iteration,meta.CurrentTimeStep = state.fh[2],Int(state.ih[3]),T(state.fh[1])
            if deadline in deadlines
                write_frame!()
            elseif floating_history !== nothing
                write_body!(floating_state(fine.attachment.floating))
            end
        end
        sync_resolution_counters!(domains)
        replace_resolution_particles!(SimParticles,frame)
        status = "finished"
    catch failure
        if failure isa InterruptException && output !== nothing && snapshots !== nothing
            GPUStepState.readback!(state)
            meta.TotalTime,meta.Iteration,meta.CurrentTimeStep = state.fh[2],Int(state.ih[3]),T(state.fh[1])
            sync_resolution_counters!(domains)
            snapshots = map(d -> solver_resolution_snapshot(d,config.region,limits;include_buffers=config.include_buffers,scheme),domains)
            replace_resolution_particles!(SimParticles,vcat(snapshots...))
            status = "interrupted"
        else
            store_log && log_line(SimLogger,"Simulation error: "*sprint(showerror,failure))
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
                if @isdefined(state)
                    log_effective_configuration(SimLogger,configuration_snapshot(state);phase="variable-resolution final",to_console=false)
                end
                LogFinal(SimLogger,timer;status)
                close(SimLogger.LoggerIo)
                AutoOpenLogFile(SimLogger,meta)
            end
        end
    end
    elapsed = LOGGER.elapsed_wall_time(SimLogger)
    steps,simulated = meta.Iteration-initial_steps,meta.TotalTime-initial_time
    configuration = configuration_snapshot(state)
    body_state = fine.attachment === nothing || fine.attachment.floating === nothing ? nothing : floating_state(fine.attachment.floating)
    springs = fine.attachment === nothing || fine.attachment.floating === nothing ? NamedTuple[] : floating_spring_state(fine.attachment.floating)
    region_pose = limits isa AttachedResolutionLimits ? first(Array(limits.frames)) : identity_resolution_pose(Val(D))
    final_state = (;particles=SimParticles,floating=body_state,time=meta.TotalTime,iteration=meta.Iteration,
        timestep=meta.CurrentTimeStep,complete=status == "finished",resolution=(;coarse=snapshots[1],fine=snapshots[2]),resolution_pose=region_pose,springs)
    performance = (;steps,simulated_time=simulated,wall_time_seconds=elapsed,mean_timestep=steps > 0 ? simulated/steps : NaN,
        milliseconds_per_step=steps > 0 ? 1e3elapsed/steps : NaN,mparticle_steps_per_second=NaN,
        output_write_seconds=write_seconds,phase_timings=copy(timer),phase_timings_synchronized=meta.GPUSyncTimers)
    cells = (coarse.cells,fine.cells,coarse.predictor.cells,fine.predictor.cells)
    diagnostics = (;particle_count=length(SimParticles),output_frames=frames,cell_list_rebuilds=sum(c.nrebuilds for c in cells),
        grid_dims=(;coarse=Tuple(coarse.cells.grid.dims),fine=Tuple(fine.cells.grid.dims)),
        periodic_source_rebuilds=0,periodic_source_count=0,frozen_particles=0,
        step_state_readbacks=state.readbacks,grid_status_readbacks=sum(c.grid_status_readbacks for c in cells),
        capacity_growths=sum(c.capacity_growths for c in cells),bbox_host_readbacks=sum(c.bbox_host_readbacks for c in cells),
        captured_graphs=length(state.graphs),resolution=(;coarse_particles=length(snapshots[1]),fine_particles=length(snapshots[2]),
            capacities=(length(coarse.particles),length(fine.particles)),
            dormant_slots=(coarse.cells.frozen_total,fine.cells.frozen_total),
            events=map(d -> Array(d.events),domains),history_path))
    AutoOpenParaview(meta,SimConstants,meta.OutputVariables;pressure_range=ParaviewPressureRange,show_damping=!isempty(damping))
    return SimulationResult(final_state,performance,diagnostics,status == "finished" ? :completed : :interrupted,configuration)
end
