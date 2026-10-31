# Normal RunSimulation integration; the standalone validation driver remains
# available, while both entry points share the same GPU phase machine.
using StructArrays: StructArray, components
import ..PreProcess
using ..SimulationMetaDataConfiguration: SimulationMetaData, NoShifting, NoKernelOutput,
    SimpleMDBC, StoreLog, resolve_output_variables!
using ..SPHCellList: run_variable_resolution
import ..SPHCellList
using ..SimulationResults: SimulationResult
using ..GPUConfiguration: effective_configuration, configuration_snapshot, log_effective_configuration
using ..DampingZones: prepare_damping
using ..ProduceHDFVTK: SetupVTKOutput
using ..SPHMeasurements: has_measurements
using ..OpenExternalPrograms: AutoOpenParaview, AutoOpenLogFile

function fine_resolution_parameters(config,constants::SimulationConstants{T},kernel) where T
    ratio = config.region.ratio
    fine_constants = if config.fine_constants === nothing
        fields = NamedTuple{fieldnames(typeof(constants))}(Tuple(getfield(constants,f) for f in fieldnames(typeof(constants))))
        SimulationConstants{T}(; merge(fields,(; dx=constants.dx/ratio,m₀=constants.m₀/ratio^2))...)
    else
        config.fine_constants
    end
    fine_kernel = config.fine_kernel === nothing ? SPHKernelInstance{2,T}(kernel.kernel;
        h=T(kernel.h/ratio),k=kernel.k,η²=T(kernel.η²/ratio^2),αD=T(kernel.αD*ratio^2)) : config.fine_kernel
    typeof(fine_constants) === typeof(constants) || throw(ArgumentError("fine and coarse constants must use the same precision"))
    validate_resolution_physics(constants,fine_constants,kernel,fine_kernel,config.region)
    return fine_constants,fine_kernel
end

function validate_resolution_solver(meta::SimulationMetaData{D,T,S,K,B},geometry,particles,
        viscosity,diffusion,scheme,measurements,external,rotation,motion) where {D,T,S,K,B}
    D == 2 && S === NoShifting && K === NoKernelOutput && B === SimpleMDBC ||
        throw(ArgumentError("variable resolution requires 2D NoShifting, NoKernelOutput and SimpleMDBC metadata modes"))
    scheme isa DualSPHysicsSymplectic || throw(ArgumentError("variable resolution requires SimTimeStepping=DualSPHysicsSymplectic()"))
    viscosity isa ArtificialViscosity && diffusion isa FourtakasDensityDiffusion && diffusion.full ||
        throw(ArgumentError("variable resolution requires ArtificialViscosity() and FourtakasDensityDiffusion(full=true)"))
    # The reference kernels currently use the default boundary viscosity and
    # unrestricted reference integration; reject other policies explicitly.
    viscosity == ArtificialViscosity() || throw(ArgumentError("variable resolution currently requires default ArtificialViscosity parameters"))
    all(!isfinite,scheme.domain_min) && all(!isfinite,scheme.domain_max) &&
        !isfinite(scheme.density_min) && !isfinite(scheme.density_max) &&
        !isfinite(scheme.smoothing_length) && !isfinite(scheme.sound_speed) && !scheme.cache_periodic_queries ||
        throw(ArgumentError("variable resolution derives per-domain timestep lengths and uses its own domain/density retirement policy; custom reference bounds are unsupported"))
    meta.Periodicity === nothing && !meta.GPUFixedGrid && meta.GPUCellSubdivision == 1 ||
        throw(ArgumentError("variable resolution requires nonperiodic expanding grids and GPUCellSubdivision=1"))
    meta.GPUTimestepDiagnosticsEvery == 0 || throw(ArgumentError("variable-resolution timestep-ID diagnostics are not supported"))
    measurements === nothing || !has_measurements(measurements) ||
        throw(ArgumentError("variable-resolution measurements require per-particle mass weighting and are not supported yet"))
    external === nothing && rotation === nothing && motion === nothing ||
        throw(ArgumentError("variable resolution currently supports fixed walls and SimDamping; external forces and prescribed motion are unsupported"))
    all(g -> g.Type in (Fixed,Fluid) && g.Motion === nothing && g.Floating === nothing,geometry) ||
        throw(ArgumentError("variable resolution supports only fluid and fixed wall geometry"))
    all(t -> t in (Fixed,Fluid),particles.Type) || throw(ArgumentError("variable resolution requires initially active fluid/fixed particles"))
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
    if normals_path !== nothing
        points,ghosts,normals = PreProcess.LoadBoundaryNormals(Val(2),Float64,normals_path)
        tolerance = Float64(spacing)*1e-3
        key(x) = ntuple(d -> round(Int,x[d]/tolerance),2)
        lookup = Dict(key(x) => i for (i,x) in enumerate(points))
        for i in eachindex(particles.Type)
            particles.Type[i] == Fixed || continue
            j = get(lookup,key(particles.Position[i]),0)
            j > 0 || throw(ArgumentError("Boundary normals do not match generated particles"))
            particles.GhostPoints[i],particles.GhostNormals[i] = ghosts[j],normals[j]
        end
    end
    fields = filter(f -> hasproperty(particles,f),(:Position,:Density,:Velocity,:ID,:GhostPoints,:GhostNormals))
    geometry = SPHGeometry{2,T}[]
    for (kind,marker) in unique(collect(zip(particles.Type,particles.GroupMarker)))
        indices = findall(i -> particles.Type[i] == kind && particles.GroupMarker[i] == marker,eachindex(particles.Type))
        data = StructArray(NamedTuple{fields}(map(f -> getproperty(particles,f)[indices],fields)))
        push!(geometry,SPHGeometry{2,T}(;Particles=data,Type=kind,GroupMarker=Int(marker)))
    end
    return geometry
end

function solver_resolution_snapshot(domain,region,limits; include_buffers=false)
    p,c = domain.particles,domain.constants
    x,types = Array(p.Position),Array(p.Type)
    owned = [inside_box(position,region.lower,region.upper) == domain.fine for position in x]
    keep = findall(i -> types[i] != Frozen && (include_buffers || owned[i]),eachindex(types))
    rho = Array(p.Density)[keep]
    return StructArray((;
        Position=x[keep],Velocity=Array(p.Velocity)[keep],Density=rho,
        Pressure=[eltype(rho)(Float64(c.Cb)*((Float64(r)/c.ρ₀)^c.γ-1)) for r in rho],
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

function refresh_resolution_output!(domains,state,interaction)
    # Rebuilds discard derivatives. Evaluate the accepted state once at a frame
    # boundary so acceleration and mDBC wall density are current in host output.
    stop = state.ih[GPUStepState.I_STOP]
    GPUStepState.set_slot!(state.i,state.ih,GPUStepState.I_STOP,0)
    for d in domains
        resident_force_stage!(d,ResolutionStep(state,d,0); interaction)
    end
    GPUStepState.set_slot!(state.i,state.ih,GPUStepState.I_STOP,stop)
    return nothing
end

function SPHCellList.run_variable_resolution(; SimGeometry,SimMetaData,SimConstants,SimKernel,SimLogger,SimParticles,
        SimViscosity,SimDensityDiffusion,SimTimeStepping,ParticleNormalsPath,SimMeasurements,SimExternalForces,
        SimDamping,SimRotation,SimMotion,ParaviewPressureRange)
    meta,config = SimMetaData,SimMetaData.VariableResolution
    T = eltype(SimParticles.Density)
    validate_resolution_solver(meta,SimGeometry,SimParticles,SimViscosity,SimDensityDiffusion,SimTimeStepping,
        SimMeasurements,SimExternalForces,SimRotation,SimMotion)
    fine_constants,fine_kernel = fine_resolution_parameters(config,SimConstants,SimKernel)
    deadlines = resolution_output_deadlines(meta)
    resolve_output_variables!(meta)
    damping = prepare_damping(SimDamping,Val(2))
    meta.TimeSteppingMode = SimTimeStepping
    initial_time,initial_steps = meta.TotalTime,meta.Iteration
    batch_steps = meta.GPUSyncTimers ? 1 : clamp(meta.GPUMaxStepsPerSync,1,256)
    use_graphs = meta.GPUUseGraph && !meta.GPUSyncTimers
    interaction = (;threads=meta.GPUInteractionThreads,always_inline=meta.GPUInteractionAlwaysInline)
    LOGGER.start_runtime!(SimLogger,initial_time)
    timer = meta.HourGlass
    output,history = nothing,nothing
    status = "failed"
    frames,write_seconds = 0,0.0
    times,files = Float64[],String[]
    snapshots = nothing
    local state,coarse,fine,domains,limits,frame
    history_path = joinpath(meta.SaveLocation,meta.SimulationName*"_History.csv")
    store_log = typeof(meta).parameters[6] === StoreLog
    try
        @timeit timer "00 Initialization" begin
            coarse_geometry = resolution_geometry(SimParticles,T; normals_path=ParticleNormalsPath,spacing=SimConstants.dx)
            limits = resolution_limits(config.region,Float64(SimKernel.h),Float64(fine_kernel.h))
            area = prod(limits.fine_outer_hi-limits.fine_outer_lo)
            reserves = map((SimConstants,fine_constants)) do c
                config.particle_reserve === nothing ? max(1024,ceil(Int,2area/c.dx^2)) : config.particle_reserve
            end
            coarse = make_resolution_domain(coarse_geometry,SimConstants,SimKernel,limits,false,reserves[1];resident=true)
            fine = make_resolution_domain(config.fine_geometry,fine_constants,fine_kernel,limits,true,reserves[2];resident=true)
            domains = (coarse,fine)
            for d in (domains...,coarse.predictor,fine.predictor)
                d.cells.max_cells = meta.GPUMaxCells
                d.cells.deterministic = meta.GPUDeterministicSort
            end
            foreach(d -> retire_particles!(d,limits),domains)
            state = initialize_resident_domains!(domains;time=initial_time,iteration=initial_steps)
            exchange_buffers!(coarse,fine,limits)
            exchange_buffers!(fine,coarse,limits)
            prepare_resolution_graph!(domains,limits,state;batch_steps,use_graphs,damping,interaction)
            state.configuration = effective_configuration(meta,SimKernel,SimDensityDiffusion,SimViscosity;
                particle_count=length(SimParticles))
            refresh_resolution_output!(domains,state,interaction)
            snapshots = map(d -> solver_resolution_snapshot(d,config.region,limits;include_buffers=config.include_buffers),domains)
            frame = vcat(snapshots...)
            replace_resolution_particles!(SimParticles,frame)
            output = SetupVTKOutput(meta,frame,SimKernel,2;constants=SimConstants,damping,variable_count=true)
            history = open(history_path,"w")
            println(history,"time,iteration,coarse_fluid,fine_fluid,represented_fluid_mass,coarse_created,fine_created,coarse_retired,fine_retired,coarse_excluded,fine_excluded,coarse_nonfinite,fine_nonfinite")
            if store_log
                log_effective_configuration(SimLogger,state.configuration;phase="variable-resolution startup")
                LOGGER.InitializeLogger(SimLogger,SimConstants,meta,SimKernel,SimViscosity,SimDensityDiffusion,
                    [coarse_geometry;config.fine_geometry],SimParticles;
                    gpu_buffers=(;particles=(coarse.particles,fine.particles,coarse.predictor.particles,fine.predictor.particles),
                        support=(coarse.support,fine.support,coarse.predictor.support,fine.predictor.support),
                        cells=(coarse.cells,fine.cells,coarse.predictor.cells,fine.predictor.cells),
                        auxiliary=(coarse,fine,state.f,state.i)))
                log_line(SimLogger,"Variable resolution: ratio=$(config.region.ratio), coarse/fine spacing=$(SimConstants.dx)/$(fine_constants.dx)")
                log_line(SimLogger,"Resolution pool capacities (coarse/fine): $(length(coarse.particles))/$(length(fine.particles)); includes dormant creation reserves")
            end
            CUDA.synchronize()
        end
        function write_frame!(; initial=false)
            started = time_ns()
            @timeit timer "13 Output Frame" begin
                events = map(d -> Array(d.events),domains)
                all(e -> e[4] == 0,events) || error("Non-finite variable-resolution particle state at t=$(meta.TotalTime)")
                if !initial
                    refresh_resolution_output!(domains,state,interaction)
                    snapshots = map(d -> solver_resolution_snapshot(d,config.region,limits;include_buffers=config.include_buffers),domains)
                    replace_resolution_particles!(frame,vcat(snapshots...))
                end
                meta.OutputIterationCounter = frames+1
                output.save_particles(meta.OutputIterationCounter,meta.TotalTime)
                if meta.ExportGridCells
                    cells = occupied_cells_host(frame.Position,SimKernel.H⁻¹)
                    output.save_grid(meta.OutputIterationCounter,cells,frame,meta.TotalTime)
                    meta.IndexCounter = length(cells)
                end
                counts = map(s -> count(==(Fluid),s.Type),snapshots)
                mass = sum(frame.Mass[i] for i in eachindex(frame.Type) if frame.Type[i] == Fluid && frame.Owned[i] == 1;init=0.0)
                println(history,join((meta.TotalTime,meta.Iteration,counts...,mass,events[1][1],events[2][1],events[1][2],events[2][2],events[1][3],events[2][3],events[1][4],events[2][4]),','))
                flush(history)
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
        for deadline in deadlines
            GPUStepState.output_time_reached(meta.TotalTime,deadline,meta.GPUClipTimeStepToOutput) && continue
            @timeit timer "03 Launch Steps" advance_resolution!(domains,limits,state,deadline;
                batch_steps,use_graphs,clip_timestep=meta.GPUClipTimeStepToOutput,damping,interaction)
            meta.TotalTime,meta.Iteration,meta.CurrentTimeStep = state.fh[2],Int(state.ih[3]),T(state.fh[1])
            write_frame!()
        end
        sync_resolution_counters!(domains)
        replace_resolution_particles!(SimParticles,frame)
        status = "finished"
    catch failure
        if failure isa InterruptException && output !== nothing && snapshots !== nothing
            GPUStepState.readback!(state)
            meta.TotalTime,meta.Iteration,meta.CurrentTimeStep = state.fh[2],Int(state.ih[3]),T(state.fh[1])
            sync_resolution_counters!(domains)
            snapshots = map(d -> solver_resolution_snapshot(d,config.region,limits;include_buffers=config.include_buffers),domains)
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
                history === nothing || close(history)
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
    final_state = (;particles=SimParticles,floating=nothing,time=meta.TotalTime,iteration=meta.Iteration,
        timestep=meta.CurrentTimeStep,complete=status == "finished",resolution=(;coarse=snapshots[1],fine=snapshots[2]))
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
