# Included in SPHCellList; the existing WCSPH driver and buffer contracts stay separate.
function validate_iisph(meta::SimulationMetaData{D,T,S,K,B}, geometry, particles,
                        constants, kernel, viscosity, diffusion, scheme;
                        normals=nothing, rigid=nothing, open_boundary=nothing,
                        damping=nothing, rotation=nothing, motion=nothing) where {D,T,S,K,B}
    D in (2,3) && T in (Float32,Float64) && S === NoShifting && K === NoKernelOutput && B === NoMDBC ||
        throw(ArgumentError("IISPH requires 2D/3D Float32/Float64, NoShifting, NoKernelOutput and NoMDBC"))
    meta.VariableResolution === nothing && meta.Periodicity === nothing && !meta.GPUFixedGrid ||
        throw(ArgumentError("IISPH requires uniform resolution, nonperiodic particles and an expanding grid"))
    meta.GPUCellSubdivision == 1 || throw(ArgumentError("the initial IISPH path requires GPUCellSubdivision=1"))
    meta.GPUTimestepDiagnosticsEvery == 0 || throw(ArgumentError("IISPH writes its own pressure/timestep history; WCSPH timestep-ID diagnostics are unsupported"))
    !meta.GPURadialInteractions && !meta.GPUScalarSPSViscosity && !meta.GPUCacheInteractionFields ||
        throw(ArgumentError("IISPH does not support WCSPH radial/SPS/interaction-cache options"))
    scheme === nothing || throw(ArgumentError("IISPH owns its projection and symplectic Euler integration; omit SimTimeStepping"))
    viscosity isa Union{ZeroViscosity,Laminar} || throw(ArgumentError("IISPH supports ZeroViscosity() or Laminar(); artificial/SPS viscosity is not supported yet"))
    diffusion isa ZeroDensityDiffusion || throw(ArgumentError("IISPH requires ZeroDensityDiffusion()"))
    all(isnothing,(normals,rigid,open_boundary,damping,rotation,motion)) ||
        throw(ArgumentError("IISPH does not yet support mDBC normals, rigid dynamics, open boundaries, damping or prescribed motion"))
    all(g -> g.Type in (Fluid,Fixed) && g.Motion === nothing && g.Floating === nothing,geometry) &&
        all(t -> t in (Fluid,Fixed),particles.Type) || throw(ArgumentError("IISPH supports only fluid and stationary fixed walls"))
    length(particles) > 0 && any(==(Fluid),particles.Type) || throw(ArgumentError("IISPH requires at least one fluid particle"))
    all(i -> particles.Type[i] == Fluid || iszero(particles.Velocity[i]),eachindex(particles.Type)) ||
        throw(ArgumentError("IISPH fixed walls must have zero velocity"))
    constants isa SimulationConstants{T} && kernel isa SPHKernelInstance{<:Any,D,T} ||
        throw(ArgumentError("IISPH constants, kernel and particles must use matching precision and dimension"))
    all(x -> all(isfinite,x),particles.Position) && all(x -> all(isfinite,x),particles.Velocity) ||
        throw(ArgumentError("IISPH requires finite initial positions and velocities"))
    all(x -> isfinite(x) && x > 0,(constants.ρ₀,constants.m₀,constants.dx,kernel.h,kernel.η²)) &&
        isfinite(constants.g) && isfinite(constants.ν₀) && constants.ν₀ >= 0 ||
        throw(ArgumentError("IISPH requires positive finite density, mass, spacing and kernel scales, finite gravity and nonnegative viscosity"))
    meta.TotalTime == 0 && meta.Iteration == 0 || throw(ArgumentError("IISPH restart from metadata is not supported; start at time and iteration zero"))
    isfinite(meta.SimulationTime) && meta.SimulationTime > 0 || throw(ArgumentError("IISPH SimulationTime must be positive and finite"))
    return nothing
end

function iisph_output_deadlines(meta)
    finish = meta.SimulationTime
    times = if meta.OutputTimes isa AbstractVector
        all(isfinite,meta.OutputTimes) && issorted(meta.OutputTimes) && all(>(0),diff(meta.OutputTimes)) &&
            all(>=(0),meta.OutputTimes) || throw(ArgumentError("OutputTimes must be finite, nonnegative and strictly increasing"))
        filter(t -> 0 < t < finish,meta.OutputTimes)
    else
        interval = meta.OutputTimes
        isfinite(interval) && interval > 0 || throw(ArgumentError("OutputTimes must be a positive finite interval"))
        [i*interval for i in 1:floor(Int,finish/interval) if i*interval < finish]
    end
    return [times;finish]
end

function rebuild_iisph!(gpu,cl,ws,kernel,constants)
    rebuild_cell_list!(gpu,cl,kernel.H⁻¹)
    # This path omits ghost buffers, so no owner compaction consumes cl.Perm.
    @cuda threads=128 blocks=cld(length(gpu),128) gather_kernel!(cl.Perm,
        (ws.pressure,),(ws.next_pressure,),Int32(length(gpu)))
    ws.pressure,ws.next_pressure = ws.next_pressure,ws.pressure
    launch_boundary_mass!(ws,gpu,cl,kernel,constants)
    launch_density!(ws,gpu,cl,kernel,constants)
    return nothing
end

function iisph_timestep(criteria,kernel,constants,solver,remaining,viscosity)
    iszero(criteria[3]) || error("Nonfinite IISPH particle state or invalid summation density")
    speed,force = Float64(criteria[1]),Float64(criteria[2])
    velocity_dt = iszero(speed) ? Inf : solver.cfl*kernel.h/speed
    force_dt = iszero(force) ? Inf : solver.cfl*sqrt(kernel.h/force)
    viscous_dt = viscosity isa Laminar && constants.ν₀ > 0 ? 0.125*kernel.h^2/constants.ν₀ : Inf
    dt = min(velocity_dt,force_dt,viscous_dt,solver.max_timestep,remaining)
    T = typeof(constants.ρ₀)
    dt = T(dt)
    isfinite(dt) && dt > 0 || error("IISPH timestep is nonfinite, nonpositive or underflows the working precision")
    return dt
end

function solve_iisph_pressure!(ws,gpu,cl,kernel,constants,dt,solver)
    for iteration in 0:solver.max_iterations
        launch_pressure_acceleration!(ws,gpu,cl,kernel,constants)
        launch_pressure_update!(ws,gpu,cl,kernel,constants,dt,solver)
        residual = pressure_residual(ws)
        isfinite(residual) || error("IISPH pressure solve produced a nonfinite residual")
        if iteration >= solver.min_iterations && residual <= solver.relative_density_tolerance
            return (;iterations=iteration,relative_density_residual=Float64(residual))
        end
        iteration == solver.max_iterations && error("IISPH pressure solve did not converge after $iteration iterations: maximum relative density residual=$residual, tolerance=$(solver.relative_density_tolerance). Reduce max_timestep or increase max_iterations.")
        ws.pressure,ws.next_pressure = ws.next_pressure,ws.pressure
    end
    error("unreachable IISPH pressure solve")
end

function run_iisph_simulation(;SimGeometry,SimMetaData,SimConstants,SimKernel,SimLogger,SimParticles,
        SimViscosity,SimDensityDiffusion,SimTimeStepping,SimSolver,ParticleNormalsPath,
        SimMeasurements,SimExternalForces,SimRigidBodyDynamics,SimOpenBoundaries,SimDamping,
        SimRotation,SimMotion,ParaviewPressureRange)
    meta,constants,kernel,solver = SimMetaData,SimConstants,SimKernel,SimSolver
    validate_iisph(meta,SimGeometry,SimParticles,constants,kernel,SimViscosity,SimDensityDiffusion,SimTimeStepping;
        normals=ParticleNormalsPath,rigid=SimRigidBodyDynamics,open_boundary=SimOpenBoundaries,
        damping=SimDamping,rotation=SimRotation,motion=SimMotion)
    deadlines = iisph_output_deadlines(meta)
    resolve_output_variables!(meta)
    unsupported_output = intersect(meta.OutputVariables,["Kernel","KernelGradient","GhostPoints","GhostNormals"])
    isempty(unsupported_output) || throw(ArgumentError("IISPH does not provide output fields $unsupported_output"))
    D,T = length(eltype(SimParticles.Position)),eltype(SimParticles.Density)
    # Reuse the caller's host columns, adding a solver-owned pressure column.
    # The shared writer and probes will therefore never reconstruct IISPH pressure from density.
    pressure = hasproperty(SimParticles,:Pressure) ? SimParticles.Pressure : zeros(T,length(SimParticles))
    frame = StructArray(merge(StructArrays.components(SimParticles),(;Pressure=pressure)))
    fill!(frame.Pressure,zero(T))
    SimulationLoggerConfiguration.start_runtime!(SimLogger,0.0)
    gpu = upload_particles(frame;position_type=position_float_type(meta),compact_metadata=meta.GPUCompactBuffers,
        kernel_output=false,ghost_data=false,reconstruct_fields=meta.GPUCompactBuffers)
    ws = IISPHWorkspace{D,T}(length(gpu))
    cl = CellListWorkspace{D,position_type(gpu)}(length(gpu);max_cells=meta.GPUMaxCells,
        deterministic=meta.GPUDeterministicSort)
    external = prepare_external_forces(SimExternalForces,frame,Val(D),T)
    state = StepState{T}()
    state.configuration = effective_configuration(meta,kernel,SimDensityDiffusion,SimViscosity;
        particle_count=length(gpu),solver)
    store_log = typeof(meta).parameters[6] === StoreLog
    output,history = nothing,nothing
    frames,write_seconds,total_iterations,max_iterations,max_residual = 0,0.0,0,0,0.0
    history_path = joinpath(meta.SaveLocation,meta.SimulationName*"_IISPH.csv")
    status = "failed"
    try
        rebuild_iisph!(gpu,cl,ws,kernel,constants)
        download_particles!(frame,gpu,cl.grid,(:Velocity,:Density,:ID,:Type,:GroupMarker,:Acceleration);cells=meta.ExportGridCells)
        copyto!(frame.Pressure,ws.pressure)
        output = SetupVTKOutput(meta,frame,kernel,D;measurements=SimMeasurements,fluid_type=Fluid,constants)
        history = open(history_path,"w")
        println(history,"Iteration,Time,TimeStep,PressureIterations,MaxRelativeDensityResidual")
        if store_log
            InitializeLogger(SimLogger,constants,meta,kernel,SimViscosity,SimDensityDiffusion,SimGeometry,frame;
                gpu_buffers=(;particles=gpu,support=ws,cells=cl,auxiliary=(external,state.f,state.i)))
            log_effective_configuration(SimLogger,state.configuration;phase="IISPH startup")
        end
        function write_frame!()
            started = time_ns()
            download_particles!(frame,gpu,cl.grid,(:Velocity,:Density,:ID,:Type,:GroupMarker,:Acceleration);cells=meta.ExportGridCells)
            copyto!(frame.Pressure,ws.pressure)
            meta.OutputIterationCounter = frames+1
            output.save_particles(meta.OutputIterationCounter,meta.TotalTime)
            if meta.ExportGridCells
                cells = occupied_cells_host(frame.Position,kernel.H⁻¹)
                output.save_grid(meta.OutputIterationCounter,cells,frame,meta.TotalTime)
                meta.IndexCounter = length(cells)
            end
            frames += 1
            write_seconds += (time_ns()-started)/1e9
            store_log && LogStep(SimLogger,meta,meta.HourGlass)
            meta.StepsTakenForLastOutput = meta.Iteration
            return nothing
        end
        write_frame!()
        grid_current = true
        for deadline in deadlines
            while meta.TotalTime < deadline
                grid_current || rebuild_iisph!(gpu,cl,ws,kernel,constants)
                grid_current = true
                launch_nonpressure!(ws,gpu,cl,kernel,constants,SimViscosity)
                launch_external_forces!(external,ws.nonpressure_acceleration,gpu.Position,gpu.Velocity,
                    gpu.Type,gpu.GroupMarker,HostStep(zero(T),meta.TotalTime),constants.g)
                criteria = timestep_criteria(ws,gpu)
                dt = iisph_timestep(criteria,kernel,constants,solver,deadline-meta.TotalTime,SimViscosity)
                meta.TotalTime+Float64(dt) > meta.TotalTime || error("IISPH timestep cannot advance the Float64 clock")
                launch_predict!(ws,gpu,dt,solver)
                launch_source!(ws,gpu,cl,kernel,constants,dt)
                info = solve_iisph_pressure!(ws,gpu,cl,kernel,constants,dt,solver)
                launch_integrate!(ws,gpu,dt)
                CUDA.synchronize()
                meta.Iteration += 1
                meta.CurrentTimeStep = dt
                next_time = meta.TotalTime+Float64(dt)
                # A clipped step uses the closest working-precision dt. Avoid
                # solving a second pressure system for a sub-ulp clock remainder.
                meta.TotalTime = abs(next_time-deadline) <= 2Float64(eps(T(deadline-meta.TotalTime))) ? deadline : next_time
                total_iterations += info.iterations
                max_iterations = max(max_iterations,info.iterations)
                max_residual = max(max_residual,info.relative_density_residual)
                state.direct_batches += 1
                println(history,join((meta.Iteration,meta.TotalTime,dt,info.iterations,info.relative_density_residual),','))
                grid_current = false
            end
            rebuild_iisph!(gpu,cl,ws,kernel,constants)
            grid_current = true
            write_frame!()
            flush(history)
        end
        status = "finished"
    finally
        try
            if output !== nothing
                started = time_ns()
                output.close_files()
                write_seconds += (time_ns()-started)/1e9
            end
        finally
            history === nothing || close(history)
            if store_log
                LogFinal(SimLogger,meta.HourGlass;status)
                close(SimLogger.LoggerIo)
                AutoOpenLogFile(SimLogger,meta)
            end
        end
    end
    elapsed = SimulationLoggerConfiguration.elapsed_wall_time(SimLogger)
    steps = meta.Iteration
    final_state = (;particles=frame,floating=nothing,time=meta.TotalTime,iteration=steps,
        timestep=meta.CurrentTimeStep,complete=true)
    performance = (;steps,simulated_time=meta.TotalTime,wall_time_seconds=elapsed,
        mean_timestep=meta.TotalTime/steps,milliseconds_per_step=1e3elapsed/steps,
        mparticle_steps_per_second=length(gpu)*steps/(1e6elapsed),output_write_seconds=write_seconds,
        phase_timings=copy(meta.HourGlass),phase_timings_synchronized=false)
    diagnostics = (;particle_count=length(gpu),output_frames=frames,cell_list_rebuilds=cl.nrebuilds,
        grid_dims=Tuple(cl.grid.dims),frozen_particles=0,step_state_readbacks=0,
        grid_status_readbacks=cl.grid_status_readbacks,capacity_growths=cl.capacity_growths,
        bbox_host_readbacks=cl.bbox_host_readbacks,captured_graphs=0,
        periodic_source_rebuilds=0,periodic_source_count=0,
        iisph=(;total_pressure_iterations=total_iterations,max_pressure_iterations=max_iterations,
            max_relative_density_residual=max_residual,
            pressure_residual_readbacks=total_iterations+steps,timestep_readbacks=steps,history_path))
    AutoOpenParaview(meta,constants,meta.OutputVariables;pressure_range=ParaviewPressureRange)
    return SimulationResult(final_state,performance,diagnostics,:completed,configuration_snapshot(state))
end
