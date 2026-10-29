# Included in SPHCellList. Keep reference-case rules separate from CPU semantics.
function dual_time_step(criteria, kernel, constants, scheme)
    iszero(criteria[4]) || throw(ArgumentError("A periodic source left the configured domain grid or became nonfinite; check the reference domain bounds"))
    acceleration = sqrt(Float64(criteria[1]))
    speed = sqrt(Float64(criteria[2]))
    h = isfinite(scheme.smoothing_length) ? scheme.smoothing_length : Float64(kernel.h)
    sound_speed = isfinite(scheme.sound_speed) ? scheme.sound_speed : Float64(constants.c₀)
    force_limit = iszero(acceleration) ? Inf : sqrt(h / acceleration)
    acoustic_limit = h / (max(sound_speed, 10speed) + h * Float64(criteria[3]))
    dt = max(0.05h / sound_speed, Float64(constants.CFL) * min(force_limit,acoustic_limit))
    isfinite(dt) && dt > 0 || error("Reference timestep is nonfinite or nonpositive; criteria=$(criteria)")
    return dt
end

function dual_force_stage!(position,velocity,density,pairs,gpu,cl,sup,kernel,constants,diffusion,viscosity,meta,step)
    images = sup.dual_state.images
    if images !== nothing
        if !all(isfinite,meta.TimeSteppingMode.domain_min) || !all(isfinite,meta.TimeSteppingMode.domain_max)
            configure_periodic_images!(images,cl.periodicity,cl.grid,kernel)
        end
        build_periodic_images!(images,position,velocity,density,gpu.Type,cl.periodicity,meta.TimeSteppingMode,kernel)
        source = images.sorted
        launch_interactions!(sup.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,sup.∇Cᵢ,sup.∇◌rᵢ,
            source.Position,source.Density,source.Velocity,source.Type,images.cl.CellStart,images.cells,images.cl.grid_dev,step,
            diffusion,viscosity,kernel,constants,Val(false),Val(false);
            threads=meta.GPUInteractionThreads,lanes=Val(1),boundary_forces=Val(false),
            pos_cells=images.pairs,viscous_dt=sup.dual_state.viscous,
            target_indices=images.targets,target_count=length(gpu),fast_math=meta.TimeSteppingMode.fast_math,
            always_inline=meta.GPUInteractionAlwaysInline,fastmath=meta.GPUInteractionFastMath)
        criteria = GPUKernels.launch_dual_criteria!(sup.dual_state,velocity,gpu.Acceleration,gpu.Type)
        return dual_time_step(criteria,kernel,constants,meta.TimeSteppingMode)
    end
    launch_pos_cells!(pairs,position,gpu.CellID,cl.grid_dev,step,kernel)
    if periodic_bounds(sup) !== nothing
        GPUKernels.launch_periodic_halo!(sup.PeriodicBounds,position,gpu.CellID,cl.grid_dev,kernel.H,step)
    end
    query_bounds = periodic_bounds(sup)
    if sup.dual_state.queries !== nothing
        GPUKernels.launch_periodic_queries!(sup.dual_state.queries,position,gpu.CellID,
            cl.grid_dev,kernel.H,query_bounds,cl.periodicity,step)
        query_bounds = sup.dual_state.queries
    end
    launch_interactions!(sup.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,sup.∇Cᵢ,sup.∇◌rᵢ,
        position,density,velocity,gpu.Type,cl.CellStart,gpu.CellID,cl.grid_dev,step,
        diffusion,viscosity,kernel,constants,Val(false),Val(false);
        threads=meta.GPUInteractionThreads, lanes=Val(1), boundary_forces=Val(false),
        pos_cells=pairs, periodicity=cl.periodicity, periodic_bounds=query_bounds,
        periodic_cell_coordinates=meta.GPUPeriodicCellCoordinates, viscous_dt=sup.dual_state.viscous,
        fast_math=meta.TimeSteppingMode.fast_math,
        always_inline=meta.GPUInteractionAlwaysInline, fastmath=meta.GPUInteractionFastMath)
    criteria = GPUKernels.launch_dual_criteria!(sup.dual_state,velocity,gpu.Acceleration,gpu.Type)
    return dual_time_step(criteria,kernel,constants,meta.TimeSteppingMode)
end

function dual_simulation_loop!(diffusion,viscosity,kernel,meta,constants,gpu,cl,sup,rotation,state)
    clock = sup.dual_state
    scheme = meta.TimeSteppingMode
    clock === nothing && throw(ArgumentError("Allocate support arrays with DualSPHysicsSymplectic metadata"))
    scheme.cache_periodic_queries == (clock.queries !== nothing) ||
        throw(ArgumentError("Periodic-query caching must match the allocated support arrays; reallocate after changing the scheme"))
    periodic_images_enabled(meta,kernel) == (clock.images !== nothing) ||
        throw(ArgumentError("Periodic-image mode must match the allocated support arrays; reallocate after changing the scheme"))
    if !clock.initialized
        clock.time = Float64(meta.TotalTime)
        h = isfinite(scheme.smoothing_length) ? scheme.smoothing_length : Float64(kernel.h)
        sound_speed = isfinite(scheme.sound_speed) ? scheme.sound_speed : Float64(constants.c₀)
        clock.dt = h / sound_speed
        clock.initialized = true
    end
    if cl.nrebuilds == 0
        rebuild_cell_list!(gpu,cl,kernel.H⁻¹; support=sup,constants,kernel,rotation)
    end
    position_half, velocity_half = predictor_arrays(sup,gpu)
    deadline = Float64(next_output_time(meta))
    while clock.time < deadline
        dt = meta.GPUClipTimeStepToOutput ? min(clock.dt,deadline-clock.time) : clock.dt
        step = HostStep(dt,clock.time)
        predictor_limit = dual_force_stage!(gpu.Position,gpu.Velocity,gpu.Density,sup.PosCells,
            gpu,cl,sup,kernel,constants,diffusion,viscosity,meta,step)
        GPUKernels.launch_dual_predictor!(position_half,velocity_half,sup.ρₙ⁺,
            gpu.Position,gpu.Velocity,gpu.Density,gpu.Acceleration,sup.dρdtI,gpu.Type,dt,constants;
            exclusions=clock.exclusions,periodicity=cl.periodicity,scheme)
        corrector_limit = dual_force_stage!(position_half,velocity_half,sup.ρₙ⁺,sup.PosCellsₙ⁺,
            gpu,cl,sup,kernel,constants,diffusion,viscosity,meta,step)
        GPUKernels.launch_dual_corrector!(gpu.Position,gpu.Velocity,gpu.Density,position_half,
            sup.ρₙ⁺,gpu.Acceleration,sup.dρdtI,gpu.Type,dt,constants)
        # DualSPHysics moves prescribed walls after the fluid corrector. Its
        # velocity is the completed displacement divided by this step's dt.
        launch_rotation!(rotation,gpu.Position,gpu.Velocity,gpu.Type,gpu.GroupMarker,gpu.ID,step; fraction=1.0)
        finish = GPUKernels.launch_dual_motion_finish!(clock,gpu.Position,gpu.Velocity,gpu.Density,
            position_half,gpu.Type,dt,cl.periodicity,scheme)
        clock.excluded_position += Int(finish[1])
        clock.excluded_density += Int(finish[2])
        cl.frozen_total = clock.excluded_position + clock.excluded_density
        clock.time += dt
        clock.dt = min(predictor_limit,corrector_limit)
        clock.travel += 2Float64(finish[4])
        state.ih[I_ITER] += Int32(1)
        state.readbacks += 3
        state.direct_batches += 1
        # Rebuilds may be reused because measured offsets protect every full
        # and predictor query. Reordering is done only after an accepted step.
        # Wrapping invalidates the image of the old cell anchors. Keeping that
        # stale assignment would widen the global halo by a whole period.
        if clock.travel >= kernel.h || finish[1] + finish[2] + finish[3] > 0
            rebuild_cell_list!(gpu,cl,kernel.H⁻¹; support=sup,constants,kernel,rotation)
            position_half, velocity_half = predictor_arrays(sup,gpu)
            clock.travel = 0.0
            cl.frozen_total = clock.excluded_position + clock.excluded_density
        end
        state.fh[F_TIME] = clock.time
        state.fh[F_DT] = clock.dt
        state.ih[I_PHASE] = PHASE_NEED_DT
        sync_meta_data!(meta,state)
    end
    copyto!(state.f,state.fh)
    copyto!(state.i,state.ih)
    state.primed = true
    return nothing
end
