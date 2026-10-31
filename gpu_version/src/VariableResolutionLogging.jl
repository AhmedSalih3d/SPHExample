# Included in VariableResolution. Reuse the normal logger's progress columns,
# monotonic runtime clock and final TimerOutput report.
function initialize_resolution_log!(logger, settings)
    log_line(logger,"Simulation: DamBreak2D_VRes (native 2D variable resolution)")
    log_line(logger,"Logger Start Time: " * logger.CurrentDataStr)
    log_line(logger,"Julia $VERSION, $(Base.Threads.nthreads()) compute threads")
    log_line(logger,"CUDA information:\n" * sprint(CUDA.versioninfo))
    log_line(logger,"GPU: $(CUDA.name(CUDA.device())), CUDA.jl $(pkgversion(CUDA))")
    log_line(logger,"Simulation time: $(settings.simulation_time) s; output interval: $(settings.output_interval) s")
    log_line(logger,"Refinement rectangle: $(Tuple(settings.region.lower)) to $(Tuple(settings.region.upper)); ratio=$(settings.region.ratio)")
    for (name,constants,kernel,geometry,reserve) in (
            ("Coarse",settings.coarse_constants,settings.coarse_kernel,settings.coarse_geometry,settings.coarse_reserve),
            ("Fine",settings.fine_constants,settings.fine_kernel,settings.fine_geometry,settings.fine_reserve))
        fluid = sum(length(g.Particles) for g in geometry if g.Type == Fluid; init = 0)
        fixed = sum(length(g.Particles) for g in geometry if g.Type == Fixed; init = 0)
        log_line(logger,"$name spacing: $(constants.dx) m; smoothing length: $(kernel.h) m; particle mass: $(constants.m₀)")
        log_line(logger,"$name initial particles: $fluid fluid, $fixed fixed; reserved slots: $reserve")
        log_line(logger,"$name precision: physics $(typeof(constants.dx)); positions Float64")
    end
    log_line(logger,"Common physical constants:\n$(settings.coarse_constants)")
    log_line(logger,"Models: Wendland C2, artificial viscosity, full Fourtakas DDT3, fixed mDBC walls, first-order MLS buffers")
    log_line(logger,"Execution: direct CUDA launches, independent predictor sorting, synchronous VTKHDF output")
    log_line(logger,"Timing scope: driver initialization/JIT, stepping and output; excludes package load/precompilation and caller-side geometry construction/export.")
    log_line(logger,"First-use compilation inside the driver is included; these timings are not automatically warmed benchmark results.")
    log_line(logger,"Phase timings synchronize CUDA at initialization/output boundaries; no per-kernel profiling or additional per-step synchronization.")
    return nothing
end

resolution_phase_seconds(timer,name) = name in keys(timer) ? TimerOutputs.time(timer[name]) / 1e9 : 0.0

function log_resolution_summary!(logger,timer,result)
    elapsed = LOGGER.elapsed_wall_time(logger)
    solver_seconds = resolution_phase_seconds(timer,"01 Solver")
    log_line(logger,@sprintf("Physical time reached: %.6f [s]; total steps: %d; output frames: %d",
        result.time,result.iterations,length(result.files)))
    log_line(logger,@sprintf("Initialization time: %.6f [s]",resolution_phase_seconds(timer,"00 Initialization")))
    log_line(logger,@sprintf("Solver stepping time: %.6f [s]",solver_seconds))
    log_line(logger,@sprintf("Output time (download + VTKHDF/CSV): %.6f [s]",resolution_phase_seconds(timer,"02 Output Frame")))
    if result.iterations > 0
        log_line(logger,@sprintf("End-to-end throughput: %.4f ms/step, %.4f steps/s; mean dt = %.6e s",
            1e3elapsed/result.iterations,result.iterations/elapsed,result.time/result.iterations))
        solver_seconds > 0 && log_line(logger,@sprintf("Stepping throughput (excludes initialization/output): %.4f ms/step, %.4f steps/s",
            1e3solver_seconds/result.iterations,result.iterations/solver_seconds))
    end
    coarse,fine = result.coarse,result.fine
    log_line(logger,"Cell list rebuilds (accepted coarse/fine): $(coarse.cells.nrebuilds)/$(fine.cells.nrebuilds)")
    log_line(logger,"Cell list rebuilds (predictor coarse/fine): $(coarse.predictor.cells.nrebuilds)/$(fine.predictor.cells.nrebuilds)")
    log_line(logger,"Final exported fluid particles (coarse/fine): $(result.particle_counts.coarse)/$(result.particle_counts.fine)")
    log_line(logger,"Final represented fluid mass: $(result.represented_fluid_mass)")
    a,b = Array(coarse.events),Array(fine.events)
    for (index,label) in enumerate(("Particle creation events","Interface retirements","Density exclusions","Non-finite states"))
        log_line(logger,"$label (coarse/fine): $(a[index])/$(b[index])")
    end
    return nothing
end

function with_resolution_logger(run,output)
    mkpath(output)
    path = joinpath(output,"SimulationOutput.log")
    logger = SimulationLogger(output)
    timer = TimerOutput()
    LOGGER.start_runtime!(logger,0.0)
    Base.exit_on_sigint(false)
    status = "failed"
    result = nothing
    try
        result = run(logger,timer)
        log_resolution_summary!(logger,timer,result)
        status = "finished"
    catch failure
        status = failure isa InterruptException ? "interrupted" : "failed"
        log_line(logger,"Simulation error: " * sprint(showerror,failure))
        rethrow()
    finally
        try
            LogFinal(logger,timer; status)
        finally
            close(logger.LoggerIo)
        end
    end
    timings = (; total_wall_seconds = LOGGER.elapsed_wall_time(logger),
        initialization_seconds = resolution_phase_seconds(timer,"00 Initialization"),
        solver_seconds = resolution_phase_seconds(timer,"01 Solver"),
        output_seconds = resolution_phase_seconds(timer,"02 Output Frame"),phase_timings = timer)
    return merge(result,(; log_path = path,timings))
end
