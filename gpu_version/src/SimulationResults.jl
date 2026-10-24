module SimulationResults

export SimulationResult

"""
    SimulationResult

Result returned by [`RunSimulation`](@ref) after output workers have finished
and files have closed. Its fields are:

- `final_state`: a named tuple with `particles`, `floating`, `time`, `iteration`,
  `timestep`, and `complete`. `particles` is the caller's `SimParticles` array,
  refreshed from the device in cell order; copy it to retain an independent
  snapshot. `floating` contains host body-state vectors, or `nothing` when no
  bodies are active. Time, iteration and timestep are scalar snapshots of the
  solver metadata. `complete` is true on normal completion. An interrupt may
  stop midway through a step, so its device snapshot has `complete = false`
  and its time/iteration identify the last committed step.
- `performance`: run-local `steps`, `simulated_time`, `wall_time_seconds`,
  `mean_timestep`, `milliseconds_per_step`, `mparticle_steps_per_second`, and
  `output_write_seconds`. Wall time includes setup, final downloads, output
  draining and finalization. Writer time can overlap GPU work. Rates without
  any completed steps are `NaN`. `phase_timings` is an independent copy of
  `SimMetaData.HourGlass`, including any entries present before the run;
  `phase_timings_synchronized` records `GPUSyncTimers`. Without synchronization,
  these section timings measure host enqueue/readback boundaries, not GPU
  kernel execution time.
- `diagnostics`: `particle_count`, `output_frames` (including the initial
  frame), `cell_list_rebuilds`, `grid_dims`, `frozen_particles`,
  `step_state_readbacks`, `grid_status_readbacks`, `capacity_growths`,
  `bbox_host_readbacks`, and `captured_graphs`, all captured at return.
- `termination_reason`: `:completed` when the requested simulation time has
  been reached (possibly overshot), or `:interrupted` for a caught
  `InterruptException`. Other errors still propagate after output cleanup;
  they do not return a successful result.
- `configuration`: host-only `requested`, `capabilities`, `effective`,
  `fallbacks` and `execution` snapshots. `effective` is the last resolved
  driver configuration; `execution.path` is `:not_run`, `:direct`, `:cuda_graph`
  or `:mixed`, based on submitted batches, with graph/direct batch and graph
  capture-failure counts. Device stop flags may make some submitted steps
  no-ops. Capture failures remain recorded even if later captures succeed or
  graph caches are invalidated. Capture fallbacks include a `detail` string
  with the underlying exception when available. Available even with `NoLog`.

The result retains host data only, so it does not extend the lifetime of the
solver's device buffers. Ignoring the return value preserves existing usage.
"""
struct SimulationResult{S, P, D, C}
    final_state::S
    performance::P
    diagnostics::D
    termination_reason::Symbol
    configuration::C
end

# Preserve construction of results by callers using the original four fields.
SimulationResult(state, performance, diagnostics, reason::Symbol) =
    SimulationResult(state, performance, diagnostics, reason, nothing)

end # module
