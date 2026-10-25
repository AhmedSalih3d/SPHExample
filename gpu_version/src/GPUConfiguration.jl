"""Resolved GPU settings and host-only snapshots of the paths submitted by the driver."""
module GPUConfiguration

using Logging
using ..SimulationMetaDataConfiguration
using ..SPHDensityDiffusionModels: ZeroDensityDiffusion
using ..SPHViscosityModels: ZeroViscosity, LaminarSPS
using ..GPUKernels: choose_lanes, interaction_options

export effective_configuration

"""
    effective_configuration(metadata, kernel, diffusion, viscosity; particle_count)

Resolve supported GPU specializations and launch settings without running a
simulation. Returns host-only `requested`, `capabilities`, `effective` and
`fallbacks` snapshots. Automatic lanes require a functional CUDA device;
explicit `GPULanesPerParticle` values can be resolved without one.

`effective` identifies the SPH kernel, interaction specialization, diffusion
operator, viscosity arithmetic, precision, pair coordinates and simulation
modes. A radial linear-diffusion operator is `:symmetric_reciprocal_density`;
`:reference` retains CPU pair orientation (CPU equivalence also requires cell
subdivision one). Capabilities describe implementation support, not accuracy.

This is a configuration preview. `RunSimulation(...).configuration` additionally
reports observed graph/direct batch counts and capture failures under `execution`.
"""
function effective_configuration(meta::SimulationMetaData{D, T, S, K, B}, kernel, diffusion, viscosity;
                                 particle_count::Integer) where {D, T, S, K, B}
    particle_count >= 0 || throw(ArgumentError("particle_count must be nonnegative"))
    options = interaction_options(kernel, diffusion, viscosity;
        radial = meta.GPURadialInteractions, scalar_sps = meta.GPUScalarSPSViscosity)
    requested = (; radial_interactions = meta.GPURadialInteractions,
        scalar_sps_viscosity = meta.GPUScalarSPSViscosity, use_graph = meta.GPUUseGraph,
        sync_timers = meta.GPUSyncTimers, lanes_per_particle = meta.GPULanesPerParticle,
        max_steps_per_sync = meta.GPUMaxStepsPerSync, double_position = meta.GPUDoublePosition)
    capabilities = (; radial_interactions = options.radial_supported,
        scalar_sps_viscosity = options.scalar_sps_supported)
    lanes = meta.GPULanesPerParticle <= 0 ? choose_lanes(particle_count, Val(D)) : meta.GPULanesPerParticle
    position_type = position_float_type(meta)
    effective = (; sph_kernel = typeof(kernel.kernel), interaction_kernel = :interaction_kernel!,
        interaction_path = options.radial ? :radial : :reference,
        radial_interactions = options.radial, scalar_sps_viscosity = options.scalar_sps,
        density_diffusion_model = typeof(diffusion),
        density_diffusion_operator = diffusion isa ZeroDensityDiffusion ? :disabled :
            options.radial ? :symmetric_reciprocal_density : :reference,
        viscosity_model = typeof(viscosity),
        viscosity_path = viscosity isa ZeroViscosity ? :disabled : options.scalar_sps ? :scalar_sps :
            options.radial && !(viscosity isa LaminarSPS) ? :radial_symmetric : :reference_ordered,
        dimensions = D, float_type = T, position_type,
        pair_coordinates = meta.Periodicity !== nothing ? :translated_absolute : position_type === T ? :absolute : :cell_relative,
        time_stepping = typeof(meta.TimeSteppingMode), boundary_mode = B, shifting_mode = S,
        kernel_output_mode = K, cell_subdivision = meta.GPUCellSubdivision,
        periodicity = meta.Periodicity,
        cache_interaction_fields = meta.GPUCacheInteractionFields,
        boundary_forces = meta.GPUBoundaryForces, deterministic_sort = meta.GPUDeterministicSort,
        clip_timestep_to_output = meta.GPUClipTimeStepToOutput,
        interaction_threads = meta.GPUInteractionThreads, lanes_per_particle = lanes,
        use_graph = meta.GPUUseGraph && !meta.GPUSyncTimers,
        max_steps_per_sync = meta.GPUSyncTimers ? 1 : max(1, meta.GPUMaxStepsPerSync))
    fallbacks = NamedTuple[]
    if requested.radial_interactions && !effective.radial_interactions
        push!(fallbacks, (; option = :GPURadialInteractions, requested = true, effective = false,
            reason = :unsupported_kernel_or_models))
    end
    if requested.scalar_sps_viscosity && !effective.scalar_sps_viscosity
        push!(fallbacks, (; option = :GPUScalarSPSViscosity, requested = true, effective = false,
            reason = :unsupported_kernel_or_viscosity))
    end
    if requested.use_graph && !effective.use_graph
        push!(fallbacks, (; option = :GPUUseGraph, requested = true, effective = false,
            reason = :synchronized_timers))
    end
    return (; requested, capabilities, effective, fallbacks = Tuple(fallbacks))
end

function configuration_snapshot(state)
    configuration = state.configuration
    configuration === nothing && return nothing
    graph_batches, direct_batches = state.graph_batches, state.direct_batches
    path = graph_batches > 0 ? (direct_batches > 0 ? :mixed : :cuda_graph) :
        (direct_batches > 0 ? :direct : :not_run)
    execution = (; path, graph_batches, direct_batches, graph_capture_failures = state.graph_capture_failures)
    fallbacks = configuration.fallbacks
    if state.graph_capture_failures > 0
        fallbacks = (fallbacks..., (; option = :GPUUseGraph, requested = true, effective = :direct_batches,
            reason = :graph_capture_failed,
            detail = something(state.graph_capture_error, "CUDA invalidated capture without exposing an underlying exception.")))
    end
    return merge(configuration, (; fallbacks, execution))
end

function log_effective_configuration(logger, configuration; phase)
    configuration === nothing && return nothing
    e = configuration.effective
    with_logger(logger.Logger) do
        @info "Effective GPU configuration ($phase):"
        @info "Requested GPU options: $(configuration.requested)"
        @info "Supported specializations: $(configuration.capabilities)"
        @info "SPH kernel: $(e.sph_kernel); interaction kernel: $(e.interaction_kernel), path=$(e.interaction_path), scalar_sps=$(e.scalar_sps_viscosity)"
        @info "Density diffusion: $(e.density_diffusion_model), operator=$(e.density_diffusion_operator); viscosity: $(e.viscosity_model), path=$(e.viscosity_path)"
        @info "Numerical configuration: $(e)"
        for fallback in configuration.fallbacks
            @info "Configuration fallback: $(fallback.option), requested=$(fallback.requested), effective=$(fallback.effective), reason=$(fallback.reason)"
            if hasproperty(fallback, :detail)
                @info "Capture failure detail: $(fallback.detail)"
            end
        end
        if hasproperty(configuration, :execution)
            @info "Observed GPU execution: $(configuration.execution)"
        end
    end
    return nothing
end

end # module
