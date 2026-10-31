"""Resolved GPU settings and host-only snapshots of the paths submitted by the driver."""
module GPUConfiguration

using Logging
using LoggingExtras: FormatLogger
using ..SimulationMetaDataConfiguration
using ..SimulationMetaDataConfiguration: periodic_images_enabled
using ..SPHDensityDiffusionModels: ZeroDensityDiffusion, FourtakasDensityDiffusion
using ..SPHViscosityModels: ZeroViscosity, LaminarSPS, ArtificialViscosity
using ..GPUKernels: choose_lanes, interaction_options

export effective_configuration

"""
    effective_configuration(metadata, kernel, diffusion, viscosity; particle_count)

Resolve supported GPU specializations and launch settings without running a
simulation. Returns host-only `requested`, `capabilities`, `effective` and
`fallbacks` snapshots. Automatic lanes require a functional CUDA device;
explicit `GPULanesPerParticle` values can be resolved without one.

`effective` identifies the SPH kernel, interaction specialization, diffusion
operator, viscosity arithmetic, interaction compiler options, precision, pair coordinates and simulation
modes. A radial linear-diffusion operator is `:symmetric_reciprocal_density`;
`:reference` retains CPU pair orientation (CPU equivalence also requires cell
subdivision one and `GPUInteractionFastMath = false`). Capabilities describe implementation support, not accuracy.

This is a configuration preview. `RunSimulation(...).configuration` additionally
reports observed graph/direct batch counts and capture failures under `execution`.
"""
function effective_configuration(meta::SimulationMetaData{D, T, S, K, B}, kernel, diffusion, viscosity;
                                 particle_count::Integer) where {D, T, S, K, B}
    particle_count >= 0 || throw(ArgumentError("particle_count must be nonnegative"))
    options = interaction_options(kernel, diffusion, viscosity;
        radial = meta.GPURadialInteractions, scalar_sps = meta.GPUScalarSPSViscosity)
    requested = (; radial_interactions = meta.GPURadialInteractions,
        variable_resolution = meta.VariableResolution !== nothing,
        interaction_always_inline = meta.GPUInteractionAlwaysInline,
        interaction_fastmath = meta.GPUInteractionFastMath,
        scalar_sps_viscosity = meta.GPUScalarSPSViscosity, use_graph = meta.GPUUseGraph,
        sync_timers = meta.GPUSyncTimers, lanes_per_particle = meta.GPULanesPerParticle,
        max_steps_per_sync = meta.GPUMaxStepsPerSync, double_position = meta.GPUDoublePosition,
        periodic_cell_coordinates = meta.GPUPeriodicCellCoordinates, periodic_images=meta.GPUPeriodicImages)
    capabilities = (; radial_interactions = options.radial_supported,
        scalar_sps_viscosity = options.scalar_sps_supported)
    lanes = meta.VariableResolution !== nothing ? 1 : meta.GPULanesPerParticle <= 0 ? choose_lanes(particle_count, Val(D)) : meta.GPULanesPerParticle
    position_type = position_float_type(meta)
    effective = (; sph_kernel = typeof(kernel.kernel), kernel_regularization = kernel.η²,
        variable_resolution = meta.VariableResolution === nothing ? nothing :
            (; method=:two_domain_mls,region=meta.VariableResolution.region,
                include_buffers=meta.VariableResolution.include_buffers,particle_reserve=meta.VariableResolution.particle_reserve,
                coarse_smoothing_length=kernel.h,fine_smoothing_length=meta.VariableResolution.fine_kernel === nothing ?
                    kernel.h/meta.VariableResolution.region.ratio : meta.VariableResolution.fine_kernel.h),
        kernel_normalization = kernel.αD, interaction_kernel = :interaction_kernel!,
        interaction_path = options.radial ? :radial : :reference,
        radial_interactions = options.radial, scalar_sps_viscosity = options.scalar_sps,
        interaction_always_inline = meta.GPUInteractionAlwaysInline,
        interaction_fastmath = meta.GPUInteractionFastMath,
        density_diffusion_model = typeof(diffusion),
        density_diffusion_operator = diffusion isa ZeroDensityDiffusion ? :disabled :
            diffusion isa FourtakasDensityDiffusion ? (diffusion.full ? :fourtakas_full : :fourtakas_inner) :
            options.radial ? :symmetric_reciprocal_density : :reference,
        viscosity_model = typeof(viscosity),
        boundary_viscosity_factor = viscosity isa ArtificialViscosity ? viscosity.boundary_factor : nothing,
        viscosity_path = viscosity isa ZeroViscosity ? :disabled : options.scalar_sps ? :scalar_sps :
            options.radial && !(viscosity isa LaminarSPS) ? :radial_symmetric : :reference_ordered,
        dimensions = D, float_type = T, position_type,
        periodic_images = periodic_images_enabled(meta,kernel),
        pair_coordinates = periodic_images_enabled(meta,kernel) ? :periodic_images_cell_relative : meta.Periodicity !== nothing ?
            (meta.GPUPeriodicCellCoordinates ? :periodic_cell_relative : :translated_absolute) :
            position_type === T ? :absolute : :cell_relative,
        time_stepping = typeof(meta.TimeSteppingMode), boundary_mode = B, shifting_mode = S,
        reference_step_parameters = meta.TimeSteppingMode isa DualSPHysicsSymplectic ?
            (; smoothing_length=meta.TimeSteppingMode.smoothing_length, sound_speed=meta.TimeSteppingMode.sound_speed,
                domain_min=meta.TimeSteppingMode.domain_min, domain_max=meta.TimeSteppingMode.domain_max,
                density_min=meta.TimeSteppingMode.density_min, density_max=meta.TimeSteppingMode.density_max,
                fast_math=meta.TimeSteppingMode.fast_math,
                cache_periodic_queries=meta.TimeSteppingMode.cache_periodic_queries,
                periodic_images=meta.TimeSteppingMode.periodic_images) : nothing,
        kernel_output_mode = K, cell_subdivision = meta.GPUCellSubdivision,
        periodicity = meta.Periodicity,
        cache_interaction_fields = meta.GPUCacheInteractionFields,
        boundary_forces = meta.GPUBoundaryForces, deterministic_sort = meta.GPUDeterministicSort,
        clip_timestep_to_output = meta.GPUClipTimeStepToOutput,
        interaction_threads = meta.GPUInteractionThreads, lanes_per_particle = lanes,
        use_graph = meta.GPUUseGraph && !meta.GPUSyncTimers,
        max_steps_per_sync = meta.GPUSyncTimers ? 1 : max(1, meta.GPUMaxStepsPerSync))
    fallbacks = NamedTuple[]
    if meta.VariableResolution !== nothing
        effective = merge(effective,(;position_type=Float64,
            interaction_fastmath=false,radial_interactions=false,scalar_sps_viscosity=false,
            interaction_path=:reference,viscosity_path=:reference_ordered,cache_interaction_fields=false,
            boundary_forces=false,lanes_per_particle=1,compact_buffers=false,async_output=false,
            output_buffer_bytes=0,max_steps_per_sync=meta.GPUSyncTimers ? 1 : clamp(meta.GPUMaxStepsPerSync,1,256)))
        if effective.reference_step_parameters !== nothing
            effective = merge(effective,(;reference_step_parameters=merge(effective.reference_step_parameters,
                (;fast_math=false,periodic_images=false))))
            meta.TimeSteppingMode.fast_math && push!(fallbacks,(;option=:DualSPHysicsSymplecticFastMath,
                requested=true,effective=false,reason=:variable_resolution_backend))
        end
        for (option,requested_value,effective_value) in (
                (:GPUDoublePosition,meta.GPUDoublePosition,true),
                (:GPUInteractionFastMath,meta.GPUInteractionFastMath,false),
                (:GPURadialInteractions,meta.GPURadialInteractions,false),
                (:GPUScalarSPSViscosity,meta.GPUScalarSPSViscosity,false),
                (:GPUCacheInteractionFields,meta.GPUCacheInteractionFields,false),
                (:GPUBoundaryForces,meta.GPUBoundaryForces,false),
                (:GPUCompactBuffers,meta.GPUCompactBuffers,false),
                (:GPUAsyncOutput,meta.GPUAsyncOutput,false),
                (:GPUOutputBufferBytes,meta.GPUOutputBufferBytes,0),
                (:GPULanesPerParticle,meta.GPULanesPerParticle,1),
                (:GPUMaxStepsPerSync,meta.GPUMaxStepsPerSync,effective.max_steps_per_sync))
            requested_value == effective_value && continue
            push!(fallbacks,(;option,requested=requested_value,effective=effective_value,reason=:variable_resolution_backend))
        end
    end
    if requested.periodic_images && meta.Periodicity !== nothing && !effective.periodic_images
        reason = B !== NoMDBC ? :unsupported_boundary_mode :
            meta.TimeSteppingMode isa DualSPHysicsSymplectic && meta.TimeSteppingMode.cache_periodic_queries ? :query_cache_selected :
            meta.TimeSteppingMode isa DualSPHysicsSymplectic && !meta.TimeSteppingMode.periodic_images ? :scheme_selected_queries :
            :period_not_larger_than_twice_support
        push!(fallbacks,(;option=:GPUPeriodicImages,requested=true,effective=false,reason))
    end
    if meta.VariableResolution === nothing && requested.radial_interactions && !effective.radial_interactions
        push!(fallbacks, (; option = :GPURadialInteractions, requested = true, effective = false,
            reason = :unsupported_kernel_or_models))
    end
    if meta.VariableResolution === nothing && requested.scalar_sps_viscosity && !effective.scalar_sps_viscosity
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

function log_effective_configuration(logger, configuration; phase, to_console::Bool = true)
    configuration === nothing && return nothing
    e = configuration.effective
    report_logger = to_console ? logger.Logger :
        FormatLogger((io, args) -> println(io, args.message), logger.LoggerIo)
    with_logger(report_logger) do
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
