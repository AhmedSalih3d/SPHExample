    # Run from gpu_version:
    # julia --project=. -t 2,0 "validation cases/spheric_test_06/manual_case.jl"
    # Edit the layout and simulation settings below for a manual SPHERIC Test 06 run.
    using SPHExampleGPU
    using CUDA
    include(joinpath(@__DIR__, "case.jl"))
    using .SphericTest06

    let
        Dimensions = 2
        FloatType  = Float32

        # Extend the tank so the 2 m sponge stays outside the original benchmark box.
        # Set padding to zero and use_sponge to false for the official closed tank.
        domain_padding = 0.0
        use_sponge     = false
        sponge_width   = 0.0
        sponge_rate    = 0.0
        isfinite(domain_padding) && domain_padding >= 0 || throw(ArgumentError("domain_padding must be finite and nonnegative"))
        !use_sponge || 0 < sponge_width <= domain_padding ||
            throw(ArgumentError("Require 0 < sponge_width <= domain_padding to keep the original box undamped"))

        # Geometry is in metres. Tank and body edges must align with dx.
        layout = (
            tank_origin = (-domain_padding, -domain_padding),
            tank_size   = (10.0 + 2domain_padding, 5.0 + 2domain_padding),
            body_center = (1.5, 2.5),
            body_size   = (1.0, 1.0),
        )

        boundary_mode    = NoMDBC  # Set to SimpleMDBC for ghost-point density correction.
        motion_kind      = :integrated  # :hermite reproduces the rounded-position history.

        reynolds         = 100
        dx               = 0.00625
        reference_speed  = 1.0
        reference_length = layout.body_size[1]
        span             = 1.0
        wall_layers      = nothing
        coefh            = 2.0
        kernel_h_factor  = coefh * sqrt(2)
        start_delay      = 0.0
        duration         = 8.0 + start_delay
        output_interval  = 0.02
        visualize_in_paraview = true

        run_tag          = "coefh"
        n                = round(Int, reference_length / dx)
        run_name         = "Re$(lpad(string(reynolds), 3, '0'))_n$(lpad(string(n), 3, '0'))" * "_Test12";
        save_location    = joinpath(@__DIR__, "runs", run_tag, run_name)
        plot_location    = joinpath(@__DIR__, "runs", run_tag, "paraview_force_plot")
        results_location = joinpath(@__DIR__, "results", run_tag, run_name)

        SimConstants = SimulationConstants{FloatType}(Dimensions;
            dx  = dx,
            ρ₀  = 1000.0,
            g   = 0.0,
            c₀  = 160.0,
            ν₀  = reference_speed * reference_length / reynolds,
            α   = 0.0,
            δᵩ  = 0.1,
            CFL = 0.2,
            BackgroundPressure = 0.0,
        )

        SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(kernel_h_factor * dx))

        # Shared sampling preserves the validation lattice and solid/fluid interface.
        sampled = SphericTest06.sample_case_particles(layout, dx, SimKernel; wall_layers)

        SimulationGeometry = SphericTest06.manual_geometry(sampled, SimConstants.ρ₀, FloatType; boundary_mode)
        Tank, Water, Square = SimulationGeometry

        # The official history gives absolute centre x; the shared trajectory removes its initial 1.5 m offset.
        SquareMotion = PrescribedMotion(Square.GroupMarker;
            translation = SphericTest06.body_trajectory(; speed_scale = reference_speed, start_delay, motion_kind),
        )

        isfinite(start_delay) && start_delay >= 0 && 0 < output_interval <= duration <= start_delay + 8.0 ||
            throw(ArgumentError("Require start_delay >= 0 and at most 8 s of motion after the hold"))
        reynolds in (50, 100, 150) || throw(ArgumentError("Reynolds number must be 50, 100, or 150"))
        isfinite(reference_length) && reference_length > 0 && isfinite(span) && span > 0 ||
            throw(ArgumentError("reference_length and span must be positive and finite"))
        output_times = collect(output_interval:output_interval:duration)
        isapprox(last(output_times), duration; atol = 1e-12, rtol = 0) || push!(output_times, duration)
        output_times[end] = duration

        SimMetaData = SimulationMetaData{
            Dimensions,
            FloatType,
            PlanarShifting,
            NoKernelOutput,
            boundary_mode,
            StoreLog,
        }(
            SimulationName             = "MovingSquare",
            SaveLocation               = abspath(save_location),
            SimulationTime             = duration,
            OutputTimes                = output_times,
            OutputVariables            = ["Density", "Pressure", "Velocity", "Type", "GroupMarker"],
            # Open the combined particle-and-force state after force analysis, not a particle-only window.
            VisualizeInParaview         = false,
            ExportSingleVTKHDF          = true,
            OpenLogFile                = false,
            GPUUseGraph                = true,
            GPUClipTimeStepToOutput     = true,
            GPUDoublePosition          = true,
            GPULegacyTimestep           = true,
            GPUCellSubdivision         = 1,
            GPUXCellSubdivision        = 1,
            GPURadialInteractions      = false,
            GPUInteractionFastMath     = false,
            GPUPreloadInteractionTypes = false,
            GPUBoundaryForces          = false,
            GPUOutputBufferBytes       = 16 * 2^20,
        )

        SimViscosity        = Laminar()
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion()
        SimTimeStepping     = SymplecticTimeStepping()
        # Quadratic velocity damping at outer edges; no damping inside the original box.
        SimDamping          = use_sponge ? SphericTest06.edge_sponge(layout; width = sponge_width, rate = sponge_rate) : nothing
        SimDivergenceCleaning = nothing

        SimViscosity isa Union{Laminar, ZeroViscosity, ArtificialViscosity} ||
            throw(ArgumentError("Drag postprocessing supports Laminar, ZeroViscosity, or ArtificialViscosity"))
        isfinite(SimConstants.BackgroundPressure) && SimConstants.BackgroundPressure >= 0 ||
            throw(ArgumentError("BackgroundPressure must be finite and nonnegative"))
        SphericTest06.verify_reference()
        CUDA.functional() || error("An NVIDIA CUDA GPU is required")
        isfile(joinpath(SimMetaData.SaveLocation, "MovingSquare.vtkhdf")) &&
            error("Refusing to overwrite an existing run: $(SimMetaData.SaveLocation)")
        mkpath(SimMetaData.SaveLocation)
        mkpath(results_location)

        SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
        SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

        result = try
            RunSimulation(
                SimGeometry         = SimulationGeometry,
                SimMetaData         = SimMetaData,
                SimConstants        = SimConstants,
                SimLogger           = SimLogger,
                SimParticles        = SimParticles,
                SimKernel           = SimKernel,
                SimViscosity        = SimViscosity,
                SimDensityDiffusion = SimDensityDiffusion,
                SimTimeStepping     = SimTimeStepping,
                SimMotion           = SquareMotion,
                SimDamping          = SimDamping,
                SimDivergenceCleaning = SimDivergenceCleaning,
            )
        finally
            isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
        end

        # Keep the manual run's force analysis and completion record identical to the batch validation.
        case = (;
            constants = SimConstants, kernel = SimKernel, geometry = SimulationGeometry,
            metadata = SimMetaData, motion = SquareMotion, reynolds, dx, duration,
            interval = output_interval, layers = sampled.layers, layout = sampled.layout,
            sampled.layout..., body_marker = Square.GroupMarker,
            reference_speed, reference_length, span, counts = sampled.counts,
            viscosity_model = SimViscosity, density_diffusion_model = SimDensityDiffusion,
            time_stepping = SimTimeStepping, shifting = typeof(SimMetaData).parameters[3],
            start_delay, motion_kind, boundary_mode, damping = SimDamping, divergence_cleaning = SimDivergenceCleaning,
            domain_padding, sponge_width = use_sponge ? sponge_width : 0.0, sponge_rate = use_sponge ? sponge_rate : 0.0,
        )
        completed = SphericTest06.save_run_results(case, result; results_dir = results_location)

        comparison_state = SphericTest06.write_force_plot_state(case;
            results_dir = results_location, output_dir = plot_location)
        if visualize_in_paraview
            paraview = Sys.which("paraview")
            if paraview === nothing
                @warn "paraview was not found; load $comparison_state in ParaView to create the charts and save the image."
            else
                run(`$paraview --state=$comparison_state`; wait = false)
            end
        end
        (; case, completed)
    end
