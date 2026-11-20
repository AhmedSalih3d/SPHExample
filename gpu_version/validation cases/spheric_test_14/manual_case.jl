# Edit the settings below and run from gpu_version:
# julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case.jl"
using SPHExampleGPU
include(joinpath(@__DIR__, "run.jl"))

let
    Dimensions = 3
    FloatType  = Float32

    # Choose float1_down, float1_up, or float2_down.
    release_name = "float1_down"
    release = only(filter(release -> release.name == release_name, RELEASES))

    # Geometry in metres. The water surface is at z = 0.
    dx         = 0.025
    half_width = 1.2
    depth      = 0.8
    wall_top_z = 0.2

    sponge_width = 0.5
    sponge_rate  = 12.0
    duration = 4.5
    output_interval = 0.01
    visualize_in_paraview = true

    run_name        = "manual_dp$(dx)_half$(half_width)_t$(duration)"
    save_location   = joinpath(@__DIR__, "runs", run_name, release_name)
    result_location = joinpath(@__DIR__, "results", run_name)

    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx  = dx,
        ρ₀  = 1000.0,
        g   = 9.81,
        c₀  = 25.0,
        α   = 0.01,
        δᵩ  = 0.1,
        CFL = 0.2,
    )

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(1.3dx))
    SimulationGeometry, equilibrium_center_z = sample_geometry(
        release, SimConstants, SimKernel; dx, half_width, depth, wall_top_z)

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName          = "FreeHeave",
        SaveLocation            = save_location,
        SimulationTime          = duration,
        OutputTimes             = output_interval,
        OutputVariables         = ["Density", "Pressure", "Velocity", "Type", "GroupMarker"],
        ExportSingleVTKHDF      = true,
        VisualizeInParaview     = false, # Open the combined state after the heave CSV is saved.
        OpenLogFile             = false,
        GPUUseGraph             = true,
        GPUDoublePosition       = true,
        GPUClipTimeStepToOutput = true,
    )

    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimTimeStepping     = SymplecticTimeStepping()
    SimDamping          = DampingCylinder((0.0, 0.0, -depth), (0.0, 0.0, wall_top_z),
        half_width - sponge_width, half_width; rate = sponge_rate)

    case = (; geometry = SimulationGeometry, metadata = SimMetaData, constants = SimConstants,
        kernel = SimKernel, damping = SimDamping, equilibrium_center_z, dx, half_width, depth, duration,
        viscosity = SimViscosity, diffusion = SimDensityDiffusion, stepping = SimTimeStepping)
    completed = run_case(release, case; output = result_location)

    state = write_heave_state(release, completed; layout = "stacked", screenshot = true)
    println("ParaView launcher saved to ", state)
    if visualize_in_paraview
        paraview = Sys.which("paraview")
        if paraview === nothing
            @warn "Load $state using File > Load State in ParaView."
        else
            run(`$paraview --state=$state`; wait = false)
        end
    end
end
