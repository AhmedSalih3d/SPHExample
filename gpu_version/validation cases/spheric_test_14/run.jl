include(joinpath(@__DIR__, "case.jl"))

function setup(release; dx = 0.025, half_width = 1.2, duration = 4.5, save)
    depth = 0.8
    T = Float32
    duration > 0 || throw(ArgumentError("Duration must be positive"))
    constants = SimulationConstants{T}(3; dx, g = 9.81, c₀ = 25.0, CFL = 0.2, α = 0.01, δᵩ = 0.1)
    kernel = SPHKernelInstance{3,T}(WendlandC2(); h = T(1.3dx))
    geometry, equilibrium_center_z = sample_geometry(release, constants, kernel; dx, half_width, depth)
    damping = DampingCylinder((0.0, 0.0, -depth), (0.0, 0.0, 0.2),
        half_width - 0.5, half_width; rate = 12)
    metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        SimulationName = "FreeHeave", SaveLocation = save, SimulationTime = duration,
        OutputTimes = 0.01, ExportSingleVTKHDF = true, OutputVariables = ["Type"],
        VisualizeInParaview = false, OpenLogFile = false, GPUDoublePosition = true,
        GPUClipTimeStepToOutput = true)
    return (; geometry, metadata, constants, kernel, damping, equilibrium_center_z, dx, half_width, depth, duration,
        viscosity = ArtificialViscosity(), diffusion = FourtakasDensityDiffusion(), stepping = SymplecticTimeStepping())
end

function run_case(release; dx, half_width, duration)
    tag = "dp$(dx)_half$(half_width)_t$(duration)"
    save = joinpath(@__DIR__, "runs", tag, release.name)
    output = joinpath(@__DIR__, "results", tag)
    return run_case(release, setup(release; dx, half_width, duration, save); output)
end

function main(args)
    visualize = "--no-paraview" ∉ args
    positional = filter(!=("--no-paraview"), args)
    length(positional) <= 4 || error("Usage: run.jl [dx] [tank half-width] [duration] [release] [--no-paraview]")
    dx, half_width, duration = (i <= length(positional) ? parse(Float64, positional[i]) : value
        for (i, value) in enumerate((0.025, 1.2, 4.5)))
    cases = length(positional) == 4 ? filter(r -> r.name == positional[4], RELEASES) : RELEASES
    isempty(cases) && error("Release must be float1_down, float1_up or float2_down")

    first_state = nothing
    for release in cases
        output = run_case(release; dx, half_width, duration)
        visualize || continue
        try
            state = write_heave_state(release, output)
            println("ParaView launcher saved to ", state)
            first_state === nothing && (first_state = state)
        catch exception
            exception isa InterruptException && rethrow()
            @warn "Could not create the combined ParaView view for $(release.name)" exception
        end
    end
    if first_state !== nothing
        paraview = Sys.which("paraview")
        if paraview === nothing
            @warn "ParaView was not found on PATH. Open $first_state with File > Load State."
        else
            try
                run(`$paraview --state=$first_state`; wait = false)
            catch exception
                exception isa InterruptException && rethrow()
                @warn "Could not open ParaView. Open $first_state with File > Load State." exception
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
