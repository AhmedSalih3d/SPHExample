include("case.jl")
using .SphericTest06
using SPHExampleGPU

function main(args)
    mode, folder = if isempty(args)
        PlanarShifting, "fine_spike_planar_uncapped"
    elseif args == ["no-shifting"]
        NoShifting, "fine_spike_no_shifting"
    elseif args == ["dual-shifting"]
        typeof(PlanarShifting(0.1)), "fine_spike_planar_capped"
    else
        error("Expected no argument, no-shifting or dual-shifting")
    end
    save = joinpath(@__DIR__, "runs", folder)
    isfile(joinpath(save, "MovingSquare.vtkhdf")) && error("Refusing to overwrite $save")
    case = build_case(; dx = 1/160, duration = 0.9, cfl = 0.1, sound_speed = 160.0,
        kernel_h_factor = 2sqrt(2), shifting = mode, motion_kind = :integrated, save)
    clip_times = copy(case.metadata.OutputTimes)
    case.metadata.OutputTimes = sort!(unique!(vcat(clip_times, collect(0.76:0.001:0.86))))
    mkpath(save)
    logger = SimulationLogger(save)
    host = AllocateDataStructures(case.geometry, case.metadata)
    result = try
        open(joinpath(save, "console.log"), "w") do stream
            redirect_stdout(stream) do
                RunSimulation(; SimGeometry = case.geometry, SimMetaData = case.metadata,
                    SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
                    SimParticles = host, SimViscosity = case.viscosity_model,
                    SimDensityDiffusion = case.density_diffusion_model, SimTimeStepping = case.time_stepping,
                    SimMotion = case.motion, SimOutputClipTimes = clip_times)
            end
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    result.final_state.complete && result.diagnostics.frozen_particles == 0 || error("Incomplete replay")
    println("Accepted $(result.performance.steps) steps and saved $(result.diagnostics.output_frames) frames")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
