module DBCPlusStudy

using SPHExampleGPU, CUDA, CSV, TOML, Test
include(joinpath(@__DIR__, "mdbc_validation.jl"))
const M = MDBCValidation
const S = M.C.SphericTest06

function run_case(; duration = 0.9, dx = 0.0125, graph = true, output_root = @__DIR__)
    tag = "n$(round(Int,1/dx))_t$(duration)_$(graph ? "graph" : "direct")"
    directory = joinpath(output_root, "results", "dbc_plus", tag)
    record = joinpath(directory, "native.toml")
    if isfile(record)
        error("Refusing to overwrite DBC+ study $directory")
    end
    case = M.build_mdbc_case(; dx, duration, sound_speed = 160.0, cfl = 0.2,
        kernel_h_factor = 1.2sqrt(2), boundary_mode = DBCPlus, legacy_timestep = true,
        motion_kind = :integrated, graph, save = joinpath(output_root, "runs", "dbc_plus", tag))
    case = merge(case, (; motion = PrescribedMotion(3; translation = S.body_trajectory(motion_kind = :integrated))))
    push!(case.metadata.OutputVariables, "Pressure")
    mkpath(case.metadata.SaveLocation)
    isfile(joinpath(case.metadata.SaveLocation, "MovingSquare.vtkhdf")) &&
        error("Refusing to overwrite partial particle history")
    logger = SimulationLogger(case.metadata.SaveLocation)
    host = AllocateDataStructures(case.geometry, case.metadata)
    result = open(joinpath(case.metadata.SaveLocation, "console.log"), "w") do io
        redirect_stdout(io) do
            try
                RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
                    SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
                    SimParticles = host, SimViscosity = Laminar(), SimDensityDiffusion = LinearDensityDiffusion(),
                    SimTimeStepping = SymplecticTimeStepping(), SimMotion = case.motion)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
        end
    end
    result.final_state.complete && result.final_state.time == duration || error("Incomplete DBC+ run")
    result.diagnostics.frozen_particles == 0 || error("DBC+ froze particles")
    mkpath(directory)
    analysis = M.analyze_mdbc_history(joinpath(case.metadata.SaveLocation, "MovingSquare.vtkhdf"), case;
        coefficient_path = joinpath(directory, "native.dat"), diagnostics_path = joinpath(directory, "native_state.csv"))
    open(record, "w") do io
        TOML.print(io, Dict("boundary" => "DBC+", "diffusion" => "LinearDensityDiffusion",
            "viscosity" => "Laminar, physical DBC wall velocity", "duration" => duration, "dx" => dx,
            "c0" => 160.0, "coefh" => 1.2, "cfl" => 0.2, "steps" => result.performance.steps,
            "frames" => result.diagnostics.output_frames, "minimum_clearance" => minimum(r.minimum_clearance for r in analysis.checks),
            "density_relative_max" => maximum(r.density_relative_max for r in analysis.checks),
            "motion" => "integrated", "reynolds" => 100,
            "background_pressure" => 0.0, "wall_layers" => case.layers,
            "particle_count" => length(host), "frozen_particles" => result.diagnostics.frozen_particles,
            "solver_commit" => strip(read(`git -C $(@__DIR__) rev-parse HEAD`, String))))
    end
    return (; result, analysis, directory)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    DBCPlusStudy.run_case()
end
