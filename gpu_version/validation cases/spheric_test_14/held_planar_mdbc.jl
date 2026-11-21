# Independent buoyancy check at the nominal draft, without release motion.
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "planar_case.jl"))
using Statistics

function held_planar_study()
    root = joinpath(@__DIR__, "results", "mdbc_study", "held")
    mkpath(root)
    summary = NamedTuple[]
    for index in (1, 3), mode in (NoMDBC, SimpleMDBC)
        release = merge(RELEASES[index], (; offset = 0.0))
        tag = "$(release.name)_$(nameof(mode))"
        history = joinpath(root, tag * ".csv")
        case = build_planar_case(release; boundary_mode = mode, pause_time = 4., duration = 3.,
            output_interval = 0.02, save_location = joinpath(@__DIR__, "runs", "mdbc_held", tag))
        if !isfile(history)
            save = case.metadata.SaveLocation
            isfile(joinpath(save, "FreeHeave.vtkhdf")) && error("Existing unfinished run: $save")
            case.metadata.OutputVariables = ["Type"]
            mkpath(save)
            logger = SimulationLogger(save)
            host = AllocateDataStructures(case.geometry, case.metadata)
            result = try
                RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata, SimConstants = case.constants,
                    SimKernel = case.kernel, SimLogger = logger, SimParticles = host, SimViscosity = case.viscosity,
                    SimDensityDiffusion = case.diffusion, SimTimeStepping = case.stepping, SimDamping = case.damping)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            result.final_state.complete && result.diagnostics.frozen_particles == 0 || error("Incomplete held run")
            rows = CSV.File(joinpath(save, "FreeHeave_Floating.csv"))
            z = getproperty(rows, Symbol("Center:1"))
            all(v -> isapprox(v, case.equilibrium_center_z; atol = 1e-8), z) || error("Body moved during hold")
            isapprox(last(rows.Time), 3.; atol = 1e-6) || error("Incomplete history")
            force = getproperty(rows, Symbol("Force:1"))
            all(isfinite, force) && any(!iszero, force) || error("Invalid force")
            CSV.write(history, (; time = rows.Time, fluid_force_n_m = force, center_z_m = z))
        end
        data = CSV.File(history)
        force = data.fluid_force_n_m[data.time .>= 2.]
        weight = case.line_mass * Float64(case.constants.g)
        push!(summary, (; release = release.name, boundary = string(nameof(mode)), weight_n_m = weight,
            mean_force_n_m = mean(force), force_std_n_m = std(force), relative_excess = mean(force)/weight-1))
        println(last(summary)); flush(stdout)
    end
    CSV.write(joinpath(root, "summary.csv"), summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    held_planar_study()
end
