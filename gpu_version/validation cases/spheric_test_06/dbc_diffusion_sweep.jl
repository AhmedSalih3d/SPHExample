include("regularized_mdbc_experiment.jl")
using .RegularizedMDBCExperiment: BOUNDARY_MODES, M, RUNS, experiment_case
using SPHExampleGPU, CSV, TOML

const MODELS = (
    "zero" => ZeroDensityDiffusion(),
    "zero-gravity" => ZeroGravityLinearDensityDiffusion(),
    "fluid-linear" => LinearDensityDiffusion(),
    "complex" => ComplexDensityDiffusion(),
    "fourtakas-inner" => FourtakasDensityDiffusion(),
    "fourtakas-full" => FourtakasDensityDiffusion(full = true),
)

for (name, diffusion) in MODELS
    suffix = name == "fluid-linear" ? "" : "-$name"
    dir = joinpath(RUNS, "dbc_n80_half-step-planar$suffix")
    if !isfile(joinpath(dir, "native.csv"))
        println("Starting $name")
        case = experiment_case(BOUNDARY_MODES["dbc"], dir; cfl = 0.1, shifting_mode = PlanarShifting)
        M.run_mdbc_case(case; results_dir = dir, diffusion, compare_stored = false)
    end
    history = CSV.File(joinpath(dir, "native.csv"))
    info = TOML.parsefile(joinpath(dir, "native.toml"))
    model = string(nameof(typeof(diffusion)))
    if diffusion isa FourtakasDensityDiffusion && diffusion.full
        model *= "_full"
    end
    info["diffusion"] == model && info["boundary_mode"] == "NoMDBC" &&
        info["shifting_mode"] == "PlanarShifting" || error("Unexpected configuration for $name")
    length(history) == 46 && first(history.time) == 0.0 && last(history.time) == 0.9 ||
        error("Incomplete output for $name")
    println("Verified $name through 0.90 s")
end
