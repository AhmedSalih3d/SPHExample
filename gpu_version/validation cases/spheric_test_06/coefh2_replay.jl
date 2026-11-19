include("mdbc_validation.jl")
using .MDBCValidation: MDBCValidation as M
using SPHExampleGPU, CSV, TOML

const RUNS = get(ENV, "SPH_COEFH_BASELINES", joinpath(@__DIR__, "runs", "regularized_mdbc_experiment"))
const OUT = joinpath(RUNS, "coefh2")
const BOUNDARY = Dict("NoMDBC" => NoMDBC, "SimpleMDBC" => SimpleMDBC, "DBCPlusImpedance" => DBCPlusImpedance)
if isdefined(SPHExampleGPU, :RegularizedMDBC)
    BOUNDARY["RegularizedMDBC"] = SPHExampleGPU.RegularizedMDBC
end
const DIFFUSION = Dict(
    "ZeroDensityDiffusion" => ZeroDensityDiffusion(),
    "ZeroGravityLinearDensityDiffusion" => ZeroGravityLinearDensityDiffusion(),
    "LinearDensityDiffusion" => LinearDensityDiffusion(),
    "ComplexDensityDiffusion" => ComplexDensityDiffusion(),
    "FourtakasDensityDiffusion" => FourtakasDensityDiffusion(),
    "FourtakasDensityDiffusion_full" => FourtakasDensityDiffusion(full = true),
)
const MATCHED = ("n", "reynolds", "dx", "cfl", "sound_speed", "motion", "frames",
    "background_pressure", "density_diffusion_coefficient", "timestep_criterion", "boundary_mode",
    "shifting_mode", "divergence_cleaning", "fluid_particles", "square_particles", "diffusion")

function replay(tag)
    baseline = TOML.parsefile(joinpath(RUNS, tag, "native.toml"))
    n = baseline["n"]
    isapprox(baseline["h"], 1.2 * sqrt(2) / n; rtol = 1e-7) || error("Unexpected baseline h for $tag")
    dir = joinpath(OUT, tag)
    duration = last(CSV.File(joinpath(RUNS, tag, "native.csv")).time)
    if !isfile(joinpath(dir, "native.csv"))
        println("Running $tag (through $duration s)")
        mode = get(BOUNDARY, baseline["boundary_mode"], nothing)
        if mode === nothing
            println("Unavailable wall mode for $tag: $(baseline["boundary_mode"])")
            return
        end
        shifting_mode = baseline["shifting_mode"] == "PlanarShifting" ? PlanarShifting : NoShifting
        case = M.build_mdbc_case(; dx = baseline["dx"], duration, interval = 0.02,
            sound_speed = baseline["sound_speed"], cfl = baseline["cfl"], kernel_h_factor = 2sqrt(2),
            boundary_mode = mode, legacy_timestep = true, motion_kind = :integrated, shifting_mode,
            save = joinpath(dir, "state"))
        motion = PrescribedMotion(3; translation = M.C.SphericTest06.body_trajectory(motion_kind = :integrated))
        push!(case.metadata.OutputVariables, "Pressure")
        case = merge(case, (; motion))
        M.run_mdbc_case(case; results_dir = dir, diffusion = DIFFUSION[baseline["diffusion"]])
    end
    info = TOML.parsefile(joinpath(dir, "native.toml"))
    all(key -> info[key] == baseline[key], MATCHED) || error("Replay settings differ for $tag")
    isapprox(info["h"], 2sqrt(2) / n; rtol = 1e-7) || error("Unexpected replay h for $tag")
    info["wall_layers"] == ceil(Int, 2 * info["h"] / info["dx"]) + 1 ||
        error("Unexpected tank wall depth for $tag")
    times = CSV.File(joinpath(dir, "native.csv")).time
    times == CSV.File(joinpath(RUNS, tag, "native.csv")).time || error("Output times differ for $tag")
    println("Verified $tag ($(info["steps"]) steps, $(length(times)) frames)")
end

tags = filter(tag -> isfile(joinpath(RUNS, tag, "native.toml")), readdir(RUNS))
sort!(tags; by = tag -> (endswith(tag, "_full"), !occursin("half-step-planar", tag), tag))
for tag in (isempty(ARGS) ? tags : ARGS)
    replay(tag)
end
