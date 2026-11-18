using SPHExampleGPU, CSV
include(joinpath(@__DIR__, "mdbc_validation.jl"))

const M = MDBCValidation
const S = M.C.SphericTest06

let
    isempty(ARGS) || ARGS == ["startup"] || ARGS == ["fluid-only-startup"] ||
        error("Usage: simple_mdbc_zero_ddt.jl [startup|fluid-only-startup]")
    startup = !isempty(ARGS)
    fluid_only = ARGS == ["fluid-only-startup"]
    diffusion = fluid_only ? LinearDensityDiffusion() : ZeroDensityDiffusion()
    duration = startup ? 0.9 : 8.0
    expected_frames = startup ? 46 : 401
    tag = fluid_only ? "simple_mdbc_fluid_ddt_startup" :
        startup ? "simple_mdbc_zero_ddt_startup" : "simple_mdbc_zero_ddt"
    results_dir = joinpath(@__DIR__, "results", tag)
    isfile(joinpath(results_dir, "native.toml")) && error("Refusing to overwrite the completed control: $results_dir")
    case = M.build_mdbc_case(; dx = 0.0125, duration, interval = 0.02,
        sound_speed = 160.0, cfl = 0.2, kernel_h_factor = 1.2sqrt(2), boundary_mode = SimpleMDBC,
        legacy_timestep = true, motion_kind = :integrated, save = joinpath(@__DIR__, "runs", tag, "Re100_n080"))
    motion = PrescribedMotion(3; translation = S.body_trajectory(motion_kind = :integrated))
    case = merge(case, (; motion))
    push!(case.metadata.OutputVariables, "Pressure")

    info = M.run_mdbc_case(case; results_dir, diffusion)
    history = CSV.File(joinpath(results_dir, "native.csv"))
    info.boundary_mode == "SimpleMDBC" && info.diffusion == string(nameof(typeof(diffusion))) ||
        error("Unexpected boundary or diffusion model")
    info.frames == length(history) == expected_frames && first(history.time) == 0.0 && last(history.time) == duration ||
        error("Expected all $expected_frames frames through $duration s")
    println("Completed $duration s of SimpleMDBC with $(info.diffusion): $(info.steps) steps, $(info.frames) frames.")
    println("Results: $results_dir")
end
