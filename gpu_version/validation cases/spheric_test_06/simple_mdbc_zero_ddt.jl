using SPHExampleGPU, CSV
include(joinpath(@__DIR__, "mdbc_validation.jl"))

const M = MDBCValidation
const S = M.C.SphericTest06

let
    isempty(ARGS) || ARGS == ["startup"] || error("Usage: simple_mdbc_zero_ddt.jl [startup]")
    startup = !isempty(ARGS)
    duration = startup ? 0.9 : 8.0
    expected_frames = startup ? 46 : 401
    tag = startup ? "simple_mdbc_zero_ddt_startup" : "simple_mdbc_zero_ddt"
    results_dir = joinpath(@__DIR__, "results", tag)
    isfile(joinpath(results_dir, "native.toml")) && error("Refusing to overwrite the completed control: $results_dir")
    case = M.build_mdbc_case(; dx = 0.0125, duration, interval = 0.02,
        sound_speed = 160.0, cfl = 0.2, kernel_h_factor = 1.2sqrt(2), boundary_mode = SimpleMDBC,
        legacy_timestep = true, motion_kind = :integrated, save = joinpath(@__DIR__, "runs", tag, "Re100_n080"))
    motion = PrescribedMotion(3; translation = S.body_trajectory(motion_kind = :integrated))
    case = merge(case, (; motion))
    push!(case.metadata.OutputVariables, "Pressure")

    info = M.run_mdbc_case(case; results_dir, diffusion = ZeroDensityDiffusion())
    history = CSV.File(joinpath(results_dir, "native.csv"))
    info.boundary_mode == "SimpleMDBC" && info.diffusion == "ZeroDensityDiffusion" ||
        error("Unexpected boundary or diffusion model")
    info.frames == length(history) == expected_frames && first(history.time) == 0.0 && last(history.time) == duration ||
        error("Expected all $expected_frames frames through $duration s")
    println("Completed the $duration s SimpleMDBC no-diffusion control: $(info.steps) steps, $(info.frames) frames.")
    println("Results: $results_dir")
end
