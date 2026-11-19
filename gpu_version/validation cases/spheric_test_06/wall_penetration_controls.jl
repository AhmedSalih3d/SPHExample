include("mdbc_validation.jl")
using .MDBCValidation
using .MDBCValidation.C.SphericTest06
using SPHExampleGPU, TOML

const TAG = "wall_penetration_controls"
const RUNS = joinpath(@__DIR__, "runs", TAG)
const RESULTS = joinpath(@__DIR__, "results", TAG)
const SETTINGS = (
    ("no-shifting", NoShifting, 0.2, 0.0),
    ("half-cfl", PlanarShifting, 0.1, 0.0),
    ("pressure-offset", PlanarShifting, 0.2, 1000.0),
    ("surface-row", PlanarShifting, 0.2, 0.0),
)

function align_square_surface!(case)
    particles = case.geometry[3].Particles
    half_width = case.body_size / 2
    outer_row = half_width .- case.dx / 2
    for i in eachindex(particles.Position)
        old = particles.Position[i]
        local_pos = old - case.body_center
        on_outer_row = isapprox.(abs.(local_pos), outer_row; atol = 1e-12, rtol = 0)
        aligned = case.body_center + ifelse.(on_outer_row, copysign.(half_width, local_pos), local_pos)
        particles.Position[i] = aligned
        if case.boundary_mode === SimpleMDBC
            particles.GhostNormals[i] = particles.GhostPoints[i] - aligned
        end
    end
end

function main(args)
    for (name, shifting, cfl, background_pressure) in SETTINGS, boundary_mode in (NoMDBC, SimpleMDBC)
        isempty(args) || name in args || continue
        tag = joinpath(string(boundary_mode), name)
        save = joinpath(RUNS, tag)
        results_dir = joinpath(RESULTS, tag)
        record = joinpath(results_dir, "Re100_n040.toml")
        if !isfile(record)
            println("Running $tag")
            case = build_case(; dx = 0.025, duration = 2.0, sound_speed = 160.0,
                kernel_h_factor = 2sqrt(2), motion_kind = :integrated, boundary_mode,
                shifting, cfl, background_pressure, save)
            name == "surface-row" && align_square_surface!(case)
            push!(case.metadata.OutputVariables, "Pressure")
            mkpath(save)
            open(joinpath(save, "console.log"), "w") do io
                redirect_stdout(io) do
                    run_case(case; results_dir)
                end
            end
        end
        result = TOML.parsefile(record)
        result["frames"] == 101 && result["duration"] == 2.0 || error("Incomplete history for $tag")
        println("$tag minimum clearance: $(1000 * result["minimum_clearance"]) mm")
        flush(stdout)
    end
    full = "impedance-full" in args || "updated-impedance-full" in args
    fine = "impedance-fine" in args
    if isempty(args) || any(in(args), ("boundary-models", "updated-impedance", "updated-impedance-full", "impedance-full", "impedance-fine"))
        duration = full ? 8.0 : 2.0
        modes = if any(in(args), ("updated-impedance", "updated-impedance-full"))
            (UpdatedMDBCWithImpedance,)
        elseif fine
            (DBCPlusImpedance, UpdatedMDBCWithImpedance)
        elseif full
            (DBCPlusImpedance,)
        else
            (UpdatedMDBC, DBCPlusImpedance, UpdatedMDBCWithImpedance)
        end
        for boundary_mode in modes
            tag = string(boundary_mode) * (fine ? "_n080" : full ? "_full" : "")
            save = joinpath(RUNS, tag)
            results_dir = joinpath(RESULTS, tag)
            record = joinpath(results_dir, "native.toml")
            if !isfile(record)
                println("Running $tag")
                case = build_mdbc_case(; dx = fine ? 0.0125 : 0.025, duration, sound_speed = 160.0,
                    kernel_h_factor = 2sqrt(2), boundary_mode, shifting_mode = PlanarShifting,
                    legacy_timestep = true, save)
                motion = PrescribedMotion(3; translation = body_trajectory(motion_kind = :integrated))
                push!(case.metadata.OutputVariables, "Pressure")
                run_mdbc_case(merge(case, (; motion)); results_dir)
            end
            result = TOML.parsefile(record)
            result["frames"] == round(Int, duration / 0.02) + 1 || error("Incomplete history for $tag")
            println("$tag minimum clearance: $(1000 * result["minimum_clearance"]) mm")
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
