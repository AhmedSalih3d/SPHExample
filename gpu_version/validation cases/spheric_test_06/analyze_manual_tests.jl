include("case.jl")
using .SphericTest06
using SPHExampleGPU, CSV

function main()
    results = mkpath(joinpath(@__DIR__, "results", "manual_tests"))
    for test in 1:4
        name = "Re100_n040_Test$test"
        path = joinpath(@__DIR__, "runs", "coefh", name, "MovingSquare.vtkhdf")
        case = build_case(; dx = 0.025, sound_speed = test == 4 ? 160.0 : 40.0,
            kernel_h_factor = 2sqrt(2), motion_kind = :integrated)
        analysis = analyze_history(path, case;
            coefficient_path = joinpath(results, "$name.dat"),
            diagnostics_path = joinpath(results, "$(name)_state.csv"),
            trajectory = case.motion.translation)
        length(analysis.rows) == 401 || error("Incomplete history for $name")
        println("Reconstructed $name")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
