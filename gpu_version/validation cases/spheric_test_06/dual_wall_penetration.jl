include("dual_compare.jl")
using .Test06DualComparison: prepare_dual, dual_run, logged, PART

function export_positions(data, directory)
    mkpath(directory)
    output = joinpath(directory, "positions")
    logged(`$PART -dirdata $data -savevtk $output -savestatscsv $output`,
        joinpath(directory, "partvtk.log"))
end

function main(args)
    matched = args == ["matched"]
    isempty(args) || matched || error("Choose the fine run or pass matched for the n40 no-shifting control")
    n, duration, shifting = matched ? (40, 2.0, 0) : (80, 8.0, 2)
    tag = matched ? "dual-n040-matched" : "dual-n080"
    directory = joinpath(@__DIR__, "runs", "wall_penetration_controls", tag)
    if !isfile(joinpath(directory, "Run.csv"))
        isdir(joinpath(directory, "data")) && error("Existing partial DualSPHysics run: $directory")
        prefix = prepare_dual(; n, reynolds = 100, sound_speed = 160, coefh = 2.0, duration,
            motion_kind = :integrated, shifting)
        println(dual_run(prefix, directory; postprocess = false))
    end
    export_positions(joinpath(directory, "data"), joinpath(directory, "vtk"))
    archived = joinpath(@__DIR__, "runs", "coefh_reynolds", "re100_dual", "data")
    if !matched && isdir(archived)
        export_positions(archived, joinpath(dirname(directory), "dual-pilot"))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
