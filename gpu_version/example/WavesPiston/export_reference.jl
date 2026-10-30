# Rebuild the bundled tables from a local DualSPHysics 5.4 distribution.
# julia --project=gpu_version gpu_version/example/WavesPiston/export_reference.jl W:/DualSPHysics_v5.4
length(ARGS) in (1, 2) || error("supply the DualSPHysics 5.4 distribution directory and optionally an output directory")
distribution = abspath(ARGS[1])
reference = joinpath(distribution, "examples", "main", "09_WavesPiston")
binary(name) = Sys.iswindows() ? joinpath(distribution, "bin", "windows", name * "_win64.exe") :
    joinpath(distribution, "bin", "linux", name * "_linux64")
destination = length(ARGS) == 2 ? abspath(ARGS[2]) : joinpath(@__DIR__, "..", "data", "waves_piston")
mkpath(destination)

mktempdir() do scratch
    for (kind, suffix) in (("Regular", "REG"), ("Irregular", "IRREG"))
        xml = read(joinpath(reference, "CasePistonBeach_$(suffix)_Def.xml"), String)
        # Motion/theory export density changes; physical and spectral settings do not.
        xml = replace(xml, "periodsteps=\"20\"" => "periodsteps=\"200\"",
            "<savemotion time=\"20\" timedt=\"0.1\"" => "<savemotion time=\"30\" timedt=\"0.01\"",
            "<saveserie " => "<_saveserie ")
        casepath = joinpath(scratch, "Piston" * kind)
        write(casepath * "_Def.xml", xml)
        run(`$(binary("GenCase")) $(casepath * "_Def") $casepath -save:all`)
        outdir = joinpath(scratch, lowercase(kind))
        # Keep the physical duration at 30 s so theoretical series remain active.
        # The solver only needs initialization and one step to export the tables.
        run(`$(binary("DualSPHysics5.4CPU")) $casepath $outdir -tmax:30 -nsteps:1 -sv:none -ompthreads:2`)
        lines = filter(readlines(joinpath(outdir, "WavePaddle_mkb0010.csv"))) do line
            occursin(r"^\d+\.\d+;", line) && parse(Float64, first(split(line, ';'))) <= 30
        end
        open(joinpath(destination, kind * "Theory.csv"), "w") do io
            println(io, "time,displacement,elevation_o1,elevation_o2,velocity_x_o1,velocity_y_o1,velocity_x_o2,velocity_y_o2")
            foreach(line -> println(io, replace(line, ';' => ',')), lines)
        end
        if kind == "Irregular"
            open(joinpath(destination, "IrregularMotion.csv"), "w") do io
                println(io, "time,displacement")
                foreach(line -> println(io, join(split(line, ';')[1:2], ',')), lines)
            end
        end
    end
end
