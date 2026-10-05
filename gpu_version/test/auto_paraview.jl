using Test
using SPHExampleGPU
using StaticArrays

@testset "ParaView particles and cell grid" begin
    pvpython = Sys.which("pvpython")
    if pvpython === nothing
        @test_skip false # Optional integration check requires ParaView.
    else
        mktempdir() do directory
            constants = SimulationConstants{Float32}(dx = 0.02f0)
            for (single_file, export_grid) in ((true, true), (false, true), (true, false))
                output = mkpath(joinpath(directory, "$(single_file)_$(export_grid)"))
                name = "Case.with.dots"
                metadata = SimulationMetaData{2, Float32, NoShifting,
                    NoKernelOutput, NoMDBC, NoLog}(
                    SimulationName = name, SaveLocation = output,
                    ExportSingleVTKHDF = single_file, ExportGridCells = export_grid,
                    VisualizeInParaview = false, OpenLogFile = false)
                suffixes = single_file ? [""] : ["_000002", "_000001"]
                for suffix in suffixes
                    SaveVTKHDF(joinpath(output, name * suffix * ".vtkhdf"),
                        [SVector(0.01, 0.01, 0.0)], ["Density"], [1000.0])
                    grid_name = single_file ? name * "_GridCells" :
                        "CellGrid_" * name * suffix
                    SaveCellGridVTKHDF(joinpath(output, grid_name * ".vtkhdf"),
                        0.04, [CartesianIndex(0, 0)])
                end
                # A similarly named old run must not enter either file series.
                touch(joinpath(output, name * "_old.vtkhdf"))
                AutoOpenParaview(metadata, constants, ["Density"]; paraview_cmd = nothing)
                state_suffix = single_file ? "_SingleVTKHDFStateFile.py" : "_StateFile.py"
                state = joinpath(output, name * state_suffix)
                open(state, "a") do io
                    write(io, "\nassert len(GetSources()) == $(export_grid ? 2 : 1)\n")
                    write(io, "assert file_list == sorted(file_list)\n")
                    write(io, "assert len(file_list) == $(length(suffixes))\n")
                    if export_grid
                        write(io, "assert grid_display.Representation == 'Wireframe'\n")
                        write(io, "assert grid_display.Visibility == 1\n")
                        write(io, "assert len(grid_files) == $(length(suffixes))\n")
                        write(io, "assert grid_files == sorted(grid_files)\n")
                    end
                end
                process = run(pipeline(ignorestatus(`$pvpython $state`);
                    stdout = stdout, stderr = stderr))
                @test process.exitcode == 0
            end
        end
    end
end
