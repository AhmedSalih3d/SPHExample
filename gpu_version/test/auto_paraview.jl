using Test
using SPHExampleGPU
using StaticArrays

@testset "ParaView particles and cell grid" begin
    pressure_range_for = SPHExampleGPU.OpenExternalPrograms.hydrostatic_pressure_range
    constants = SimulationConstants{Float64}(dx = 0.02, g = 10.0)
    for positions in ([SVector(0.0, 1.0), SVector(0.0, 3.0), SVector(0.0, 50.0)],
                      [SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 3.0),
                       SVector(0.0, 0.0, 50.0)])
        particles = (Position = positions, Type = [Fluid, Fluid, Fixed])
        @test pressure_range_for(particles, constants) == (0.0, 20200.0)
    end
    @test pressure_range_for((Position = [SVector(0.0, 0.0)], Type = [Fixed]),
        constants) === nothing
    @test pressure_range_for((Position = [SVector(0.0, 0.0)], Type = [Fluid]),
        SimulationConstants{Float64}(g = 0.0)) === nothing
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
                        [SVector(0.01, 0.01, 0.0)], ["Density", "Pressure"],
                        [1000.0], [0.0])
                    grid_name = single_file ? name * "_GridCells" :
                        "CellGrid_" * name * suffix
                    SaveCellGridVTKHDF(joinpath(output, grid_name * ".vtkhdf"),
                        0.04, [CartesianIndex(0, 0)])
                end
                # A similarly named old run must not enter either file series.
                touch(joinpath(output, name * "_old.vtkhdf"))
                pressure_range = single_file ? (-1000.0, 20000.0) : nothing
                AutoOpenParaview(metadata, constants, ["Density", "Pressure"];
                    paraview_cmd = nothing, pressure_range = pressure_range)
                @test_throws ArgumentError AutoOpenParaview(metadata, constants,
                    ["Pressure"]; paraview_cmd = nothing, pressure_range = (1.0, 0.0))
                state_suffix = single_file ? "_SingleVTKHDFStateFile.py" : "_StateFile.py"
                state = joinpath(output, name * state_suffix)
                open(state, "a") do io
                    write(io, "\nassert len(GetSources()) == $(export_grid ? 2 : 1)\n")
                    write(io, "assert file_list == sorted(file_list)\n")
                    write(io, "assert len(file_list) == $(length(suffixes))\n")
                    write(io, "assert Simulation_vtkhdfDisplay.ColorArrayName[1] == " *
                        "'Pressure'\n")
                    expected_range = single_file ? [-1000.0, 20000.0] : [0.0, 1.0]
                    write(io, "assert pressure_lut.RGBPoints[0] == $(expected_range[1])\n")
                    write(io, "assert pressure_lut.RGBPoints[-4] == $(expected_range[2])\n")
                    write(io, "assert pressure_lut.AutomaticRescaleRangeMode == 'Never'\n")
                    write(io, "assert pressure_bar.Title == 'Pressure [Pa]'\n")
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
