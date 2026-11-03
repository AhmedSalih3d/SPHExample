using Test
using SPHExampleGPU
using StaticArrays
using StructArrays

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
    @testset "damping blocks get a wireframe display" begin
        mktempdir() do directory
            metadata = SimulationMetaData{2, Float32}(
                SimulationName = "DampingView",
                SaveLocation = directory,
                VisualizeInParaview = false,
                OpenLogFile = false,
            )
            AutoOpenParaview(
                metadata, SimulationConstants{Float32}(), ["Density"];
                paraview_cmd = nothing, show_damping = true,
            )
            state = read(
                joinpath(directory, "DampingView_SingleVTKHDFStateFile.py"),
                String,
            )
            @test occursin("if True and data_type == 'vtkMultiBlockDataSet':", state)
            @test occursin("point_array_status.append('NormalizedDamping')", state)
            @test occursin(
                "DampingBlocks.Selectors = ['/Root/DampingZones', '/Root/DampingProfiles']",
                state,
            )
            @test occursin("DampingBlocksDisplay.SetRepresentationType('Wireframe')", state)
            @test occursin(
                "ColorBy(DampingBlocksDisplay, ('POINTS', 'NormalizedDamping'))",
                state,
            )
            @test occursin("damping_lut.RescaleTransferFunction(0.0, 1.0)", state)
            @test occursin(
                "damping_bar.Title = 'Normalized damping weight (dimensionless)'",
                state,
            )
        end
    end
    pvpython = Sys.which("pvpython")
    if pvpython === nothing
        @test_skip false # Optional integration check requires ParaView.
    else
        @testset "floating body force vectors and generated glyphs" begin
            for D in (2, 3), single in (false, true)
                mktempdir() do directory
                    metadata = SimulationMetaData{D,Float32}(SimulationName = "BodyLoads", SaveLocation = directory,
                        ExportSingleVTKHDF = single, SimulationTime = 0.1, OutputTimes = 0.1,
                        OutputVariables = ["Density"], VisualizeInParaview = false, OpenLogFile = false)
                    constants = SimulationConstants{Float32}(dx = 1e-5)
                    kernel = SPHKernelInstance{D,Float32}(WendlandC2(); dx = constants.dx)
                    v = SVector{D,Float32}(ntuple(d -> Float32(d), D))
                    body = (; center = [v], velocity = [zero(v)], load_force = [2v],
                        load_torque = D == 2 ? Float32[3] : [3v])
                    zero_body = merge(body, (; load_force = [zero(v)]))
                    particles = StructArray((; Position = [v .- 20, v .+ 20], Density = Float32[1000, 1000]))
                    output = SetupVTKOutput(metadata, particles, kernel, D; constants,
                        floating_state = body, floating_markers = [7])
                    try
                        output.save_particles(1, 0.0; body_state = zero_body)
                        output.save_particles(2, 0.1; body_state = body)
                    finally
                        output.close_files()
                    end
                    AutoOpenParaview(metadata, constants, ["Density"]; paraview_cmd = nothing)
                    suffix = single ? "_SingleVTKHDFStateFile.py" : "_StateFile.py"
                    state = joinpath(directory, "BodyLoads" * suffix)
                    open(state, "a") do io
                        write(io, "\nfrom paraview import servermanager\n")
                        write(io, "FloatingBodies.UpdatePipeline(list(Simulation_vtkhdf.TimestepValues)[-1])\n")
                        write(io, "body_data = servermanager.Fetch(FloatingBodies).GetBlock(0)\n")
                        write(io, "assert body_data.GetNumberOfPoints() == 1\n")
                        write(io, "assert body_data.GetPointData().GetArray('Force').GetTuple3(0) == $(D == 2 ? "(2.0, 4.0, 0.0)" : "(2.0, 4.0, 6.0)")\n")
                        write(io, "FloatingForceGlyphs.UpdatePipeline(list(Simulation_vtkhdf.TimestepValues)[-1])\n")
                        write(io, "assert servermanager.Fetch(FloatingForceGlyphs).GetNumberOfPoints() > 0\n")
                        write(io, "assert FloatingForceDisplay.Visibility == 1\n")
                        write(io, "assert FloatingForceDisplay.Representation == 'Surface'\n")
                        write(io, "assert FloatingForceGlyphs.ScaleFactor > 1.0\n")
                        write(io, "assert FloatingForceDisplay.UseSeparateColorMap == 1\n")
                        write(io, "assert floating_force_bar.Visibility == 1\n")
                        write(io, "assert floating_force_bar.Title == 'Floating body force [N]'\n")
                        write(io, "FloatingForceGlyphs.UpdatePipeline(list(Simulation_vtkhdf.TimestepValues)[0])\n")
                        write(io, "zero_bounds = FloatingForceGlyphs.GetDataInformation().GetBounds()\n")
                        write(io, "assert max(zero_bounds[2*d+1] - zero_bounds[2*d] for d in range(3)) < 1e-6 * FloatingForceGlyphs.ScaleFactor\n")
                    end
                    process = run(pipeline(ignorestatus(`$pvpython --force-offscreen-rendering --no-mpi --dr $state`);
                        stdout = stdout, stderr = stderr))
                    @test process.exitcode == 0
                end
            end
        end
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
                    paraview_cmd = nothing, pressure_range = pressure_range,
                    color_variable = single_file ? "Pressure" : "Density")
                @test_throws ArgumentError AutoOpenParaview(metadata, constants,
                    ["Pressure"]; paraview_cmd = nothing, pressure_range = (1.0, 0.0))
                state_suffix = single_file ? "_SingleVTKHDFStateFile.py" : "_StateFile.py"
                state = joinpath(output, name * state_suffix)
                open(state, "a") do io
                    write(io, "\nassert len(GetSources()) == $(export_grid ? 2 : 1)\n")
                    write(io, "assert file_list == sorted(file_list)\n")
                    write(io, "assert len(file_list) == $(length(suffixes))\n")
                    write(io, "assert Simulation_vtkhdfDisplay.BlockSelectors == ['/']\n")
                    write(io, "assert Simulation_vtkhdfDisplay.ColorArrayName[1] == " *
                        "'$(single_file ? "Pressure" : "Density")'\n")
                    expected_range = single_file ? [-1000.0, 20000.0] : [0.0, 1.0]
                    write(io, "assert pressure_lut.RGBPoints[0] == $(expected_range[1])\n")
                    write(io, "assert pressure_lut.RGBPoints[-4] == $(expected_range[2])\n")
                    write(io, "assert pressure_lut.AutomaticRescaleRangeMode == 'Never'\n")
                    write(io, "assert pressure_bar.Title == 'Pressure [Pa]'\n")
                    write(io, "assert pressure_bar.Visibility == $(single_file ? 1 : 0)\n")
                    write(io, "ColorBy(Simulation_vtkhdfDisplay, ('POINTS', 'Pressure'))\n")
                    write(io, "Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1, True)\n")
                    write(io, "assert pressure_bar.Visibility == 1\n")
                    write(io, "ColorBy(Simulation_vtkhdfDisplay, ('POINTS', 'Density'))\n")
                    write(io, "HideScalarBarIfNotNeeded(pressure_lut, renderView1)\n")
                    write(io, "assert pressure_bar.Visibility == 0\n")
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
