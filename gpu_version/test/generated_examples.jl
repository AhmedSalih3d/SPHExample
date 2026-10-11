using Test
using SPHExampleGPU
using CUDA

@testset "main examples generate particle inputs" begin
    for name in ("StillWedgeMDBC", "StillWedgeMiddleSquareMDBC", "Dambreak2dMDBC",
                 "MovingSquare2d", "FloatingCylinder2d", "LidDrivenCavity2d")
        mktempdir() do dir
            # Capture prepared inputs without running the examples' full simulations.
            scope = Module(gensym(:GeneratedExample))
            Core.eval(scope, :(include(path) = Base.include($scope, path)))
            Core.eval(scope, quote
                captured = Ref{Any}(nothing)
                function RunSimulation(; kwargs...)
                    captured[] = (; kwargs...)
                    close(kwargs[:SimLogger].LoggerIo)
                    return nothing
                end
            end)
            path = joinpath(@__DIR__, "..", "example", name * ".jl")
            source = read(path, String)
            source = replace(source, r"SaveLocation\s*=\s*\x22[^\x22]*\x22" =>
                "SaveLocation = " * repr(dir))
            Base.include_string(scope, source, path)
            if name == "StillWedgeMDBC"
                @test Base.invokelatest(getproperty, scope, :captured)[] === nothing
                Base.invokelatest(Base.invokelatest(getproperty, scope, :run_still_wedge);
                    save_location = dir, visualize = false, open_log_file = false)
            elseif name == "LidDrivenCavity2d"
                Base.invokelatest(Base.invokelatest(getproperty, scope, :run_lid_driven_cavity_2d);
                    dx = 0.05, simulation_time = 0.01,
                    visualize = false, open_log_file = false, save_location = dir,
                    input_dir = joinpath(dir, "does_not_exist"))
            end
            inputs = Base.invokelatest(getproperty, scope, :captured)[]
            @test inputs !== nothing
            @test all(geom -> isempty(geom.CSVFile), inputs.SimGeometry)
            @test !isempty(inputs.SimParticles)
            @test all(isfinite, inputs.SimParticles.Density)
            @test length(unique(inputs.SimParticles.ID)) == length(inputs.SimParticles)
            @test occursin("# CSVFile", source)
            if name == "StillWedgeMDBC"
                @test inputs.SimTimeStepping isa SymplecticTimeStepping
                @test inputs.SimMetaData.GPUDoublePosition
                @test inputs.SimMetaData.OutputTimes == 0.01f0
                @test inputs.SimMetaData.ExportGridCells
                @test eltype(eltype(inputs.SimParticles.Position)) === Float64
                @test eltype(inputs.SimParticles.Density) === Float32
                Base.invokelatest(Base.invokelatest(getproperty, scope, :run_still_wedge);
                    time_stepping = SingleNeighborTimeStepping(), simulation_time = 0.01,
                    save_location = dir, visualize = false, open_log_file = false)
                original_inputs = Base.invokelatest(getproperty, scope, :captured)[]
                @test original_inputs.SimTimeStepping isa SingleNeighborTimeStepping
                @test original_inputs.SimMetaData.SimulationTime == 0.01f0
            end
            if name in ("Dambreak2dMDBC", "LidDrivenCavity2d")
                walls = filter(geom -> geom.Type != Fluid, inputs.SimGeometry)
                @test all(geom -> hasproperty(geom.Particles, :GhostPoints), walls)
                @test all(geom -> hasproperty(geom.Particles, :GhostNormals), walls)
                @test get(inputs, :ParticleNormalsPath, nothing) === nothing
            end

            if CUDA.functional()
                meta = inputs.SimMetaData
                meta.SimulationTime = 0.002f0
                meta.OutputTimes = 0.002f0
                meta.VisualizeInParaview = false
                meta.OpenLogFile = false
                logger = SimulationLogger(dir)
                SPHExampleGPU.RunSimulation(; merge(inputs, (; SimLogger = logger))...)
                @test meta.TotalTime >= meta.SimulationTime
                @test all(isfinite, inputs.SimParticles.Density)
                @test all(>(0), inputs.SimParticles.Density)
            end
        end
    end
end
