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
            if name == "LidDrivenCavity2d"
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
                logger = SimulationLogger(dir; to_console = false)
                SPHExampleGPU.RunSimulation(; merge(inputs, (; SimLogger = logger))...)
                @test meta.TotalTime >= meta.SimulationTime
                @test all(isfinite, inputs.SimParticles.Density)
                @test all(>(0), inputs.SimParticles.Density)
            end
        end
    end
end
