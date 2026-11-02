module WaveRunupTests

using Test, SPHExampleGPU, CUDA, HDF5

function prepare_example(directory)
    scope = Module(gensym(:WaveRunupExample))
    Core.eval(scope, quote
        captured = Ref{Any}(nothing)
        function RunSimulation(; kwargs...)
            captured[] = (; kwargs...)
            close(kwargs[:SimLogger].LoggerIo)
            return nothing
        end
    end)
    path = joinpath(@__DIR__, "..", "example", "WaveRunup3d.jl")
    source = replace(read(path, String),
        "joinpath(@__DIR__, \"..\", \"output\", \"wave_runup_3d\")" => repr(directory),
        "VisualizeInParaview = true" => "VisualizeInParaview = false",
        "OpenLogFile         = true" => "OpenLogFile         = false")
    Base.include_string(scope, source, path)
    return Base.invokelatest(getproperty, scope, :captured)[]
end

@testset "Case 17 wave run-up" begin
    mktempdir() do directory
        inputs = prepare_example(directory)
        (; SimGeometry, SimParticles, SimConstants, SimKernel, SimMotion, SimMeasurements, SimMetaData) = inputs
        tank, slope, blocks, piston, water = SimGeometry

        @test length(SimGeometry) == 5
        @test all(g -> isempty(g.CSVFile) && !isempty(g.Particles), SimGeometry)
        @test getproperty.(SimGeometry, :Type) == [Fixed, Fixed, Fixed, Moving, Fluid]
        @test length(unique(SimParticles.Position)) == length(SimParticles)
        @test length(unique(SimParticles.ID)) == length(SimParticles)
        @test all(isfinite, SimParticles.Density)
        @test collect(extrema(last, water.Particles.Position)) ≈ [0.008, 0.24]
        @test SimConstants.c₀ ≈ 20sqrt(9.81 * 0.232)
        @test SimConstants.m₀ ≈ 1000 * 0.008^3
        @test SimKernel.h ≈ 1.5sqrt(3) * 0.008
        @test inputs.SimViscosity.boundary_factor == 0
        @test inputs.SimDensityDiffusion isa FourtakasDensityDiffusion
        @test inputs.SimTimeStepping isa SymplecticTimeStepping
        @test SimMetaData.GPUDoublePosition
        @test !SimMetaData.GPURadialInteractions
        @test SimMetaData.GPUCellSubdivision == 1
        @test SimMetaData.SimulationTime == 20
        @test SimMetaData.OutputTimes == 0.02

        periodicity = SimMetaData.Periodicity
        @test periodicity.axis == 2
        @test periodicity.lower ≈ -0.004
        @test periodicity.upper ≈ 0.364
        @test length(unique(p[2] for p in water.Particles.Position)) == 46
        @test all(p -> periodicity.lower <= p[2] < periodicity.upper, SimParticles.Position)
        @test maximum(first, piston.Particles.Position) ≈ 0
        @test minimum(last, piston.Particles.Position) ≈ 0.008
        @test minimum(last, tank.Particles.Position) ≈ -0.04

        # Known vertices after the XML's rotation and translation. Export must
        # retain the 8 slope facets and all 98 independently oriented blocks.
        h5open(joinpath(directory, "WaveRunup3D_Geometry.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            regions = read(root["CellData/Region"])
            @test count(==(2), regions) == 8
            @test count(==(3), regions) == 98 * 12
            points = read(root["Points"])
            @test any(p -> isapprox(p[1], 5.953258244367316) && isapprox(p[3], 1e-6), eachcol(points))
            @test all(isfinite, points)
            water_cell = findfirst(==(5), regions)
            offsets = read(root["Polygons/Offsets"])
            connectivity = read(root["Polygons/Connectivity"])
            outline = Int.(connectivity[(offsets[water_cell] + 1):offsets[water_cell + 1]]) .+ 1
            @test maximum(points[1, outline]) < 6.5
            @test maximum(points[3, outline]) ≈ 0.248
        end

        trajectory = SimMotion.translation
        @test length(trajectory.times) == 601
        @test all(>(0), diff(trajectory.times))
        @test trajectory.times[end] == 15
        @test !trajectory.cubic
        for (time, displacement) in ((0.0, 0.0), (0.125, 0.0026), (0.2, 0.0052), (15.0, 0.0), (20.0, 0.0))
            value, velocity, _ = trajectory_state(trajectory, time)
            @test value[1] ≈ displacement
            @test iszero(value[2]) && iszero(value[3])
            time >= 15 && @test iszero(velocity)
        end
        @test first(trajectory_state(trajectory, 0.1125))[1] ≈ 0.0013
        @test [probe.location for probe in SimMeasurements.water_column_probes] ==
            [(x, 0.18, 0.0) for x in (3.10, 3.20, 3.34, 3.63)]
        @test SimMeasurements.free_surface !== nothing

        if CUDA.functional()
            # Exercise the actual 8 mm case, prescribed periodic piston, graph
            # replay, measurements and geometry/grid output together.
            SimMetaData.SimulationTime = 0.15
            SimMetaData.OutputTimes = 0.15
            logger = SimulationLogger(directory)
            SPHExampleGPU.RunSimulation(; merge(inputs, (; SimLogger = logger))...)
            @test SimMetaData.TotalTime >= 0.15
            @test all(point -> all(isfinite, point), SimParticles.Position)
            @test all(isfinite, SimParticles.Density)
            @test all(ρ -> 700 < ρ < 1300, SimParticles.Density)
            @test all(velocity -> all(isfinite, velocity), SimParticles.Velocity)
            moving = SimParticles.Position[SimParticles.Type .== Moving]
            @test maximum(first, moving) ≈ 0.0026 atol = 1e-7
            h5open(joinpath(directory, "WaveRunup3D.vtkhdf"), "r") do file
                root = file["VTKHDF"]
                @test haskey(root, "WaterColumnProbes")
                @test haskey(root, "FreeSurface")
                heights = read(root["WaterColumnProbes/PointData/WaterColumnHeight"])
                @test all(isfinite, heights)
                @test all(height -> 0.20 < height < 0.28, heights)
                @test length(read(root["WaterColumnProbes/Steps/Values"])) == 2
            end
        end
    end
end

end
