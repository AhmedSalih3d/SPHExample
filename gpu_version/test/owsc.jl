module OWSCTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra,CSV
include(joinpath(@__DIR__,"..","benchmark","chrono_case.jl"))

@testset "OWSC reference geometry, inertia and ramped waves" begin
    case = prepare_chrono_case("OWSC3d.jl")
    @test length.(getproperty.(case.geometry,:Particles)) == [25926,2294,2436,142732]
    @test only(filter(g -> g.Type == Floating,case.geometry)).Floating.Center == (0.,0.,0.33)
    @test only(case.dynamics.joints).stiffness == 15
    @test only(case.dynamics.joints).damping == 100
    @test case.meta.Periodicity === nothing
    scope = Module(gensym(:WaveTest))
    Base.include(scope,joinpath(@__DIR__,"..","example","data","regular_piston.jl"))
    wave = Base.invokelatest(scope.regular_piston_state,0.15,2.,1.3;ramp_periods = 1)
    @test Base.invokelatest(wave,0.) == (0.,0.)
    # Printed displacement samples from the installed reference paddle CSV.
    for (time,target) in ((0.1,0.000235),(0.5,0.014405),(0.7,0.019607),(1.5,-0.052244),(2.1,0.016799))
        @test Base.invokelatest(wave,time)[1] ≈ target atol = 5e-7
    end
    for time in (0.2,1.,2.,2.5)
        epsilon = 1e-5
        derivative = (Base.invokelatest(wave,time+epsilon)[1]-Base.invokelatest(wave,time-epsilon)[1])/(2epsilon)
        @test Base.invokelatest(wave,time)[2] ≈ derivative atol = 1e-9
    end
end

if CUDA.functional()
    @testset "custom floating mass centre and inertia" begin
        F = SPHExampleGPU.GPUFloating
        for D in (2,3)
            points = D == 2 ? [SVector(0.05i,0.05j) for i in (-1,1) for j in (-1,1)] :
                [SVector(0.05i,0.05j,0.05k) for i in (-1,1) for j in (-1,1) for k in (-1,1)]
            center = D == 2 ? (0.02,0.03) : (0.02,0.03,0.04)
            inertia = D == 2 ? 1.5 : (1.5,2.,3.)
            function allocate(properties)
                geometry = [SPHGeometry{D,Float64}(points;Density = 1000,GroupMarker = 1,Type = Floating,Floating = properties)]
                constants = SimulationConstants{Float64}(D;dx = 0.1)
                meta = SimulationMetaData{D,Float64}(SimulationName = "MassProperties",SaveLocation = tempdir(),
                    SimulationTime = 0.01,OutputTimes = 0.01,VisualizeInParaview = false,OpenLogFile = false)
                return F.FloatingArrays(geometry,AllocateDataStructures(geometry,meta),constants)
            end
            f = allocate(FloatingDetails{Float64}(RelativeWeight = 1,Center = center,Inertia = inertia))
            @test only(Array(f.center)) == SVector{D,Float64}(center)
            @test D == 2 ? only(Array(f.inertia)) == inertia : diag(only(Array(f.inertia))) == SVector(inertia)
            @test_throws ArgumentError allocate(FloatingDetails{Float64}(RelativeWeight = 1,Center = (NaN,0)))
            @test_throws ArgumentError allocate(FloatingDetails{Float64}(RelativeWeight = 1,Inertia = D == 2 ? -1 : (1,1,-1)))
            @test_throws ArgumentError allocate(FloatingDetails{Float64}(RelativeWeight = 1,Inertia = D == 2 ? (1,2) : 2))
        end
    end
    @testset "OWSC fluid and prescribed piston couple to native hinge" begin
        mktempdir() do directory
            result,case = redirect_stdout(devnull) do
                run_chrono_case("OWSC3d.jl";simulation_time = 0.04,save_location = directory)
            end
            @test result.final_state.complete
            @test result.diagnostics.frozen_particles == 0
            @test maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,result.final_state.floating)) < 1e-6
            @test all(isfinite,only(result.final_state.floating.orientation))
            @test norm(only(result.final_state.floating.load_force)) > 0
            @test Base.invokelatest(case.save_surfaces,directory,case.geometry) isa String
        end
    end
end
end
