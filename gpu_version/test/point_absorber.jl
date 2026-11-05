module PointAbsorberTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra,CSV,HDF5
include(joinpath(@__DIR__,"..","benchmark","chrono_case.jl"))
const F = SPHExampleGPU.GPUFloating
const S = SPHExampleGPU.GPUStepState

@testset "Coulomb damper law and validation" begin
    for D in (2,3),T in (Float32,Float64)
        vector(x) = SVector{D,T}(ntuple(d -> d == D ? T(x) : zero(T),D))
        @test axial_coulomb_force(vector(1),vector(0.2),T(10)) == vector(-10)
        @test axial_coulomb_force(vector(1),vector(-0.2),T(10),T(2)) == vector(2)
        @test axial_coulomb_force(vector(1),vector(0),T(10)) == vector(0)
        @test axial_coulomb_force(vector(0),vector(0.2),T(10)) == vector(0)
        @test dot(axial_coulomb_force(vector(1),vector(0.2),T(10)),vector(0.2)) < 0
        @test_throws ArgumentError CoulombDamper{D,T}(Anchor = vector(0),Attachment = vector(1),Force = -1)
    end
end

@testset "PointAbsorber reference variants" begin
    for variant in (:direct,:point_line,:spring,:coulomb)
        case = prepare_chrono_case("PointAbsorber2d.jl";variant)
        @test length.(getproperty.(case.geometry,:Particles)) == [1081,360,527,17929]
        @test case.meta.Periodicity === nothing
        body = only(filter(g -> g.Type == Floating,case.geometry))
        @test body.Floating.RelativeWeight == 0.5
        @test length(body.Floating.Springs) == (variant in (:spring,:coulomb) ? 1 : 0)
        @test variant == :point_line ? only(case.dynamics.joints).kind == 3 : body.Floating.Constraints.rotation_free == SVector(false)
    end
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "Coulomb GPU stopping impulse D=$D T=$T" for D in (2,3),T in (Float32,Float64)
        vector(x) = SVector{D,T}(ntuple(d -> d == D ? T(x) : zero(T),D))
        points = [SVector{D,T}(ntuple(d -> T(0.01*((i >> (d-1)) & 1)-0.005),D)) for i in 0:2^D-1]
        damper = CoulombDamper{D,T}(Anchor = vector(-1),Attachment = vector(0),Force = 10)
        constants = SimulationConstants{T}(D;dx = 0.01,m₀ = T(0.4/length(points)),g = 0,c₀ = 20)
        geometry = [SPHGeometry{D,T}(points;Density = 1000,GroupMarker = 1,Type = Floating,
            Floating = FloatingDetails{T}(RelativeWeight = 1,Springs = (damper,),InitialVelocity = vector(0.01)))]
        host = AllocateDataStructures(geometry)
        f = F.FloatingArrays(geometry,host,constants)
        F.launch_floating_springs!(f,S.HostStep(T(0.1),0.),false)
        @test Array(f.force)[1:D] ≈ vector(-0.08) rtol = 1e-5
        @test all(iszero,Array(f.force)[D+1:end])
    end

    @testset "Coulomb damping stops without reversal or residual creep" begin
        for scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping()),graph in (false,true)
            points = [SVector(0.005i,0.005j) for i in (-1,1) for j in (-1,1)]
            geometry = [SPHGeometry{2,Float64}(points;Density = 1000,GroupMarker = 1,Type = Floating,
                Floating = FloatingDetails{Float64}(RelativeWeight = 1,InitialVelocity = (0.,0.1),
                    Springs = (CoulombDamper{2,Float64}(Anchor = (0,-1),Attachment = (0,0),Force = 10),)))]
            mktempdir() do directory
                meta = SimulationMetaData{2,Float64}(SimulationName = "CoulombStop",SaveLocation = directory,
                    SimulationTime = 0.01,OutputTimes = 0.001,VisualizeInParaview = false,OpenLogFile = false,
                    GPUUseGraph = graph,GPUDoublePosition = true,GPUClipTimeStepToOutput = true)
                logger = SimulationLogger(directory)
                result = try
                    redirect_stdout(devnull) do
                        RunSimulation(SimGeometry = geometry,SimMetaData = meta,SimConstants = SimulationConstants{Float64}(2;dx = 0.01,g = 0,c₀ = 20),
                            SimParticles = AllocateDataStructures(geometry,meta),SimLogger = logger,
                            SimKernel = SPHKernelInstance{2,Float64}(WendlandC2();h = 0.02),
                            SimViscosity = ZeroViscosity(),SimDensityDiffusion = ZeroDensityDiffusion(),SimTimeStepping = scheme)
                    end
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                rows = collect(CSV.File(joinpath(directory,"CoulombStop_Floating.csv")))
                speed = [r[Symbol("Velocity:1")] for r in rows]
                @test minimum(speed) >= -1e-12
                @test maximum(diff(speed)) <= 1e-12
                @test abs(last(speed)) < 1e-8
                @test only(result.final_state.floating.center)[2] ≈ 0.0002 atol = 1e-5
            end
        end
    end

    @testset "PointAbsorber pure heave and PTO output" begin
        for variant in (:direct,:point_line,:spring,:coulomb)
            mktempdir() do directory
                result,case = redirect_stdout(devnull) do
                    run_chrono_case("PointAbsorber2d.jl";variant,simulation_time = 0.1,save_location = directory)
                end
                state = result.final_state.floating
                @test result.final_state.complete
                @test result.diagnostics.frozen_particles == 0
                @test only(state.center)[1] ≈ 2 atol = 1e-8
                @test abs(only(state.angle)) < 1e-8
                @test abs(only(state.omega)) < 1e-8
                @test all(isfinite,only(state.velocity))
                file = Base.invokelatest(case.save_surfaces,directory,case.geometry)
                h5open(file,"r") do io
                    @test haskey(io,"VTKHDF/Springs") == (variant in (:spring,:coulomb))
                    @test haskey(io,"VTKHDF/RigidSurfaces")
                end
            end
        end
    end
end
end
