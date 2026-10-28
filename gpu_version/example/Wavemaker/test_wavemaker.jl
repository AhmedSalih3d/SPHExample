module WavemakerTests
using Test, SPHExampleGPU, StaticArrays, CUDA, LinearAlgebra
include(joinpath(@__DIR__, "Wavemaker.jl"))
const GK = SPHExampleGPU.GPUKernels
const CL = SPHExampleGPU.SPHCellList

@testset "matched GenCase geometry" begin
    case = wavemaker_geometry()
    @test length.(values(case.positions)) == (31600,30000,401600)
    points = vcat(values(case.positions)...)
    @test length(unique(points)) == 463200
    @test extrema(x -> x[2],points) == (0.01,1.99)
    @test case.periodicity.lower == 0
    @test case.periodicity.upper == 2
    @test extrema(last,case.positions.water) == (0.03,0.73)
    @test minimum(p -> WavemakerGeometry.bed_clearance(p[1],p[3]),case.positions.water) > 0.01
    @test_throws ArgumentError wavemaker_geometry(; dx=0.05)
end

@testset "reference sinusoidal motion" begin
    motion = wavemaker_motion()
    phase = 4.71238898
    for t in (0.,0.25,0.5,1.,2.,4.75)
        @test motion_position(motion,(0,0,0),t) ≈ SVector(0.25*(sin(pi*t+phase)-sin(phase)),0,0) atol=2e-15
        @test motion_velocity(motion,(0,0,0),t) ≈ SVector(0.25pi*cos(pi*t+phase),0,0) atol=2e-15
    end
    @test iszero(motion_velocity(motion,(0,0,0),100.1))
end

@testset "reference adaptive timestep" begin
    constants = SimulationConstants{Float32}(3; dx=0.02,c₀=52.4099227246139)
    kernel = SPHKernelInstance{3,Float32}(WendlandC2();h=Float32(1.5sqrt(3)*0.02))
    scheme = DualSPHysicsSymplectic(; smoothing_length=1.5sqrt(3)*0.02,sound_speed=52.4099227246139)
    base = CL.dual_time_step(SVector(16f0,9f0,100f0,0f0),kernel,constants,scheme)
    expected = Float64(constants.CFL)*scheme.smoothing_length/(scheme.sound_speed+100*scheme.smoothing_length)
    @test base ≈ expected
    reference_kernel = SPHKernelInstance{3,Float32}(WendlandC2();h=kernel.h,η²=Float32((0.1*scheme.smoothing_length)^2))
    @test reference_kernel.η² ≈ 100*kernel.η²
    @test CL.dual_time_step(SVector(16f0,100f0,100f0,0f0),kernel,constants,scheme) < base
    @test_throws ErrorException CL.dual_time_step(SVector(NaN32,0f0,0f0,0f0),kernel,constants,scheme)
end

if CUDA.functional()
    @testset "reference corrector and gravity direction" begin
        T=Float32
        constants=SimulationConstants{T}(3;dx=0.02,c₀=20,g=0)
        x=CuArray([SVector(0.2,0.4,0.6),SVector(0.,0.4,0.01)])
        v=CuArray([SVector(1f0,2f0,3f0),zero(SVector{3,T})])
        rho=CuArray(T[1001,1000]); types=CuArray([Fluid,Fixed])
        xp=similar(x); vp=similar(v); rp=similar(rho)
        acceleration=CUDA.zeros(SVector{3,T},2); rate=CUDA.zeros(T,2)
        GK.launch_dual_predictor!(xp,vp,rp,x,v,rho,acceleration,rate,types,0.1,constants)
        copyto!(acceleration,[SVector(2f0,0f0,0f0),zero(SVector{3,T})])
        GK.launch_dual_corrector!(x,v,rho,xp,rp,acceleration,rate,types,0.1,constants)
        @test Array(x)[1] ≈ SVector(0.31,0.6,0.9) atol=5e-9
        @test Array(v)[1] == SVector(1.2f0,2f0,3f0)
        @test Array(x)[2] == SVector(0.,0.4,0.01)
        @test Array(rho)[2] == 1000
        support=GK.DualStepSupport{T}(2)
        criteria=GK.launch_dual_criteria!(support,v,acceleration,types)
        @test criteria[1] == 4f0
        gravity=SimulationConstants{T}(3;dx=0.02,c₀=20,g=9.81)
        fill!(x,zero(SVector{3,Float64})); fill!(v,zero(SVector{3,T})); fill!(acceleration,zero(SVector{3,T}))
        GK.launch_dual_predictor!(xp,vp,rp,x,v,rho,acceleration,rate,types,0.1,gravity)
        @test Array(vp)[1][3] ≈ -0.4905f0
        GK.launch_dual_corrector!(x,v,rho,xp,rp,acceleration,rate,types,0.1,gravity)
        @test Array(x)[1][3] ≈ -0.04905 atol=5e-9
    end
    @testset "reference predictor exclusions and event accounting" begin
        T=Float32
        constants=SimulationConstants{T}(3;dx=0.02,c₀=20,g=0)
        x=CuArray([SVector(0.2,0.4,0.01),SVector(0.2,0.4,0.2),SVector(0.,0.4,0.01)])
        v=CuArray([SVector(0f0,0f0,-1f0),zero(SVector{3,T}),zero(SVector{3,T})])
        rho=CUDA.fill(1000f0,3); types=CuArray([Fluid,Fluid,Fixed])
        acceleration=CUDA.zeros(SVector{3,T},3); rate=CuArray(T[0,-8000,0])
        xp=similar(x);vp=similar(v);rp=similar(rho);support=GK.DualStepSupport{T}(3)
        scheme=DualSPHysicsSymplectic(;domain_min=(-1.,-1.,0.),domain_max=(1.,1.,1.),density_min=700.,density_max=1300.)
        GK.launch_dual_predictor!(xp,vp,rp,x,v,rho,acceleration,rate,types,0.1,constants;
            exclusions=support.exclusions,scheme)
        @test Array(types) == [Frozen,Frozen,Fixed]
        @test Array(support.exclusions) == [1,2,0]
        @test Array(x)[1][3] < 0
        counts=GK.launch_dual_motion_finish!(support,x,v,rho,xp,types,0.1,nothing,scheme)
        @test counts[1:2] == [1,1]
        @test all(iszero,Array(support.exclusions))
        @test GK.launch_dual_motion_finish!(support,x,v,rho,xp,types,0.1,nothing,scheme)[1:2] == [0,0]
        copyto!(x,[SVector(0.2,0.01,0.2),SVector(0.2,0.4,0.2),SVector(0.,0.4,0.01)])
        copyto!(v,[SVector(0f0,-1f0,0f0),zero(SVector{3,T}),zero(SVector{3,T})])
        copyto!(types,[Fluid,Fluid,Fixed]);fill!(rho,1000f0);fill!(rate,0f0)
        periodic=PeriodicBoundary{3}(2;min=0,max=2)
        scheme=DualSPHysicsSymplectic(;domain_min=(-1.,0.,0.),domain_max=(1.,2.,1.),density_min=700.,density_max=1300.)
        GK.launch_dual_predictor!(xp,vp,rp,x,v,rho,acceleration,rate,types,0.1,constants;
            exclusions=support.exclusions,periodicity=periodic,scheme)
        @test Array(xp)[1][2] < 0
        @test Array(types)[1] == Fluid
        @test all(iszero,Array(support.exclusions))
    end
    @testset "matched DBC short run" begin
        mktempdir() do dir
            result=run_wavemaker(; simulation_time=0.003,output_interval=0.003,
                visualize=false,open_log_file=false,save_location=dir)
            @test result.final_state.complete
            @test result.configuration.effective.boundary_mode === NoMDBC
            @test result.configuration.effective.pair_coordinates === :periodic_cell_relative
            @test all(isfinite,result.final_state.particles.Density)
            @test result.diagnostics.frozen_particles == 0
        end
    end
end
end
include(joinpath(@__DIR__,"test_periodic_bounds.jl"))
