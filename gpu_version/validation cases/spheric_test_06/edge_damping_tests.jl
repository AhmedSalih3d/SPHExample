module Test06EdgeDampingTests
using Test, SPHExampleGPU, StaticArrays, CUDA, HDF5, CSV
include(joinpath(@__DIR__,"edge_damping.jl"))
const E = Test06EdgeDamping
const C = E.C
const DZ = SPHExampleGPU.DampingZones

@testset "four-edge damping profile and square clearance" begin
    zone = E.edge_zone(rate = 10.)
    points = [SVector(0.25,2.5),SVector(9.75,2.5),SVector(5.,0.25),SVector(5.,4.75)]
    @test all(p -> DZ.normalized_damping(zone,p) == 0.25,points)
    @test DZ.normalized_damping(zone,SVector(0.25,0.25)) == 0.25
    @test DZ.normalized_damping(zone,SVector(0.,0.)) == 1.
    @test DZ.normalized_damping(zone,SVector(5.,2.5)) == 0.
    @test DZ.normalized_damping(zone,SVector(0.5,2.5)) == 0.
    velocity = SVector(2.,-3.)
    @test all(p -> DZ.damped_velocity((zone,),p,velocity,Fluid,0.01) ≈ 0.975velocity,points)
    for kind in (Fixed,Moving,Floating,Frozen)
        @test DZ.damped_velocity((zone,),first(points),velocity,kind,0.01) == velocity
    end
    trajectory = C.body_trajectory()
    corners = [SVector(1.5,2.5)+first(trajectory_state(trajectory,t))+SVector(x,y)
        for t in 0:0.01:8 for x in (-0.5,0.5) for y in (-0.5,0.5)]
    @test all(p -> DZ.normalized_damping(zone,p) == 0.,corners)
    clearance = minimum(p -> min(p[1]-0.5,9.5-p[1],p[2]-0.5,4.5-p[2]),corners)
    @test 0.026 < clearance < 0.027
    # This 3D box represents the Dual x/z configuration, with inactive y sides.
    dual_box = DampingBox(((0.5,-1.,0.5),(9.5,1.,4.5)),((0.,-1.,0.),(10.,1.,5.));
        directions = (:left,:right,:bottom,:top),rate = 10.,overlimit = 0.1,factors = (1.,0.,1.))
    @test all(p -> DZ.normalized_damping(zone,p) == DZ.normalized_damping(dual_box,SVector(p[1],0.,p[2])),
        (points...,SVector(0.25,0.25),SVector(5.,2.5)))
    @test_throws ArgumentError E.edge_zone(width = 1.)
    @test_throws ArgumentError E.edge_zone(rate = -1.)
end

if CUDA.functional()
    @testset "edge damping kernel and composite history analysis" begin
        for T in (Float32,Float64)
            zone = E.edge_zone(rate = 10.)
            positions = CuArray([SVector(0.25,2.5),SVector(9.75,2.5),SVector(5.,0.25),SVector(5.,4.75),SVector(5.,2.5)])
            velocity = SVector{2,T}(2,-3)
            velocities = CuArray(fill(velocity,5))
            types = CuArray([Fluid,Fluid,Fluid,Moving,Fluid])
            DZ.launch_damping!((zone,),positions,velocities,types,HostStep(T(0.01),zero(T)))
            actual = Array(velocities)
            @test all(v -> v ≈ T(0.975)*velocity,actual[1:3])
            @test actual[4] == actual[5] == velocity
        end
        mktempdir() do directory
            configuration = merge(C.config("smoke"),(; n = 10,duration = 0.2,interval = 0.02))
            result = C.native_run(configuration,joinpath(directory,"run"); results = directory,damping = E.edge_zone(rate = 100.))
            @test result.frames == 11
            @test result.particles == 6264
            h5open(joinpath(directory,"run","MovingSquare.vtkhdf"),"r") do file
                @test haskey(file,"VTKHDF/Particles")
                @test haskey(file,"VTKHDF/DampingZones")
                @test length(read(file["VTKHDF/Particles/Steps/Values"])) == 11
            end
            @test length(CSV.File(joinpath(directory,"native.csv"))) == 11
        end
    end
end
end
