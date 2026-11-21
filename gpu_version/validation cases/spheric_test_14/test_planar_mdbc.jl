module PlanarMDBCGeometryTests
using Test,SPHExampleGPU,StaticArrays,LinearAlgebra
include(joinpath(@__DIR__,"case.jl"))
include(joinpath(@__DIR__,"planar_case.jl"))

@testset "matched planar DBC/mDBC geometry and exact reflected surfaces" begin
    for release in RELEASES
        dbc = build_planar_case(release;dx=0.025,duration=0.01)
        mdbc = build_planar_case(release;dx=0.025,duration=0.01,boundary_mode=SimpleMDBC)
        @test dbc.line_mass == mdbc.line_mass
        @test dbc.geometry[3].Floating == mdbc.geometry[3].Floating
        @test all(a.Particles.Position == b.Particles.Position for (a,b) in zip(dbc.geometry,mdbc.geometry))
        base = release.offset-release.draft
        for (p,g) in zip(mdbc.geometry[3].Particles.Position,mdbc.body_ghosts)
            q = (p+g)/2
            @test norm(q-float_section_projection(p,release,base)) < 1e-12
            @test !in_float(SVector(g[1],0.,g[2]),release,base)
        end
        for (p,g) in zip(mdbc.geometry[1].Particles.Position,mdbc.wall_ghosts)
            @test -mdbc.half_width <= g[1] <= mdbc.half_width
            @test g[2] >= -mdbc.depth
        end
    end
    for mode in (UpdatedMDBC, UpdatedMDBCWithImpedance, UpdatedMDBCNoPenetration)
        case = build_planar_case(first(RELEASES); dx=0.025, boundary_mode=mode, pause_time=0.06)
        @test case.metadata isa SimulationMetaData{2,Float32,NoShifting,NoKernelOutput,mode}
        @test case.geometry[3].Floating.PauseTime <= case.release_time == 0.06
    end
    @test_throws ArgumentError build_planar_case(first(RELEASES); boundary_mode=DBCPlus)
end
end
