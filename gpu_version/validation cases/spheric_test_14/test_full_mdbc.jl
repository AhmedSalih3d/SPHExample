module FullMDBCGeometryTests
using Test, SPHExampleGPU, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "run.jl"))

@testset "full 3D mDBC surfaces, experimental mass and free body" begin
    for release in RELEASES
        dbc = build_full_case(release; dx = 0.05, duration = 0.01, boundary_mode = NoMDBC)
        mdbc = build_full_case(release; dx = 0.05, duration = 0.01)
        @test all(a.Particles.Position == b.Particles.Position for (a,b) in zip(dbc.geometry, mdbc.geometry))
        @test all(a.Particles.Density == b.Particles.Density for (a,b) in zip(dbc.geometry, mdbc.geometry))
        @test mdbc.geometry[3].Floating.Constraints === nothing
        @test mdbc.geometry[3].Floating == dbc.geometry[3].Floating
        @test mdbc.geometry[3].Floating.RelativeWeight * mdbc.constants.m₀ * length(mdbc.geometry[3].Particles) ≈ release.mass rtol=2e-7
        @test mdbc.metadata isa SimulationMetaData{3,Float32,NoShifting,NoKernelOutput,UpdatedMDBC}
        @test mdbc.metadata.OutputTimes == 0.005
        @test mdbc.viscosity isa Laminar
        @test mdbc.diffusion isa LinearDensityDiffusion
        base = release.offset-release.draft
        for (p,g,n) in zip(mdbc.geometry[3].Particles.Position, mdbc.geometry[3].Particles.GhostPoints, mdbc.geometry[3].Particles.GhostNormals)
            q = (p+g)/2
            @test q ≈ float_surface_projection(p, release, base) atol=1e-12
            @test g-p ≈ n atol=1e-12
            @test !in_float(g, release, base)
        end
        for g in mdbc.geometry[1].Particles.GhostPoints
            @test abs(g[1]) <= mdbc.half_width && abs(g[2]) <= mdbc.half_width && g[3] >= -mdbc.depth
        end
        @test float_surface_projection(SVector(0.,0.,base+0.001), release, base) ≈ SVector(0.,0.,base)
        @test float_surface_projection(SVector(0.,0.,base+release.height-0.001), release, base) ≈ SVector(0.,0.,base+release.height)
        @test float_surface_projection(SVector(release.radius-0.001,0.,base+release.height-0.01), release, base) ≈
            SVector(release.radius,0.,base+release.height-0.01)
    end
    @test build_full_case(first(RELEASES); dx = 0.05, boundary_mode = UpdatedMDBC).metadata isa
        SimulationMetaData{3,Float32,NoShifting,NoKernelOutput,UpdatedMDBC}
    @test_throws ArgumentError build_full_case(first(RELEASES); sponge_width = 1.1)
    @test setup(first(RELEASES); dx = 0.05, duration = 0.01, save = "unused").boundary_mode == "UpdatedMDBC"
end
end
