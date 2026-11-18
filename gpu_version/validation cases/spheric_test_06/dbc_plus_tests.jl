using Test, SPHExampleGPU, StaticArrays, LinearAlgebra
using HDF5
include(joinpath(@__DIR__, "dbc_plus.jl"))
const M = DBCPlusStudy.M

@testset "DBC+ analytical pressure traction" begin
    for mode in (DBCPlus, DBCPlusImpedance)
        uniform = M.build_submerged_square_fixture(g = 0.0, c0 = 160.0, boundary_mode = mode)
        fill!(uniform.density, 1000.1)
        load = M.mdbc_frame_loads(uniform.position, uniform.velocity, uniform.density,
            uniform.types, uniform.markers, uniform.kernel, uniform.constants;
            boundary_mode = mode(), center = uniform.center)
        @test norm(load.force_pressure) < 1e-7
        for dx in (0.05, 0.025)
            accelerated = M.build_submerged_square_fixture(; dx, g = 0.0, ax = 1.0,
                boundary_mode = mode)
            load = M.mdbc_frame_loads(accelerated.position, accelerated.velocity, accelerated.density,
                accelerated.types, accelerated.markers, accelerated.kernel, accelerated.constants;
                boundary_mode = mode(), center = accelerated.center,
                wall_acceleration = SVector(1.0, 0.0))
            @test load.force_pressure[1] ≈ 1000 rtol = 0.02
            @test abs(load.force_pressure[2]) < 1e-7
        end
    end
end

@testset "DBC+ graph replay and driver stages" begin
    mktempdir() do directory
        results = map((false, true)) do impedance
            a = DBCPlusStudy.run_case(dx = 0.1, duration = 0.04, graph = false, output_root = directory; impedance)
            b = DBCPlusStudy.run_case(dx = 0.1, duration = 0.04, graph = true, output_root = directory; impedance)
            @test a.result.performance.steps == b.result.performance.steps
            @test a.analysis.rows == b.analysis.rows
            @test b.result.final_state.complete
            @test maximum(r.motion_error for r in b.analysis.checks) < 1e-7
            path = joinpath(directory, "runs", "dbc_plus", basename(b.directory), "MovingSquare.vtkhdf")
            h5open(path, "r") do file
                root = haskey(file, "VTKHDF/Particles") ? file["VTKHDF/Particles"] : file["VTKHDF"]
                density = read(root["PointData/Density"])
                pressure = read(root["PointData/Pressure"])
                types = read(root["PointData/Type"])
                walls = types .!= Int(Fluid)
                @test pressure[walls] ≈ 160f0^2 .* (density[walls] .- 1000f0) atol = .01
            end
            b.analysis.rows
        end
        @test results[1] != results[2]
    end
end
