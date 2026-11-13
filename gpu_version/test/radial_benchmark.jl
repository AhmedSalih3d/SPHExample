module RadialBenchmarkTests

using Test, HDF5

include(joinpath(@__DIR__, "..", "benchmark", "radial_interactions.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "operator_fusion.jl"))

@testset "radial pressure-viscosity fusion and benchmark guards" begin
    fusion = OperatorFusionBenchmark
    @test_throws ArgumentError fusion.main(["--rounds=1"])
    @test_throws ArgumentError fusion.main(["--duration=NaN"])
    @test_throws ArgumentError fusion.main(["--max-position-dx=-1"])
    @test_throws ArgumentError fusion.main(["--candidate=unknown"])
    @test_throws ArgumentError fusion.main(["--unknown"])
    @test fusion.check_accuracy((; position_dx = 0.1), 0.1) === nothing
    @test_throws r"Position drift" fusion.check_accuracy((; position_dx = 0.11), 0.1)
    @test_throws r"Position drift" fusion.check_accuracy((; position_dx = NaN), 0.1)
    @test_throws r"source anchor" fusion.unfused_source("")
    @test_throws r"source anchor" fusion.unit_wall_source("")
    @test_throws r"source anchor" fusion.resources_source("")
    fields = ([SVector(0.0, 0.0)], [SVector(0.0, 0.0)], [1000.0], [SVector(0.0, 0.0)])
    fixture = (; work = (; steps = 2, rebuilds = 1, readbacks = 2), ids = [1], types = [Fluid],
        state = fields, time = 0.1,
        output = (; counts = [1], times = [0.1], datasets = Dict("Points" => (Float64, (2, 1), "hash"))))
    @test fusion.differences(fixture, fixture, 0.02).exact_output
    @test_throws r"counts changed" fusion.differences(fixture,
        merge(fixture, (; work = (; steps = 3, rebuilds = 1, readbacks = 2))), 0.02)
    @test_throws r"identities" fusion.differences(fixture, merge(fixture, (; ids = [2])), 0.02)
    @test_throws r"shape/type" fusion.differences(fixture,
        merge(fixture, (; output = merge(fixture.output, (; datasets = Dict("Points" => (Float64, (3, 1), "hash")))))), 0.02)
    for T in (Float32, Float64), D in (2, 3)
        constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(35), m₀ = T(1000 * 0.02^D))
        kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx)
        for scenario in ((0, -1, 1, 1), (0.2, 1, 1, 1), (1, -1, 1, 1), (1, 0, 1, 1),
                         (1, -1, 0, 1), (1, -1, 0.5, 2), (2, -1, 2, 0.5))
            q, velocity_scale, wall, mass_ratio = T.(scenario)
            x = normalize(SVector{D,T}(ntuple(identity, D))) * (q * kernel.h)
            gradient = ∇Wᵢⱼ(kernel, q, x)
            inverse_distance = inv(dot(x, x) + kernel.η²)
            v = velocity_scale * x
            coefficient = SPHExampleGPU.SPHViscosityModels.radial_viscosity(ArtificialViscosity(),
                kernel, constants, x, v, one(T), T(995), T(1007), inverse_distance)
            @test coefficient isa T
            visc = SPHExampleGPU.SPHViscosityModels.radial_viscosity(ArtificialViscosity(),
                kernel, constants, x, v, gradient, T(995), T(1007), inverse_distance)
            for pressure in (T(-3), zero(T), T(3), -coefficient * wall * mass_ratio)
                expected = pressure * gradient + (visc * wall) * mass_ratio
                actual = (pressure + (coefficient * wall) * mass_ratio) * gradient
                scale = norm(pressure * gradient) + norm((visc * wall) * mass_ratio)
                @test norm(actual - expected) <= T(16) * eps(T) * scale
            end
        end
    end
end

function write_output_fixture(folder; reverse_order = false, position_error = 0.0, density_error = 0.0)
    name = "Fixture"
    points = [0.0 1.0 0.1 1.1; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    points[1, 3] += position_error
    density = [1000.0, 1001.0, 1000.0 + density_error, 1001.0]
    order = reverse_order ? [2, 1, 4, 3] : [1, 2, 3, 4]
    h5open(joinpath(folder, name * ".vtkhdf"), "w") do file
        root = create_group(file, "VTKHDF")
        root["Points"] = points[:, order]
        root["NumberOfPoints"] = [2, 2]
        steps = create_group(root, "Steps")
        steps["Values"] = [0.0, 0.1]
        steps["PointOffsets"] = [0, 2]
        fields = create_group(root, "PointData")
        fields["ID"] = [1, 2, 1, 2][order]
        fields["Type"] = fill(Int8(1), 4)
        fields["GroupMarker"] = fill(UInt(1), 4)
        fields["Density"] = density[order]
        fields["Velocity"] = zeros(3, 4)
    end
    h5open(joinpath(folder, name * "_GridCells.vtkhdf"), "w") do file
        root = create_group(file, "VTKHDF")
        root["Points"] = zeros(3, 4)
        root["NumberOfCells"] = [1]
        root["Connectivity"] = [0, 1, 2, 3]
        steps = create_group(root, "Steps")
        steps["Values"] = [0.1]
    end
    return name
end

@testset "radial benchmark validates input and accuracy thresholds" begin
    @test_throws ArgumentError benchmark_radial(rounds = 1)
    @test_throws ArgumentError benchmark_radial(duration = 0.0)
    @test_throws ArgumentError benchmark_radial(long_time = 0.01)
    @test_throws ArgumentError benchmark_radial(minimum_reduction = 1.0)
    @test_throws ArgumentError benchmark_radial(maximum_position_dx = -1.0)
    @test_throws ArgumentError main_radial(["--unknown"])
    @test_throws ArgumentError main_radial(["--time"])
    @test_throws ArgumentError main_radial(["--viscosity"])
    @test_throws ArgumentError main_radial(["--viscosity", "unknown"])
    @test check_radial_accuracy((; position_dx = 0.05), 0.1) === nothing
    @test check_radial_accuracy((; position_dx = 0.1), 0.1) === nothing
    @test_throws r"maximum position difference.*exceeds" check_radial_accuracy((; position_dx = 0.11), 0.1)
    @test_throws r"maximum position difference.*exceeds" check_radial_accuracy((; position_dx = 215.85), 0.1)
end

@testset "SPS benchmark overrides both compared variants consistently" begin
    original = only(select_cases(["DamBreak3D_dp0.0085"]))
    @test with_benchmark_viscosity(original, nothing) === original
    viscosity = LaminarSPS()
    case = with_benchmark_viscosity(original, viscosity)
    @test case.name == original.name * "_LaminarSPS"
    @test case.dims == original.dims
    mktempdir() do save
        expected = original.build(Float32, save)
        kwargs = case.build(Float32, save)
        @test kwargs.SimViscosity === viscosity
        @test kwargs.SimDensityDiffusion === expected.SimDensityDiffusion
        @test kwargs.SimConstants == expected.SimConstants
        @test kwargs.SimKernel == expected.SimKernel
        @test kwargs.SimTimeStepping === expected.SimTimeStepping
        @test [geom.CSVFile for geom in kwargs.SimGeometry] == [geom.CSVFile for geom in expected.SimGeometry]
    end
end

@testset "streamed radial output comparison reports differences" begin
    mktempdir() do folder
        original = joinpath(folder, "reference")
        candidate = joinpath(folder, "candidate")
        mkpath(original)
        mkpath(candidate)
        name = write_output_fixture(original)
        write_output_fixture(candidate; reverse_order = true)
        exact = streamed_output_differences(original, candidate, name)
        @test exact.exact_output
        @test exact.output_position == exact.output_density == exact.output_time == 0.0
        write_output_fixture(candidate; reverse_order = true, position_error = 0.02, density_error = 0.5)
        changed = streamed_output_differences(original, candidate, name)
        @test !changed.exact_output
        @test changed.output_position ≈ 0.02
        @test changed.output_density == 0.5
        @test changed.output_time == 0.0
    end
end

end
