module RadialBenchmarkTests

using Test, HDF5

include(joinpath(@__DIR__, "..", "benchmark", "radial_interactions.jl"))

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
    @test check_radial_accuracy((; position_dx = 0.05), 0.1) === nothing
    @test check_radial_accuracy((; position_dx = 0.1), 0.1) === nothing
    @test_throws r"maximum position difference.*exceeds" check_radial_accuracy((; position_dx = 0.11), 0.1)
    @test_throws r"maximum position difference.*exceeds" check_radial_accuracy((; position_dx = 215.85), 0.1)
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
