using Test
using SPHExampleGPU
using CSV
using HDF5

include(joinpath(@__DIR__, "..", "example",
                 "GenerateStillWedgeMiddleSquareMDBC.jl"))

@testset "StillWedge middle-square particle geometry" begin
    input_dir = normpath(joinpath(@__DIR__, "..", "input",
                                  "still_wedge_middle_square_mdbc"))
    bound_rows = collect(CSV.File(joinpath(
        input_dir, "StillWedge_MiddleSquare_Dp0.02_Bound.csv")))
    fluid_rows = collect(CSV.File(joinpath(
        input_dir, "StillWedge_MiddleSquare_Dp0.02_Fluid.csv")))
    expected_x = vcat(
        Float64[row[Symbol("Points:0")] for row in bound_rows],
        Float64[row[Symbol("Points:0")] for row in fluid_rows],
    )
    expected_y = vcat(
        Float64[row[Symbol("Points:2")] for row in bound_rows],
        Float64[row[Symbol("Points:2")] for row in fluid_rows],
    )
    expected_density = vcat(
        Float64[row.Rhop for row in bound_rows],
        Float64[row.Rhop for row in fluid_rows],
    )
    expected_pressure = vcat(
        Float64[row.Press for row in bound_rows],
        Float64[row.Press for row in fluid_rows],
    )
    expected_types = vcat(fill(Int8(Fixed), length(bound_rows)),
                          fill(Int8(Fluid), length(fluid_rows)))
    expected_markers = vcat(fill(1, length(bound_rows)),
                            fill(2, length(fluid_rows)))

    mktempdir() do output_dir
        output_path = generate_still_wedge_middle_square_geometry(output_dir)
        @test isfile(output_path)
        @test basename(output_path) ==
              "StillWedge_MiddleSquare_Dp0.02_Particles.vtkhdf"

        h5open(output_path, "r") do file
            root = file["VTKHDF"]
            points = read(root["Points"])
            point_data = root["PointData"]
            @test read(root["NumberOfPoints"]) == [length(expected_x)]
            @test size(points) == (3, length(expected_x))
            @test points[1, :] == expected_x
            @test points[2, :] == expected_y
            @test all(iszero, points[3, :])
            @test read(point_data["Density"]) == expected_density
            @test read(point_data["Pressure"]) == expected_pressure
            @test read(point_data["Type"]) == expected_types
            @test read(point_data["GroupMarker"]) == expected_markers
        end
    end
end
