# Standalone measurement-output test; run with:
# julia --project=. test/measurements_standalone.jl

using Test
using HDF5
using StaticArrays

include(joinpath(@__DIR__, "..", "src", "SPHMeasurements.jl"))
using .SPHMeasurements

const SM = SPHMeasurements
const FLUID = UInt8(1)
const FIXED = UInt8(2)

function measurement_frame(height_shift)
    return (
        Position = SVector{2, Float64}[
            SVector(0.1, 0.2 + height_shift),
            SVector(0.1, 1.2 + height_shift),
            SVector(1.1, 0.4 + height_shift),
            SVector(0.1, 2.0 + height_shift),
        ],
        Type = UInt8[FLUID, FLUID, FLUID, FIXED],
        Pressure = [10.0, 30.0, 20.0, 900.0],
        Velocity = SVector{2, Float64}[
            SVector(1.0, 2.0),
            SVector(5.0, 6.0),
            SVector(3.0, 4.0),
            zero(SVector{2, Float64}),
        ],
    )
end

@testset "Standalone SPH measurements" begin
    empty_config = MeasurementConfig()
    @test !SM.has_measurements(empty_config)
    @test SM.required_particle_fields(empty_config) == Symbol[]

    config = MeasurementConfig(
        pressure_probes = [MeasurementProbe("gauge", (0.05, 1.15))],
        velocity_probes = [MeasurementProbe("outlet", (1.0, 0.3))],
        water_column_probes = [
            WaterColumnProbe("near", (0.1, 0.0); radius = 0.05),
            WaterColumnProbe("default-radius", (0.5, 0.0)),
            WaterColumnProbe("above", (0.1, 1.5); radius = 0.05),
            WaterColumnProbe("empty", (10.0, 0.0); radius = 0.1),
        ],
        free_surface = FreeSurfaceDomain((0.0, 0.0), (2.0, 0.0), 1.0),
        vertical_axis = 2,
        sample_every = 2,
    )
    @test SM.has_measurements(config)
    @test SM.required_particle_fields(config) == [:Type, :Pressure, :Velocity]

    plan = SM.resolve_measurements(config, 2, 0.7)
    @test plan.vertical_axis == 2
    @test plan.column_radii == [0.05, 0.7, 0.05, 0.1]
    @test plan.surface_shape == [2]
    @test plan.surface_locations == reshape([0.5, 1.5], 1, 2)

    mktempdir() do directory
        path = joinpath(directory, "measurements.h5")
        h5open(path, "w") do file
            writer = SM.MeasurementWriter(file, plan, FLUID; capacity = 3)
            SM.append_measurements!(writer, 0.0, measurement_frame(0.0))
            SM.append_measurements!(writer, 1.0, measurement_frame(0.1))
            SM.append_measurements!(writer, 2.0, measurement_frame(0.2))
            SM.flush_measurements!(writer)
        end

        h5open(path, "r") do file
            group = file["Measurements"]
            @test read(group["Time"]) == [0.0, 2.0]
            @test HDF5.attrs(group)["VerticalAxis"] == 2
            @test HDF5.attrs(group)["SampleEveryOutputFrames"] == 2

            pressure = group["Pressure"]
            @test HDF5.attrs(pressure)["Names"] == "gauge"
            @test vec(read(pressure["Values"])) == [30.0, 30.0]

            velocity = group["Velocity"]
            @test HDF5.attrs(velocity)["Names"] == "outlet"
            @test read(velocity["Values"])[:, 1, 1] == [3.0, 4.0]

            columns = group["WaterColumnHeight"]
            @test HDF5.attrs(columns)["Names"] ==
                  "near\ndefault-radius\nabove\nempty"
            @test read(columns["Radii"]) == [0.05, 0.7, 0.05, 0.1]
            heights = read(columns["Values"])
            @test heights[1:3, :] ≈ [1.2 1.4; 1.2 1.4; 0.0 0.0]
            @test all(isnan, heights[4, :])

            surface = group["FreeSurface"]
            @test read(surface["GridShape"]) == [2]
            @test read(surface["Values"]) ≈ [1.2 1.4; 0.4 0.6]
        end
    end

    # The vertical axis can be changed for coordinate systems where height is
    # not the final position component.
    axis_config = MeasurementConfig(
        free_surface = FreeSurfaceDomain(
            (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), 1.0,
        ),
        vertical_axis = 1,
    )
    axis_plan = SM.resolve_measurements(axis_config, 3, 0.7)
    @test axis_plan.vertical_axis == 1
    @test axis_plan.surface_axes == [2, 3]
    @test axis_plan.surface_shape == [2, 2]
end
