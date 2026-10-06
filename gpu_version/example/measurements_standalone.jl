# Standalone SPH measurements example. Uses synthetic particle frames to test
# the available measurements without running a simulation.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/measurements_standalone.jl

using SPHExampleGPU
using Test
using HDF5
using StaticArrays

const Measurements = SPHExampleGPU.SPHMeasurements

let
    Dimensions = 2
    FloatType = Float64

    function measurement_frame(height_shift)
        return (
            Position = SVector{Dimensions, FloatType}[
                SVector(0.1, 0.2 + height_shift),
                SVector(0.1, 1.2 + height_shift),
                SVector(1.1, 0.4 + height_shift),
                SVector(0.1, 2.0 + height_shift),
            ],
            Type = [Fluid, Fluid, Fluid, Fixed],
            Pressure = [10.0, 30.0, 20.0, 900.0],
            Velocity = SVector{Dimensions, FloatType}[
                SVector(1.0, 2.0),
                SVector(5.0, 6.0),
                SVector(3.0, 4.0),
                zero(SVector{Dimensions, FloatType}),
            ],
        )
    end

    @testset "SPH measurements" begin
        empty_config = MeasurementConfig()
        @test !Measurements.has_measurements(empty_config)
        @test Measurements.required_particle_fields(empty_config) == Symbol[]

        config = MeasurementConfig(
            pressure_probes = [MeasurementProbe("gauge", (0.05, 1.15))],
            velocity_probes = [MeasurementProbe("outlet", (1.0, 0.3))],
            water_column_probes = [
                WaterColumnProbe("near", (0.1, 0.0); radius = 0.05),
                WaterColumnProbe("default-radius", (0.5, 0.0)),
                WaterColumnProbe("above", (0.1, 1.5); radius = 0.05),
                WaterColumnProbe("empty", (10.0, 0.0); radius = 0.1),
            ],
            free_surface = FreeSurfaceDomain(
                (0.0, 0.0), (2.0, 0.0), 1.0,
            ),
            vertical_axis = 2,
            sample_every = 2,
        )
        @test Measurements.has_measurements(config)
        @test Measurements.required_particle_fields(config) ==
              [:Type, :Pressure, :Velocity]

        SimMeasurements = Measurements.resolve_measurements(
            config, Dimensions, 0.7,
        )
        @test SimMeasurements.vertical_axis == 2
        @test SimMeasurements.column_radii == [0.05, 0.7, 0.05, 0.1]
        @test SimMeasurements.surface_shape == [2]
        @test SimMeasurements.surface_locations == reshape([0.5, 1.5], 1, 2)

        mktempdir() do SaveLocation
            MeasurementFile = joinpath(SaveLocation, "measurements.h5")
            h5open(MeasurementFile, "w") do file
                writer = Measurements.MeasurementWriter(
                    file, SimMeasurements, Fluid; capacity = 3,
                )
                Measurements.append_measurements!(
                    writer, 0.0, measurement_frame(0.0),
                )
                Measurements.append_measurements!(
                    writer, 1.0, measurement_frame(0.1),
                )
                Measurements.append_measurements!(
                    writer, 2.0, measurement_frame(0.2),
                )
                Measurements.flush_measurements!(writer)
            end

            h5open(MeasurementFile, "r") do file
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
                @test heights[1:3, :] ≈
                      [1.2 1.4; 1.2 1.4; 0.0 0.0]
                @test all(isnan, heights[4, :])

                surface = group["FreeSurface"]
                @test read(surface["GridShape"]) == [2]
                @test read(surface["Values"]) ≈ [1.2 1.4; 0.4 0.6]
            end
        end

        # Height can also use a coordinate other than the last axis.
        SimMeasurements3D = MeasurementConfig(
            free_surface = FreeSurfaceDomain(
                (0.0, 0.0, 0.0), (2.0, 2.0, 2.0), 1.0,
            ),
            vertical_axis = 1,
        )
        ResolvedMeasurements3D = Measurements.resolve_measurements(
            SimMeasurements3D, 3, 0.7,
        )
        @test ResolvedMeasurements3D.vertical_axis == 1
        @test ResolvedMeasurements3D.surface_axes == [2, 3]
        @test ResolvedMeasurements3D.surface_shape == [2, 2]
    end
end
