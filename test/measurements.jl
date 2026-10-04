using HDF5
using StaticArrays
using StructArrays

function measurement_output_tests(SPH)
    @testset "SPH measurements" begin
        T = Float64
        constants = SPH.SimulationConstants{T}()
        kernel = SPH.SPHKernelInstance{2, T}(
            SPH.WendlandC2(); dx = constants.dx,
        )
        positions = SVector{2, T}[
            SVector(0.25, 0.2),
            SVector(0.25, 1.2),
            SVector(1.25, 0.4),
            SVector(0.25, 2.0),
        ]
        particles = StructArray((
            Position = positions,
            Velocity = SVector{2, T}[
                SVector(1.0, 2.0),
                SVector(3.0, 4.0),
                SVector(5.0, 6.0),
                SVector(0.0, 0.0),
            ],
            Pressure = T[4, 6, 8, 999],
            Type = [SPH.Fluid, SPH.Fluid, SPH.Fluid, SPH.Fixed],
            Density = fill(constants.ρ₀, 4),
            Acceleration = fill(zero(SVector{2, T}), 4),
            ID = collect(1:4),
            GroupMarker = fill(UInt(1), 4),
            ChunkID = zeros(Int, 4),
            Kernel = zeros(T, 4),
            KernelGradient = fill(zero(SVector{2, T}), 4),
            BoundaryBool = UInt8[0, 0, 0, 1],
            GhostPoints = fill(zero(SVector{2, T}), 4),
            GhostNormals = fill(zero(SVector{2, T}), 4),
        ))

        measurements = SPH.MeasurementConfig(
            pressure_probes = [
                SPH.MeasurementProbe("gauge", (0.2, 0.3)),
            ],
            velocity_probes = [
                SPH.MeasurementProbe("outlet", (1.1, 0.3)),
            ],
            water_column_probes = [
                SPH.WaterColumnProbe("column", (0.25, 0.0); radius = 0.05),
                SPH.WaterColumnProbe("above", (0.25, 1.5); radius = 0.05),
                SPH.WaterColumnProbe("empty", (10.0, 0.0); radius = 0.05),
            ],
            free_surface = SPH.FreeSurfaceDomain(
                (0.0, 0.0), (3.0, 0.0), 1.0,
            ),
            sample_every = 2,
        )

        make_metadata(save_location, simulation_name; single_file = true) =
            SPH.SimulationMetaData{2, T}(
                SimulationName = simulation_name,
                SaveLocation = save_location,
                OutputVariables = ["Pressure"],
                ExportSingleVTKHDF = single_file,
                ExportGridCells = false,
            )

        mktempdir() do save_location
            metadata = make_metadata(save_location, "with_measurements")
            output = SPH.SetupVTKOutput(
                metadata, particles, kernel, 2;
                measurements = measurements, fluid_type = SPH.Fluid,
            )
            output_times = collect(0.0:0.5:3.0)
            for (iteration, time) in enumerate(output_times)
                metadata.TotalTime = time
                output.save_particles(iteration)
            end
            output.close_files()

            measurement_file = joinpath(save_location, "with_measurements.vtkhdf")
            h5open(measurement_file, "r") do file
                @test haskey(file, "Measurements")
                @test read(file["VTKHDF/Steps/Values"]) ≈ output_times
                @test read(file["Measurements/Time"]) ≈ [0.0, 1.0, 2.0, 3.0]
                @test read(file["Measurements/Pressure/Values"]) ≈
                      [4.0 4.0 4.0 4.0]
                @test read(file["Measurements/Velocity/Values"])[:, 1, 1] ≈
                      [5.0, 6.0]
                water_height = read(file["Measurements/WaterColumnHeight/Values"])
                @test water_height[1:2, :] ≈
                      [1.2 1.2 1.2 1.2; 0.0 0.0 0.0 0.0]
                @test all(isnan, water_height[3, :])
                @test read(file["Measurements/WaterColumnHeight/Radii"]) ≈
                      [0.05, 0.05, 0.05]
                free_surface = read(file["Measurements/FreeSurface/Values"])
                @test free_surface[1, :] ≈ fill(1.2, 4)
                @test free_surface[2, :] ≈ fill(0.4, 4)
                @test all(isnan, free_surface[3, :])
                @test read(file["Measurements/FreeSurface/GridShape"]) == [3]
            end

            no_measurements = make_metadata(save_location, "without_measurements")
            output = SPH.SetupVTKOutput(
                no_measurements, particles, kernel, 2,
            )
            output.save_particles(1)
            output.close_files()
            h5open(
                    joinpath(save_location, "without_measurements.vtkhdf"),
                    "r",
                ) do file
                @test !haskey(file, "Measurements")
            end

            multi_file_metadata = make_metadata(
                save_location, "multi_file"; single_file = false,
            )
            @test_throws ArgumentError SPH.SetupVTKOutput(
                multi_file_metadata, particles, kernel, 2;
                measurements = measurements, fluid_type = SPH.Fluid,
            )
            @test !isfile(joinpath(save_location, "multi_file.vtkhdf"))
        end

        @test SPH.SPHMeasurements.required_particle_fields(nothing) ==
              Symbol[]
        @test SPH.SPHMeasurements.required_particle_fields(measurements) ==
              [:Type, :Pressure, :Velocity]
        @test_throws ArgumentError SPH.MeasurementProbe("bad", (0.0, NaN))

        domain_3d = SPH.FreeSurfaceDomain(
            (0.0, 0.0, 0.0), (2.0, 2.0, 0.0), 1.0,
        )
        measurements_3d = SPH.MeasurementConfig(
            pressure_probes = [
                SPH.MeasurementProbe("gauge3d", (0.25, 0.25, 0.2)),
            ],
            velocity_probes = [
                SPH.MeasurementProbe("velocity3d", (0.25, 0.25, 0.2)),
            ],
            water_column_probes = [
                SPH.WaterColumnProbe(
                    "column3d", (0.25, 0.25, 0.0); radius = 0.1,
                ),
            ],
            free_surface = domain_3d,
        )
        plan_3d = SPH.SPHMeasurements.resolve_measurements(
            measurements_3d, 3, 0.5,
        )
        @test plan_3d.vertical_axis == 3
        @test plan_3d.surface_axes == [1, 2]
        @test plan_3d.surface_shape == [2, 2]
        @test plan_3d.surface_locations ==
              [0.5 1.5 0.5 1.5; 0.5 0.5 1.5 1.5]

        particles_3d = (
            Position = SVector{3, T}[
                SVector(0.25, 0.25, 0.2),
                SVector(1.25, 0.25, 0.6),
                SVector(0.25, 1.25, 0.8),
                SVector(1.25, 1.25, 1.0),
                SVector(0.25, 0.25, 2.0),
            ],
            Type = [
                SPH.Fluid, SPH.Fluid, SPH.Fluid, SPH.Fluid, SPH.Fixed,
            ],
            Pressure = T[7, 9, 11, 13, 999],
            Velocity = SVector{3, T}[
                SVector(1.0, 2.0, 3.0),
                SVector(4.0, 5.0, 6.0),
                SVector(7.0, 8.0, 9.0),
                SVector(10.0, 11.0, 12.0),
                zero(SVector{3, T}),
            ],
        )
        mktempdir() do save_location
            path = joinpath(save_location, "measurements_3d.h5")
            file = h5open(path, "w")
            writer = SPH.SPHMeasurements.MeasurementWriter(
                file, plan_3d, SPH.Fluid; capacity = 1,
            )
            SPH.SPHMeasurements.append_measurements!(
                writer, 0.0, particles_3d,
            )
            close(file)

            h5open(path, "r") do file
                @test vec(read(file["Measurements/Pressure/Values"])) == [7.0]
                @test read(file["Measurements/Velocity/Values"])[:, 1, 1] ==
                      [1.0, 2.0, 3.0]
                @test vec(read(file["Measurements/WaterColumnHeight/Values"])) ==
                      [0.2]
                @test vec(read(file["Measurements/FreeSurface/Values"])) ≈
                      [0.2, 0.6, 0.8, 1.0]
            end
        end
    end
end
