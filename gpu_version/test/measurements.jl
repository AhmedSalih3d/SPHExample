using HDF5
using StaticArrays
using StructArrays

# GPU measurement coverage is local because its probes use SPH interpolation.
# Group `name` is a PolyData block of the composite `root`.
function check_block(root, name)
    block = root[name]
    @test HDF5.attrs(block)["Type"] == "PolyData"
    @test HDF5.attrs(block)["Version"] == [2, 3]
    return block
end

function measurement_interpolation_tests(SPH)
    Measurements = SPH.SPHMeasurements
    # Kernel normalization and the uniform mass cancel in the probe quotient.
    shape(kernel, q) = kernel isa SPH.WendlandC2 ?
        (1 - q / 2)^4 * (2q + 1) :
        q <= 1 ? 1 - 1.5q^2 + 0.75q^3 : 0.25 * (2 - q)^3

    @testset "volume-weighted SPH probe interpolation" begin
        for D in (2, 3), T in (Float32, Float64), family in (SPH.WendlandC2(), SPH.CubicSpline{T}()), h in (0.2, 0.4)
            kernel = SPH.SPHKernelInstance{D,T}(family; h = T(h))
            # Float64 positions with Float32 fields exercise mixed precision.
            point(x) = SVector{D,Float64}(ntuple(axis -> axis == 1 ? x : 0.0, D))
            velocity = SVector{D,T}(ntuple(T, D))
            constants = SPH.SimulationConstants{T}(D; ρ₀ = 975, c₀ = 30, m₀ = 3)
            particles = (
                Position = point.([0.1, 0.3, 0.0, 1.5, Float64(kernel.H), 0.0, 0.0, 0.0, 5.0]),
                Type = [SPH.Fluid, SPH.Fluid, SPH.Fixed, SPH.Fluid, SPH.Fluid,
                        SPH.Floating, SPH.Moving, SPH.Frozen, SPH.Fixed],
                Density = T[1000, 2000, NaN, 1000, 1000, NaN, NaN, NaN, NaN],
                Pressure = T[10, 50, NaN, NaN, 900, NaN, NaN, NaN, NaN],
                Velocity = [velocity, -2velocity, SVector{D,T}(ntuple(_ -> T(NaN), D)),
                            velocity, velocity, velocity, velocity, velocity, velocity],
            )
            locations = [SPH.MeasurementProbe("wet", Tuple(point(0))),
                         SPH.MeasurementProbe("dry", Tuple(point(5)))]
            config = SPH.MeasurementConfig(; pressure_probes = locations, velocity_probes = locations)
            weights = [shape(family, 0.1 / h) / 1000, shape(family, 0.3 / h) / 2000]
            pressure_expected = sum(weights .* [10, 50]) / sum(weights)
            velocity_expected = (weights[1] * velocity - 2weights[2] * velocity) / sum(weights)
            tolerance = T === Float32 ? 2e-6 : 1e-12

            mktempdir() do dir
                metadata = SPH.SimulationMetaData{D,T}(;
                    SaveLocation = dir, SimulationName = "interpolation",
                    OutputVariables = ["Type"], ExportGridCells = false,
                    ExportSingleVTKHDF = true)
                output = SPH.SetupVTKOutput(metadata, particles, kernel, D;
                    measurements = config, fluid_type = SPH.Fluid, constants)
                try
                    output.save_particles(1)
                    # Constant fields survive incomplete support at the probe.
                    particles.Pressure .= 12
                    particles.Velocity .= Ref(velocity)
                    metadata.TotalTime = T(1)
                    output.save_particles(2)
                finally
                    output.close_files()
                end
                h5open(joinpath(dir, "interpolation.vtkhdf"), "r") do file
                    pressure = file["VTKHDF/PressureProbes"]
                    values = reshape(read(pressure["PointData/Pressure"]), 2, :)
                    @test values[1, 1] ≈ pressure_expected rtol = tolerance
                    @test abs(values[1, 1] - 10) > 0.1
                    @test values[1, 2] ≈ 12 rtol = tolerance
                    @test all(isnan, values[2, :])
                    @test HDF5.attrs(pressure)["Kernel"] == string(nameof(typeof(family)))
                    @test HDF5.attrs(pressure)["SmoothingLength"] == Float64(kernel.h)
                    @test HDF5.attrs(pressure)["SupportRadius"] == Float64(kernel.H)
                    values = reshape(read(file["VTKHDF/VelocityProbes/PointData/Velocity"]), 3, 2, :)
                    @test values[1:D, 1, 1] ≈ velocity_expected rtol = tolerance
                    @test values[1:D, 1, 2] ≈ velocity rtol = tolerance
                    @test all(isnan, values[:, 2, :])
                    D == 2 && @test all(iszero, values[3, 1, :])
                end

                # Compact host frames have no stored pressure. Interpolate the
                # per-particle EOS values, using the supplied simulation constants.
                compact = Base.structdiff(particles, (; Pressure = nothing))
                metadata.SimulationName = "derived_pressure"
                metadata.TotalTime = zero(T)
                output = SPH.SetupVTKOutput(metadata, compact, kernel, D;
                    measurements = config, fluid_type = SPH.Fluid, constants)
                try
                    output.save_particles(1)
                finally
                    output.close_files()
                end
                pressure_from_density = [Float64(SPH.SimulationEquations.EquationOfStateGamma7(
                    T(ρ), constants.c₀, constants.ρ₀)) for ρ in (1000, 2000)]
                expected = sum(weights .* pressure_from_density) / sum(weights)
                h5open(joinpath(dir, "derived_pressure.vtkhdf"), "r") do file
                    @test read(file["VTKHDF/PressureProbes/PointData/Pressure"])[1] ≈ expected rtol = tolerance
                end
            end
        end
    end

    @testset "SPH probes across shifted periodic seams" begin
        for D in (2, 3)
            point(x, z) = SVector{D,Float64}(ntuple(axis -> axis == 1 ? x : axis == D ? z : 0.0, D))
            periodicity = SPH.PeriodicBoundary{D}(1; min = 0, max = 2, shift = point(0, 0.3))
            kernel = SPH.SPHKernelInstance{D,Float64}(SPH.WendlandC2(); h = 0.2)
            particles = (Position = [point(0.1, 0), point(1.9, -0.3)],
                Type = [SPH.Fluid, SPH.Fluid], Density = [1000., 2000.],
                Pressure = [10., 50.], Velocity = [point(1, 2), point(3, 4)])
            # These are the same physical point in two periodic images.
            probes = [SPH.MeasurementProbe("seam", Tuple(point(0, 0))),
                      SPH.MeasurementProbe("image", Tuple(point(2, -0.3)))]
            config = SPH.MeasurementConfig(; pressure_probes = probes, velocity_probes = probes)
            mktempdir() do dir
                metadata = SPH.SimulationMetaData{D,Float64}(;
                    SaveLocation = dir, SimulationName = "seam", Periodicity = periodicity,
                    OutputVariables = ["Type"], ExportGridCells = false, ExportSingleVTKHDF = true)
                output = SPH.SetupVTKOutput(metadata, particles, kernel, D;
                    measurements = config, fluid_type = SPH.Fluid)
                try
                    output.save_particles(1)
                finally
                    output.close_files()
                end
                h5open(joinpath(dir, "seam.vtkhdf"), "r") do file
                    @test read(file["VTKHDF/PressureProbes/PointData/Pressure"]) ≈ fill(70 / 3, 2)
                    expected_velocity = (2particles.Velocity[1] + particles.Velocity[2]) / 3
                    values = read(file["VTKHDF/VelocityProbes/PointData/Velocity"])
                    @test values[1:D, :] ≈ repeat(expected_velocity, 1, 2)
                end
            end
        end
    end

    @testset "SPH probe input requirements" begin
        kernel = SPH.SPHKernelInstance{2,Float64}(SPH.WendlandC2(); h = 0.1)
        config = SPH.MeasurementConfig(; velocity_probes = [SPH.MeasurementProbe("v", (0, 0))])
        @test Measurements.required_particle_fields(config) == [:Type, :Density, :Velocity]
        columns = SPH.MeasurementConfig(; water_column_probes = [SPH.WaterColumnProbe("h", (0, 0))])
        @test Measurements.required_particle_fields(columns) == [:Type]
        mktempdir() do dir
            h5open(joinpath(dir, "validation.h5"), "w") do file
                root = HDF5.create_group(file, "VTKHDF")
                plan = Measurements.resolve_measurements(config, 2, kernel.H)
                @test_throws ArgumentError Measurements.MeasurementWriter(root, plan, SPH.Fluid)
                wrong_kernel = SPH.SPHKernelInstance{3,Float64}(SPH.WendlandC2(); h = 0.1)
                @test_throws DimensionMismatch Measurements.MeasurementWriter(root, plan, SPH.Fluid; kernel = wrong_kernel)
                writer = Measurements.MeasurementWriter(root, plan, SPH.Fluid; kernel, capacity = 1)
                particles = (Position = [SVector(0., 0.)], Type = [SPH.Fluid], Velocity = [SVector(1., 2.)])
                @test_throws ArgumentError Measurements.append_measurements!(writer, 0., particles)
                @test_throws DimensionMismatch Measurements.append_measurements!(writer, 0., merge(particles, (; Density = Float64[])))
                @test_throws ArgumentError Measurements.append_measurements!(writer, 0., merge(particles, (; Density = [0.])))
                for density in (-1., NaN, Inf)
                    @test_throws ArgumentError Measurements.append_measurements!(writer, 0., merge(particles, (; Density = [density])))
                end
                Measurements.append_measurements!(writer, 0., merge(particles, (; Density = [1000.])))
                @test writer.velocity_values[:, 1] ≈ [1, 2, 0]
                dry = merge(particles, (; Position = [SVector(10., 0.)], Density = [1000.]))
                Measurements.append_measurements!(writer, 1., dry)
                @test all(isnan, writer.velocity_values)
                empty_frame = map(empty, merge(particles, (; Density = [1000.])))
                Measurements.append_measurements!(writer, 2., empty_frame)
                @test all(isnan, writer.velocity_values)
                height_plan = Measurements.resolve_measurements(columns, 2, kernel.H)
                height_writer = Measurements.MeasurementWriter(root, height_plan, SPH.Fluid; capacity = 1)
                Measurements.append_measurements!(height_writer, 0., particles)
                @test height_writer.column_values == [0.]
            end
        end
    end
end

function measurement_gpu_tests(SPH)
    @testset "velocity probes stage density without particle-field output" begin
        if SPH.CUDA.functional()
            for async in (false, true), scheme in (SPH.SymplecticTimeStepping(), SPH.SingleNeighborTimeStepping())
                mktempdir() do dir
                    T = Float32
                    points = [SVector{2,T}(x, y) for x in T.((0, 0.02, 0.04)), y in T.((0, 0.02, 0.04))]
                    geometry = [SPH.SPHGeometry{2,T}(vec(points); Density = 1000, GroupMarker = 1, Type = SPH.Fluid)]
                    metadata = SPH.SimulationMetaData{2,T}(;
                        SaveLocation = dir, SimulationName = "velocity_only", SimulationTime = T(0.001),
                        OutputTimes = T(0.0005), OutputVariables = ["Type"], ExportGridCells = false,
                        ExportSingleVTKHDF = true, GPUAsyncOutput = async, GPUUseGraph = false)
                    constants = SPH.SimulationConstants{T}(2; g = 0, c₀ = 20)
                    kernel = SPH.SPHKernelInstance{2,T}(SPH.CubicSpline{T}(); h = T(0.03))
                    particles = SPH.AllocateDataStructures(geometry, metadata)
                    velocity = SVector{2,T}(0.125, -0.25)
                    particles.Velocity .= Ref(velocity)
                    config = SPH.MeasurementConfig(; velocity_probes = [SPH.MeasurementProbe("v", (0.02, 0.02))])
                    logger = SPH.SimulationLogger(dir)
                    try
                        redirect_stdout(devnull) do
                            SPH.RunSimulation(; SimGeometry = geometry, SimMetaData = metadata,
                                SimConstants = constants, SimKernel = kernel, SimParticles = particles,
                                SimLogger = logger, SimViscosity = SPH.ZeroViscosity(),
                                SimDensityDiffusion = SPH.ZeroDensityDiffusion(), SimTimeStepping = scheme,
                                SimMeasurements = config)
                        end
                    finally
                        close(logger.LoggerIo)
                    end
                    h5open(joinpath(dir, "velocity_only.vtkhdf"), "r") do file
                        @test !haskey(file["VTKHDF/Particles/PointData"], "Density")
                        @test !haskey(file["VTKHDF/Particles/PointData"], "Velocity")
                        block = file["VTKHDF/VelocityProbes"]
                        values = read(block["PointData/Velocity"])
                        @test size(values, 2) >= 2
                        @test values ≈ repeat([Float64.(velocity); 0], 1, size(values, 2)) rtol = 1e-6
                        @test HDF5.attrs(block)["Kernel"] == "CubicSpline"
                    end
                end
            end
        else
            @test_skip SPH.CUDA.functional()
        end
    end
end

# The target path of the soft link `name` of `group`; `H5Lget_val` fails for
# any other kind of link.
function soft_link_target(group, name)
    buffer = zeros(UInt8, 256)
    status = ccall(
        (:H5Lget_val, HDF5.API.libhdf5), HDF5.API.herr_t,
        (HDF5.API.hid_t, Cstring, Ptr{Cvoid}, Csize_t, HDF5.API.hid_t),
        group, name, buffer, length(buffer), HDF5.API.H5P_DEFAULT,
    )
    status < 0 && error("`$name` is not a soft link")
    return unsafe_string(pointer(buffer))
end

# Link `name` of `group` is a soft link to the block `/VTKHDF/name`.
function check_soft_link(group, name)
    @test soft_link_target(group, name) == "/VTKHDF/" * name
    @test HDF5.attrs(group[name])["Type"] == "PolyData"
end

# The temporal bookkeeping of a measurement block with `npoints` vertices
# over `times`.
function check_block_steps(block, npoints, times; dynamic_points = false)
    nsteps = length(times)
    steps = block["Steps"]
    @test HDF5.attrs(steps)["NSteps"] == nsteps
    @test read(steps["Values"]) ≈ times
    @test read(steps["PartOffsets"]) == 0:(nsteps - 1)
    @test read(steps["NumberOfParts"]) == ones(nsteps)
    expected_point_offsets = dynamic_points ?
        npoints .* (0:(nsteps - 1)) : zeros(nsteps)
    @test read(steps["PointOffsets"]) == expected_point_offsets
    @test read(steps["CellOffsets"]) == zeros(4, nsteps)
    @test read(steps["ConnectivityIdOffsets"]) == zeros(4, nsteps)
    @test read(block["NumberOfPoints"]) == fill(npoints, nsteps)
    point_columns = dynamic_points ? npoints * nsteps : npoints
    @test size(read(block["Points"])) == (3, point_columns)
    @test read(block["Vertices/NumberOfCells"]) == fill(npoints, nsteps)
    @test read(block["Vertices/NumberOfConnectivityIds"]) == fill(npoints, nsteps)
    @test read(block["Vertices/Connectivity"]) == 0:(npoints - 1)
    @test read(block["Vertices/Offsets"]) == 0:npoints
    for topology in ("Lines", "Polygons", "Strips")
        @test read(block["$topology/NumberOfCells"]) == zeros(nsteps)
        @test read(block["$topology/NumberOfConnectivityIds"]) == zeros(nsteps)
        @test read(block["$topology/Offsets"]) == [0]
        @test isempty(read(block["$topology/Connectivity"]))
    end
    for name in ("Sampled", "SampleTime")
        @test read(steps["PointDataOffsets"][name]) == npoints .* (0:(nsteps - 1))
    end
end

# The `npoints × nsteps` values of the scalar point array `name`.
series(block, name, npoints) = reshape(read(block["PointData"][name]), npoints, :)

function measurement_output_tests(SPH)
    @testset "SPH measurements" begin
        T = Float64
        constants = SPH.SimulationConstants{T}()
        kernel = SPH.SPHKernelInstance{2, T}(
            SPH.WendlandC2(); h = 0.1,
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
        block_names = SPH.SPHMeasurements.measurement_block_names
        @test block_names(SPH.SPHMeasurements.resolve_measurements(measurements, 2, 0.1)) ==
              ["PressureProbes", "VelocityProbes", "WaterColumnProbes", "FreeSurface"]

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
            # Seven frames: with `sample_every = 2` the odd frames are sampled;
            # the gauge pressure changes every frame so that repeated samples
            # are distinguishable from fresh ones.
            output_times = collect(0.0:0.5:3.0)
            for (iteration, time) in enumerate(output_times)
                metadata.TotalTime = time
                particles.Pressure[1] = 4 + iteration
                particles.Position[2] = SVector(
                    0.25, 1.2 + 0.1 * (iteration - 1),
                )
                output.save_particles(iteration)
            end
            output.close_files()
            particles.Pressure[1] = 4
            particles.Position[2] = positions[2]

            nsteps = length(output_times)
            sampled = UInt8[1, 0, 1, 0, 1, 0, 1]
            sample_times = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0]

            measurement_file = joinpath(save_location, "with_measurements.vtkhdf")
            h5open(measurement_file, "r") do file
                @test !haskey(file, "Measurements")
                root = file["VTKHDF"]
                @test HDF5.attrs(root)["Type"] == "MultiBlockDataSet"
                @test HDF5.attrs(root)["Version"] == [2, 3]
                @test HDF5.get_create_properties(root).track_order
                @test keys(root) == [
                    "Particles", "PressureProbes", "VelocityProbes",
                    "WaterColumnProbes", "FreeSurface", "Assembly",
                ]

                particle_block = check_block(root, "Particles")
                @test HDF5.attrs(particle_block["Steps"])["NSteps"] == nsteps
                @test read(particle_block["Steps/Values"]) ≈ output_times
                @test haskey(particle_block, "PointData/Pressure")

                assembly = root["Assembly"]
                @test HDF5.get_create_properties(assembly).track_order
                @test keys(assembly) == ["Particles", "Measurements"]
                check_soft_link(assembly, "Particles")
                node = assembly["Measurements"]
                @test HDF5.get_create_properties(node).track_order
                @test keys(node) == [
                    "PressureProbes", "VelocityProbes", "WaterColumnProbes",
                    "FreeSurface",
                ]
                foreach(name -> check_soft_link(node, name), keys(node))

                pressure = check_block(root, "PressureProbes")
                check_block_steps(pressure, 1, output_times)
                @test read(pressure["Points"]) ≈ [0.2; 0.3; 0.0;;]
                @test read(pressure["FieldData/Names"]) == ["gauge"]
                @test HDF5.attrs(pressure)["Names"] == "gauge"
                @test read(pressure["Steps/FieldDataOffsets/Names"]) == zeros(nsteps)
                @test read(pressure["Steps/FieldDataSizes/Names"]) ==
                      repeat([1, 1], 1, nsteps)
                @test read(pressure["Steps/PointDataOffsets/Pressure"]) == 0:(nsteps - 1)
                @test vec(series(pressure, "Pressure", 1)) ≈ [5, 5, 7, 7, 9, 9, 11]
                @test vec(series(pressure, "Sampled", 1)) == sampled
                @test vec(series(pressure, "SampleTime", 1)) ≈ sample_times
                @test HDF5.attrs(pressure)["SampleEveryOutputFrames"] == 2
                @test HDF5.attrs(pressure)["Sampling"] == "normalized SPH interpolation"
                @test HDF5.attrs(pressure)["Kernel"] == "WendlandC2"
                @test HDF5.attrs(pressure)["SmoothingLength"] == kernel.h
                @test HDF5.attrs(pressure)["SupportRadius"] == kernel.H
                @test HDF5.attrs(pressure)["VerticalAxis"] == 2

                velocity = check_block(root, "VelocityProbes")
                check_block_steps(velocity, 1, output_times)
                @test read(velocity["Points"]) ≈ [1.1; 0.3; 0.0;;]
                @test read(velocity["FieldData/Names"]) == ["outlet"]
                @test read(velocity["Steps/PointDataOffsets/Velocity"]) == 0:(nsteps - 1)
                @test read(velocity["PointData/Velocity"]) ≈
                      repeat([5.0, 6.0, 0.0], 1, nsteps)
                @test vec(series(velocity, "Sampled", 1)) == sampled

                column = check_block(root, "WaterColumnProbes")
                check_block_steps(column, 3, output_times; dynamic_points = true)
                column_points = reshape(read(column["Points"]), 3, 3, nsteps)
                @test column_points[1, :, :] ≈ repeat(
                    reshape([0.25, 0.25, 10.0], 3, 1), 1, nsteps,
                )
                @test column_points[2, 1, :] ≈
                      [1.2, 1.2, 1.4, 1.4, 1.6, 1.6, 1.8]
                @test column_points[2, 2, :] ≈
                      [1.5, 1.5, 1.5, 1.5, 1.6, 1.6, 1.8]
                @test column_points[2, 3, :] == zeros(nsteps)
                @test read(column["FieldData/Names"]) == ["column", "above", "empty"]
                @test HDF5.attrs(column)["Names"] == "column\nabove\nempty"
                @test read(column["PointData/Radius"]) ≈ [0.05, 0.05, 0.05]
                @test read(column["Steps/PointDataOffsets/Radius"]) == zeros(nsteps)
                @test read(column["Steps/PointDataOffsets/WaterColumnHeight"]) ==
                      3 .* (0:(nsteps - 1))
                water_height = series(column, "WaterColumnHeight", 3)
                @test water_height[1, :] ≈
                      [1.2, 1.2, 1.4, 1.4, 1.6, 1.6, 1.8]
                @test water_height[2, :] ≈
                      [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.3]
                @test all(isnan, water_height[3, :])
                @test series(column, "Sampled", 3) == repeat(sampled', 3, 1)
                @test series(column, "SampleTime", 3) ≈ repeat(sample_times', 3, 1)

                surface = check_block(root, "FreeSurface")
                check_block_steps(surface, 3, output_times; dynamic_points = true)
                surface_points = reshape(read(surface["Points"]), 3, 3, nsteps)
                @test surface_points[1, :, :] ≈
                      repeat(reshape([0.5, 1.5, 2.5], 3, 1), 1, nsteps)
                @test surface_points[2, 1, :] ≈
                      [1.2, 1.2, 1.4, 1.4, 1.6, 1.6, 1.8]
                @test surface_points[2, 2, :] == fill(0.4, nsteps)
                @test surface_points[2, 3, :] == zeros(nsteps)
                @test read(surface["FieldData/GridShape"]) == [3]
                @test read(surface["FieldData/HorizontalAxes"]) == [1]
                @test read(surface["FieldData/Spacing"]) ≈ [1.0]
                @test read(surface["FieldData/DomainLower"]) ≈ [0.0, 0.0]
                @test read(surface["FieldData/DomainUpper"]) ≈ [3.0, 0.0]
                @test read(surface["Steps/FieldDataSizes/DomainLower"]) ==
                      repeat([1, 2], 1, nsteps)
                free_surface = series(surface, "FreeSurfaceHeight", 3)
                @test free_surface[1, :] ≈
                      [1.2, 1.2, 1.4, 1.4, 1.6, 1.6, 1.8]
                @test free_surface[2, :] ≈ fill(0.4, nsteps)
                @test all(isnan, free_surface[3, :])
                @test HDF5.attrs(surface)["Ordering"] ==
                      "first horizontal axis varies fastest"
            end

            # Without measurements the file stays a plain temporal PolyData.
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
                root = file["VTKHDF"]
                @test HDF5.attrs(root)["Type"] == "PolyData"
                @test !haskey(root, "Assembly")
                @test !haskey(root, "Particles")
                @test haskey(root, "Steps")
                @test read(root["Steps/Values"]) == [0.0]
            end

            # An empty configuration counts as disabled.
            output = SPH.SetupVTKOutput(
                make_metadata(save_location, "empty_config"), particles, kernel, 2;
                measurements = SPH.MeasurementConfig(), fluid_type = SPH.Fluid,
            )
            output.save_particles(1)
            output.close_files()
            h5open(joinpath(save_location, "empty_config.vtkhdf"), "r") do file
                @test HDF5.attrs(file["VTKHDF"])["Type"] == "PolyData"
            end

            # Only the requested categories become blocks.
            pressure_only = SPH.MeasurementConfig(
                pressure_probes = [SPH.MeasurementProbe("gauge", (0.2, 0.3))],
            )
            pressure_plan = SPH.SPHMeasurements.resolve_measurements(pressure_only, 2, 0.1)
            @test block_names(pressure_plan) == ["PressureProbes"]
            output = SPH.SetupVTKOutput(
                make_metadata(save_location, "pressure_only"), particles, kernel, 2;
                measurements = pressure_only, fluid_type = SPH.Fluid,
            )
            output.save_particles(1)
            output.close_files()
            h5open(joinpath(save_location, "pressure_only.vtkhdf"), "r") do file
                root = file["VTKHDF"]
                @test keys(root) == ["Particles", "PressureProbes", "Assembly"]
                @test keys(root["Assembly/Measurements"]) == ["PressureProbes"]
                pressure = check_block(root, "PressureProbes")
                check_block_steps(pressure, 1, [0.0])
                @test vec(series(pressure, "Pressure", 1)) ≈ [4.0]
                @test vec(series(pressure, "Sampled", 1)) == [1]
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
              [:Type, :Density, :Pressure, :Velocity]
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
            Density = fill(constants.ρ₀, 5),
            Pressure = T[7, 9, 11, 13, 999],
            Velocity = SVector{3, T}[
                SVector(1.0, 2.0, 3.0),
                SVector(4.0, 5.0, 6.0),
                SVector(7.0, 8.0, 9.0),
                SVector(10.0, 11.0, 12.0),
                zero(SVector{3, T}),
            ],
        )
        Measurements = SPH.SPHMeasurements
        kernel_3d = SPH.SPHKernelInstance{3,T}(SPH.WendlandC2(); h = 0.25)
        mktempdir() do save_location
            path = joinpath(save_location, "measurements_3d.vtkhdf")
            file = h5open(path, "w")
            root = SPH.ProduceHDFVTK.create_collection_root(file)
            writer = Measurements.MeasurementWriter(
                root, plan_3d, SPH.Fluid; kernel = kernel_3d, capacity = 1,
            )
            @test [block.name for block in writer.blocks] ==
                  Measurements.measurement_block_names(plan_3d)
            Measurements.append_measurements!(writer, 0.0, particles_3d)
            @test Measurements.frames_written(writer) == 1
            close(file)

            h5open(path, "r") do file
                root = file["VTKHDF"]
                pressure = check_block(root, "PressureProbes")
                @test read(pressure["Points"]) ≈ [0.25; 0.25; 0.2;;]
                @test vec(read(pressure["PointData/Pressure"])) ≈ [7.0]
                velocity = check_block(root, "VelocityProbes")
                @test read(velocity["PointData/Velocity"]) ≈ [1.0; 2.0; 3.0;;]
                column = check_block(root, "WaterColumnProbes")
                @test vec(read(column["PointData/WaterColumnHeight"])) == [0.2]
                @test read(column["PointData/Radius"]) == [0.1]
                @test read(column["Points"]) ≈ [0.25; 0.25; 0.2;;]
                surface = check_block(root, "FreeSurface")
                surface_height = read(surface["PointData/FreeSurfaceHeight"])
                @test vec(surface_height) ≈ [0.2, 0.6, 0.8, 1.0]
                @test read(surface["Points"]) ≈
                      [0.5 1.5 0.5 1.5; 0.5 0.5 1.5 1.5; 0.2 0.6 0.8 1.0]
                @test read(surface["FieldData/GridShape"]) == [2, 2]
                @test read(surface["FieldData/HorizontalAxes"]) == [1, 2]
            end
        end

        # Buffering: frames are written `capacity` at a time, the rest on flush.
        mktempdir() do save_location
            path = joinpath(save_location, "buffered.vtkhdf")
            plan = Measurements.resolve_measurements(measurements, 2, 0.1)
            h5open(path, "w") do file
                root = SPH.ProduceHDFVTK.create_collection_root(file)
                writer = Measurements.MeasurementWriter(
                    root, plan, SPH.Fluid; kernel, capacity = 2,
                )
                nsteps(name) = HDF5.attrs(root[name]["Steps"])["NSteps"]
                pressure = root["PressureProbes"]
                moving_column = root["WaterColumnProbes"]
                moving_points_offsets = moving_column["Steps/PointOffsets"]
                column_offsets = root["WaterColumnProbes/Steps/PointDataOffsets"]
                for (frame, time) in enumerate([0.0, 0.5, 1.0])
                    particles.Pressure[1] = 4 + frame
                    particles.Position[2] = SVector(
                        0.25, 1.2 + 0.1 * (frame - 1),
                    )
                    Measurements.append_measurements!(writer, time, particles)
                end
                particles.Pressure[1] = 4
                particles.Position[2] = positions[2]
                @test Measurements.frames_written(writer) == 2
                @test Measurements.frames_pending(writer) == 1
                @test nsteps("PressureProbes") == 2
                @test nsteps("FreeSurface") == 2
                @test read(pressure["Steps/Values"]) == [0.0, 0.5]
                @test vec(read(pressure["PointData/Pressure"])) ≈ [5.0, 5.0]
                @test read(moving_points_offsets) == [0, 3]
                @test reshape(read(moving_column["Points"]), 3, 3, 2)[2, 1, :] ==
                      [1.2, 1.2]
                @test read(column_offsets["WaterColumnHeight"]) == [0, 3]
                Measurements.flush_measurements!(writer)
                @test Measurements.frames_written(writer) == 3
                @test Measurements.frames_pending(writer) == 0
                # Flushing without pending frames changes nothing.
                Measurements.flush_measurements!(writer)
                @test Measurements.frames_written(writer) == 3
                for name in Measurements.measurement_block_names(plan)
                    @test nsteps(name) == 3
                    @test read(root[name]["Steps/Values"]) == [0.0, 0.5, 1.0]
                end
                @test vec(read(pressure["PointData/Pressure"])) ≈ [5.0, 5.0, 7.0]
                @test vec(read(pressure["PointData/Sampled"])) == [1, 0, 1]
                @test vec(read(pressure["PointData/SampleTime"])) == [0.0, 0.0, 1.0]
                @test read(moving_points_offsets) == [0, 3, 6]
                @test reshape(read(moving_column["Points"]), 3, 3, 3)[2, 1, :] ==
                      [1.2, 1.2, 1.4]
                @test read(column_offsets["WaterColumnHeight"]) == [0, 3, 6]
                @test read(root["FreeSurface/Steps/FieldDataSizes/GridShape"]) ==
                      [1 1 1; 1 1 1]
            end
        end
    end
end
