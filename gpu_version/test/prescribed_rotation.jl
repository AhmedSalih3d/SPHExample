module PrescribedRotationTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5

const ROTATION = SPHExampleGPU.PrescribedRotation
const MEASUREMENTS = SPHExampleGPU.SPHMeasurements

@testset "absolute prescribed rotation" begin
    motion = RotationMotion(3; times = [0, 1, 2], angles = [0, π/2, 0], center = (1, 0))
    @test rotation_state(motion, -1) == (0, 0)
    @test rotation_state(motion, 0.5) == (π/4, π/2)
    @test rotation_state(motion, 1.5) == (π/4, -π/2)
    @test rotation_state(motion, 2) == (0, 0)
    @test rotation_position(motion, (2, 0), 1) ≈ SVector(1., 1.) atol = 1e-15
    @test rotation_position(motion, (2, 0), 0.5) ≈ SVector(1 + sqrt(0.5), sqrt(0.5)) atol = 1e-15
    @test_throws ArgumentError RotationMotion(1; times = [0], angles = [0])
    @test_throws ArgumentError RotationMotion(1; times = [0, 0], angles = [0, 1])
    @test_throws ArgumentError RotationMotion(1; times = [-1, 0], angles = [0, 1])
    @test_throws ArgumentError RotationMotion(1; times = [0, 1], angles = [0, Inf])
    @test_throws DimensionMismatch RotationMotion(1; times = [0, 1], angles = [0])
    @test_throws DimensionMismatch RotationMotion(1; times = [0, 1], angles = [0, 1], center = (0, 0, 0))

    if CUDA.functional()
        for TP in (Float32, Float64)
            reference = SVector{2,TP}[(2, 0), (0, 0), (3, 2)]
            host = (Position = reference, Velocity = zeros(SVector{2,Float32}, 3),
                Type = [Moving, Moving, Fluid], GroupMarker = [3, 3, 2], ID = [1001, 8, 73])
            rotation = ROTATION.prepare_rotation(motion, (), host)
            # Spatial order differs from the reference and IDs are deliberately sparse.
            order = [3, 2, 1]
            p, v = CuArray(reference[order]), CUDA.zeros(SVector{2,Float32}, 3)
            ty, gm, ids = CuArray(host.Type[order]), CuArray(host.GroupMarker[order]), CuArray(host.ID[order])
            for _ in 1:20
                ROTATION.launch_rotation!(rotation, p, v, ty, gm, ids, HostStep(1.0, 0.0); fraction = 0.5)
                @test Array(p)[3] ≈ SVector(1 + sqrt(0.5), sqrt(0.5)) atol = 1e-7
                @test Array(v)[3] ≈ SVector(-π/2*sqrt(0.5), π/2*sqrt(0.5)) atol = 1e-7
                ROTATION.launch_rotation!(rotation, p, v, ty, gm, ids, HostStep(0.0, 2.0))
                @test Array(p) == reference[order]
                @test Array(v) == zeros(SVector{2,Float32}, 3)
            end
            state = StepState{Float32}()
            SPHExampleGPU.GPUStepState.set_slot!(state.i, state.ih, SPHExampleGPU.GPUStepState.I_STOP,
                SPHExampleGPU.GPUStepState.STOP_OUTPUT)
            ROTATION.launch_rotation!(rotation, p, v, ty, gm, ids, state; fraction = 1)
            @test Array(p) == reference[order]
            @test_throws ArgumentError ROTATION.prepare_rotation(RotationMotion(9; times = [0, 1], angles = [0, 1]), (), host)
            @test_throws ArgumentError ROTATION.prepare_rotation(RotationMotion(2; times = [0, 1], angles = [0, 1]), (), host)
        end
    end
end

@testset "moving probes sample and export their frame positions" begin
    mktempdir() do dir
        config = MeasurementConfig(
            pressure_probes = [MeasurementProbe("moving_pressure", (1, 0); trajectory = t -> (1-t, t))],
            velocity_probes = [MeasurementProbe("moving_velocity", (1, 0); trajectory = t -> (1-t, t))],
            sample_every = 2)
        plan = MEASUREMENTS.resolve_measurements(config, 2, 0.1)
        particles = (Position = [SVector(1., 0.), SVector(0., 1.)], Type = [Fluid, Fluid],
            Pressure = [10., 20.], Velocity = [SVector(2., 0.), SVector(0., 3.)])
        path = joinpath(dir, "moving_probes.h5")
        h5open(path, "w") do file
            writer = MEASUREMENTS.MeasurementWriter(create_group(file, "VTKHDF"), plan, Fluid; capacity = 2)
            for t in (0., 0.5, 1.)
                MEASUREMENTS.append_measurements!(writer, t, particles)
            end
            MEASUREMENTS.flush_measurements!(writer)
        end
        h5open(path, "r") do file
            pressure = file["VTKHDF/PressureProbes"]
            velocity = file["VTKHDF/VelocityProbes"]
            expected = [1. 0.5 0.; 0. 0.5 1.; 0. 0. 0.]
            @test read(pressure["Points"]) == expected
            @test read(velocity["Points"]) == expected
            @test read(pressure["PointData/Pressure"]) == [10, 10, 20]
            @test read(pressure["PointData/Sampled"]) == [1, 0, 1]
            @test read(pressure["PointData/SampleTime"]) == [0, 0, 1]
            @test read(pressure["Steps/PointOffsets"]) == [0, 1, 2]
            @test read(velocity["PointData/Velocity"]) == [2. 2. 0.; 0. 0. 3.; 0. 0. 0.]
        end
    end
end

end
