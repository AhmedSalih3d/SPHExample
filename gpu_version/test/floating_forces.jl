module FloatingForceTests

using Test, SPHExampleGPU, CUDA, StaticArrays, StructArrays, HDF5, LinearAlgebra

const F = SPHExampleGPU.GPUFloating
const S = SPHExampleGPU.GPUStepState

@testset "body loads survive accumulator clearing and constraints" begin
    for D in (2, 3), T in (Float32, Float64), constrained in (false, true)
        points = [SVector{D,T}(ntuple(d -> T((i >> (d - 1)) & 1), D)) for i in 0:2^D-1]
        constraints = constrained ? FloatingConstraints{D,T}(
            TranslationLocks = ntuple(_ -> true, D), RotationLocks = ntuple(_ -> true, D == 2 ? 1 : 3)) : nothing
        geometry = [SPHGeometry{D,T}(points; Density = 1000, GroupMarker = 7, Type = Floating,
            Floating = FloatingDetails{T}(RelativeWeight = 1.2, PauseTime = 1, Constraints = constraints))]
        constants = SimulationConstants{T}(dx = 0.1)
        host = AllocateDataStructures(geometry; position_type = Float64)
        f = FloatingArrays(geometry, host, constants; position_type = Float64)
        gpu = upload_particles(host)
        acceleration = [SVector{D,T}(ntuple(d -> T(i + d) / 8, D)) for i in eachindex(points)]
        copyto!(gpu.Acceleration, acceleration)
        center = only(Array(f.center))
        expected_force = constants.m₀ * sum(acceleration)
        expected_torque = constants.m₀ * sum(F.floating_torque(SVector{D,T}(host.Position[i] - center), acceleration[i]) for i in eachindex(points))
        step = S.HostStep(T(0.001), zero(T))
        reduce!() = F.launch_floating_forces!(f, gpu.Acceleration, gpu.Position, gpu.Type, gpu.GroupMarker,
            f.center, constants.m₀, step)
        reduce!()
        F.launch_floating_update!(f, step, constants.g, false)
        @test all(iszero, Array(f.load_force))
        @test all(iszero, Array(f.force))
        reduce!()
        F.launch_floating_update!(f, step, constants.g, true)
        state = floating_state(f)
        @test only(state.load_force) ≈ expected_force
        @test only(state.load_torque) ≈ expected_torque
        @test all(iszero, Array(f.force))
        @test only(state.velocity) == zero(SVector{D,T}) # held, but the applied load remains observable
        reduce!()
        F.launch_floating_derivative!(f, step, constants.g)
        @test only(floating_state(f).load_force) ≈ expected_force
        @test only(floating_state(f).load_torque) ≈ expected_torque
        @test all(iszero, Array(f.force))
        fill!(f.frozen, Int32(1))
        F.freeze_body_state!(f)
        @test all(iszero, floating_state(f).load_force)
        @test all(iszero, floating_state(f).load_torque)
    end
end

@testset "body load output in temporal, numbered and variable-count files" begin
    for D in (2, 3), T in (Float32, Float64), single in (false, true), variable_count in (false, true)
        mktempdir() do directory
            markers = [7, 3]
            function state(k)
                vector(scale) = [SVector{D,T}(ntuple(d -> T(scale * k + b + d / 4), D)) for b in 1:2]
                (; center = SVector{D,Float64}.(vector(1)), velocity = vector(2), load_force = vector(3),
                    load_torque = D == 2 ? T[k, -2k] : vector(4))
            end
            particles = StructArray((; Position = copy(state(0).center), ID = [1, 2]))
            metadata = SimulationMetaData{D,T}(SimulationName = "Loads", SaveLocation = directory,
                SimulationTime = 0.2, OutputTimes = 0.1, ExportSingleVTKHDF = single,
                OutputVariables = ["ID"], GPUOutputBufferBytes = 0)
            kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.1))
            output = SetupVTKOutput(metadata, particles, kernel, D; variable_count,
                floating_state = state(0), floating_markers = markers)
            try
                @test_throws ArgumentError output.save_particles(1)
                for k in 0:2
                    body = state(k)
                    particles.Position .= body.center
                    output.save_particles(k + 1, 0.1k; body_state = body)
                end
            finally
                output.close_files()
            end
            for k in 0:2
                filename = single ? "Loads.vtkhdf" : "Loads_$(lpad(k+1,6,'0')).vtkhdf"
                h5open(joinpath(directory, filename), "r") do file
                    @test HDF5.read_attribute(file["VTKHDF"], "Type") == "MultiBlockDataSet"
                    @test haskey(file, "VTKHDF/Assembly/FloatingBodies")
                    root = file["VTKHDF/FloatingBodies"]
                    cols = single ? (2k+1:2k+2) : (1:2)
                    body = state(k)
                    @test read(root["Steps/Values"]) == (single ? [0.0, 0.1, 0.2] : [0.1k])
                    @test root["PointData/Body"][cols] == [1, 2]
                    @test root["PointData/GroupMarker"][cols] == markers
                    @test root["Points"][1:D, cols] == stack(body.center)
                    @test root["PointData/Force"][1:D, cols] == stack(body.load_force)
                    @test root["PointData/ForceMagnitude"][cols] ≈ norm.(body.load_force)
                    if D == 2
                        @test all(iszero, root["PointData/Force"][3, cols])
                        @test all(iszero, root["PointData/Torque"][1:2, cols])
                        @test root["PointData/Torque"][3, cols] == body.load_torque
                    else
                        @test root["PointData/Torque"][:, cols] == stack(body.load_torque)
                    end
                end
            end
        end
    end
end

end
