module SphericTest06Tests

using Test
using SPHExampleGPU
using StaticArrays: SVector
using LinearAlgebra: norm
using CUDA
using TOML
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06

@testset "Official Test 06 data and absolute motion" begin
    @test verify_reference()
    motion = read_reference(joinpath(REFERENCE_DIR, "Motion_Body.dat"))
    @test size(motion, 2) == 4
    @test first(motion[:, 1]) == 0
    @test last(motion[:, 1]) > 8
    @test motion[1, 4] == 1.5
    @test maximum(motion[:, 3]) ≈ 1 atol = 0.0002
    trajectory = body_trajectory()
    for i in (1, 50, 100, 300, 700, 900)
        offset, velocity, _ = trajectory_state(trajectory, motion[i, 1])
        @test offset ≈ SVector(motion[i, 4] - 1.5, 0)
        @test velocity ≈ SVector(motion[i, 3], 0)
    end
    for reynolds in (50, 100, 150)
        table = read_reference(joinpath(REFERENCE_DIR, "Force_Re$(lpad(reynolds, 3, '0')).dat"))
        @test size(table, 2) == 3
        @test table[end, 1] >= 8
        @test all(>(0), table[:, 3])
    end
    mktempdir() do directory
        write(joinpath(directory, "bad.dat"), "variables=t,a,b\nzone\n0 1 2\n0 2 3\n")
        @test_throws ErrorException read_reference(joinpath(directory, "bad.dat"))
    end
end

@testset "Benchmark geometry, viscosity and numerical controls" begin
    for reynolds in (50, 100, 150), dx in (0.05, 0.025, 1/60)
        case = build_case(; reynolds, dx)
        @test case.counts.fluid * dx^2 ≈ 49
        @test case.counts.square * dx^2 ≈ 1
        @test case.constants.ν₀ ≈ 1/reynolds
        @test case.constants.m₀ ≈ 1000dx^2
        @test case.constants.g == 0
        @test case.metadata.GPUCellSubdivision == case.metadata.GPUXCellSubdivision == 1
        @test !case.metadata.GPURadialInteractions
        @test !case.metadata.GPUInteractionFastMath
        @test case.metadata.GPUClipTimeStepToOutput
        @test case.layers * dx > case.kernel.H
        @test case.metadata.OutputTimes[end] == 8.0
        square = case.geometry[3].Particles.Position
        @test maximum(p -> abs(p[1]-1.5), square) ≈ 0.5-dx/2
        @test maximum(p -> abs(p[2]-2.5), square) ≈ 0.5-dx/2
        @test all(p -> 0 < p[1] < 10 && 0 < p[2] < 5 &&
            !(1 < p[1] < 2 && 2 < p[2] < 3), case.geometry[2].Particles.Position)
        @test isempty(intersect(Set(case.geometry[2].Particles.Position), Set(square)))
    end
    @test_throws ArgumentError build_case(reynolds = 200)
    @test_throws ArgumentError build_case(dx = 0.03)
    @test_throws ArgumentError build_case(duration = 8.1)
    @test_throws ArgumentError build_case(duration = 0.1, interval = 0.2)
end

@testset "Drag sign, pair selection and time integration" begin
    constants = SimulationConstants{Float64}(2; dx = 0.025, c₀ = 40, g = 0, ν₀ = 0.01)
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = sqrt(2)*0.025)
    body, water = SVector(1.99, 2.5), SVector(2.02, 2.5)
    velocity, rest = SVector(1., 0.), SVector(0., 0.)
    pressure, viscous = pair_loads(body, water, velocity, rest, 1001., 1002., kernel, constants)
    @test pressure[1] < 0
    @test viscous[1] < 0
    reverse = pair_loads(water, body, rest, velocity, 1002., 1001., kernel, constants)
    @test reverse[1] ≈ -pressure
    @test reverse[2] ≈ -viscous
    @test all(iszero, pair_loads(body, SVector(3., 2.5), velocity, rest, 1001., 1002., kernel, constants))
    positions = [body, water, SVector(1.98, 2.5)]
    velocities = [velocity, rest, rest]
    loads = frame_loads(positions, velocities, [1001., 1002., 1010.], Int.([Moving, Fluid, Fixed]), [3,2,1], kernel, constants)
    @test loads.cd_pressure ≈ -2pressure[1]/1000
    @test loads.cd_viscous ≈ -2viscous[1]/1000
    @test loads.cy_pressure == loads.cy_viscous == 0
    # Unevenly sampled linear functions must have their exact integral.
    @test bin_average([0.,0.01,0.3,1.], [1.,1.02,1.6,3.], 0.1, 0.8) ≈ 1.9
    @test_throws ArgumentError bin_average([0.,1.], [0.,1.], -0.01, 0.5)
    @test_throws ArgumentError bin_average([0.,1.], [0.,1.], 0.5, 1.1)
end

if CUDA.functional()
    @testset "Force postprocessing agrees with the GPU solver" begin
        for T in (Float32, Float64)
            constants = SimulationConstants{T}(2; dx = 0.025, c₀ = 40, g = 0, ν₀ = 0.01)
            kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(sqrt(2)*0.025))
            geometry = [SPHGeometry{2,T}([SVector(1.99,2.5)]; Density = 1001, Type = Moving, GroupMarker = 3),
                SPHGeometry{2,T}([SVector(2.02,2.5)]; Density = 1002, Type = Fluid, GroupMarker = 2)]
            metadata = SimulationMetaData{2,T}(SimulationName = "Pair", SaveLocation = tempdir(),
                GPUBoundaryForces = true, GPUCellSubdivision = 1, GPUXCellSubdivision = 1,
                GPUInteractionFastMath = false)
            host = AllocateDataStructures(geometry, metadata)
            host.Velocity[1] = SVector{2,T}(1, 0)
            gpu = upload_particles(host, metadata)
            support = GPUSupportArrays{2,T}(gpu, metadata)
            cells = CellListWorkspace{2,T}(2)
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; support, constants, kernel)
            function acceleration(viscosity)
                launch_interactions!(support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
                    support.∇Cᵢ, support.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
                    cells.CellStart, gpu.CellID, cells.grid_dev, HostStep(T(0.001), zero(T)),
                    ZeroDensityDiffusion(), viscosity, kernel, constants, Val(false), Val(false);
                    lanes = Val(1), boundary_forces = Val(true), fastmath = false)
                return Array(gpu.Acceleration)
            end
            pressure = acceleration(ZeroViscosity())
            total = acceleration(Laminar())
            position, velocity, density = Array(gpu.Position), Array(gpu.Velocity), Array(gpu.Density)
            i = findfirst(==(3), Array(gpu.GroupMarker))
            j = findfirst(==(2), Array(gpu.GroupMarker))
            expected_pressure, expected_viscous = pair_loads(position[i], position[j], velocity[i], velocity[j], density[i], density[j], kernel, constants)
            tolerance = T == Float32 ? 1e-4 : 1e-11
            @test constants.m₀ * pressure[i] ≈ expected_pressure rtol = tolerance
            @test constants.m₀ * (total[i] - pressure[i]) ≈ expected_viscous rtol = tolerance
            @test pressure[i] ≈ -pressure[j] rtol = tolerance
            @test total[i] ≈ -total[j] rtol = tolerance
        end
    end

    @testset "GPU run, HDF history and incomplete-validation guard" begin
        mktempdir() do directory
            case = build_case(dx = 0.1, duration = 0.2, interval = 0.02, save = joinpath(directory, "run"))
            result = run_case(case; results_dir = directory)
            @test result.result.final_state.time == 0.2
            @test result.result.diagnostics.frozen_particles == 0
            @test length(result.analysis.rows) == 11
            @test result.metadata["motion_error_max"] < 1e-7
            @test result.metadata["motion_velocity_error_max"] < 1e-6
            @test result.metadata["minimum_clearance"] > 0
            comparison = compare_histories(directory)
            @test length(comparison.metrics) == 2
            @test all(row -> !row.full_interval && !row.drag_pass, comparison.metrics)
            @test_throws ErrorException run_case(case; results_dir = directory)
        end
    end
end

end
