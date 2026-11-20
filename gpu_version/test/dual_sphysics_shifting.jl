module PlanarShiftingTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

const DRIVER = SPHExampleGPU.SPHCellList
const KERNELS = SPHExampleGPU.GPUKernels
const CLASSIC = Val(typeof(PlanarShifting(0.1)))

@testset "DualSPHysics shifting" begin
    @testset "neighbor volume, full boundary participation and advection-only predictor" begin
        for (D, T) in ((2, Float64), (3, Float32)), lanes in (1, 4), cached in (false, true)
            V = SVector{D,T}
            tolerance = max(T(1e-6), 30eps(T))
            constants = SimulationConstants{T}(D; dx = T(0.02), c₀ = T(20), g = zero(T))
            kernel = SPHKernelInstance{D,T}(WendlandC2(); h = T(0.04), k = T(1.5))
            offset = V(ntuple(d -> d == 1 ? kernel.h / 2 : zero(T), D))
            geometry = [SPHGeometry{D,T}([zero(V)]; Density = T(1000), Type = Fluid, GroupMarker = 1),
                SPHGeometry{D,T}([offset]; Density = T(1250), Type = Fixed, GroupMarker = 2)]
            meta = SimulationMetaData{D,T,typeof(PlanarShifting(0.1))}(
                SimulationName = "DualShifting", SaveLocation = tempdir(),
                GPUCompactBuffers = true, GPUCacheInteractionFields = cached)
            host = AllocateDataStructures(geometry, meta)
            speed = V(ntuple(d -> d == 1 ? T(0.3) : zero(T), D))
            host.Velocity[1] = speed
            gpu = upload_particles(host, meta)
            support = GPUSupportArrays{D,T}(gpu, meta; fuse_predictor = true)
            @test length(support.∇Cᵢ) == length(gpu)
            @test isempty(support.∇◌rᵢ)
            cells = CellListWorkspace{D,position_float_type(meta)}(length(gpu))
            DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; support, kernel, constants)
            step = HostStep(T(0.001), zero(T))
            KERNELS.launch_prepare_interactions!(gpu.Position, step; inverse_density = support.InvDensity,
                pressure = support.Pressure, Density = gpu.Density, ParticleType = gpu.Type,
                SimConstants = constants, boundary_mode = NoMDBC())
            masses = T[constants.m₀, 3 * constants.m₀]
            interaction_mass = (; values = CuArray(masses), groups = gpu.GroupMarker)
            function interactions!(predictor_output = nothing)
                KERNELS.launch_interactions!(support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
                    support.∇Cᵢ, support.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
                    cells.CellStart, gpu.CellID, cells.grid_dev, step, ZeroDensityDiffusion(), ZeroViscosity(),
                    kernel, constants, Val(false), CLASSIC; lanes = Val(lanes),
                    interaction_mass, inverse_density = support.InvDensity, pressure = support.Pressure,
                    predictor_output)
            end
            interactions!()
            types = Array(gpu.Type)
            fluid = only(findall(==(Fluid), types))
            expected = (masses[2] / T(1250)) * ∇Wᵢⱼ(kernel, norm(offset) / kernel.h, -offset)
            @test !iszero(expected)
            @test Array(support.∇Cᵢ)[fluid] ≈ expected rtol = tolerance

            motion = MotionArrays(geometry, host)
            position_half, velocity_half = DRIVER.predictor_arrays(support, gpu)
            KERNELS.launch_half_step!(position_half, velocity_half, support.ρₙ⁺,
                gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, support.dρdtI,
                gpu.Type, gpu.GroupMarker, motion, step, constants;
                SimKernel = kernel, boundary_mode = NoMDBC())
            unfused = map(Array, (position_half, velocity_half, support.ρₙ⁺))
            @test unfused[1][fluid] ≈ speed * T(0.0005) rtol = tolerance
            @test unfused[1][only(findall(==(Fixed), types))] == offset

            output = DRIVER.predictor_outputs((; gpu, sup = support, cl = cells, SimKernel = kernel))
            interactions!(output)
            for (fused, reference) in zip(map(Array, (position_half, velocity_half, support.ρₙ⁺)), unfused)
                @test all(isapprox(a, b; rtol = tolerance) for (a, b) in zip(fused, reference))
            end
        end
    end

    @testset "component clamp and zero free-surface threshold" begin
        T = Float64
        V = SVector{3,T}
        constants = SimulationConstants{T}(dx = 0.02, c₀ = 20, g = 0)
        kernel = SPHKernelInstance{3,T}(WendlandC2(); dx = constants.dx)
        speed = V(3, 4, 0)
        gradient = V(-100, 80, 0.25)
        dt = T(0.01)
        limit = T(0.1) * constants.dx
        raw_shift = -2 * kernel.h * norm(speed) * dt * gradient
        for (mode, expected_shift) in ((CLASSIC, clamp.(raw_shift, -limit, limit)),
                                      (Val(PlanarShifting), raw_shift))
            position, predictor_position, velocity, acceleration = (CUDA.zeros(V, 2) for _ in 1:4)
            density = CUDA.fill(T(1000), 2)
            KERNELS.launch_final_step!(position, velocity, acceleration, density, CUDA.zeros(T, 2),
                copy(density), predictor_position, CuArray([speed, speed]), CuArray([Fluid, Fixed]),
                CuArray([gradient, gradient]), CUDA.fill(T(-1), 2), HostStep(dt, zero(T)),
                kernel, constants, ReductionWorkspace{SVector{3,T}}(2), mode; boundary_mode = NoMDBC())
            @test Array(position)[1] ≈ dt * speed + expected_shift rtol = 30eps(T)
            @test iszero(Array(position)[2])
            @test all(iszero, Array(velocity))
        end
    end
end

end
