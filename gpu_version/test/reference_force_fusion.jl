module ReferenceForceFusionTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

const DRIVER = SPHExampleGPU.SPHCellList
const KERNELS = SPHExampleGPU.GPUKernels

function fixture(D, T; cached, kind = WendlandC2())
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(35), m₀ = T(1000 * 0.02^D))
    kernel = SPHKernelInstance{D,T}(kind; dx = constants.dx)
    points = [SVector{D,Float64}(ntuple(d -> 0.02 * index[d], D))
        for index in CartesianIndices(ntuple(_ -> 0:3, D))][:]
    push!(points, first(points))
    host = AllocateDataStructures([SPHGeometry{D,T}(points; Density = T(1000), Type = Fluid, GroupMarker = 1)];
        position_type = Float64)
    for i in eachindex(host)
        host.Type[i] = i == 1 ? Fixed : i == 2 ? Moving : i == 3 ? Floating : Fluid
        host.Density[i] = T(995 + i % 17)
        host.Velocity[i] = SVector{D,T}(ntuple(d -> T(0.04) * sin(T(i * d)), D))
    end
    gpu = upload_particles(host; position_type = Float64)
    cells = CellListWorkspace{D,Float64}(length(gpu))
    DRIVER.rebuild_cell_list!(gpu, cells, kernel.H⁻¹)
    support = GPUSupportArrays{D,T}(length(gpu); position_type = Float64, cache_interaction_fields = cached)
    step = HostStep(T(1e-4), 0.0)
    KERNELS.launch_pos_cells!(support.PosCells, gpu.Position, gpu.CellID, cells.grid_dev, step, kernel)
    KERNELS.launch_prepare_interactions!(gpu.Position, step; inverse_density = support.InvDensity,
        pressure = support.Pressure, Density = gpu.Density, ParticleType = gpu.Type,
        SimConstants = constants, boundary_mode = NoMDBC())
    return (; gpu, cells, support, constants, kernel, step)
end

function evaluate(f, diffusion, viscosity; fused = nothing, fastmath = true, lanes = 1, boundary_forces = true,
                  interaction_mass = nothing)
    (; gpu, cells, support) = f
    option = fused === nothing ? (;) : (; fuse_reference_forces = fused)
    KERNELS.launch_interactions!(support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
        support.∇Cᵢ, support.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
        cells.CellStart, gpu.CellID, cells.grid_dev, f.step, diffusion, viscosity, f.kernel, f.constants,
        Val(true), Val(true); pos_cells = support.PosCells, inverse_density = support.InvDensity,
        pressure = support.Pressure, radial = Val(false), lanes = Val(lanes),
        boundary_forces = Val(boundary_forces), fastmath, interaction_mass, option...)
    return map(Array, (support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, support.∇Cᵢ, support.∇◌rᵢ))
end

function check_close(actual, expected, T)
    tolerance = T === Float32 ? 3e-5 : 5e-13
    for (a, b) in zip(actual, expected)
        @test all(v -> all(isfinite, v), a)
        @test maximum(norm.(a .- b); init = 0.0) <= tolerance * max(1, maximum(norm.(b); init = 0.0))
    end
end

@testset "reference force fusion retains diffusion and other interaction outputs" begin
    for D in (2, 3), T in (Float32, Float64), cached in (false, true)
        f = fixture(D, T; cached)
        for diffusion in (LinearDensityDiffusion(), ComplexDensityDiffusion(), FourtakasDensityDiffusion()),
            factor in (0.0, 1.0, 2.0), lanes in (1, 4)
            viscosity = ArtificialViscosity(boundary_factor = factor)
            reference = evaluate(f, diffusion, viscosity; fused = false, lanes)
            actual = evaluate(f, diffusion, viscosity; fused = true, lanes)
            check_close(actual, reference, T)
        end
        model = LinearDensityDiffusion()
        viscosity = ArtificialViscosity()
        @test isequal(evaluate(f, model, viscosity), evaluate(f, model, viscosity; fused = true))
        @test isequal(evaluate(f, model, viscosity; fastmath = false),
            evaluate(f, model, viscosity; fastmath = false, fused = false))
        check_close(evaluate(f, model, viscosity; fastmath = false, fused = true),
            evaluate(f, model, viscosity; fastmath = false, fused = false), T)
        for viscosity in (ZeroViscosity(), Laminar())
            @test isequal(evaluate(f, model, viscosity; fused = false), evaluate(f, model, viscosity; fused = true))
        end
        reference = evaluate(f, model, ArtificialViscosity(); fused = false, boundary_forces = false)
        actual = evaluate(f, model, ArtificialViscosity(); fused = true, boundary_forces = false)
        check_close(actual, reference, T)
        mass = (; values = CuArray(T[2 * f.constants.m₀]), groups = f.gpu.GroupMarker)
        check_close(evaluate(f, ZeroDensityDiffusion(), ArtificialViscosity(); fused = true, interaction_mass = mass),
            evaluate(f, ZeroDensityDiffusion(), ArtificialViscosity(); fused = false, interaction_mass = mass), T)
        cubic = fixture(D, T; cached, kind = CubicSpline{T}())
        check_close(evaluate(cubic, model, ArtificialViscosity(); fused = true),
            evaluate(cubic, model, ArtificialViscosity(); fused = false), T)
    end
end

@testset "effective force fusion is independent of reference diffusion" begin
    for T in (Float32, Float64), fastmath in (false, true)
        meta = SimulationMetaData{2,T}(SimulationName = "ReferenceForceFusion", SaveLocation = tempdir(),
            GPUInteractionFastMath = fastmath, GPURadialInteractions = false, GPULanesPerParticle = 1)
        kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = T(0.02))
        config = effective_configuration(meta, kernel, LinearDensityDiffusion(), ArtificialViscosity(); particle_count = 1)
        @test config.effective.density_diffusion_operator === :reference
        @test config.effective.force_fusion == fastmath
        meta.GPURadialInteractions = true
        @test effective_configuration(meta, kernel, LinearDensityDiffusion(), ArtificialViscosity(); particle_count = 1).effective.force_fusion
        meta.TimeSteppingMode = DualSPHysicsSymplectic()
        @test !effective_configuration(meta, kernel, LinearDensityDiffusion(), ArtificialViscosity(); particle_count = 1).effective.force_fusion
    end
end

end
