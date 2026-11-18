using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
isdefined(@__MODULE__, :updated_mdbc_fixture) || include(joinpath(@__DIR__, "updated_mdbc.jl"))

if CUDA.functional()
@testset "DBC+ pressure-only correction and DBC viscosity" begin
    GK = SPHExampleGPU.GPUKernels
    CL = SPHExampleGPU.SPHCellList
    for D in (2, 3), T in (Float32, Float64), compact in (false, true), cache in (false, true)
        f = updated_mdbc_fixture(Val(D), T, T; boundary_mode = DBCPlus)
        gpu = upload_particles(f.host; boundary_only = compact)
        cl = CellListWorkspace{D,T}(length(gpu))
        CL.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
        support = GPUSupportArrays{D,T}(length(gpu); updated_mdbc = true,
            boundary_slots = gpu.BoundarySlot, boundary_count = length(gpu.GhostIndex),
            cache_interaction_fields = cache)
        step = HostStep(T(1e-4), 0.0)
        original = copy(gpu.Density)
        correct = function (mirror)
            copyto!(gpu.Density, original)
            GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity,
                gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart,
                cl.grid_dev, step, f.kernel, f.constants, support.boundary_data;
                compact_ghosts = Val(compact), inverse_density = support.InvDensity,
                pressure = support.Pressure, mirror_viscosity = mirror)
            return Array(gpu.Density)
        end
        updated = correct(true)
        plus = correct(false)
        @test plus == updated
        @test any(<(T(1000)), plus)
        types = Array(gpu.Type)
        slots = gpu.BoundarySlot === nothing ? collect(eachindex(types)) : Array(gpu.BoundarySlot)
        wall = findfirst(==(Fixed), types)
        data = support.boundary_data
        @test Array(data.tangent_velocity)[slots[wall]] == Array(gpu.Velocity)[wall]
        @test Array(data.velocity)[slots[wall]] == Array(gpu.Velocity)[wall]
        @test particle_pressure(plus[wall], Fixed, f.constants.c₀, f.constants.ρ₀, DBCPlus()) ==
            particle_pressure(plus[wall], Fixed, f.constants.c₀, f.constants.ρ₀, UpdatedMDBC())
        if cache
            @test Array(support.Pressure)[wall] ≈
                particle_pressure(plus[wall], Fixed, f.constants.c₀, f.constants.ρ₀, DBCPlus()) atol=T(.5)
        end
    end

    @testset "DBC+ impedance acts only on relative normal approach" begin
        GK = SPHExampleGPU.GPUKernels
        for T in (Float32, Float64)
            f = updated_mdbc_fixture(Val(2), T, T; boundary_mode = DBCPlusImpedance)
            gpu = upload_particles(f.host; boundary_only = true)
            cl = CellListWorkspace{2,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
            support = GPUSupportArrays{2,T}(length(gpu); updated_mdbc = true,
                boundary_slots = gpu.BoundarySlot, boundary_count = length(gpu.GhostIndex),
                cache_interaction_fields = true)
            step = HostStep(T(1e-4), 0.0)
            types = Array(gpu.Type)
            wall = only(findall(==(Fixed), types))
            slot = Array(gpu.BoundarySlot)[wall]
            data = support.boundary_data
            original_density = copy(gpu.Density)
            correct = function (impedance, wall_velocity, fluid_velocity)
                copyto!(gpu.Density, original_density)
                copyto!(gpu.Velocity, [type == Fixed ? wall_velocity : fluid_velocity for type in types])
                GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity, gpu.GhostPoints,
                    gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart, cl.grid_dev,
                    step, f.kernel, f.constants, data; compact_ghosts = Val(true),
                    pressure = support.Pressure, mirror_viscosity = false, normal_impedance = impedance)
                return Array(support.Pressure)[wall]
            end
            # The outward normal is +y. A pure common translation must not change the correction.
            stationary = zero(SVector{2,T})
            approach = SVector{2,T}(0.25, -0.125)
            baseline = correct(false, stationary, approach)
            pressure = correct(true, stationary, approach)
            @test pressure - baseline ≈ 2500 atol = 0.02
            @test particle_pressure(Array(gpu.Density)[wall], Fixed,
                f.constants.c₀, f.constants.ρ₀, DBCPlusImpedance()) ≈ pressure atol = 0.03
            boost = SVector{2,T}(0.5, 0.125)
            @test correct(true, boost, boost + approach) ≈ pressure atol = 0.02
            @test Array(data.velocity)[slot] == Array(data.tangent_velocity)[slot] == boost
            @test correct(true, stationary, -approach) == baseline
            @test correct(true, stationary, SVector{2,T}(0.25, 0)) == baseline
        end
    end
end
end
