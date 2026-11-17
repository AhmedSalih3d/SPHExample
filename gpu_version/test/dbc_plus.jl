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
end
end
