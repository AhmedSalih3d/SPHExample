using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

@testset "Updated mDBC numerical checks and cloning" begin
    GK = SPHExampleGPU.GPUKernels
    for T in (Float32, Float64), D in (2, 3)
        N = D + 1
        A = SMatrix{N, N, T}(I)
        b = SVector{N, T}(ntuple(k -> k == 1 ? 999 : 2, N))
        @test GK.updated_ghost_density(A, b, T(1000)) ≈ 999
        # Singular, under-supported and ill-conditioned systems use Shepard.
        for diag in (ntuple(k -> k == 1 ? T(0.05) : one(T), N),
                     ntuple(k -> k == N ? zero(T) : one(T), N),
                     ntuple(k -> k == N ? T(100) : one(T), N))
            matrix = SMatrix{N, N, T}(Diagonal(SVector{N, T}(diag)))
            @test GK.updated_ghost_density(matrix, b, T(1000)) ≈ b[1] / matrix[1, 1]
        end
        @test GK.updated_ghost_density(zero(A), b, T(1000)) == T(1000)
        coupled = setindex(A, T(0.2), 1, 2)
        exact = SVector{N, T}(ntuple(k -> k == 1 ? 1001 : 10, N))
        rhs = coupled * exact
        @test GK.updated_ghost_density(coupled, rhs, T(1000), T(0.04)) ≈ exact[1]
        @test rhs[1] / coupled[1, 1] != exact[1]
        normal = SVector{D, T}(ntuple(k -> k == D ? one(T) : zero(T), D))
        rho, pressure = GK.cloned_boundary_state(T(1000), normal, -T(0.1) * normal,
            -T(9.81) * normal, zero(normal), T(20), T(1000))
        @test pressure ≈ T(981)
        @test rho ≈ T(1000) + T(981 / 400)
        rho, pressure = GK.cloned_boundary_state(T(999), normal, -T(0.1) * normal,
            zero(normal), zero(normal), T(20), T(1000))
        @test pressure ≈ -T(400)
        @test rho ≈ T(999)
    end
end

function updated_mdbc_fixture(::Val{D}, ::Type{T}, ::Type{TP}; dry = false) where {D, T, TP}
    V = SVector{D, TP}
    normal = SVector{D, T}(ntuple(k -> k == D ? T(0.1) : zero(T), D))
    wall = V(ntuple(k -> k == D ? zero(TP) : TP(0.5), D))
    ghost = wall + V(normal)
    fluid = vec(V[ghost + V(TP(0.04) .* Tuple(idx)) for idx in CartesianIndices(ntuple(_ -> -2:2, D))])
    dry && (fluid = [x + V(ntuple(_ -> TP(10), D)) for x in fluid])
    geometry = [
        SPHGeometry{D, T}(Particles = particle_struct_array([wall], T(1000);
            GhostPoints = [ghost], GhostNormals = [normal]), GroupMarker = 1, Type = Fixed),
        SPHGeometry{D, T}(Particles = particle_struct_array(fluid, T(999);
            Velocity = fill(SVector{D, T}(ntuple(_ -> T(0.2), D)), length(fluid))), GroupMarker = 2, Type = Fluid),
    ]
    metadata = SimulationMetaData{D, T, NoShifting, NoKernelOutput, UpdatedMDBC}(
        SimulationName = "UpdatedMDBC", SaveLocation = mktempdir(), GPUDoublePosition = TP !== T,
        SimulationTime = T(0.0002), OutputTimes = T(0.0002), VisualizeInParaview = false,
        OpenLogFile = false, GPUUseGraph = false)
    constants = SimulationConstants{T}(dx = T(0.04), m₀ = T(1000 * 0.04^D), c₀ = T(20), g = zero(T))
    kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
    host = AllocateDataStructures(geometry, metadata)
    return (; geometry, metadata, constants, kernel, host)
end

if CUDA.functional()
    @testset "Updated mDBC CUDA ghost correction" begin
        GK = SPHExampleGPU.GPUKernels
        CL = SPHExampleGPU.SPHCellList
        for D in (2, 3), (T, TP) in ((Float32, Float32), (Float32, Float64), (Float64, Float64)), dry in (false, true)
            f = updated_mdbc_fixture(Val(D), T, TP; dry)
            gpu = upload_particles(f.host; position_type = TP)
            cl = CellListWorkspace{D, TP}(length(gpu))
            CL.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
            sup = GPUSupportArrays{D, T}(length(gpu); position_type = TP, updated_mdbc = true)
            step = HostStep(T(1e-4), zero(T))
            if TP !== T
                launch_pos_cells!(sup.PosCells, gpu.Position, gpu.CellID, cl.grid_dev, step, f.kernel)
            end
            for K in (1, 4, 32)
                GK.launch_updated_mdbc!(gpu.Density, gpu.Pressure, gpu.Position, gpu.Velocity,
                    gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart,
                    cl.grid_dev, step, f.kernel, f.constants, sup.boundary_data; lanes = Val(K), pos_cells = sup.PosCells)
                idx = only(Array(gpu.GhostIndex))
                @test Array(sup.boundary_data.active)[idx] == !dry
                @test Array(gpu.Density)[idx] ≈ (dry ? T(1000) : T(999)) rtol = 1e-5
                @test Array(gpu.Pressure)[idx] ≈ (dry ? zero(T) : -T(400)) atol = T(0.5)
                @test Array(gpu.Velocity)[idx] == zero(SVector{D, T})
                @test Array(sup.boundary_data.velocity)[idx] ≈ (dry ? zero(SVector{D, T}) : SVector{D, T}(ntuple(_ -> -T(0.2), D)))
                tangent = Array(sup.boundary_data.tangent_velocity)[idx]
                @test tangent[D] ≈ zero(T) atol = eps(T)
                @test tangent[1] ≈ (dry ? zero(T) : -T(0.2))
            end
        end
    end

    @testset "Inactive boundaries contribute no mass or forces" begin
        f = updated_mdbc_fixture(Val(2), Float64, Float64)
        gpu = upload_particles(f.host[[1, 2]])
        cl = CellListWorkspace{2, Float64}(2)
        SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
        sup = GPUSupportArrays{2, Float64}(2; updated_mdbc = true)
        fill!(gpu.Density, 1000)
        fill!(gpu.Pressure, 100)
        step = HostStep(1e-4, 0.0)
        SPHExampleGPU.GPUKernels.launch_inv_density!(sup.InvDensity, gpu.Density, step)
        fluid = findfirst(==(Fluid), Array(gpu.Type))
        for active in (true, false)
            fill!(sup.boundary_data.active, active)
            launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                gpu.Position, gpu.Density, sup.InvDensity, gpu.Pressure, gpu.Velocity, gpu.Type,
                cl.CellStart, gpu.CellID, cl.grid_dev, step, ZeroDensityDiffusion(), ZeroViscosity(),
                f.kernel, f.constants, Val(false), Val(false); boundary_data = sup.boundary_data)
            acceleration = Array(gpu.Acceleration)[fluid]
            rate = Array(sup.dρdtI)[fluid]
            @test iszero(acceleration) == !active
            @test iszero(rate) == !active
        end
    end

    @testset "Updated mDBC simulation stages and graph replay" begin
        for graph in (false, true), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            f = updated_mdbc_fixture(Val(2), Float32, Float64)
            f.metadata.GPUUseGraph = graph
            RunSimulation(SimGeometry = f.geometry, SimMetaData = f.metadata, SimConstants = f.constants,
                SimKernel = f.kernel, SimParticles = f.host, SimLogger = SimulationLogger(f.metadata.SaveLocation),
                SimViscosity = Laminar(), SimDensityDiffusion = ZeroDensityDiffusion(), SimTimeStepping = scheme)
            wall = findfirst(==(Fixed), f.host.Type)
            @test all(isfinite, f.host.Density)
            @test f.host.Pressure[wall] < 0
            @test f.host.Density[wall] < 1000
            @test f.host.Pressure[wall] ≈ f.constants.c₀^2 * (f.host.Density[wall] - f.constants.ρ₀)
            @test f.host.Velocity[wall] == zero(SVector{2, Float32})
            @test f.host.Position[wall] ≈ SVector(0.5, 0.0)
            @test f.metadata.Iteration > 0
        end
    end

    @testset "Prescribed moving wall retains physical velocity and follows its ghost" begin
        f = updated_mdbc_fixture(Val(2), Float32, Float64)
        original = f.geometry[1]
        motion = MotionDetails{2, Float32}(Velocity = 0.1f0, StartTime = -1f0, Duration = 2f0,
            Direction = SVector(1f0, 0f0))
        f.geometry[1] = SPHGeometry{2, Float32}(Particles = original.Particles, GroupMarker = 1,
            Type = Moving, Motion = motion)
        host = AllocateDataStructures(f.geometry, f.metadata)
        RunSimulation(SimGeometry = f.geometry, SimMetaData = f.metadata, SimConstants = f.constants,
            SimKernel = f.kernel, SimParticles = host, SimLogger = SimulationLogger(f.metadata.SaveLocation),
            SimViscosity = Laminar(), SimDensityDiffusion = ZeroDensityDiffusion(), SimTimeStepping = SymplecticTimeStepping())
        wall = findfirst(==(Moving), host.Type)
        @test host.Velocity[wall] == SVector(0.1f0, 0f0)
        @test host.Position[wall][1] > 0.5
        @test host.GhostPoints[wall] ≈ host.Position[wall] + host.GhostNormals[wall]
    end
end
