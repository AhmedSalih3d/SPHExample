using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

@testset "Updated mDBC numerical checks and cloning" begin
    GK = SPHExampleGPU.GPUKernels
    for T in (Float32, Float64), D in (2, 3)
        cubic = SPHKernelInstance{D, T}(CubicSpline{T}(); dx = T(0.04))
        for pressures in ((T(-100), T(200)), (T(300), T(400)))
            pi, pj = pressures
            reference = tensile_correction(cubic, pi, T(997), pj, T(1005), T(0.7), T(0.04))
            cached = tensile_correction(cubic, pi, T(997), pj, T(1005), T(0.7), T(0.04), inv(T(997)), inv(T(1005)))
            @test cached ≈ reference rtol = 5eps(T)
        end
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
    @testset "cached ghost mass/density weights and corrected reciprocals" begin
        GK = SPHExampleGPU.GPUKernels
        CL = SPHExampleGPU.SPHCellList
        for D in (2, 3), T in (Float32, Float64), updated in (false, true)
            f = updated_mdbc_fixture(Val(D), T, T)
            gpu = upload_particles(f.host; boundary_only = true)
            cl = CellListWorkspace{D, T}(length(gpu))
            CL.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
            sup = GPUSupportArrays{D, T}(length(gpu); updated_mdbc = true,
                cache_interaction_fields = true, boundary_slots = gpu.BoundarySlot,
                boundary_count = length(gpu.GhostIndex))
            step = HostStep(T(1e-4), zero(T))
            original = copy(gpu.Density)
            mode = updated ? UpdatedMDBC() : SimpleMDBC()
            function correct(cache, pcache = nothing)
                if updated
                    GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity,
                        gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart,
                        cl.grid_dev, step, f.kernel, f.constants, sup.boundary_data;
                        compact_ghosts = Val(true), inverse_density = cache, pressure = pcache)
                else
                    GK.launch_mdbc!(gpu.Density, gpu.Position,
                        gpu.GhostPoints, gpu.GhostIndex, gpu.Type, cl.CellStart, cl.grid_dev,
                        step, f.kernel, f.constants; compact_ghosts = Val(true), inverse_density = cache, pressure = pcache)
                end
                densities = Array(gpu.Density)
                types = Array(gpu.Type)
                pressures = [particle_pressure(densities[i], types[i], f.constants.c₀, f.constants.ρ₀, mode)
                    for i in eachindex(densities)]
                return densities, pressures
            end
            reference = correct(nothing)
            copyto!(gpu.Density, original)
            GK.launch_prepare_interactions!(gpu.Position, step;
                inverse_density = sup.InvDensity, pressure = sup.Pressure, Density = gpu.Density,
                ParticleType = gpu.Type, SimConstants = f.constants, boundary_mode = mode)
            cached = correct(sup.InvDensity, sup.Pressure)
            @test cached[1] ≈ reference[1] rtol = 20eps(T)
            @test cached[2] ≈ reference[2] rtol = 100eps(T) atol = T(0.5)
            @test Array(sup.InvDensity) ≈ inv.(cached[1]) rtol = 2eps(T)
            @test Array(sup.Pressure) ≈ cached[2] rtol = 100eps(T) atol = T(0.5)
        end
    end

    @testset "Updated mDBC CUDA ghost correction" begin
        GK = SPHExampleGPU.GPUKernels
        CL = SPHExampleGPU.SPHCellList
        for D in (2, 3), (T, TP) in ((Float32, Float32), (Float32, Float64), (Float64, Float64)),
            dry in (false, true), compact in (false, true), cache in (false, true)
            f = updated_mdbc_fixture(Val(D), T, TP; dry)
            gpu = upload_particles(f.host; position_type = TP, boundary_only = compact)
            cl = CellListWorkspace{D, TP}(length(gpu))
            CL.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
            sup = GPUSupportArrays{D, T}(length(gpu); position_type = TP, updated_mdbc = true,
                cache_interaction_fields = cache,
                boundary_slots = gpu.BoundarySlot, boundary_count = compact ? length(gpu.GhostIndex) : length(gpu))
            step = HostStep(T(1e-4), zero(T))
            if TP !== T
                launch_pos_cells!(sup.PosCells, gpu.Position, gpu.CellID, cl.grid_dev, step, f.kernel)
            end
            for K in (1, 4, 32)
                GK.launch_prepare_interactions!(gpu.Position, step;
                    inverse_density = sup.InvDensity, pressure = sup.Pressure, Density = gpu.Density,
                    ParticleType = gpu.Type, SimConstants = f.constants, boundary_mode = UpdatedMDBC())
                GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity,
                    gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart,
                    cl.grid_dev, step, f.kernel, f.constants, sup.boundary_data; lanes = Val(K), pos_cells = sup.PosCells,
                    compact_ghosts = Val(compact), inverse_density = sup.InvDensity, pressure = sup.Pressure)
                if cache
                    @test Array(sup.InvDensity) ≈ inv.(Array(gpu.Density)) rtol = 2eps(T)
                    @test Array(sup.Pressure)[only(Array(gpu.GhostIndex))] ≈ (dry ? zero(T) : -T(400)) atol = T(0.5)
                end
                idx = only(Array(gpu.GhostIndex))
                slot = compact ? 1 : idx
                @test Array(sup.boundary_data.active)[slot] == !dry
                @test Array(gpu.Density)[idx] ≈ (dry ? T(1000) : T(999)) rtol = 1e-5
                @test particle_pressure(Array(gpu.Density)[idx], Array(gpu.Type)[idx],
                    f.constants.c₀, f.constants.ρ₀, UpdatedMDBC()) ≈ (dry ? zero(T) : -T(400)) atol = T(0.5)
                @test Array(gpu.Velocity)[idx] == zero(SVector{D, T})
                @test Array(sup.boundary_data.velocity)[slot] ≈ (dry ? zero(SVector{D, T}) : SVector{D, T}(ntuple(_ -> -T(0.2), D)))
                tangent = Array(sup.boundary_data.tangent_velocity)[slot]
                @test tangent[D] ≈ zero(T) atol = eps(T)
                @test tangent[1] ≈ (dry ? zero(T) : -T(0.2))
            end
        end
    end

    @testset "Inactive boundaries contribute no mass or forces" begin
        f = updated_mdbc_fixture(Val(2), Float64, Float64)
        for compact in (false, true)
            gpu = upload_particles(f.host[[1, 2]]; boundary_only = compact)
            cl = CellListWorkspace{2, Float64}(2)
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, f.kernel.H⁻¹)
            sup = GPUSupportArrays{2, Float64}(2; updated_mdbc = true, boundary_slots = gpu.BoundarySlot,
                boundary_count = compact ? length(gpu.GhostIndex) : length(gpu))
            fill!(gpu.Density, 1001)
            step = HostStep(1e-4, 0.0)
            fluid = findfirst(==(Fluid), Array(gpu.Type))
            for active in (true, false)
                fill!(sup.boundary_data.active, active)
                launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                    gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
                    cl.CellStart, gpu.CellID, cl.grid_dev, step, ZeroDensityDiffusion(), ZeroViscosity(),
                    f.kernel, f.constants, Val(false), Val(false); boundary_data = sup.boundary_data)
                acceleration = Array(gpu.Acceleration)[fluid]
                rate = Array(sup.dρdtI)[fluid]
                @test iszero(acceleration) == !active
                @test iszero(rate) == !active
            end
        end
    end

    @testset "Updated mDBC simulation stages and graph replay" begin
        for graph in (false, true), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping()), cache in (false, true)
            f = updated_mdbc_fixture(Val(2), Float32, Float64)
            f.metadata.GPUUseGraph = graph
            f.metadata.GPUCacheInteractionFields = cache
            RunSimulation(SimGeometry = f.geometry, SimMetaData = f.metadata, SimConstants = f.constants,
                SimKernel = f.kernel, SimParticles = f.host, SimLogger = SimulationLogger(f.metadata.SaveLocation),
                SimViscosity = Laminar(), SimDensityDiffusion = ZeroDensityDiffusion(), SimTimeStepping = scheme)
            wall = findfirst(==(Fixed), f.host.Type)
            @test all(isfinite, f.host.Density)
            @test f.host.Density[wall] < 1000
            wall_pressure = particle_pressure(f.host.Density[wall], f.host.Type[wall],
                f.constants.c₀, f.constants.ρ₀, UpdatedMDBC())
            @test wall_pressure < 0
            @test wall_pressure ≈ f.constants.c₀^2 * (f.host.Density[wall] - f.constants.ρ₀)
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

    @testset "Galilean-invariant viscous wall velocity and SimMotion acceleration cloning" begin
        GK = SPHExampleGPU.GPUKernels
        CL = SPHExampleGPU.SPHCellList
        PM = SPHExampleGPU.PrescribedMotions
        for T in (Float32, Float64)
            dx = T(0.04)
            d = T(0.04)
            normal_vec = SVector{2, T}(2d, zero(T)) # x_g - x_b along +x
            wall_pos = SVector{2, Float64}(0.0, 0.5)
            ghost_pos = wall_pos + SVector{2, Float64}(normal_vec)
            fluid_pos = [ghost_pos + SVector(0.04 * ix, 0.04 * iy) for ix in 0:2, iy in -2:2] |> vec
            U_wall = SVector{2, T}(T(0.75), -T(0.35))
            # 1. Co-moving uniform velocity must give zero viscous & continuity rates (Galilean invariance).
            geometry = [
                SPHGeometry{2, T}(Particles = particle_struct_array([wall_pos], T(1000);
                    Velocity = [U_wall], GhostPoints = [ghost_pos], GhostNormals = [normal_vec]), GroupMarker = 3, Type = Moving),
                SPHGeometry{2, T}(Particles = particle_struct_array(fluid_pos, T(1000);
                    Velocity = fill(U_wall, length(fluid_pos))), GroupMarker = 2, Type = Fluid),
            ]
            meta = SimulationMetaData{2, T, NoShifting, NoKernelOutput, UpdatedMDBC}(
                SimulationName = "GalileanUpdatedMDBC", SaveLocation = mktempdir(), GPUDoublePosition = true,
                GPUCompactBuffers = true, VisualizeInParaview = false, OpenLogFile = false)
            constants = SimulationConstants{T}(2; dx, c₀ = T(20), g = zero(T), ν₀ = T(0.02))
            kernel = SPHKernelInstance{2, T}(WendlandC2(); dx)
            host = AllocateDataStructures(geometry, meta)
            gpu = upload_particles(host, meta)
            cl = CellListWorkspace{2, Float64}(length(gpu))
            CL.rebuild_cell_list!(gpu, cl, kernel.H⁻¹)
            sup = GPUSupportArrays{2, T}(gpu, meta; kernel)
            step = HostStep(T(1e-4), 0.0)
            launch_pos_cells!(sup.PosCells, gpu.Position, gpu.CellID, cl.grid_dev, step, kernel)
            GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity,
                gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart,
                cl.grid_dev, step, kernel, constants, sup.boundary_data; pos_cells = sup.PosCells,
                compact_ghosts = Val(true))
            @test only(Array(sup.boundary_data.velocity)) ≈ U_wall atol = 20eps(T)
            @test only(Array(sup.boundary_data.tangent_velocity)) ≈ U_wall atol = 20eps(T)
            launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                gpu.Position, gpu.Density, gpu.Velocity, gpu.Type, cl.CellStart, gpu.CellID, cl.grid_dev, step,
                ZeroDensityDiffusion(), ZeroViscosity(), kernel, constants, Val(false), Val(false);
                pos_cells = sup.PosCells, boundary_data = sup.boundary_data, boundary_forces = Val(true))
            acc_inviscid = Array(gpu.Acceleration)
            rate_inviscid = Array(sup.dρdtI)
            launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                gpu.Position, gpu.Density, gpu.Velocity, gpu.Type, cl.CellStart, gpu.CellID, cl.grid_dev, step,
                ZeroDensityDiffusion(), Laminar(), kernel, constants, Val(false), Val(false);
                pos_cells = sup.PosCells, boundary_data = sup.boundary_data, boundary_forces = Val(true))
            @test Array(gpu.Acceleration) ≈ acc_inviscid atol = 20eps(T)
            @test Array(sup.dρdtI) ≈ rate_inviscid atol = 20eps(T)
            @test maximum(abs, rate_inviscid) < 20eps(T) * constants.ρ₀

            # 2. Shear + normal relative velocity: tangential relative part mirrors, normal stays wall normal.
            delta_u = SVector{2, T}(T(0.12), T(0.24))
            host2 = AllocateDataStructures(geometry, meta)
            for k in eachindex(host2.Type)
                host2.Type[k] == Fluid && (host2.Velocity[k] = U_wall + delta_u)
            end
            gpu2 = upload_particles(host2, meta)
            CL.rebuild_cell_list!(gpu2, cl, kernel.H⁻¹)
            launch_pos_cells!(sup.PosCells, gpu2.Position, gpu2.CellID, cl.grid_dev, step, kernel)
            GK.launch_updated_mdbc!(gpu2.Density, gpu2.Position, gpu2.Velocity,
                gpu2.GhostPoints, gpu2.GhostNormals, gpu2.GhostIndex, gpu2.Type, cl.CellStart,
                cl.grid_dev, step, kernel, constants, sup.boundary_data; pos_cells = sup.PosCells,
                compact_ghosts = Val(true))
            @test only(Array(sup.boundary_data.velocity)) ≈ U_wall - delta_u atol = 50eps(T)
            @test only(Array(sup.boundary_data.tangent_velocity)) ≈ SVector{2, T}(U_wall[1], U_wall[2] - delta_u[2]) atol = 50eps(T)

            # 3. Exact prescribed SimMotion acceleration clones advancing (+p) and receding (-p) wall pressure.
            traj = MotionSequence([KinematicSegment(1.0; velocity = (0.2, 0.0), acceleration = (5.0, 0.0))])
            prescribed = PM.prepare_prescribed_motion(PrescribedMotion(3; translation = traj), geometry, host)
            CL.update_rotation_indices!(prescribed, cl)
            GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity,
                gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart,
                cl.grid_dev, HostStep(T(0.01), 0.1), kernel, constants, sup.boundary_data;
                pos_cells = sup.PosCells, groups = gpu.GroupMarker, prescribed_motion = prescribed,
                fraction = 0.5, compact_ghosts = Val(true))
            wall_idx = only(Array(gpu.GhostIndex))
            # Displacement x_b - x_g = -2d along +x, normal = +x, acceleration = +5:
            # Δp = ρ0 * dot(-a, n) * dot(x_b - x_g, n) = 1000 * (-5) * (-2d) = +10000 * d.
            expected_p = T(1000) * T(5.0) * (2d)
            @test particle_pressure(Array(gpu.Density)[wall_idx], Moving, constants.c₀, constants.ρ₀, UpdatedMDBC()) ≈
                expected_p atol = (T === Float32 ? T(0.5) : T(1e-9))
        end

        # 4. End-to-end RunSimulation with UpdatedMDBC + SimMotion across schemes, graphs, and caches.
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping()), cache in (false, true)
            function run_updated_sim_motion(use_graph::Bool)
                f = updated_mdbc_fixture(Val(2), Float32, Float64)
                original = f.geometry[1]
                f.geometry[1] = SPHGeometry{2, Float32}(Particles = original.Particles, GroupMarker = 1, Type = Moving)
                f.metadata.SaveLocation = mktempdir()
                f.metadata.GPUUseGraph = use_graph
                f.metadata.GPUCacheInteractionFields = cache
                f.metadata.GPUClipTimeStepToOutput = true
                f.metadata.SimulationTime = 0.002
                f.metadata.OutputTimes = 0.001
                host = AllocateDataStructures(f.geometry, f.metadata)
                motion = MotionSystem((
                    PrescribedMotion(10; group_marker = nothing, translation = MotionSequence([KinematicSegment(1.0; velocity = (0.1, 0.0), acceleration = (2.0, 0.0))])),
                    PrescribedMotion(1; parent = 10, translation = MotionSequence([KinematicSegment(1.0; velocity = (0.0, -0.05), acceleration = (0.0, -1.5))])),
                ))
                logger = SimulationLogger(f.metadata.SaveLocation)
                result = try
                    RunSimulation(SimGeometry = f.geometry, SimMetaData = f.metadata, SimConstants = f.constants,
                        SimKernel = f.kernel, SimParticles = host, SimLogger = logger,
                        SimViscosity = Laminar(), SimDensityDiffusion = ZeroDensityDiffusion(),
                        SimTimeStepping = scheme, SimMotion = motion)
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                @test result.final_state.complete
                @test result.diagnostics.frozen_particles == 0
                order = sortperm(host.ID)
                wall = findfirst(==(Moving), host.Type)
                @test host.Position[wall] ≈ motion_position(motion, 1, SVector(0.5, 0.0), 0.002) atol = 1e-9
                @test host.Velocity[wall] ≈ SVector{2, Float32}(motion_velocity(motion, 1, SVector(0.5, 0.0), 0.002)) atol = 1f-6
                @test host.GhostPoints[wall] ≈ host.Position[wall] + SVector{2, Float64}(host.GhostNormals[wall]) atol = 1e-6
                return (; pos = host.Position[order], vel = host.Velocity[order], rho = host.Density[order])
            end
            direct = run_updated_sim_motion(false)
            graphed = run_updated_sim_motion(true)
            @test direct == graphed
        end
    end
end
