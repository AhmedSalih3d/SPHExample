module MDBCValidationTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "mdbc_validation.jl"))
using .MDBCValidation

@testset "Tank and square ghost offsets lie strictly in fluid and preserve symmetry" begin
    for n in (20, 40, 60)
        dx = 1.0 / n
        case_simple = build_mdbc_case(; reynolds = 100, dx, boundary_mode = SimpleMDBC)
        case_updated = build_mdbc_case(; reynolds = 100, dx, boundary_mode = UpdatedMDBC)
        @test !case_simple.metadata.GPULegacyTimestep
        @test !case_updated.metadata.GPULegacyTimestep
        @test case_updated.counts == (fixed = 4 * (10n + 5n + 4 * case_updated.layers) * case_updated.layers ÷ 2,
            fluid = 49 * n^2, square = n^2) || (case_updated.counts.fluid == 49 * n^2 && case_updated.counts.square == n^2)
        for p in case_updated.geometry[1].Particles
            g = p.Position + SVector{2,Float64}(p.GhostNormals)
            @test p.GhostPoints ≈ g atol = 1e-6
            @test 0.0 < g[1] < 10.0 && 0.0 < g[2] < 5.0
            @test !(1.0 <= g[1] <= 2.0 && 2.0 <= g[2] <= 3.0)
        end
        center = SVector(1.5, 2.5)
        for p in case_updated.geometry[3].Particles
            g = p.Position + SVector{2,Float64}(p.GhostNormals)
            @test p.GhostPoints ≈ g atol = 1e-6
            @test 0.0 < g[1] < 10.0 && 0.0 < g[2] < 5.0
            @test max(abs(g[1] - center[1]), abs(g[2] - center[2])) > 0.5
            local_p = p.Position - center
            mirrored_y = SVector(local_p[1], -local_p[2])
            off1 = square_ghost_offset(local_p)
            off2 = square_ghost_offset(mirrored_y)
            @test off1[1] ≈ off2[1] atol = 1e-14
            @test off1[2] ≈ -off2[2] atol = 1e-14
        end
    end
end

if CUDA.functional()
    @testset "mDBC force postprocessing agrees with GPU solver in Float32 and Float64 (including suction & acceleration)" begin
        GK = SPHExampleGPU.GPUKernels
        CL = SPHExampleGPU.SPHCellList
        PM = SPHExampleGPU.PrescribedMotions
        for T in (Float32, Float64), B in (SimpleMDBC, UpdatedMDBC)
            dx = 0.05
            case = build_mdbc_case(; reynolds = 50, dx, duration = 0.1, interval = 0.05,
                sound_speed = 40.0, boundary_mode = B, float_type = T, save = mktempdir())
            host = AllocateDataStructures(case.geometry, case.metadata)
            # Impose asymmetric wake suction (rho < rho0 behind square, rho > rho0 ahead) + nonzero shear & wall motion!
            u_wall = SVector{2,T}(T(0.8), zero(T))
            acc_wall = SVector{2,Float64}(2.5, 0.0)
            traj = MotionSequence([KinematicSegment(1.0; velocity = (0.8, 0.0), acceleration = Tuple(acc_wall))])
            for i in eachindex(host.Type)
                if host.Type[i] == Fluid
                    x, y = host.Position[i]
                    host.Density[i] = T(1000.0 + 3.5 * (x - 1.5) + 0.5 * (y - 2.5))
                    host.Velocity[i] = SVector{2,T}(T(0.15 * (y - 2.5)), T(-0.1 * (x - 1.5)))
                elseif host.GroupMarker[i] == 3
                    host.Velocity[i] = u_wall
                end
            end
            meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,B}(
                SimulationName = "MDBCForceParity", SaveLocation = mktempdir(),
                GPUDoublePosition = true, GPUBoundaryForces = true,
                GPUCellSubdivision = 1, GPUXCellSubdivision = 1, GPUInteractionFastMath = false)
            gpu = upload_particles(host, meta)
            sup = GPUSupportArrays{2,T}(gpu, meta; kernel = case.kernel)
            cl = CellListWorkspace{2,Float64}(length(gpu))
            prescribed = PM.prepare_prescribed_motion(PrescribedMotion(3; translation = traj), case.geometry, host)
            CL.rebuild_cell_list!(gpu, cl, case.kernel.H⁻¹; support = sup, constants = case.constants,
                kernel = case.kernel, boundary_mode = B(), rotation = prescribed)
            step = HostStep(T(1e-4), 0.0)
            launch_pos_cells!(sup.PosCells, gpu.Position, gpu.CellID, cl.grid_dev, step, case.kernel)
            if B === UpdatedMDBC
                GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity,
                    gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart,
                    cl.grid_dev, step, case.kernel, case.constants, sup.boundary_data;
                    pos_cells = sup.PosCells, groups = gpu.GroupMarker, prescribed_motion = prescribed,
                    fraction = 0.0, compact_ghosts = Val(gpu.BoundarySlot !== nothing))
            else
                launch_mdbc!(gpu.Density, gpu.Position, gpu.GhostPoints, gpu.GhostIndex, gpu.Type,
                    cl.CellStart, cl.grid_dev, step, case.kernel, case.constants;
                    pos_cells = sup.PosCells, ghost_normals = gpu.GhostNormals,
                    compact_ghosts = Val(gpu.BoundarySlot !== nothing))
            end
            function square_gpu_force(visc)
                launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
                    sup.∇Cᵢ, sup.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
                    cl.CellStart, gpu.CellID, cl.grid_dev, step, ZeroDensityDiffusion(), visc,
                    case.kernel, case.constants, Val(false), Val(false);
                    lanes = Val(1), boundary_forces = Val(true), pos_cells = sup.PosCells,
                    boundary_data = sup.boundary_data, fastmath = false)
                acc = Array(gpu.Acceleration)
                mk = Array(gpu.GroupMarker)
                # Internal square-square pair forces cancel identically; tank walls are > H away.
                return sum(Float64(case.constants.m₀) .* SVector{2,Float64}(acc[k]) for k in eachindex(acc) if mk[k] == 3)
            end
            pos64 = [SVector{2,Float64}(p) for p in Array(gpu.Position)]
            vel64 = [SVector{2,Float64}(v) for v in Array(gpu.Velocity)]
            rho64 = Float64.(Array(gpu.Density))
            ty64 = Int.(Array(gpu.Type))
            mk64 = Int.(Array(gpu.GroupMarker))
            c64 = SimulationConstants{Float64}(2; dx, ρ₀ = 1000.0, c₀ = 40.0, g = 0.0, ν₀ = Float64(case.constants.ν₀))
            k64 = SPHKernelInstance{2,Float64}(WendlandC2(); h = Float64(case.kernel.h),
                αD = Float64(case.kernel.αD), η² = Float64(case.kernel.η²))
            loads = mdbc_frame_loads(pos64, vel64, rho64, ty64, mk64, k64, c64;
                boundary_mode = B(), center = SVector(1.5, 2.5), wall_acceleration = acc_wall)

            gpu_p = square_gpu_force(ZeroViscosity())
            gpu_tot = square_gpu_force(Laminar())
            gpu_v = gpu_tot - gpu_p
            tol = T === Float32 ? 2e-3 : 1e-10
            @test gpu_p ≈ loads.force_pressure rtol = tol
            @test gpu_v ≈ loads.force_viscous rtol = tol
        end
    end
end

@testset "Analytical suction-acceleration pressure and Couette Galilean invariance" begin
    variants = MDBCValidation.moving_square_variants()
    @test length(variants) == 35
    @test length(unique(v.id for v in variants)) == 35

    fix_dbc = MDBCValidation.build_submerged_square_fixture(; dx = 0.05, g = 0.0, ax = 1.0, boundary_mode = NoMDBC)
    loads_dbc = mdbc_frame_loads(fix_dbc.position, fix_dbc.velocity, fix_dbc.density, fix_dbc.types, fix_dbc.markers,
        fix_dbc.kernel, fix_dbc.constants; boundary_mode = NoMDBC(), center = fix_dbc.center, wall_acceleration = SVector(1.0, 0.0))
    fix_upd = MDBCValidation.build_submerged_square_fixture(; dx = 0.05, g = 0.0, ax = 1.0, boundary_mode = UpdatedMDBC)
    loads_upd = mdbc_frame_loads(fix_upd.position, fix_upd.velocity, fix_upd.density, fix_upd.types, fix_upd.markers,
        fix_upd.kernel, fix_upd.constants; boundary_mode = UpdatedMDBC(), center = fix_upd.center, wall_acceleration = SVector(1.0, 0.0))
    @test abs(loads_upd.force_pressure[1] - 1000.0) / 1000.0 < 0.01
    @test abs(loads_dbc.force_pressure[1] - 1000.0) / 1000.0 > 0.40

    for mode in (NoMDBC, SimpleMDBC, UpdatedMDBC)
        rest = MDBCValidation.couette_wall_shear(20, mode; boost = SVector(0.0, 0.0))
        boosted = MDBCValidation.couette_wall_shear(20, mode; boost = SVector(0.65, 0.40))
        @test abs(boosted.tau_top - rest.tau_top) < 1e-11
        @test abs(boosted.normal_top - rest.normal_top) < 1e-12
    end
end

end
