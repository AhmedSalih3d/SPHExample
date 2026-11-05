module DemolisherTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, SHA, HDF5
include(joinpath(@__DIR__,"..","benchmark","demolisher_case.jl"))
include(joinpath(@__DIR__,"..","benchmark","demolisher_validation.jl"))

@testset "Demolisher GenCase lattice and collision model" begin
    case = prepare_demolisher_3d()
    @test length.(getproperty.(case.geometry,:Particles)) == [67541,161,161,161,160,1839,fill(1331,16)...]
    @test sum(length(g.Particles) for g in case.geometry) == 91319
    @test case.meta.Periodicity === nothing
    @test all(g -> g.Type != Fluid,case.geometry)
    @test length(case.dynamics.joints) == 5
    @test all(j -> j.kind == 0,case.dynamics.joints)
    @test case.dynamics.contacts.fixed_groups == (11,)
    @test case.dynamics.contacts.floating_groups == (5,(52:67)...)
    @test_throws ArgumentError prepare_demolisher_3d(;dx = 0.025)
    lattice = sort!([(Int(g.GroupMarker), (round(Int,(p[d]+0.5)/0.05) for d in 1:3)...)
        for g in case.geometry for p in g.Particles.Position])
    @test bytes2hex(sha256(join((join(row,",")*"\n" for row in lattice)))) ==
        "4b3effa637621aa835872ded5408e69e38f6cf6a163bddcc23462debad7d415c"
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "Demolisher material pairs and inactive arm contacts" begin
        case = prepare_demolisher_3d()
        particles = AllocateDataStructures(case.geometry,case.meta)
        F = SPHExampleGPU.GPUFloating
        floating = F.FloatingArrays(case.geometry,particles,case.constants;position_type = Float64)
        native = F.attach_rigid_dynamics(floating,case.dynamics,particles,case.kernel;gravity = case.constants.g)
        @test Array(native.mechanics.floating_enabled) == [0,0,0,0,fill(1,17)...]
        @test length(native.mechanics.pair_order) == 153
        pairs = Array(native.mechanics.pairs)
        for index in Array(native.mechanics.pair_order)
            a,b = pairs[index]
            @test a >= 5 && (b == 0 || b >= 5)
        end
        friction, restitution = Array(native.mechanics.contact_friction),Array(native.mechanics.contact_restitution)
        @test friction[F.rigid_pair_index(Int32(6),Int32(5))] ≈ 0.45f0
        @test restitution[F.rigid_pair_index(Int32(6),Int32(5))] ≈ 0.5f0
        @test friction[F.rigid_pair_index(Int32(7),Int32(6))] ≈ 0.75f0
        @test restitution[F.rigid_pair_index(Int32(5),Int32(0))] ≈ 0.8f0
        @test Array(native.mass)[1:5] ≈ Float32[24.15,24.15,24.15,24.,574.6875] rtol = 2e-6
        bad = RigidBodyDynamics(Contacts = RigidContacts(ParticleDiameter = 0.05,Margin = 0.02,
            FloatingGroups = (11,)))
        @test_throws ArgumentError F.attach_rigid_dynamics(floating,bad,particles,case.kernel)
    end

    @testset "Demolisher graph/direct state" begin
        mktempdir() do directory
            results = [redirect_stdout(devnull) do
                run_demolisher_3d(;simulation_time = 0.04,output_interval = 0.02,
                    save_location = joinpath(directory,string(graph)),use_graph = graph,
                    visualize = false,open_log_file = false)
            end for graph in (false,true)]
            @test results[1].performance.steps == results[2].performance.steps
            for field in (:center,:velocity,:orientation,:omega)
                @test getproperty(results[1].final_state.floating,field) == getproperty(results[2].final_state.floating,field)
            end
            a,b = results[1].final_state.particles,results[2].final_state.particles
            @test a.Position[sortperm(a.ID)] == b.Position[sortperm(b.ID)]
        end
    end

    @testset "linked ball knocks movable blocks" begin
        mktempdir() do directory
            case = prepare_demolisher_3d()
            result = redirect_stdout(devnull) do
                run_demolisher_3d(;simulation_time = 1.8,output_interval = 0.02,
                    save_location = directory,visualize = false,open_log_file = false)
            end
            metrics = demolisher_validation(case,directory)
            @test result.final_state.complete
            @test result.diagnostics.frozen_particles == 0
            @test metrics.frames == 91
            @test metrics.max_joint_error_m < 1e-4
            @test metrics.early_block_horizontal_drift_m < 0.01
            @test 0.5 < metrics.first_block_horizontal_motion_s < 1.6
            @test metrics.displaced_blocks >= 3
            @test metrics.max_body_speed_m_s < 30
            @test metrics.max_relative_energy_growth < 0.02
            @test metrics.minimum_block_corner_height_m > -0.03
            @test metrics.max_sphere_box_overlap_m < 0.06
            vtk = Base.invokelatest(case.save_surfaces,directory,case.geometry)
            h5open(vtk,"r") do io
                @test first(read(io["VTKHDF/DemolisherSurfaces/NumberOfPoints"])) == 878
                @test length(read(io["VTKHDF/DemolisherSurfaces/Steps/Values"])) == 91
                @test length(unique(read(io["VTKHDF/DemolisherSurfaces/CellData/Region"]))) == 21
            end
            script = read(joinpath(directory,"Demolisher3D_SingleVTKHDFStateFile.py"),String)
            @test occursin("Wrecking ball forces [N]",script)
        end
    end
end
end # module
