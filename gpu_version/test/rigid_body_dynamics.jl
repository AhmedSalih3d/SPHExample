module RigidBodyDynamicsTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, HDF5

const F = SPHExampleGPU.GPUFloating
include(joinpath(@__DIR__, "..", "benchmark", "zipline_case.jl"))

@testset "ZipLine reference geometry" begin
    case = prepare_zipline_3d()
    @test length.(getproperty.(case.geometry, :Particles)) == [31049,111,204,35,47,32,26,69,71,40,43]
    @test sum(length(g.Particles) for g in case.geometry) == 31727
    @test all(g -> g.GroupMarker != 1, case.geometry)
    @test count(g -> g.Type == Floating, case.geometry) == 10
    @test case.meta.Periodicity === nothing
    @test length(case.dynamics.joints) == 10
    @test case.dynamics.contacts.fixed_groups == (2,)
    @test_throws ArgumentError prepare_zipline_3d(; dx = 0.02)
end

@testset "native rigid configuration" begin
    @test isbitstype(typeof(BallJoint(1, 2; Anchor = (0, 0, 0))))
    @test PointLineJoint(1; Point = (0, 0, 0), Direction = (2, 0, 0)).axis == SVector(1., 0., 0.)
    @test_throws ArgumentError BallJoint(1, 1; Anchor = (0, 0, 0))
    @test_throws ArgumentError HingeJoint(1; Anchor = (0, 0, 0), Axis = (0, 0, 0))
    @test_throws ArgumentError RigidContacts(ParticleDiameter = 0)
    @test_throws ArgumentError RigidContacts(ParticleDiameter = 1, Slop = 1)
    @test_throws ArgumentError RigidContacts(ParticleDiameter = 1, Friction = -1)
    @test_throws ArgumentError RigidBodyDynamics(Iterations = 0)
    @test_throws ArgumentError RigidBodyDynamics(MaxTimeStep = 0)
    @test_throws ArgumentError RigidBodyDynamics(MaxTimeStep = Inf)
end

function cube(center; spacing = 0.05)
    return [SVector{3,Float64}(center) + spacing * SVector(i, j, k) / 2
        for i in (-1, 1), j in (-1, 1), k in (-1, 1)][:]
end

function simulate(geometry, dynamics, directory; T = Float64, duration = 0.1, graph = false, gravity = 9.81, dx = 0.05, scheme = SymplecticTimeStepping(), cfl = 0.2)
    mkpath(directory)
    constants = SimulationConstants{T}(3; dx, g = gravity, c₀ = 20, CFL = cfl)
    meta = SimulationMetaData{3,T}(SimulationName = "RigidDynamics", SaveLocation = directory,
        SimulationTime = duration, OutputTimes = duration, VisualizeInParaview = false,
        OpenLogFile = false, GPUUseGraph = graph, GPUDoublePosition = true, GPUBoundaryForces = false)
    particles = AllocateDataStructures(geometry, meta)
    logger = SimulationLogger(directory)
    try
        return RunSimulation(SimGeometry = geometry, SimMetaData = meta, SimParticles = particles,
            SimConstants = constants, SimKernel = SPHKernelInstance{3,T}(WendlandC2(); h = T(2dx)),
            SimLogger = logger, SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion(),
            SimTimeStepping = scheme, SimRigidBodyDynamics = dynamics)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "ZipLine graph/direct state and CAD motion output" begin
        mktempdir() do directory
            results = [redirect_stdout(devnull) do
                run_zipline_3d(; simulation_time = 0.04, output_interval = 0.02,
                    save_location = joinpath(directory, string(graph)), use_graph = graph,
                    visualize = false, open_log_file = false)
            end for graph in (false, true)]
            a, b = results
            @test a.performance.steps == b.performance.steps
            @test a.final_state.floating.center == b.final_state.floating.center
            @test a.final_state.floating.velocity == b.final_state.floating.velocity
            @test a.final_state.floating.orientation == b.final_state.floating.orientation
            for field in (:Position, :Velocity, :Density)
                ap, bp = a.final_state.particles, b.final_state.particles
                @test getproperty(ap, field)[sortperm(ap.ID)] == getproperty(bp, field)[sortperm(bp.ID)]
            end
            @test b.configuration.execution.graph_batches > 0
            case = prepare_zipline_3d()
            path = Base.invokelatest(case.save_surfaces, joinpath(directory, "true"), case.geometry)
            h5open(path, "r") do io
                @test haskey(io, "VTKHDF/RagdollSurfaces/Points")
                @test read(io["VTKHDF/RagdollSurfaces/Steps/Values"]) == [0, 0.02, 0.04]
                points = read(io["VTKHDF/RagdollSurfaces/Points"])
                n = first(read(io["VTKHDF/RagdollSurfaces/NumberOfPoints"]))
                @test n == 127 # shared vertices copied from GenCase's actual mesh
                @test all(isfinite, points)
                @test norm(points[:,1:n] - points[:,end-n+1:end]) > 0.001
                fields = io["VTKHDF/FloatingBodies/PointData"]
                gravity = read(fields["GravityForce"])
                reaction = read(fields["ConstraintForce"])
                total = read(fields["TotalForce"])
                @test all(iszero, gravity[1:2,:])
                @test all(<(0), gravity[3,:])
                @test any(!iszero, reaction)
                @test total ≈ gravity + reaction rtol = 2e-6 atol = 1e-8
                @test all(iszero, read(fields["Force"]))
            end
            script = read(joinpath(directory, "true", "ZipLine3D_SingleVTKHDFStateFile.py"), String)
            @test occursin("Analytical sliding axis (visual guide)", script)
            @test occursin("Joint and contact reaction (red)", script)
        end
    end

    @testset "off-centre world hinge swings under gravity" begin
        center, anchor = SVector(0.2, 0., 1.), SVector(0., 0., 1.)
        geometry = [SPHGeometry{3,Float64}(cube(center); Density = 1000, GroupMarker = 1, Type = Floating,
            Floating = FloatingDetails{Float64}(RelativeWeight = 1))]
        dynamics = RigidBodyDynamics(Joints = (HingeJoint(1; Anchor = anchor, Axis = (0, 1, 0)),))
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            mktempdir() do directory
                result = simulate(geometry, dynamics, directory; duration = 0.12, graph = true, scheme)
                state = result.final_state.floating
                errors = only(rigid_joint_errors(dynamics, geometry, state))
                @test errors.translation_error < 1e-7
                @test errors.rotation_error < 1e-7
                @test state.center[1][3] < 0.99
                @test state.omega[1][2] > 0.1
                @test abs(state.omega[1][1]) + abs(state.omega[1][3]) < 1e-8
                @test norm(state.center[1] - anchor) ≈ 0.2 atol = 1e-7
            end
        end
    end

    @testset "internal ball and hinge impulses conserve momentum" begin
        for constructor in (BallJoint, HingeJoint)
            centers = (SVector(-0.1, 0., 1.), SVector(0.1, 0., 1.))
            initial_velocity = (SVector(0., -0.2, 0.), SVector(0., 0.1, 0.))
            geometry = [SPHGeometry{3,Float64}(cube(c); Density = 1000, GroupMarker = b, Type = Floating,
                Floating = FloatingDetails{Float64}(RelativeWeight = b, InitialVelocity = initial_velocity[b])) for (b,c) in enumerate(centers)]
            joint = constructor === BallJoint ? BallJoint(1, 2; Anchor = (0, 0, 1)) : HingeJoint(1, 2; Anchor = (0, 0, 1), Axis = (0, 0, 1))
            dynamics = RigidBodyDynamics(Joints = (joint,))
            mass = [1000 * 0.05^3 * 8b for b in 1:2]
            inertia = [m * 0.05^2 / 2 * one(SMatrix{3,3,Float64,9}) for m in mass]
            initial_angular = sum(cross(centers[b], mass[b] * initial_velocity[b]) for b in 1:2)
            mktempdir() do directory
                result = simulate(geometry, dynamics, directory; duration = 0.1, gravity = 0, graph = true)
                state = result.final_state.floating
                linear = sum(mass .* state.velocity)
                angular = sum(cross(state.center[b], mass[b] * state.velocity[b]) +
                    F.rotate_vector(state.orientation[b], inertia[b] * F.rotate_vector(F.quaternion_conjugate(state.orientation[b]), state.omega[b])) for b in 1:2)
                @test norm(linear) < 1e-9
                @test angular ≈ initial_angular rtol = 0.005 atol = 1e-9
                @test only(rigid_joint_errors(dynamics, geometry, state)).translation_error < 1e-7
                @test norm(state.omega[1]) + norm(state.omega[2]) > 0.1
            end
        end
    end
    @testset "point-on-line analytical inclined slide" begin
        T = Float64
        center = SVector(0., 0., 1.)
        geometry = [SPHGeometry{3,T}(cube(center); Density = 1000, GroupMarker = 1, Type = Floating,
            Floating = FloatingDetails{T}(RelativeWeight = 1))]
        for ceiling in (nothing, 0.002)
            dynamics = RigidBodyDynamics(Joints = (PointLineJoint(1; Point = center, Direction = (1, 0, -1)),), MaxTimeStep = ceiling)
            mktempdir() do directory
                result = simulate(geometry, dynamics, directory)
                state, time = result.final_state.floating, result.final_state.time
                @test result.final_state.complete
                @test state.center[1] ≈ center + SVector(1., 0., -1.) * (9.81time^2 / 4) atol = 1e-8
                @test state.velocity[1] ≈ SVector(1., 0., -1.) * (9.81time / 2) atol = 1e-8
                @test norm(state.omega[1]) < 1e-9
                @test state.orientation[1] ≈ SVector(1., 0., 0., 0.) atol = 1e-9
                @test ceiling === nothing || result.performance.steps < 55
            end
        end
    end

    @testset "native particle contact supports a resting body" begin
        T = Float64
        floor = [SVector(x, y, 0.0) for x in -0.45:0.1:0.45 for y in -0.45:0.1:0.45]
        geometry = [SPHGeometry{3,T}(floor; Density = 1000, GroupMarker = 1, Type = Fixed),
            SPHGeometry{3,T}(cube((0, 0, 0.15); spacing = 0.1); Density = 1000, GroupMarker = 2, Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = 1))]
        contacts = RigidContacts(ParticleDiameter = 0.1, Margin = 0.1, Friction = 0.5)
        dynamics = RigidBodyDynamics(Contacts = contacts, Iterations = 40)
        mktempdir() do directory
            result = simulate(geometry, dynamics, directory; duration = 0.1, dx = 0.1)
            state = result.final_state.floating
            @test result.final_state.complete
            @test state.center[1] ≈ SVector(0., 0., 0.15) atol = 1e-6
            @test norm(state.velocity[1]) < 1e-5
            @test norm(state.omega[1]) < 1e-4
        end
    end

    @testset "Coulomb sticking on an inclined particle plane" begin
        T = Float64
        angle = deg2rad(20.0)
        rotation = @SMatrix [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)]
        floor = [rotation * SVector(x, y, 0.0) for x in -0.45:0.1:0.45 for y in -0.45:0.1:0.45]
        center = rotation * SVector(0.0, 0.0, 0.15)
        geometry = [SPHGeometry{3,T}(floor; Density = 1000, GroupMarker = 1, Type = Fixed),
            SPHGeometry{3,T}([rotation * p for p in cube((0, 0, 0.15); spacing = 0.1)];
                Density = 1000, GroupMarker = 2, Type = Floating, Floating = FloatingDetails{T}(RelativeWeight = 1))]
        dynamics = RigidBodyDynamics(Contacts = RigidContacts(ParticleDiameter = 0.1, Friction = 0.7), Iterations = 64)
        mktempdir() do directory
            result = simulate(geometry, dynamics, directory; duration = 0.1, dx = 0.1)
            state = result.final_state.floating
            @test norm(state.center[1] - center) < 1e-5
            @test norm(state.velocity[1]) < 1e-4
            @test norm(state.omega[1]) < 1e-3
            mass = 1000 * 0.1^3 * 8
            @test result.final_state.rigid.reaction_force[1] ≈ SVector(0., 0., mass * 9.81) rtol = 1e-3
        end
    end

    @testset "restitution of a dropped body" begin
        T = Float64
        floor = [SVector(x, y, 0.0) for x in -0.45:0.1:0.45 for y in -0.45:0.1:0.45]
        geometry = [SPHGeometry{3,T}(floor; Density = 1000, GroupMarker = 1, Type = Fixed),
            SPHGeometry{3,T}(cube((0, 0, 0.5); spacing = 0.1); Density = 1000, GroupMarker = 2, Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = 1))]
        restitution = 0.6
        impact_time = sqrt(2 * (0.5 - 0.15) / 9.81)
        apex_time = impact_time * (1 + restitution)
        expected_height = 0.15 + restitution^2 * (0.5 - 0.15)
        dynamics = RigidBodyDynamics(Contacts = RigidContacts(ParticleDiameter = 0.1,
            Restitution = restitution, Friction = 0.5), Iterations = 48)
        mktempdir() do directory
            result = simulate(geometry, dynamics, directory; duration = apex_time, dx = 0.1)
            @test result.final_state.floating.center[1][3] ≈ expected_height atol = 0.006
            @test abs(result.final_state.floating.velocity[1][3]) < 0.08
            @test norm(result.final_state.floating.omega[1]) < 0.01
        end
    end

    @testset "coupled hinges and balls preserve total translation" begin
        for T in (Float32, Float64), graph in (false, true)
            centers = (SVector(-0.1, 0., 1.), SVector(0.1, 0., 1.))
            geometry = [SPHGeometry{3,T}(cube(c); Density = 1000, GroupMarker = b, Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = b, InitialVelocity = (0.2, -0.1, 0.3))) for (b, c) in enumerate(centers)]
            dynamics = RigidBodyDynamics(Joints = (HingeJoint(1, 2; Anchor = (0, 0, 1), Axis = (0, 0, 1)),))
            mktempdir() do directory
                result = simulate(geometry, dynamics, directory; T, duration = 0.03, graph, gravity = 0)
                state, time = result.final_state.floating, result.final_state.time
                for b in 1:2
                    @test state.center[b] ≈ centers[b] + SVector(0.2, -0.1, 0.3) * time atol = 2e-6
                    @test state.velocity[b] ≈ SVector(0.2, -0.1, 0.3) atol = 2e-6
                end
                errors = rigid_joint_errors(dynamics, geometry, state)
                @test errors[1].translation_error < 2e-6
                @test errors[1].rotation_error < 2e-6
                @test !graph || result.configuration.execution.graph_batches > 0
            end
        end
    end

    @testset "two-body impact conserves momentum and rebounds" begin
        geometry = [SPHGeometry{3,Float64}(cube((x, 0, 1); spacing = 0.1);
            Density = 1000, GroupMarker = b, Type = Floating,
            Floating = FloatingDetails{Float64}(RelativeWeight = 1, InitialVelocity = (b == 1 ? 1. : -1., 0., 0.)))
            for (b,x) in enumerate((-0.25, 0.25))]
        dynamics = RigidBodyDynamics(Contacts = RigidContacts(ParticleDiameter = 0.1,
            Friction = 0, Restitution = 0.8), Iterations = 48, MaxTimeStep = 0.002)
        mktempdir() do directory
            result = simulate(geometry, dynamics, directory; duration = 0.2, dx = 0.1, gravity = 0, graph = true)
            state = result.final_state.floating
            @test sum(state.velocity) ≈ zeros(3) atol = 1e-10
            @test state.velocity[1] ≈ SVector(-0.8, 0., 0.) atol = 1e-3
            @test state.velocity[2] ≈ SVector(0.8, 0., 0.) atol = 1e-3
            @test maximum(norm, state.omega) < 1e-3
            @test state.center[1][1] ≈ -0.14 atol = 0.004
        end
    end

    @testset "native free body retains fluid coupling" begin
        geometry = [SPHGeometry{3,Float64}(cube((0, 0, 0.3)); Density = 1001, GroupMarker = 1, Type = Fluid),
            SPHGeometry{3,Float64}(cube((0, 0, 0.15)); Density = 1000, GroupMarker = 2, Type = Floating,
                Floating = FloatingDetails{Float64}(RelativeWeight = 1))]
        mktempdir() do directory
            a = simulate(geometry, nothing, joinpath(directory, "original"); duration = 0.004, gravity = 0)
            b = simulate(geometry, RigidBodyDynamics(), joinpath(directory, "native"); duration = 0.004, gravity = 0)
            @test norm(only(b.final_state.floating.velocity)) > 1e-6
            @test only(a.final_state.floating.center) ≈ only(b.final_state.floating.center) atol = 1e-9
            for field in (:Position, :Velocity, :Density)
                ap, bp = a.final_state.particles, b.final_state.particles
                @test getproperty(ap, field)[sortperm(ap.ID)] ≈ getproperty(bp, field)[sortperm(bp.ID)] atol = 1e-8
            end
            @test_throws ArgumentError simulate(geometry, RigidBodyDynamics(MaxTimeStep = 0.002),
                joinpath(directory, "invalid"); duration = 0.004, gravity = 0)
        end
    end
end

end
