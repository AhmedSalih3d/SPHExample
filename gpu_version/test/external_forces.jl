module ExternalForcesTests

using Test, SPHExampleGPU, StaticArrays, CUDA, Adapt

const EF = SPHExampleGPU.ExternalForces
const STEP = SPHExampleGPU.GPUStepState

@testset "external force histories" begin
    for T in (Float32, Float64)
        force = ExternalForce{3,T}(3; times = [0, 1],
            linear_acceleration = [(0, 0, 0), (0, 0, 4)],
            angular_acceleration = [(0, 0, 0), (0, 2, 0)], center = (1, 0, 1))
        x, v = SVector{3,T}(2, 0, 3), SVector{3,T}(4, 0, 5)
        # At t=0.5: linear=(0,0,2), alpha=omega=(0,1,0), r=(1,0,2).
        # Tangential=(2,0,-1), centripetal=(-1,0,-2), reference Coriolis=(10,0,-8).
        @test external_acceleration(force, x, v, T(0.5), T(9.81)) ≈ T[11, 0, -9]
        @test force.linear_velocity[end] == SVector{3,T}(0, 0, 4)
        @test force.angular_velocity[end] == SVector{3,T}(0, 2, 0)
        @test external_acceleration(force, x, v, T(2), T(9.81)) == external_acceleration(force, x, v, one(T), T(9.81))
        @test iszero(external_acceleration(force, x, v, -one(T), T(9.81)))

        # Match the reference's angular gate even when integrated omega is nonzero.
        gate = ExternalForce{3,T}(3; times = [0, 1, 2], angular_acceleration = [(0, 0, 0), (0, 2, 0), (0, 0, 0)])
        @test !iszero(gate.angular_velocity[end])
        @test iszero(external_acceleration(gate, x, v, T(2), T(9.81)))

        f2 = ExternalForce{2,T}(3; times = [0, 1], linear_acceleration = [(0, 0), (0, 4)],
            angular_acceleration = [0, -2], center = (1, 1))
        @test external_acceleration(f2, SVector{2,T}(2, 3), SVector{2,T}(4, 5), T(0.5), T(9.81)) ≈ T[11, -9]
        for D in (2, 3)
            p = zero(SVector{D,T})
            f = ExternalForce{D,T}(3; times = [1, 2], global_gravity = false, start_time = 0.5, end_time = 3)
            expected = SVector{D,T}(ntuple(d -> d == D ? T(9.81) : zero(T), D))
            @test external_acceleration(f, p, p, T(0.5), T(9.81)) == expected
            @test external_acceleration(f, p, p, T(3), T(9.81)) == expected
            @test iszero(external_acceleration(f, p, p, T(0.49), T(9.81)))
            @test iszero(external_acceleration(f, p, p, T(3.01), T(9.81)))
        end

        # Preserve subtraction of large world coordinates before reducing to Float32.
        precise = ExternalForce{3,T}(3; times = [0, 1], center = (1e8 + 1, 0, 0),
            angular_acceleration = [(0, 0, 1), (0, 0, 1)])
        @test external_acceleration(precise, SVector(1e8 + 3, 0.0, 0.0), zero(SVector{3,T}), zero(T), zero(T)) == SVector{3,T}(0, 2, 0)
    end

    @test_throws ArgumentError ExternalForce{1,Float32}(1; times = [0, 1])
    for times in ([0], [0, 0], [1, 0], [-1, 0], [0, Inf], [0, NaN])
        @test_throws ArgumentError ExternalForce{2,Float32}(1; times)
    end
    @test_throws DimensionMismatch ExternalForce{2,Float32}(1; times = [0, 1], linear_acceleration = [(0, 0)])
    @test_throws DimensionMismatch ExternalForce{2,Float32}(1; times = [0, 1], center = (0, 0, 0))
    @test_throws ArgumentError ExternalForce{2,Float32}(1; times = [0, 1], angular_acceleration = [(0,), (1,)])
    @test_throws ArgumentError ExternalForce{2,Float32}(1; times = [0, 1], linear_acceleration = [(0, 0), (Inf, 0)])
    @test_throws ArgumentError ExternalForce{2,Float32}(1; times = [0, 1], start_time = 2, end_time = 1)
    history = ExternalForce{2,Float32}(3; times = [0, 1])
    @test_throws ArgumentError EF.prepare_external_forces(history, (GroupMarker = [3], Type = [Fixed]), Val(2), Float32)
    @test_throws ArgumentError EF.prepare_external_forces(history, (GroupMarker = [3], Type = [Fluid]), Val(3), Float32)

    mktempdir() do dir
        path = joinpath(dir, "acceleration.csv")
        write(path, "#Time;ax;ay;az;alpha_x;alpha_y;alpha_z\n0;0;0;0;0;0;0\n1;0;0;4;0;2;0\n")
        f3 = ExternalForce{3,Float64}(3, path; center = (1, 0, 1))
        f2 = ExternalForce{2,Float64}(3, path; center = (1, 1))
        @test external_acceleration(f3, SVector(2., 0., 3.), SVector(4., 0., 5.), 0.5, 9.81) == SVector(11., 0., -9.)
        @test external_acceleration(f2, SVector(2., 3.), SVector(4., 5.), 0.5, 9.81) == SVector(11., -9.)
        write(path, "0;0;1;0;0;0;0\n1;0;0;0;0;0;0\n")
        @test_throws ArgumentError ExternalForce{2,Float64}(3, path)
        write(path, "0;0;0\n1;0;0\n")
        @test_throws ArgumentError ExternalForce{3,Float64}(3, path)
    end
end

if CUDA.functional()
    @testset "shared force samples match per-particle full and half stages" begin
        for D in (2, 3), T in (Float32, Float64), TP in (Float32, Float64)
            V = SVector{D,T}
            points = [SVector{D,TP}(ntuple(d -> 1 + i + d / 8, D)) for i in 1:5]
            velocities = [V(ntuple(d -> i - d / 4, D)) for i in 1:5]
            types, groups = [Fluid, Fluid, Fixed, Frozen, Fluid], [3, 4, 3, 3, 3]
            force = ExternalForce{D,T}(3; times = [0.1, 0.3, 0.7],
                linear_acceleration = [ntuple(d -> k*d / 7, D) for k in 1:3],
                angular_acceleration = D == 2 ? [1, -2, 0] : [(1, 2, 3), (-2, 3, -1), (0, 0, 0)],
                center = ntuple(d -> d / 8, D), global_gravity = false, start_time = 0.05, end_time = 0.8)
            host = (Type = fill(Fluid, 5), GroupMarker = groups)
            prepared = EF.prepare_external_forces(force, host, Val(D), T; sample_once = true)
            legacy = (Adapt.adapt(CuArray, force),)
            p, v, ty, gm = CuArray(points), CuArray(velocities), CuArray(types), CuArray(groups)
            for time in T.((0, 0.05, 0.1, 0.3, 0.7, 0.8, 0.9)), half_time in (false, true)
                state = StepState{T}(; time, dt = T(0.13))
                initial = fill(one(T) * V(ntuple(identity, D)), 5)
                a, b = CuArray(initial), CuArray(initial)
                EF.launch_external_forces!(legacy, a, p, v, ty, gm, state, T(9.81); half_time)
                EF.launch_external_forces!(prepared, b, p, v, ty, gm, state, T(9.81); half_time)
                @test Array(a) == Array(b)
                before = (Array(b), Array(prepared[1].sample))
                STEP.set_slot!(state.i, state.ih, STEP.I_STOP, STEP.STOP_REBUILD)
                EF.launch_external_forces!(prepared, b, p, v, ty, gm, state, T(9.81); half_time)
                @test (Array(b), Array(prepared[1].sample)) == before
            end
            # Host callers may sample Float32 histories with a Float64 time.
            @test all(isfinite, external_acceleration(force, points[1], velocities[1], 0.7, T(9.81)))
            @test EF.sample_force(force, 0.700000001) isa EF.ExternalForceSample{T}
        end
    end

    @testset "external force device selection and stop flags" begin
        for D in (2, 3), T in (Float32, Float64)
            V = SVector{D,T}
            force = ExternalForce{D,T}(3; times = [0, 1], global_gravity = false,
                linear_acceleration = fill(ntuple(d -> d == 1 ? 2 : 0, D), 2),
                angular_acceleration = D == 2 ? [0, 2] : [(0, 0, 0), (0, 2, 0)])
            points = fill(SVector{D,Float64}(ntuple(d -> d, D)), 5)
            velocities = fill(V(ntuple(d -> 2d, D)), 5)
            types = [Fluid, Fluid, Fixed, Frozen, Fluid]
            groups = [3, 4, 3, 3, 3]
            a, p, v = CUDA.zeros(V, 5), CuArray(points), CuArray(velocities)
            ty, gm = CuArray(types), CuArray(groups)
            history = (Adapt.adapt(CuArray, force),)
            EF.launch_external_forces!(history, a, p, v, ty, gm, HostStep(T(0.2), T(0.4)), T(9.81); half_time = true)
            expected = external_acceleration(force, points[1], velocities[1], T(0.5), T(9.81))
            @test Array(a) ≈ [expected, zero(V), zero(V), zero(V), expected]
            kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.01))
            red = ReductionWorkspace{SVector{3,T}}(5)
            SPHExampleGPU.GPUKernels.launch_step_reduction!(red, p, v, a, p, kernel)
            @test Array(red.partial)[1][2] ≈ sqrt(kernel.h / sqrt(sum(abs2, expected)))
            state = StepState{T}(; time = T(0.4))
            STEP.set_slot!(state.i, state.ih, STEP.I_STOP, STEP.STOP_OUTPUT)
            before = Array(a)
            EF.launch_external_forces!(history, a, p, v, ty, gm, state, T(9.81))
            @test Array(a) == before
        end
    end

    @testset "initial external acceleration limits the first timestep" begin
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            T = Float64
            geometry = [SPHGeometry{2,T}([SVector(0., 0.)]; Density = 1000, GroupMarker = 1, Type = Fluid)]
            metadata = SimulationMetaData{2,T}(SimulationName = "InitialForce", SaveLocation = tempdir(),
                SimulationTime = 1e-8, OutputTimes = 1e-8, GPUUseGraph = false, GPUMaxStepsPerSync = 1)
            metadata.TimeSteppingMode = scheme
            host = AllocateDataStructures(geometry, metadata)
            gpu = upload_particles(host, metadata)
            support = GPUSupportArrays{2,T}(gpu, metadata)
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = 0.01)
            constants = SimulationConstants{T}(2; dx = 0.01, c₀ = 10, g = 9.81)
            state = StepState{T}(; dx = 1 + kernel.h)
            force = ExternalForce{2,T}(1; times = [0, 1], linear_acceleration = [(1e8, 0), (1e8, 0)], global_gravity = false)
            SimulationLoop(ZeroDensityDiffusion(), ZeroViscosity(), kernel, metadata, constants, gpu,
                CellListWorkspace{2,T}(1), support, ReductionWorkspace{SVector{3,T}}(1),
                MotionArrays(geometry, host), state;
                external_forces = EF.prepare_external_forces(force, host, Val(2), T))
            @test metadata.Iteration == 1
            @test metadata.TotalTime ≈ constants.CFL * sqrt(kernel.h / 1e8)
        end
    end

    function run_isolated(D, T, scheme, graph; forced = true, varying = false)
        mktempdir() do dir
            V = SVector{D,T}
            point(x) = V(ntuple(d -> d == 1 ? x : 0, D))
            # Reverse spatial order deliberately; the first rebuild sorts these markers.
            geometry = [SPHGeometry{D,T}([point(x)]; Density = 1000, GroupMarker = marker, Type = Fluid)
                for (marker, x) in ((3, 0.4), (4, 0.2), (5, 0.0))]
            metadata = SimulationMetaData{D,T}(SimulationName = "ExternalForces", SaveLocation = dir,
                SimulationTime = 0.008, OutputTimes = 0.002, GPUUseGraph = graph,
                GPUMaxStepsPerSync = graph ? 4 : 1, GPUClipTimeStepToOutput = true,
                VisualizeInParaview = false, OpenLogFile = false, ExportGridCells = false)
            particles = AllocateDataStructures(geometry, metadata)
            fill!(particles.Velocity, point(2))
            histories = Tuple(ExternalForce{D,T}(marker; times = [0, 0.004, 0.008],
                linear_acceleration = [point(varying ? 0 : acceleration), point(acceleration), point(varying ? -acceleration : acceleration)],
                global_gravity = false) for (marker, acceleration) in ((3, 20), (4, -10)))
            @test_throws ArgumentError EF.prepare_external_forces((histories[1], histories[1]), particles, Val(D), T)
            @test_throws ArgumentError EF.prepare_external_forces(ExternalForce{D,T}(9; times = [0, 1]), particles, Val(D), T)
            logger = SimulationLogger(dir)
            result = try
                RunSimulation(SimGeometry = geometry, SimMetaData = metadata, SimParticles = particles,
                    SimConstants = SimulationConstants{T}(D; dx = 0.01, c₀ = 10, g = 9.81),
                    SimKernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.01)),
                    SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion(),
                    SimTimeStepping = scheme, SimLogger = logger,
                    SimExternalForces = forced ? histories : nothing)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            order = sortperm(particles.ID)
            @test result.final_state.complete
            @test result.diagnostics.cell_list_rebuilds > 1
            @test result.configuration.execution.graph_capture_failures == 0
            graph && @test result.configuration.execution.graph_batches > 0
            if forced && !varying
                t = metadata.TotalTime
                for (i, acceleration) in ((1, 20), (2, -10))
                    @test particles.Velocity[order[i]][1] ≈ 2 + acceleration * t rtol = 2e-5
                    @test particles.Position[order[i]][1] ≈ geometry[i].Particles.Position[1][1] + 2t + acceleration * t^2 / 2 atol = 2e-7
                    @test abs(particles.Velocity[order[i]][D]) < 2e-7
                end
                @test particles.Velocity[order[3]][D] ≈ -T(9.81) * t rtol = 2e-5
            end
            return (; position = particles.Position[order], velocity = particles.Velocity[order],
                density = particles.Density[order], time = metadata.TotalTime, iteration = metadata.Iteration)
        end
    end

    @testset "external forces in both integrators and CUDA graphs" begin
        for D in (2, 3), T in (Float32, Float64), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            direct = run_isolated(D, T, scheme, false)
            graph = run_isolated(D, T, scheme, true)
            @test direct == graph
            @test direct != run_isolated(D, T, scheme, false; forced = false)
            @test run_isolated(D, T, scheme, false; varying = true) == run_isolated(D, T, scheme, true; varying = true)
        end
    end
end

@testset "generated external-forces example" begin
    scope = Module(gensym(:ExternalForcesExample))
    path = joinpath(@__DIR__, "..", "example", "ExternalForces.jl")
    source = first(split(read(path, String), "\nrun_external_forces_3d(;"))
    Base.include_string(scope, source, path)
    case = scope.external_forces_3d_geometry(; dx = 0.05)
    @test length(case.geometry) == 4
    @test length(case.forces) == 2
    @test all(g -> !isempty(g.Particles.Position), case.geometry)
    @test length(unique(vcat([g.Particles.Position for g in case.geometry]...))) == sum(g -> length(g.Particles.Position), case.geometry)
    @test case.forces[1].linear[2] == SVector(0f0, 0f0, 15f0)
    @test case.forces[1].angular[8] == SVector(0f0, 20f0, 0f0)
    @test iszero(case.forces[1].angular[end])
    @test_throws ArgumentError scope.external_forces_3d_geometry(; dx = 0.2)
end

end
