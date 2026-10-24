module FloatingConstraintTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

const F = SPHExampleGPU.GPUFloating
const S = SPHExampleGPU.GPUStepState

@testset "floating constraint configuration" begin
    @test !FloatingConstraints{2,Float32}().active
    @test isbitstype(typeof(FloatingConstraints{3,Float64}()))
    @test FloatingDetails{Float64}(1.2, 0.0).Constraints === nothing
    @test FloatingDetails(1.2, 0.0).Constraints === nothing
    @test_throws ArgumentError FloatingConstraints{1,Float64}()
    @test_throws ArgumentError FloatingConstraints{2,Float64}(TranslationLocks = (true,))
    @test_throws ArgumentError FloatingConstraints{3,Float64}(RotationLocks = (true,))
    @test_throws ArgumentError FloatingConstraints{2,Float64}(TranslationLocks = (1, 0))
    @test_throws ArgumentError FloatingConstraints{2,Float64}(TranslationLocks = (true, false), PrescribedVelocity = (1, nothing))
    @test_throws ArgumentError FloatingConstraints{2,Float32}(PrescribedOmega = (Inf,))
    @test_throws ArgumentError FloatingConstraints{2,Float32}(PrescribedVelocity = (1e100, nothing))
    @test_throws ArgumentError FloatingConstraints{3,Float64}(HingePoint = (0, 0, 0))
    @test_throws ArgumentError FloatingConstraints{3,Float64}(HingePoint = (0, 0, 0), HingeAxis = (0, 0, 0))
    @test_throws ArgumentError FloatingConstraints{2,Float64}(HingePoint = (NaN, 0))
    @test_throws ArgumentError FloatingConstraints{2,Float64}(HingePoint = (0, 0), TranslationLocks = (true, false))
    @test_throws ArgumentError FloatingConstraints{2,Float64}(HingePoint = (0, 0), HingeAxis = (1,))
    @test_throws ArgumentError FloatingConstraints{2,Float64}(HingeVelocity = 1)
    @test_throws ArgumentError FloatingConstraints{3,Float64}(HingeAxis = (1, 0, 0))
    @test_throws ArgumentError FloatingConstraints{2,Float64}(HingePoint = (0, 0), HingeVelocity = NaN)
    axis = FloatingConstraints{3,Float32}(HingePoint = (0, 0, 0), HingeAxis = (1, 2, 3)).axis
    @test norm(axis) ≈ 1
    for T in (Float32, Float64)
        config = FloatingConstraints{3,T}(HingePoint = (0, 0, 0), HingeAxis = (1, 2, 3))
        omega = config.axis * T(0.7)
        projected = omega
        for _ in 1:10_000
            projected = F.constrained_omega(projected, config, Val(3))
        end
        @test projected ≈ omega rtol = 4eps(T)
    end
end

function fixture(D, T, constraints; pause = 0, time = 0, position_type = Float64)
    # An asymmetric body whose centre is offset from the origin/pivot.
    points = [SVector{D,Float64}(ntuple(d -> 0.3d + ((i >> (d - 1)) & 1) * (0.7 - 0.1d), D))
              for i in 0:(2^D - 1)]
    geometry = [SPHGeometry{D,T}(points; Density = 1000, GroupMarker = 1, Type = Floating,
        Floating = FloatingDetails{T}(RelativeWeight = 1.2, PauseTime = pause, Constraints = constraints))]
    host = AllocateDataStructures(geometry; position_type)
    constants = SimulationConstants{T}(dx = T(0.2), c₀ = T(20))
    f = FloatingArrays(geometry, host, constants; position_type, time)
    gpu = upload_particles(host)
    F.initialize_floating_velocities!(f, gpu)
    return (; f, host, gpu, constants, geometry)
end

function stage!(case, step, load, final; carried = false)
    (; f, gpu) = case
    copyto!(f.force, load)
    F.launch_floating_update!(f, step, zero(eltype(load)), final; carried)
    # Predictor placement must read accepted positions without overwriting them.
    positions = final ? gpu.Position : similar(gpu.Position)
    velocities = final ? gpu.Velocity : similar(gpu.Velocity)
    F.launch_floating_particles!(f, positions, velocities, gpu.Position, gpu.Type, gpu.GroupMarker, step, final)
    return Array(positions), Array(velocities)
end

if CUDA.functional()
    @testset "constrained body dynamics" begin
        @test_throws ArgumentError fixture(2, Float32, FloatingConstraints{2,Float64}())
        @test_throws ArgumentError fixture(2, Float32, FloatingConstraints{3,Float32}())
        @testset "Float32 positions and resumed prescribed motion D=$D" for D in (2, 3)
            config = D == 2 ? FloatingConstraints{D,Float32}(HingePoint = (0.1, 0.2), HingeVelocity = 0.7) :
                FloatingConstraints{D,Float32}(HingePoint = (0.1, 0.2, 0.3), HingeAxis = (1, 2, 3), HingeVelocity = 0.7)
            case = fixture(D, Float32, config; pause = 0.1, time = 0.2, position_type = Float32)
            initial = floating_state(case.f)
            @test eltype(initial.center) == SVector{D,Float32}
            @test !iszero(initial.omega[1])
            step = S.HostStep(0.01f0, 0.2f0)
            load = zeros(Float32, D == 2 ? 3 : 6)
            stage!(case, step, load, false)
            stage!(case, step, load, true)
            pivot = SVector{D,Float32}(config.point)
            @test norm(floating_state(case.f).center[1] - pivot) ≈ norm(initial.center[1] - pivot) rtol = 2e-6
        end
        @testset "locks, prescribed DOFs, carried state and graph replay D=$D T=$T" for D in (2, 3), T in (Float32, Float64)
            R = D == 2 ? 1 : 3
            config = FloatingConstraints{D,T}(
                TranslationLocks = ntuple(d -> d == 1, D),
                PrescribedVelocity = ntuple(d -> d == D ? T(0.25) : nothing, D),
                RotationLocks = ntuple(d -> d < R, R),
                PrescribedOmega = ntuple(d -> d == R ? T(0.4) : nothing, R))
            case = fixture(D, T, config)
            (; f, gpu, host) = case
            initial = floating_state(f)
            @test initial.velocity[1][D] == T(0.25)
            @test initial.velocity[1][1] == 0
            r = host.Position[1] - initial.center[1]
            expected_v = initial.velocity[1] + F.angular_cross(initial.omega[1], SVector{D,T}(r))
            @test Array(gpu.Velocity)[1] ≈ expected_v
            dt = T(0.01)
            step = StepState{T}(; dt)
            load = T.(1:(D == 2 ? 3 : 6))
            copyto!(f.force, load)
            F.launch_floating_derivative!(f, step, T(9.81))
            @test Array(f.acceleration)[1][1] == 0
            @test Array(f.acceleration)[1][D] == 0
            @test iszero(Array(f.angular_acceleration)[1])
            # Keep device force input fixed so graph replay repeats the same loads.
            device_load = CuArray(load)
            advance() = begin
                copyto!(f.force, device_load)
                F.launch_floating_update!(f, step, T(9.81), false; carried = true)
                copyto!(f.force, device_load)
                F.launch_floating_update!(f, step, T(9.81), true)
                F.launch_floating_particles!(f, gpu.Position, gpu.Velocity, gpu.Position,
                    gpu.Type, gpu.GroupMarker, step, true)
            end
            advance()
            CUDA.synchronize()
            graph = CUDA.instantiate(CUDA.capture(advance))
            before = floating_state(f)
            for _ in 1:10
                CUDA.launch(graph)
            end
            state = floating_state(f)
            @test state.center[1][1] == initial.center[1][1]
            @test state.center[1][D] ≈ before.center[1][D] + 10 * dt * T(0.25) atol = 1e-7
            @test state.omega == initial.omega
            @test state.velocity[1][D] == T(0.25)
            @test all(iszero, Array(f.force))
            if D == 2
                @test state.angle[1] ≈ before.angle[1] + 10 * dt * T(0.4) rtol = 2e-6
            end
            distances(x) = [norm(x[i] - x[j]) for i in eachindex(x) for j in (i + 1):length(x)]
            @test distances(Array(gpu.Position)) ≈ distances(host.Position) rtol = 2e-6
            # Reordering must leave per-body constraints attached to the marker.
            cl = CellListWorkspace{D,Float64}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, 2.0; floating = f)
            advance()
            @test floating_state(f).velocity[1][D] == T(0.25)
            @test Array(f.constraints) == [config]
        end

        @testset "rotation locks use constrained world inertia T=$T" for T in (Float32, Float64)
            config = FloatingConstraints{3,T}(RotationLocks = (true, false, false))
            (; f) = fixture(3, T, config)
            q = F.rotation_quaternion(SVector{3,T}(0.4, 0.3, -0.2))
            w = SVector{3,T}(0, 0.2, -0.1)
            copyto!(f.orientation, [q])
            copyto!(f.omega, [w])
            inertia = F.world_inertia(q, Array(f.inertia)[1])
            torque = SVector{3,T}(0.7, -0.4, 0.9)
            load = T[0, 0, 0, torque...]
            expected = inertia[2:3,2:3] \ (torque - cross(w, inertia * w))[2:3]
            copyto!(f.force, load)
            F.launch_floating_derivative!(f, S.HostStep(T(0.01), zero(T)), zero(T))
            alpha = Array(f.angular_acceleration)[1]
            @test alpha[1] == 0
            @test alpha[2:3] ≈ expected rtol = 2e-6
            unconstrained = inertia \ (torque - cross(w, inertia * w))
            @test norm(alpha[2:3] - unconstrained[2:3]) > T(0.001)
        end

        @testset "off-centre hinge force, gravity and parallel-axis inertia D=$D T=$T" for D in (2, 3), T in (Float32, Float64)
            config = D == 2 ? FloatingConstraints{D,T}(HingePoint = (0, 0)) :
                FloatingConstraints{D,T}(HingePoint = (0, 0, 0), HingeAxis = (1, 2, 3))
            case = fixture(D, T, config)
            (; f) = case
            mass = Array(f.mass)[1]
            r = SVector{D,T}(Array(f.center)[1])
            force = SVector{D,T}(ntuple(d -> T(d / 10), D))
            torque = D == 2 ? T(0.2) : SVector{3,T}(0.2, -0.3, 0.4)
            gvec = SVector{D,T}(ntuple(d -> d == D ? -T(9.81) : zero(T), D))
            inertia = Array(f.inertia)[1]
            expected = if D == 2
                (torque + F.floating_torque(r, force + mass * gvec)) / (inertia + mass * dot(r, r))
            else
                axis = config.axis
                axis * (dot(axis, torque + cross(r, force + mass * gvec)) /
                    (dot(axis, inertia * axis) + mass * norm(cross(r, axis))^2))
            end
            load = D == 2 ? T[force..., torque] : T[force..., torque...]
            copyto!(f.force, load)
            step = S.HostStep(T(0.01), zero(T))
            F.launch_floating_derivative!(f, step, T(9.81))
            @test Array(f.angular_acceleration)[1] ≈ expected rtol = 3e-6
            @test Array(f.acceleration)[1] ≈ F.angular_cross(expected, r) rtol = 3e-6
            F.launch_floating_update!(f, step, T(9.81), false; carried = true)
            @test Array(f.omega_half)[1] ≈ expected * (step.dt / 2) rtol = 3e-6
            copyto!(f.force, load)
            F.launch_floating_update!(f, step, T(9.81), true)
            state = floating_state(f)
            @test norm(state.center[1]) ≈ norm(r) rtol = 2e-6
            @test state.velocity[1] ≈ F.angular_cross(state.omega[1], SVector{D,T}(state.center[1])) rtol = 2e-6
            @test norm(state.center[1] - Array(f.center_n)[1]) > 0
        end

        @testset "prescribed hinge preserves pivot and rigid velocities D=$D T=$T" for D in (2, 3), T in (Float32, Float64)
            point = ntuple(d -> 0.1d, D)
            config = D == 2 ? FloatingConstraints{D,T}(HingePoint = point, HingeVelocity = 0.7) :
                FloatingConstraints{D,T}(HingePoint = point, HingeAxis = (1, -2, 3), HingeVelocity = 0.7)
            case = fixture(D, T, config)
            (; f, gpu, host) = case
            initial = floating_state(f)
            dt = T(0.01)
            step = S.HostStep(dt, zero(T))
            load = T.(1:(D == 2 ? 3 : 6))
            for _ in 1:20
                stage!(case, step, load, false)
                positions, velocities = stage!(case, step, load, true)
                state = floating_state(f)
                @test norm(state.center[1] - SVector{D,Float64}(point)) ≈ norm(initial.center[1] - SVector{D,Float64}(point)) rtol = 2e-6
                @test state.omega[1] ≈ initial.omega[1] rtol = 2e-6
                for i in eachindex(positions)
                    r = SVector{D,T}(positions[i] - SVector{D,Float64}(point))
                    @test velocities[i] ≈ F.angular_cross(state.omega[1], r) atol = 2e-6
                end
            end
            turn = initial.omega[1] * (20dt)
            expected = SVector{D,Float64}(point) + F.rotate_hinge_offset(initial.center[1] - SVector{D,Float64}(point), turn)
            @test floating_state(f).center[1] ≈ expected atol = 2e-6
        end

        @testset "free hinge follows analytical constant torque D=$D T=$T" for D in (2, 3), T in (Float32, Float64)
            config = D == 2 ? FloatingConstraints{D,T}(HingePoint = (0, 0)) :
                FloatingConstraints{D,T}(HingePoint = (0, 0, 0), HingeAxis = (1, -2, 3))
            case = fixture(D, T, config)
            (; f) = case
            origin = Array(f.center)[1]
            mass, inertia = Array(f.mass)[1], Array(f.inertia)[1]
            axis = config.axis
            radius = SVector{D,T}(origin)
            pivot_inertia = D == 2 ? inertia + mass * dot(radius, radius) :
                dot(axis, inertia * axis) + mass * norm(cross(radius, axis))^2
            alpha = T(0.3)
            torque = D == 2 ? alpha * pivot_inertia : axis * (alpha * pivot_inertia)
            load = D == 2 ? T[0, 0, torque] : T[0, 0, 0, torque...]
            step = S.HostStep(T(0.005), zero(T))
            copyto!(f.force, load)
            F.launch_floating_derivative!(f, step, zero(T))
            for _ in 1:20
                stage!(case, step, load, false; carried = true)
                stage!(case, step, load, true)
            end
            time = 20 * step.dt
            theta = Float64(alpha * time^2 / 2)
            expected = if D == 2
                SVector(cos(theta) * origin[1] - sin(theta) * origin[2],
                        sin(theta) * origin[1] + cos(theta) * origin[2])
            else
                unit_axis = SVector{3,Float64}(axis)
                cos(theta) * origin + sin(theta) * cross(unit_axis, origin) +
                    (1 - cos(theta)) * dot(unit_axis, origin) * unit_axis
            end
            state = floating_state(f)
            @test state.center[1] ≈ expected atol = 2e-6
            @test state.omega[1] ≈ (D == 2 ? alpha * time : axis * (alpha * time)) rtol = 3e-6
        end

        @testset "pause and freezing override prescribed motion D=$D" for D in (2, 3)
            T = Float64
            config = FloatingConstraints{D,T}(PrescribedVelocity = ntuple(_ -> 0.5, D))
            case = fixture(D, T, config; pause = 0.1)
            (; f, gpu) = case
            initial = floating_state(f)
            @test all(iszero, Array(gpu.Velocity))
            load = zeros(T, D == 2 ? 3 : 6)
            for final in (false, true)
                stage!(case, S.HostStep(0.01, 0.0), load, final)
            end
            @test floating_state(f) == initial
            for final in (false, true)
                stage!(case, S.HostStep(0.01, 0.1), load, final)
            end
            released = floating_state(f)
            @test released.center[1] ≈ initial.center[1] .+ 0.005
            @test released.velocity[1] == SVector{D,T}(ntuple(_ -> 0.5, D))
            fill!(f.frozen, Int32(1))
            F.freeze_body_state!(f)
            F.launch_floating_update!(f, S.HostStep(0.01, 0.2), 9.81, false; carried = true)
            F.launch_floating_update!(f, S.HostStep(0.01, 0.2), 9.81, true)
            @test floating_state(f).center == released.center
            @test all(iszero, Array(f.velocity))
        end

        @testset "constrained simulation D=$D T=$T $(nameof(typeof(scheme))) graph=$graph" for D in (2, 3), T in (Float32, Float64),
                scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping()), graph in (false, true)
            R = D == 2 ? 1 : 3
            config = FloatingConstraints{D,T}(
                TranslationLocks = ntuple(d -> D == 3 && d == 2, D),
                PrescribedVelocity = ntuple(d -> d == 1 ? T(0.2) : nothing, D),
                RotationLocks = ntuple(_ -> true, R))
            case = fixture(D, T, config)
            origin = Array(case.f.center)[1]
            pivot = origin .+ 4
            hinge = D == 2 ? FloatingConstraints{D,T}(HingePoint = pivot, HingeVelocity = T(0.7)) :
                FloatingConstraints{D,T}(HingePoint = pivot, HingeAxis = (1, -2, 3), HingeVelocity = T(0.7))
            points = case.host.Position .+ Ref(SVector{D,Float64}(ntuple(d -> 4 + 0.1d, D)))
            push!(case.geometry, SPHGeometry{D,T}(points; Density = 1000, GroupMarker = 2, Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = 1.2, Constraints = hinge)))
            dir = mktempdir()
            meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
                SimulationName = "Constraints", SaveLocation = dir, SimulationTime = T(0.006), OutputTimes = T(0.002),
                GPUDoublePosition = true, GPUUseGraph = graph, GPUClipTimeStepToOutput = true,
                VisualizeInParaview = false, OpenLogFile = false)
            host = AllocateDataStructures(case.geometry, meta)
            result = RunSimulation(SimGeometry = case.geometry, SimMetaData = meta, SimParticles = host,
                SimConstants = case.constants, SimKernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.2), k = T(sqrt(D))),
                SimLogger = SimulationLogger(dir), SimViscosity = Laminar(), SimDensityDiffusion = LinearDensityDiffusion(),
                SimTimeStepping = scheme)
            t = result.final_state.time
            body = result.final_state.floating
            @test body.center[1][1] ≈ origin[1] + T(0.2) * t atol = 2e-7
            @test body.center[1][D] ≈ origin[D] - case.constants.g * t^2 / 2 atol = 2e-7
            @test body.velocity[1][D] ≈ -case.constants.g * t atol = 2e-6
            @test iszero(body.omega[1])
            if D == 3
                @test body.center[1][2] == origin[2]
            end
            @test body.omega[2] ≈ F.constrained_omega(zero(body.omega[2]), hinge, Val(D)) rtol = 2e-6
            initial_center = sum(points) / length(points)
            turn = body.omega[2] * t
            expected_center = pivot + F.rotate_hinge_offset(initial_center - pivot, turn)
            @test body.center[2] ≈ expected_center atol = 2e-6
            @test all(isfinite, result.final_state.particles.Density)
            rows = readlines(joinpath(dir, "Constraints_Floating.csv"))
            @test length(rows) >= 5
            # The first particle frame and CSV row must already carry prescribed rates.
            first_row = split(rows[2], ',')
            @test parse(T, first_row[4 + D]) == T(0.2)
        end
    end
else
    @test_skip CUDA.functional()
end

end # module FloatingConstraintTests
