module FloatingBodyUpdateTests

using Test, SPHExampleGPU, CUDA, StaticArrays

const F = SPHExampleGPU.GPUFloating
const S = SPHExampleGPU.GPUStepState

function fixture(D, T; nbodies = 259)
    geometry = SPHGeometry{D,T}[]
    for b in 1:nbodies
        points = [SVector{D,Float64}(ntuple(d -> 0.1b + 0.3d +
            ((i >> (d - 1)) & 1) * (0.7 - 0.1d), D)) for i in 0:(2^D - 1)]
        mode = mod(b, 5)
        constraints = if mode == 1
            FloatingConstraints{D,T}(
                TranslationLocks = ntuple(d -> d == 1, D),
                PrescribedVelocity = ntuple(d -> d == D ? T(0.25) : nothing, D),
                RotationLocks = ntuple(d -> d == 1, D == 2 ? 1 : 3))
        elseif mode in (2, 3)
            point = ntuple(d -> 0.1b + 0.2d, D)
            rate = mode == 3 ? T(0.4) : nothing
            D == 2 ? FloatingConstraints{D,T}(HingePoint = point, HingeVelocity = rate) :
                FloatingConstraints{D,T}(HingePoint = point, HingeAxis = (1, 2, 3), HingeVelocity = rate)
        else
            nothing
        end
        push!(geometry, SPHGeometry{D,T}(points; Density = 1000, Type = Floating,
            GroupMarker = b, Floating = FloatingDetails{T}(RelativeWeight = 1 + 0.01b,
                PauseTime = mod(b, 7) == 0 ? 1 : 0, Constraints = constraints)))
    end
    host = AllocateDataStructures(geometry; position_type = Float64)
    f = FloatingArrays(geometry, host, SimulationConstants{T}(); position_type = Float64)
    frozen = Int32.(mod.(1:nbodies, 11) .== 0)
    copyto!(f.frozen, frozen)
    velocities = [SVector{D,T}(ntuple(d -> T(0.01b / d), D)) for b in 1:nbodies]
    omegas = D == 2 ? T.(0.002 .* (1:nbodies)) :
        [SVector{3,T}(T(0.001b), T(-0.002b), T(0.003b)) for b in 1:nbodies]
    copyto!(f.velocity, velocities)
    copyto!(f.omega, omegas)
    if D == 3
        orientations = [F.rotation_quaternion(SVector{3,T}(T(0.001b), T(0.002b), T(-0.001b)))
                        for b in 1:nbodies]
        copyto!(f.orientation, orientations)
    end
    F.freeze_body_state!(f)
    return f
end

function singleton_views(f)
    fields = (:frozen, :constraints, :pause, :mass, :inertia, :center, :center_n, :center_half,
              :velocity, :velocity_half, :omega, :omega_half, :turn, :rotation,
              :acceleration, :angular_acceleration)
    fields = hasproperty(f, :angle) ? (fields..., :angle) :
        (fields..., :inertia_inv, :orientation, :orientation_half)
    stride = hasproperty(f, :angle) ? 3 : 6
    return [merge(f, (; nbodies = Int32(1)),
        NamedTuple{fields}(Tuple(view(getproperty(f, name), b:b) for name in fields)),
        (; force = view(f.force, (stride * (b - 1) + 1):(stride * b))))
        for b in 1:Int(f.nbodies)]
end

function check_state(f, reference)
    for name in keys(f)
        actual = getproperty(f, name)
        actual isa CuArray || continue
        @test Array(actual) == Array(getproperty(reference, name))
    end
end

if CUDA.functional()
    @testset "parallel body updates D=$D T=$T" for D in (2, 3), T in (Float32, Float64)
        f = fixture(D, T)
        reference = map(x -> x isa CuArray ? copy(x) : x, f)
        singles = singleton_views(reference)
        step = StepState{T}(; dt = T(0.001))
        g = T(0.3)
        stride = D == 2 ? 3 : 6
        load = CuArray([T(mod(i, 17) - 8) / T(10) for i in 1:(stride * Int(f.nbodies))])
        half_load = T(2) .* load

        derivative!(body) = F.launch_floating_derivative!(body, step, g)
        predict!(body, carried) = F.launch_floating_update!(body, step, g, false; carried)
        correct!(body) = F.launch_floating_update!(body, step, g, true)
        for carried in (false, true)
            copyto!(f.force, load)
            copyto!(reference.force, load)
            derivative!(f)
            foreach(derivative!, singles)
            check_state(f, reference)
            predict!(f, carried)
            foreach(body -> predict!(body, carried), singles)
            check_state(f, reference)
            copyto!(f.force, half_load)
            copyto!(reference.force, half_load)
            correct!(f)
            foreach(correct!, singles)
            check_state(f, reference)

            advance!(body) = begin
                derivative!(body)
                predict!(body, carried)
                correct!(body)
            end
            # Warm every specialization before capturing, then compare replay
            # with independent singleton launches sharing the reference buffers.
            advance!(f)
            foreach(advance!, singles)
            CUDA.synchronize()
            graph = CUDA.instantiate(CUDA.capture(() -> advance!(f)))
            for _ in 1:3
                copyto!(f.force, load)
                copyto!(reference.force, load)
                CUDA.launch(graph)
                foreach(advance!, singles)
                check_state(f, reference)
            end
        end

        state_i = Array(step.i)
        state_i[S.I_STOP] = S.STOP_REBUILD
        copyto!(step.i, state_i)
        copyto!(f.force, load)
        before = map(x -> x isa CuArray ? copy(x) : x, f)
        derivative!(f)
        predict!(f, false)
        predict!(f, true)
        correct!(f)
        check_state(f, before)

        empty = merge(f, (; active = false, nbodies = Int32(0)))
        derivative!(empty)
        predict!(empty, false)
        correct!(empty)
        check_state(f, before)
    end
else
    @testset "parallel body updates require CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
