function floating_angular_reference(inertia, q, omega, torque)
    w, x, y, z = q
    rotation = @SMatrix [1-2(y*y+z*z) 2(x*y-w*z) 2(x*z+w*y);
                        2(x*y+w*z) 1-2(x*x+z*z) 2(y*z-w*x);
                        2(x*z-w*y) 2(y*z+w*x) 1-2(x*x+y*y)]
    inertia_world = rotation * inertia * transpose(rotation)
    return inertia_world \ (torque - cross(omega, inertia_world * omega))
end

@testset "floating carried derivatives" begin
    F = SPHExampleGPU.GPUFloating
    for D in (2, 3), T in (Float32, Float64)
        corners = [SVector{D, T}(ntuple(d -> T(((i >> (d - 1)) & 1) * (1.2 - 0.2d)), D))
                   for i in 0:(2^D - 1)]
        geometry = [SPHGeometry{D, T}(corners; Density = 1000, Type = Floating,
            GroupMarker = 1, Floating = FloatingDetails{T}(RelativeWeight = 1.2))]
        particles = AllocateDataStructures(geometry)
        f = FloatingArrays(geometry, particles, SimulationConstants{T}())
        dt, g = T(0.01), T(0.3)
        step = StepState{T}(; dt)
        stride = D == 2 ? 3 : 6
        load = T.(collect(1:stride)) / T(10)
        copyto!(f.force, load)
        if D == 3
            angle = T(0.4)
            q = SVector{4, T}(cos(angle / 2), 0, 0, sin(angle / 2))
            omega = SVector{3, T}(0.2, -0.1, 0.3)
            copyto!(f.orientation, [q])
            copyto!(f.omega, [omega])
            c, s = cos(angle), sin(angle)
            rotation = @SMatrix [c -s 0; s c 0; 0 0 1]
            inertia_world = rotation * Array(f.inertia)[1] * transpose(rotation)
            torque = SVector{3, T}(load[4:6])
            expected_alpha = inertia_world \ (torque - cross(omega, inertia_world * omega))
        else
            expected_alpha = load[3] / Array(f.inertia)[1]
        end
        expected_a = SVector{D, T}(load[1:D]) / Array(f.mass)[1] +
                     SVector{D, T}(ntuple(d -> d == D ? -g : zero(T), D))
        F.launch_floating_derivative!(f, step, g)
        @test Array(f.acceleration)[1] == expected_a
        @test Array(f.angular_acceleration)[1] ≈ expected_alpha rtol = 2e-6
        @test all(iszero, Array(f.force))

        predict() = F.launch_floating_update!(f, step, g, false; carried = true)
        predict()
        graph = CUDA.instantiate(CUDA.capture(predict))
        CUDA.launch(graph)
        @test Array(f.velocity_half)[1] ≈ Array(f.velocity)[1] + expected_a * (dt / 2)
        @test Array(f.omega_half)[1] ≈ Array(f.omega)[1] + expected_alpha * (dt / 2)

        # Refresh at the evaluated half state, then move the body to the full
        # state. The next predictor must reuse this derivative without taking
        # torque about the new centre or recalculating the 3D gyroscopic term.
        expected_half_alpha = if D == 2
            T(2) * load[3] / Array(f.inertia)[1]
        else
            floating_angular_reference(Array(f.inertia)[1], Array(f.orientation_half)[1],
                Array(f.omega_half)[1], SVector{3, T}(T(2) .* load[4:6]))
        end
        copyto!(f.force, T(2) .* load)
        F.launch_floating_update!(f, step, g, true)
        cached_a = Array(f.acceleration)[1]
        cached_alpha = Array(f.angular_acceleration)[1]
        @test isapprox(cached_alpha, expected_half_alpha; rtol = 2e-6)
        velocity_n = Array(f.velocity)[1]
        omega_n = Array(f.omega)[1]
        @test cached_a ≈ SVector{D, T}(T(2) .* load[1:D]) / Array(f.mass)[1] +
            SVector{D, T}(ntuple(d -> d == D ? -g : zero(T), D))
        CUDA.launch(graph)
        @test Array(f.velocity_half)[1] ≈ velocity_n + cached_a * (dt / 2)
        @test Array(f.omega_half)[1] ≈ omega_n + cached_alpha * (dt / 2)
        @test all(iszero, Array(f.force))

        # Startup/rebuild refresh uses the full orientation and angular velocity.
        copyto!(f.force, load)
        expected_full_alpha = if D == 2
            expected_alpha
        else
            floating_angular_reference(Array(f.inertia)[1], Array(f.orientation)[1],
                Array(f.omega)[1], SVector{3, T}(load[4:6]))
        end
        F.launch_floating_derivative!(f, step, g)
        @test Array(f.acceleration)[1] == expected_a
        @test isapprox(Array(f.angular_acceleration)[1], expected_full_alpha; rtol = 2e-6)
        @test all(iszero, Array(f.force))

        state_i = Array(step.i)
        state_i[SPHExampleGPU.GPUStepState.I_STOP] = SPHExampleGPU.GPUStepState.STOP_REBUILD
        copyto!(step.i, state_i)
        copyto!(f.force, load)
        before = Array(f.acceleration)
        F.launch_floating_derivative!(f, step, g)
        CUDA.launch(graph)
        @test Array(f.acceleration) == before
        @test Array(f.force) == load
    end
end

@testset "single-neighbor floating graph replay agrees with direct launches" begin
    direct, direct_rows, _ = run_floating_case(; FloatType = Float32, relative_weight = 1.2,
        pause = 0.005, water = true, simtime = 0.04, double = true,
        scheme = SingleNeighborTimeStepping(), graph = false)
    graphed, graph_rows, _ = run_floating_case(; FloatType = Float32, relative_weight = 1.2,
        pause = 0.005, water = true, simtime = 0.04, double = true,
        scheme = SingleNeighborTimeStepping(), graph = true)
    @test direct.ID == graphed.ID
    @test maximum(norm.(direct.Position .- graphed.Position)) < 1e-7
    @test maximum(norm.(direct.Velocity .- graphed.Velocity)) < 1e-5
    @test maximum(abs.(direct.Density .- graphed.Density)) < 1e-3
    @test graph_rows[end].Time == direct_rows[end].Time
end
