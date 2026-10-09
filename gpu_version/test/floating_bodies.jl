# GPU tests of the floating rigid bodies (`Type = Floating`, see `GPUFloating`).
# Included by runtests.jl inside the "SPHExampleGPU" test set.
using CSV

"""
    run_floating_case(; FloatType, relative_weight, pause, water, simtime, double)

Run a small 2D case on the GPU: a conforming disc of radius 0.1 m centred at
`(0.5, 0.4)` as a floating body in a closed 1 m tank, with water up to 0.8 m
when `water` is set. Returns the final particles (sorted by ID) and the rows of
the floating body log, and the particles as loaded (sorted by ID).
"""
function run_floating_case(; FloatType = Float64, relative_weight, pause = 0.0, water = true,
                           simtime, double = false, scheme = SymplecticTimeStepping(), graph = true)
    T  = FloatType
    dx = 0.02
    dir = mktempdir()
    generator_constants = SimulationConstants{Float64}(; dx, c₀ = 20.0)
    regions = [ParticleRegion("Body", circle((0.5, 0.4), 0.1), Floating; sampling = :conforming),
               ParticleRegion("Tank", outline(rectangle((0, 0), 1, 1); thickness = 3dx), Fixed)]
    water && push!(regions, ParticleRegion("Fluid", rectangle((0, 0), 1, 0.8), Fluid))
    sampled = sample_particles(regions, dx)
    level = water ? maximum(last, sampled[end].positions) : 0.0
    next_id = 0
    for region in sampled
        density = hydrostatic_density(region.positions, generator_constants; water_level = level)
        next_id = write_particle_csv(joinpath(dir, region.name * ".csv"), region.positions;
                                     density, first_id = next_id)
    end

    geometry = [SPHGeometry{2, T}(CSVFile = joinpath(dir, "Tank.csv"), GroupMarker = 1, Type = Fixed),
                SPHGeometry{2, T}(CSVFile = joinpath(dir, "Body.csv"), GroupMarker = 3, Type = Floating,
                                  Floating = FloatingDetails{T}(RelativeWeight = relative_weight,
                                                                PauseTime = pause))]
    water && push!(geometry, SPHGeometry{2, T}(CSVFile = joinpath(dir, "Fluid.csv"), GroupMarker = 2,
                                               Type = Fluid))
    meta = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, NoLog}(
        SimulationName = "Floating", SaveLocation = dir, SimulationTime = simtime,
        OutputTimes = simtime / 5, GPUDoublePosition = double, GPUUseGraph = graph,
        VisualizeInParaview = false, OpenLogFile = false)
    particles = AllocateDataStructures(geometry, meta)
    initial   = deepcopy(particles)
    RunSimulation(SimGeometry = geometry, SimMetaData = meta,
                  SimConstants = SimulationConstants{T}(; dx = T(dx), c₀ = T(20)),
                  SimKernel = SPHKernelInstance{2, T}(WendlandC2(); dx = T(dx), k = T(sqrt(2))),
                  SimLogger = SimulationLogger(dir), SimParticles = particles,
                  SimViscosity = Laminar(), SimDensityDiffusion = LinearDensityDiffusion(),
                  SimTimeStepping = scheme)
    rows = CSV.File(joinpath(dir, "Floating_Floating.csv"))
    return particles[sortperm(particles.ID)], rows, initial[sortperm(initial.ID)]
end

body_distances(p) = (x = p.Position[p.Type .== Floating];
                     [norm(x[i] - x[j]) for i in eachindex(x) for j in (i + 1):lastindex(x)])

@testset "floating bodies" begin
    @testset "compact membership survives sorting and graph replay" begin
        for D in (2, 3), T in (Float32, Float64)
            corners = [SVector{D, T}(ntuple(d -> T((i >> (d - 1)) & 1), D))
                for i in 0:(2^D - 1)]
            geometry = [SPHGeometry{D, T}(corners;
                Density = 1000, Type = Floating, GroupMarker = 1,
                Floating = FloatingDetails{T}(RelativeWeight = 1.2)),
                SPHGeometry{D, T}(corners .+ Ref(SVector{D, T}(ntuple(_ -> T(4), D)));
                    Density = 1000, Type = Floating, GroupMarker = 2,
                    Floating = FloatingDetails{T}(RelativeWeight = 1.5)),
                SPHGeometry{D, T}([SVector{D, T}(ntuple(_ -> T(i / 3), D))
                    for i in 1:37]; Density = 1000, Type = Fluid, GroupMarker = 3)]
            particles = AllocateDataStructures(geometry)
            constants = SimulationConstants{T}()
            gpu = upload_particles(particles)
            floating = FloatingArrays(geometry, particles, constants)
            workspace = CellListWorkspace{D, T}(length(gpu))
            step = StepState{T}(dt = T(0.01))
            mod = SPHExampleGPU.GPUFloating
            indices_pointer = pointer(floating.indices)
            for _ in 1:2
                SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, workspace, T(2))
                mod.update_floating_indices!(floating, gpu.Type, workspace)
                @test Array(floating.indices) == findall(==(Floating), Array(gpu.Type))
                @test pointer(floating.indices) == indices_pointer
                acceleration = [SVector{D, T}(ntuple(d -> T(d * g), D))
                    for g in Array(gpu.GroupMarker)]
                copyto!(gpu.Acceleration, acceleration)
                reduce_forces() = begin
                    fill!(floating.force, zero(T))
                    mod.launch_floating_forces!(floating, gpu.Acceleration, gpu.Position,
                        gpu.Type, gpu.GroupMarker, floating.center, constants.m₀, step)
                end
                reduce_forces()
                CUDA.synchronize()
                graph = CUDA.instantiate(CUDA.capture(reduce_forces))
                CUDA.launch(graph)
                forces = Array(floating.force)
                stride = D == 2 ? 3 : 6
                for b in 1:2
                    expected = constants.m₀ * length(corners) *
                        SVector{D, T}(ntuple(d -> T(d * b), D))
                    @test forces[((b - 1) * stride + 1):((b - 1) * stride + D)] ≈
                        expected rtol = 1e-5
                    @test maximum(abs, forces[((b - 1) * stride + D + 1):(b * stride)]) <
                        T(1e-5)
                end
                # Change the spatial ordering before the second rebuild.
                gpu.Position .= .-gpu.Position
                floating.center .= .-floating.center
            end
        end
    end
    @testset "free fall in air is exact and rigid" for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        g, pause = 9.81, 0.02
        p, rows, initial = run_floating_case(; relative_weight = 1.2, pause, water = false, simtime = 0.1, scheme)
        y0 = rows[1][Symbol("Center:1")]
        @test y0 ≈ 0.4 atol = 1e-12
        held = [r for r in rows if r.Time <= pause]
        @test all(r -> r[Symbol("Center:1")] == y0 && r[Symbol("Velocity:1")] == 0, held)

        last_row = rows[end]
        v = last_row[Symbol("Velocity:1")]
        released = last_row.Time + v / g           # the start of the first step after the pause
        @test pause <= released < pause + 1e-3
        # the symplectic update is exact for a constant acceleration
        @test last_row[Symbol("Center:1")] ≈ y0 - v^2 / (2g) rtol = 1e-10
        @test last_row[Symbol("Center:0")] == rows[1][Symbol("Center:0")]
        @test last_row[Symbol("Velocity:0")] == 0 && last_row.Angle == 0
        @test maximum(abs.(body_distances(p) .- body_distances(initial))) < 1e-12
        body = p.Type .== Floating
        @test all(x -> x ≈ SVector(0.0, v), p.Velocity[body])
        @test sum(p.Position[body]) / count(body) ≈
              SVector(last_row[Symbol("Center:0")], last_row[Symbol("Center:1")]) atol = 1e-12
    end

    @testset "buoyancy: $(nameof(typeof(scheme))) (Float32, double positions)" for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        sink(rows) = rows[1][Symbol("Center:1")] - rows[end][Symbol("Center:1")]
        _, neutral, _ = run_floating_case(FloatType = Float32, relative_weight = 1.0, pause = 0.1,
                                          simtime = 0.3, double = true, scheme = scheme)
        heavy_particles, heavy, _ = run_floating_case(FloatType = Float32, relative_weight = 2.0,
                                                      pause = 0.1, simtime = 0.3, double = true, scheme = scheme)
        # a cylinder of twice the density of water sinks with about g/3 (added mass)
        @test 0.03 < sink(heavy) < 0.1
        @test abs(sink(neutral)) < 0.2 * sink(heavy)
        @test all(isfinite, heavy_particles.Density)
        @test abs(heavy[end].Angle) < 0.05
    end

    @testset "3D floating bodies use vector rotation and double positions" begin
        T = Float32
        positions = SVector{3, Float64}[
            SVector{3, Float64}(1, 0, 0), SVector{3, Float64}(-1, 0, 0),
            SVector{3, Float64}(0, 1, 0), SVector{3, Float64}(0, -1, 0),
            SVector{3, Float64}(0, 0, 1), SVector{3, Float64}(0, 0, -1),
        ]
        geometry = [SPHGeometry{3, T}(
            Particles = StructArray((Position = positions, Density = fill(T(1000), length(positions)))),
            GroupMarker = 1, Type = Floating,
            Floating = FloatingDetails{T}(RelativeWeight = one(T)))]
        constants = SimulationConstants{T}(; dx = T(0.2), c₀ = T(20))
        particles = AllocateDataStructures(geometry; position_type = Float64)
        floating = FloatingArrays(geometry, particles, constants; position_type = Float64)

        torque_axis = SVector{3, T}(T(0.3), T(-0.4), one(T))
        acceleration = [SVector{3, T}(cross(torque_axis, SVector{3, T}(x)))
                        for x in positions]
        device_positions = CUDA.CuArray(particles.Position)
        device_acceleration = CUDA.CuArray(acceleration)
        device_type = CUDA.CuArray(particles.Type)
        device_group = CUDA.CuArray(particles.GroupMarker)
        step = SPHExampleGPU.GPUStepState.HostStep(T(0.1), zero(T))
        floating_gpu = SPHExampleGPU.GPUFloating

        floating_gpu.launch_floating_forces!(floating, device_acceleration, device_positions,
            device_type, device_group, floating.center, constants.m₀, step)
        floating_gpu.launch_floating_update!(floating, step, zero(T), false)
        floating_gpu.launch_floating_forces!(floating, device_acceleration, device_positions,
            device_type, device_group, floating.center_half, constants.m₀, step)
        floating_gpu.launch_floating_update!(floating, step, zero(T), true)

        device_positions_out = CUDA.zeros(SVector{3, Float64}, length(positions))
        device_velocity_out = CUDA.zeros(SVector{3, T}, length(positions))
        floating_gpu.launch_floating_particles!(floating, device_positions_out, device_velocity_out,
            device_positions, device_type, device_group, step, true)

        state = floating_state(floating)
        turn = torque_axis * T(0.005)
        angle = norm(turn)
        axis = SVector{3, Float64}(turn) / Float64(angle)
        q_vector = (sin(angle / 2) / angle) * turn
        @test state.center[1] ≈ SVector{3, Float64}(0, 0, 0) atol = 1e-12
        @test state.omega[1] ≈ torque_axis * T(0.1) atol = 1e-6
        @test state.orientation[1] ≈
              SVector{4, T}(cos(angle / 2), q_vector...) atol = 1e-6
        initial_position = positions[1]
        expected_position = cos(Float64(angle)) * initial_position +
            sin(Float64(angle)) * cross(axis, initial_position) +
            (1 - cos(Float64(angle))) * dot(axis, initial_position) * axis
        @test Array(device_positions_out)[1] ≈ expected_position atol = 1e-6
        @test Array(device_velocity_out)[1] ≈
              cross(state.omega[1], SVector{3, T}(expected_position)) atol = 1e-6
    end

    @testset "3D simulation logs quaternion floating state: $(nameof(typeof(scheme)))" for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        T = Float64
        dx = 0.2
        dir = mktempdir()
        offsets = SVector{3, T}[
            SVector{3, T}(dx / 2, 0, 0), SVector{3, T}(-dx / 2, 0, 0),
            SVector{3, T}(0, dx / 2, 0), SVector{3, T}(0, -dx / 2, 0),
            SVector{3, T}(0, 0, dx / 2), SVector{3, T}(0, 0, -dx / 2),
        ]
        body_positions = [x + SVector{3, T}(0, 0, 0.5) for x in offsets]
        fluid_positions = [SVector{3, T}(0, 0, 0.5)]
        geometry = [
            SPHGeometry{3, T}(
                Particles = StructArray((Position = body_positions,
                                         Density = fill(T(1000), length(body_positions)))),
                GroupMarker = 1, Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = one(T))),
            SPHGeometry{3, T}(
                Particles = StructArray((Position = fluid_positions,
                                         Density = fill(T(1000), length(fluid_positions)))),
                GroupMarker = 2, Type = Fluid),
        ]
        meta = SimulationMetaData{3, T, NoShifting, NoKernelOutput, NoMDBC, NoLog}(
            SimulationName = "Floating3D", SaveLocation = dir,
            SimulationTime = T(0.003), OutputTimes = T(0.001),
            VisualizeInParaview = false, OpenLogFile = false)
        particles = AllocateDataStructures(geometry, meta)
        RunSimulation(
            SimGeometry = geometry, SimMetaData = meta,
            SimConstants = SimulationConstants{T}(; dx, c₀ = 20.0),
            SimKernel = SPHKernelInstance{3, T}(WendlandC2(); dx, k = T(sqrt(3))),
            SimLogger = SimulationLogger(dir), SimParticles = particles,
            SimViscosity = Laminar(), SimDensityDiffusion = LinearDensityDiffusion(),
            SimTimeStepping = scheme)

        rows = CSV.File(joinpath(dir, "Floating3D_Floating.csv"))
        @test Symbol("Center:2") in propertynames(rows)
        @test Symbol("Orientation:3") in propertynames(rows)
        @test Symbol("Omega:2") in propertynames(rows)
        @test rows[1][Symbol("Center:2")] ≈ 0.5
        @test rows[1][Symbol("Orientation:0")] ≈ 1.0
    end

    @testset "floating bodies need their details" begin
        @test_throws ErrorException FloatingArrays(
            [SPHGeometry{2, Float64}(CSVFile = "", GroupMarker = 1, Type = Floating)],
            StructArray((Position = [SVector(0.0, 0.0)], GroupMarker = UInt[1], Type = [Floating])),
            SimulationConstants{Float64}())
    end
end

include("floating_single_neighbor.jl")
