module SloshingTankTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5

let path = joinpath(@__DIR__, "..", "example", "SloshingTank.jl")
    source = first(split(read(path, String), "# Including this file starts the simulation."))
    include_string(@__MODULE__, source, path)
end

@testset "generated sloshing tank geometry" begin
    for T in (Float32, Float64)
        case = sloshing_tank_geometry(; FloatType = T)
        wall, water = case.geometry
        @test length(wall.Particles.Position) == 4248
        @test length(water.Particles.Position) == 449 * 46
        @test wall.Type == Fixed && water.Type == Fluid
        @test wall.GroupMarker == 1 && water.GroupMarker == 2
        @test isempty(intersect(Set(wall.Particles.Position), Set(water.Particles.Position)))
        @test length(unique(wall.Particles.Position)) == length(wall.Particles.Position)
        @test all(p -> -0.45 < p[1] < 0.45 && 0 < p[2] < 0.093, water.Particles.Position)
        @test maximum(last, water.Particles.Position) ≈ 0.092
        @test all(p -> !(abs(p[1]) < 0.45 - 1e-10 && 1e-10 < p[2] < 0.508 - 1e-10), wall.Particles.Position)
        @test minimum(first, wall.Particles.Position) ≈ -0.454
        @test maximum(last, wall.Particles.Position) ≈ 0.512
        @test case.constants.m₀ ≈ T(1000 * 0.002^2)
        @test case.constants.c₀ ≈ T(30sqrt(9.81 * 0.09))
        @test case.kernel.h ≈ T(0.91924sqrt(2) * 0.002)
        pressure = EquationOfStateGamma7.(water.Particles.Density, case.constants.c₀, case.constants.ρ₀)
        expected = 1000 * 9.81 .* (case.water_level .- last.(water.Particles.Position))
        @test maximum(abs.(pressure .- expected)) <= (T === Float32 ? 0.15 : 2e-7)
    end
    @test_throws ArgumentError sloshing_tank_geometry(; dx = 0)
    @test_throws ArgumentError sloshing_tank_geometry(; dx = 0.1)
    @test_throws ArgumentError sloshing_tank_geometry(; dx = NaN)
end

@testset "complete measured sloshing excitation" begin
    for T in (Float32, Float64)
        force = sloshing_tank_force(T)
        @test length(force.times) == 167001
        @test first(force.times) == 0 && last(force.times) == T(8.35)
        @test all(>(zero(T)), diff(force.times))
        @test !force.global_gravity
        @test force.center == SVector(0.45, 0.0, 0.0)
        @test force.linear[1] ≈ SVector{3,T}(-1.96e-10, 0, -9.81)
        @test force.angular[1] ≈ SVector{3,T}(0, 0.312057592, 0)
        @test force.angular[end] ≈ SVector{3,T}(0, -0.18887913, 0)
        # At the first nonzero time: tangential vertical acceleration at x=0
        # is 0.45*alpha_y. Linear gravity is cancelled before the solver adds it.
        acceleration = external_acceleration(force, zero(SVector{2,T}), zero(SVector{2,T}), T(0.00005), T(9.81))
        @test acceleration ≈ SVector{2,T}(-0.000204, 0.14052414645) atol = T === Float32 ? 1e-6 : 2e-10
        @test force.linear_velocity[2] ≈ force.linear[2] * T(0.00005)
    end
    centered = sloshing_tank_force(Float64; center = (0, 0))
    @test centered.center == zero(SVector{3,Float64})
    @test_throws ArgumentError run_sloshing_tank(; simulation_time = 9, visualize = false, open_log_file = false)
    @test_throws ArgumentError run_sloshing_tank(; simulation_time = 0)
    @test_throws ArgumentError run_sloshing_tank(; output_interval = 0)
end

@testset "measured physical tank rotation" begin
    rotation = sloshing_tank_rotation()
    @test SLOSHING_DRIVING_MODE == :rotation
    @test length(rotation.times) == 167001
    @test first(rotation.times) == 0 && last(rotation.times) == 8.35
    @test rotation.center == SVector(0., 0.)
    @test rotation.angles[2] == -deg2rad(2.23547e-8)
    @test sloshing_tank_geometry(; dx = 0.01, driving_mode = :rotation).geometry[1].Type == Moving
    @test_throws ArgumentError run_sloshing_tank(; driving_mode = :invalid)
end

if CUDA.functional()
    @testset "sloshing tank GPU forcing, probes and graph replay" begin
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            results = map((false, true)) do graph
                mktempdir() do save
                    result = run_sloshing_tank(; driving_mode = :acceleration, dx = 0.01, simulation_time = 0.03,
                        output_interval = 0.015, visualize = false, open_log_file = false,
                        time_stepping = scheme, use_graph = graph, double_position = true,
                        motion_path = joinpath(save, "unused_motion.dat"),
                        save_location = save)
                    particles = result.final_state.particles
                    @test result.final_state.complete
                    @test result.termination_reason == :completed
                    @test result.diagnostics.frozen_particles == 0
                    @test all(isfinite, particles.Density)
                    @test all(p -> all(isfinite, p), particles.Position)
                    @test all(v -> all(isfinite, v), particles.Velocity)
                    @test eltype(particles.Position) == SVector{2,Float64}
                    @test result.configuration.execution.graph_capture_failures == 0
                    graph && @test result.configuration.execution.graph_batches > 0
                    wall = particles.Type .== Fixed
                    initial_wall = sloshing_tank_geometry(; dx = 0.01).geometry[1].Particles.Position
                    @test Set(particles.Position[wall]) == Set(initial_wall)
                    @test all(iszero, particles.Velocity[wall])
                    @test maximum(v -> sqrt(sum(abs2, v)), particles.Velocity[.!wall]) > 1e-4
                    h5open(joinpath(save, "SloshingTank.vtkhdf"), "r") do file
                        pressure = file["VTKHDF/PressureProbes"]
                        heights = file["VTKHDF/WaterColumnProbes"]
                        @test all(isfinite, read(pressure["PointData/Pressure"]))
                        @test all(isfinite, read(heights["PointData/WaterColumnHeight"]))
                        @test length(read(pressure["Steps/Values"])) == result.diagnostics.output_frames
                        @test HDF5.attributes(pressure)["Sampling"][] == "normalized SPH interpolation"
                    end
                    order = sortperm(particles.ID)
                    return (; position = particles.Position[order], velocity = particles.Velocity[order],
                        density = particles.Density[order], time = result.final_state.time,
                        iteration = result.final_state.iteration)
                end
            end
            @test results[1] == results[2]
        end
    end

    @testset "physical tank rotation and attached pressure probe" begin
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            runs = map((false, true)) do graph
                mktempdir() do save
                    result = run_sloshing_tank(; driving_mode = :rotation, dx = 0.01,
                        simulation_time = 0.25, output_interval = 0.1,
                        visualize = false, open_log_file = false, double_position = true,
                        time_stepping = scheme, use_graph = graph,
                        acceleration_path = joinpath(save, "unused_acceleration.csv"), save_location = save)
                    p = result.final_state.particles
                    initial = AllocateDataStructures(sloshing_tank_geometry(; dx = 0.01, driving_mode = :rotation).geometry;
                        position_type = Float64)
                    reference = Dict(id => position for (id, position) in zip(initial.ID, initial.Position))
                    rotation = sloshing_tank_rotation()
                    time = result.final_state.time
                    _, omega = rotation_state(rotation, time)
                    wall = findall(==(Moving), p.Type)
                    @test result.final_state.complete
                    @test result.diagnostics.cell_list_rebuilds > 1
                    @test result.diagnostics.frozen_particles == 0
                    @test result.configuration.execution.graph_capture_failures == 0
                    graph && @test result.configuration.execution.graph_batches > 0
                    @test all(isfinite, p.Density)
                    for i in wall
                        expected = rotation_position(rotation, reference[p.ID[i]], time)
                        @test p.Position[i] ≈ expected atol = 2e-12
                        @test p.Velocity[i] ≈ SVector(-omega*expected[2], omega*expected[1]) atol = 2e-7
                    end
                    h5open(joinpath(save, "SloshingTank.vtkhdf"), "r") do file
                        root = file["VTKHDF"]
                        probe = root["PressureProbes"]
                        times = read(probe["Steps/Values"])
                        points = read(probe["Points"])
                        @test !haskey(root, "WaterColumnProbes")
                        @test all(isfinite, read(probe["PointData/Pressure"]))
                        @test read(probe["Steps/PointOffsets"]) == collect(0:(length(times)-1))
                        for (frame, t) in enumerate(times)
                            expected = rotation_position(rotation, (-0.4461, 0.093), t)
                            @test points[:, frame] ≈ [expected[1], expected[2], 0] atol = 1e-12
                        end
                    end
                    order = sortperm(p.ID)
                    return (; position = p.Position[order], velocity = p.Velocity[order], density = p.Density[order], time)
                end
            end
            @test runs[1] == runs[2]
        end
    end
end

end
