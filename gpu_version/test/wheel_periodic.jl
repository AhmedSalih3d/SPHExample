module WheelPeriodicTests

using Test, SPHExampleGPU, StaticArrays, LinearAlgebra, CUDA, HDF5

# Load the module without starting the example's interactive simulation.
Base.include(expr -> expr isa Expr && expr.head === :module ? expr : nothing,
    @__MODULE__, joinpath(@__DIR__, "..", "example", "WheelPeriodic3d.jl"))
using .PeriodicWheelCase

const PB = SPHExampleGPU.PeriodicBoundaries
const F = SPHExampleGPU.GPUFloating
const CL = SPHExampleGPU.SPHCellList

@testset "periodic wheel configuration and 3D geometry" begin
    c = WheelPeriodicConfig()
    setup = prepare_wheel_periodic_3d(c)
    case, p = setup.case, setup.particles
    @test all(g -> isempty(g.CSVFile) && g.Motion === nothing, case.geometry)
    @test getproperty.(case.geometry, :Type) == [Fixed, Floating, Fluid]
    @test length(unique(p.Position)) == length(p)
    @test length(unique(p.ID)) == length(p)
    @test case.constants.m₀ ≈ case.constants.ρ₀ * case.constants.dx^3
    @test setup.meta.Periodicity == case.periodicity
    @test setup.meta.GPUCellSubdivision == 2
    @test setup.meta.GPUDoublePosition
    @test setup.meta.GPUClipTimeStepToOutput
    @test case.periodicity.axis == 1
    @test case.periodicity.lower == c.domain_min[1]
    @test case.periodicity.upper == c.domain_max[1]
    @test case.periodicity.translation == SVector(c.periodic_min - c.periodic_max, 0, 0)
    @test case.constraints.hinge && case.constraints.hinge_prescribed
    @test case.constraints.point == c.wheel_center
    @test case.constraints.axis == SVector(0f0, 1f0, 0f0)
    @test case.constraints.hinge_velocity == Float32(c.wheel_angular_speed)
    @test case.geometry[2].Floating.PauseTime == 0

    fluid = p.Position[p.Type .== Fluid]
    fixed = p.Position[p.Type .== Fixed]
    wheel = p.Position[p.Type .== Floating]
    @test all(x -> c.periodic_min <= x[1] < c.periodic_max, vcat(fluid, fixed))
    # Both end planes contain water and identical wall sections; there are no
    # end caps and no duplicated periodic particles.
    first_x, last_x = extrema(first, fluid)
    @test first_x ≈ c.periodic_min
    @test last_x ≈ c.periodic_max - c.particle_spacing
    sections(xs, x) = Set((p[2], p[3]) for p in xs if isapprox(p[1], x; atol = 1e-10))
    @test sections(fluid, first_x) == sections(fluid, last_x)
    @test sections(fixed, first_x) == sections(fixed, last_x)
    @test all(x -> x[3] <= c.fluid_level, fluid)
    @test all(x -> x[3] >= c.domain_min[3], fluid)
    @test maximum(x -> x[2], wheel) - minimum(x -> x[2], wheel) > c.wheel_width
    @test minimum(x -> x[3], wheel) > c.domain_min[3]
    @test maximum(x -> x[3], wheel) > c.fluid_level
    @test all(x -> hypot(x[1] - c.wheel_center[1], x[3] - c.wheel_center[3]) <= c.wheel_radius + 1e-10, wheel)
    # The rim/spoke model retains open water between spokes, rather than a disk.
    @test any(x -> norm(x - c.wheel_center) < 0.9c.wheel_radius, fluid)

    seam_x = SVector(c.periodic_max + 0.1, 0.1, 0.1)
    @test PB.wrap_periodic_position(seam_x, case.periodicity) ≈ seam_x + case.periodicity.translation
    @test PB.wrap_periodic_position(seam_x - 2case.periodicity.translation, case.periodicity) ≈
        PB.wrap_periodic_position(seam_x, case.periodicity)
end

@testset "every fluid particle gets precisely the configured initial velocity" begin
    for T in (Float32, Float64), velocity in (SVector(0.13, -0.07, 0.021), SVector(0., 0., 0.))
        c = WheelPeriodicConfig(initial_fluid_velocity = velocity, wheel_angular_speed = 1.25)
        setup = prepare_wheel_periodic_3d(c; FloatType = T)
        p = setup.particles
        fluid, nonfluid = p.Type .== Fluid, p.Type .!= Fluid
        @test any(fluid) && any(nonfluid)
        @test all(==(SVector{3,T}(velocity)), p.Velocity[fluid])
        @test all(iszero, p.Velocity[nonfluid])
        @test all(==(SVector{3,T}(velocity)), setup.case.geometry[3].Particles.Velocity)
        # Include distant water upstream AND downstream, not just wheel neighbors.
        for side in (-1, 1)
            far = findall(i -> fluid[i] && side * p.Position[i][1] > 2c.wheel_radius, eachindex(fluid))
            @test !isempty(far)
            @test all(==(SVector{3,T}(velocity)), p.Velocity[far])
        end
        # Allocation copies input fields; a previous run cannot mutate the IC.
        p.Velocity[findfirst(fluid)] = SVector{3,T}(9, 8, 7)
        fresh = AllocateDataStructures(setup.case.geometry, setup.meta)
        @test all(==(SVector{3,T}(velocity)), fresh.Velocity[fluid])
    end
end

@testset "wheel rejects incompatible or unresolved inputs" begin
    for options in ((; particle_spacing = 0), (; particle_spacing = 0.04),
            (; wheel_radius = NaN), (; periodic_axis = 0), (; periodic_axis = 2), (; periodic_axis = 4),
            (; periodic_shift = SVector(0., 0.1, 0.)),
            (; periodic_max = 2.0), (; particle_spacing = 0.026),
            (; wheel_axis = SVector(1., 0., 0.)), (; wheel_axis = SVector(0., 0., 0.)),
            (; initial_fluid_velocity = SVector(Inf, 0., 0.)), (; wheel_angular_speed = NaN),
            (; sound_speed = 1), (; simulation_time = 0), (; output_interval = Inf),
            (; paddle_count = 2), (; shaft_length = 2.0), (; fluid_level = 0.4),
            (; wheel_center = SVector(0., 0., 0.1)))
        @test_throws ArgumentError wheel_periodic_geometry(WheelPeriodicConfig(; options...))
    end
    mktempdir() do save
        sentinel = joinpath(save, "existing.txt")
        write(sentinel, "keep me")
        @test_throws ArgumentError run_wheel_periodic_3d(; save_location = save)
        @test read(sentinel, String) == "keep me"
        @test readdir(save) == ["existing.txt"]
    end
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "GPU initialization, hinge rotation and rigid seam ownership" begin
        # Deliberately straddle the seam: the body remains whole and its fixed
        # hinge stays in the same image through sorting and integration.
        c = WheelPeriodicConfig(wheel_center = SVector(0.875, 0., 0.375),
            initial_fluid_velocity = SVector(0.2, -0.03, 0.01), wheel_angular_speed = -1.5)
        setup = prepare_wheel_periodic_3d(c)
        case, host = setup.case, setup.particles
        gpu = upload_particles(host, setup.meta)
        floating = FloatingArrays(case.geometry, host, case.constants; position_type = Float64)
        F.initialize_floating_velocities!(floating, gpu)
        velocity = Array(gpu.Velocity)
        omega = SVector(0f0, Float32(c.wheel_angular_speed), 0f0)
        for i in eachindex(host.Type)
            expected = host.Type[i] == Fluid ? SVector{3,Float32}(c.initial_fluid_velocity) :
                host.Type[i] == Floating ? cross(omega, SVector{3,Float32}(host.Position[i] - c.wheel_center)) : zero(omega)
            @test velocity[i] ≈ expected atol = 2e-7
        end
        cl = CellListWorkspace{3,Float64}(length(gpu); periodicity = case.periodicity)
        CL.rebuild_cell_list!(gpu, cl, case.kernel.H⁻¹; floating)
        sorted_ids = Array(gpu.ID)
        @test Array(gpu.Velocity) ≈ velocity[sorted_ids] atol = 2e-7
        original = copy(host.Position)
        step = HostStep(0.01f0, 0f0)
        for _ in 1:10
            F.launch_floating_update!(floating, step, case.constants.g, false)
            F.launch_floating_update!(floating, step, case.constants.g, true)
            F.launch_floating_particles!(floating, gpu.Position, gpu.Velocity, gpu.Position,
                gpu.Type, gpu.GroupMarker, step, true)
        end
        state = floating_state(floating)
        @test only(state.omega) ≈ omega
        @test only(state.center) ≈ c.wheel_center atol = 2e-6
        @test only(state.velocity) ≈ zero(omega) atol = 2e-6
        positions = Array(gpu.Position)[sortperm(Array(gpu.ID))]
        angle = c.wheel_angular_speed * 0.1
        for i in findall(==(Floating), host.Type)
            r = original[i] - c.wheel_center
            expected = c.wheel_center + SVector(cos(angle)*r[1] + sin(angle)*r[3], r[2], -sin(angle)*r[1] + cos(angle)*r[3])
            @test positions[i] ≈ expected atol = 2e-6
        end
        @test any(x -> x[1] >= c.periodic_max, positions[host.Type .== Floating])
        @test only(Array(floating.constraints)).point == c.wheel_center
    end

    @testset "wheel solver startup and initial VTKHDF frame" begin
        outcomes = map((false, true)) do graph
            mktempdir() do save
                c = WheelPeriodicConfig(simulation_time = 0.02, output_interval = 0.01,
                    initial_fluid_velocity = SVector(0.4, 0.03, -0.02))
                result = redirect_stdout(devnull) do
                    run_wheel_periodic_3d(c; save_location = save, use_graph = graph)
                end
                p = result.final_state.particles
                @test result.final_state.complete
                @test result.diagnostics.frozen_particles == 0
                @test all(isfinite, p.Density) && all(>(0), p.Density)
                @test all(x -> all(isfinite, x), p.Position)
                @test result.configuration.effective.periodicity.axis == 1
                @test only(result.final_state.floating.omega) ≈ SVector(0f0, -2f0, 0f0)
                @test only(result.final_state.floating.center) ≈ c.wheel_center atol = 2e-6
                @test result.configuration.execution.graph_capture_failures == 0
                graph && @test result.configuration.execution.graph_batches > 0
                h5open(joinpath(save, "WheelPeriodic3D.vtkhdf"), "r") do io
                    root = io["VTKHDF"]
                    types = read(root["PointData/Type"])[1:length(p)]
                    v = read(root["PointData/Velocity"])[:, 1:length(p)]
                    x = read(root["Points"])[:, 1:length(p)]
                    @test first(read(root["Steps/Values"])) == 0
                    for i in eachindex(types)
                        if types[i] == UInt8(Fluid)
                            @test v[:, i] == Float32.(c.initial_fluid_velocity)
                        elseif types[i] == UInt8(Fixed)
                            @test iszero(v[:, i])
                        else
                            @test v[:, i] ≈ cross(SVector(0f0, -2f0, 0f0), Float32.(x[:, i] - c.wheel_center)) atol = 2e-7
                        end
                    end
                end
                @test occursin("initial_fluid_velocity = [0.4, 0.03, -0.02]", read(joinpath(save, "WheelPeriodic3D_config.txt"), String))
                order = sortperm(p.ID)
                return (; positions = p.Position[order], velocities = p.Velocity[order], density = p.Density[order])
            end
        end
        @test outcomes[1] == outcomes[2]
    end
else
    @info "CUDA unavailable: wheel geometry/configuration and host initialization passed; GPU hinge, sorting, stepping and output checks skipped."
end

end # module WheelPeriodicTests
