module ImpingingJetTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const RUN_IMPINGING_JET = false
include(joinpath(@__DIR__, "..", "example", "ImpingingJet2d.jl"))

@testset "impinging jet geometry and mDBC configuration" begin
    @test_throws ArgumentError impinging_jet_setup(duration = 0)
    @test_throws ArgumentError impinging_jet_setup(spacing = 0.003)
    for T in (Float32, Float64)
        setup = impinging_jet_setup(float_type = T)
        geometry, flow = setup.geometry, setup.flow
        @test typeof(setup.metadata).parameters[5] === SimpleMDBC
        @test length(geometry[1].Particles) == 812 * 4
        @test length(geometry[2].Particles) == 61 * 10
        @test length(geometry[3].Particles) == 61 * 4
        @test length(geometry[4].Particles) == length(geometry[5].Particles) == 4 * 60
        @test all(==(SVector{2,T}(0,-20)), geometry[2].Particles.Velocity)
        @test all(p -> p[2] >= 0.125, geometry[3].Particles.Position)
        @test geometry[6].Type == Frozen
        @test length(geometry[6].Particles) == 61 * (ceil(Int,flow.duration * flow.velocity / flow.spacing) + 1)
        wall = geometry[1].Particles
        @test all(i -> isapprox(wall.GhostPoints[i], SVector(wall.Position[i][1], flow.spacing-wall.Position[i][2])), eachindex(wall.Position))
        @test all(i -> isapprox(wall.GhostNormals[i], wall.GhostPoints[i]-wall.Position[i]), eachindex(wall.Position))
        @test all(i -> isapprox((wall.Position[i][2]+wall.GhostPoints[i][2])/2, flow.spacing/2), eachindex(wall.Position))
        @test all(p -> p[2] > 0, wall.GhostNormals)
    end
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "inactive inlet reserves stay outside neighbor cells" begin
        for T in (Float32, Float64)
            geometry = [SPHGeometry{2,T}([SVector(0.,0.), SVector(0.01,0.)]; Density = 1000, Type = Fluid, GroupMarker = 1),
                SPHGeometry{2,T}(fill(SVector(0.,1e6),4096); Density = 1000, Type = Frozen, GroupMarker = 5)]
            host = AllocateDataStructures(geometry)
            gpu = upload_particles(host)
            cl = CellListWorkspace{2,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, T(50); inactive_particles = true, sort_inactive = false)
            types, cells, ids, pos = Array(gpu.Type), Array(gpu.CellID), Array(gpu.ID), Array(gpu.Position)
            @test all(i -> cells[i] == 1, findall(==(Frozen),types))
            @test all(i -> cells[i] > 1, findall(==(Fluid),types))
            @test cl.grid.dims[2] < 10
            @test all(i -> pos[i] == host.Position[ids[i]], eachindex(ids))
        end
    end
    @testset "jet prescribed density and extrapolated outlet velocity" begin
        O = SPHExampleGPU.OpenBoundaries
        for T in (Float32, Float64)
            source = [SVector(-0.194,0.01), SVector(0.194,0.01)]
            geometry = [SPHGeometry{2,T}(; Particles = particle_struct_array(source,T(1010);
                    Velocity = fill(SVector{2,T}(3,-2),2)), Type = Fluid, GroupMarker = 1),
                SPHGeometry{2,T}([SVector(0.,0.125)]; Density = 900, Type = Fluid, GroupMarker = 2),
                SPHGeometry{2,T}([SVector(-0.205,0.01)]; Density = 900, Type = Fluid, GroupMarker = 3),
                SPHGeometry{2,T}(; Particles = particle_struct_array([SVector(0.205,0.01), SVector(0.205,0.1)],T(900);
                    Velocity = [SVector{2,T}(0,0),SVector{2,T}(7,1)]), Type = Fluid, GroupMarker = 4)]
            gpu = upload_particles(AllocateDataStructures(geometry))
            setup = impinging_jet_setup(spacing = 0.005, duration = 0.01, float_type = T)
            kernel, constants = setup.kernel, setup.constants
            cl = CellListWorkspace{2,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, kernel.H⁻¹)
            O.launch_open_boundary_values!((;configuration = setup.flow), gpu.Position, gpu.Velocity, gpu.Density,
                gpu.Type, gpu.GroupMarker, cl.CellStart, cl.grid_dev, kernel, constants,
                SPHExampleGPU.GPUStepState.HostStep(T(0.001),0.))
            groups, pos, rho, vel = Array(gpu.GroupMarker), Array(gpu.Position), Array(gpu.Density), Array(gpu.Velocity)
            for i in eachindex(groups)
                if groups[i] == 2
                    @test rho[i] == 1000
                    @test vel[i] == SVector{2,T}(0,-20)
                elseif groups[i] in (3,4)
                    @test rho[i] == 1000
                    @test pos[i][2] < 0.02 ? isapprox(vel[i],SVector{2,T}(3,-2)) : vel[i] == SVector{2,T}(7,1)
                end
            end
        end
    end
    @testset "jet events respect finite inlet and outlet zones" begin
        O = SPHExampleGPU.OpenBoundaries
        for T in (Float32, Float64)
            setup = impinging_jet_setup(spacing = 0.005, duration = 0.001, float_type = T)
            interior = [SVector(0.,0.13), SVector(0.05,0.13),
                SVector(-0.201,0.015), SVector(0.201,0.015),
                SVector(-0.201,0.04), SVector(0.201,0.04)]
            region(points,marker) = SPHGeometry{2,T}(; Particles = particle_struct_array(points,T(1000);
                Velocity = fill(SVector{2,T}(1,1),length(points))), Type = Fluid, GroupMarker = marker)
            geometry = [region(interior,1), region([SVector(-0.199,0.015)],3), region([SVector(0.199,0.015)],4)]
            host = AllocateDataStructures(geometry)
            gpu = upload_particles(host)
            support = GPUSupportArrays{2,T}(length(gpu))
            cl = CellListWorkspace{2,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, setup.kernel.H⁻¹)
            state = (; configuration = setup.flow, emission_time = CUDA.fill(Inf,length(gpu)),
                emission_position = CUDA.zeros(SVector{2,Float64},length(gpu)), counters = CUDA.zeros(Int32,2))
            step = SPHExampleGPU.GPUStepState.StepState{T}(dt = T(0.001), time = 0.)
            O.launch_open_boundary_events!(state, gpu, support, cl, setup.kernel, setup.constants, step)
            order = sortperm(Array(gpu.ID))
            types, groups, velocity = Array(gpu.Type)[order], Array(gpu.GroupMarker)[order], Array(gpu.Velocity)[order]
            @test types == [Frozen; fill(Fluid,7)]
            @test groups == [1,1,3,4,1,1,1,1]
            @test Array(state.counters) == [0,1]
            @test iszero(velocity[1])
            @test all(==(SVector{2,T}(1,1)),velocity[2:end])
            @test isapprox(Array(gpu.Position)[order],host.Position; rtol = 5eps(T))
        end
    end
    @testset "driver mDBC correction refreshes sorted wall caches and resets dry walls" begin
        CL = SPHExampleGPU.SPHCellList
        for T in (Float32, Float64), cache in (false,true)
            setup = impinging_jet_setup(spacing = 0.005, duration = 0.001, float_type = T)
            wall = setup.geometry[1].Particles
            owner = findfirst(p -> abs(p[1]) < 1e-10 && abs(p[2]) < 1e-10, wall.Position)
            @test owner !== nothing
            fluid = [SVector(x,y) for x in -0.01:0.005:0.01 for y in 0.005:0.005:0.015]
            geometry = [SPHGeometry{2,T}(; Particles = particle_struct_array([wall.Position[owner]],T(900);
                    GhostPoints = [wall.GhostPoints[owner]], GhostNormals = [wall.GhostNormals[owner]]), Type = Fixed, GroupMarker = 10),
                SPHGeometry{2,T}(fluid; Density = 1010, Type = Fluid, GroupMarker = 1)]
            gpu = upload_particles(AllocateDataStructures(geometry); boundary_only = true)
            cl = CellListWorkspace{2,T}(length(gpu))
            CL.rebuild_cell_list!(gpu, cl, setup.kernel.H⁻¹)
            support = GPUSupportArrays{2,T}(length(gpu); cache_interaction_fields = cache,
                boundary_slots = gpu.BoundarySlot, boundary_count = length(gpu.GhostIndex))
            step = SPHExampleGPU.GPUStepState.HostStep(T(0.0001),0.)
            ctx = (; gpu, sup = support, cl, SimKernel = setup.kernel, SimConstants = setup.constants,
                threads = 128, lanes = Val(1), boundary_mode = SimpleMDBC(), open_boundary = (; configuration = setup.flow))
            CL.enqueue_boundary_correction!(ctx, step, false)
            index = only(Array(gpu.GhostIndex))
            @test Array(gpu.GroupMarker)[index] == 10
            @test isapprox(Array(gpu.Density)[index], T(1010); rtol = 1e-5)
            @test isapprox(only(Array(gpu.GhostPoints)), wall.GhostPoints[owner]; rtol = 5eps(T))
            @test isapprox(only(Array(gpu.GhostNormals)), wall.GhostNormals[owner]; rtol = 5eps(T))
            if cache
                @test isapprox(Array(support.InvDensity)[index], inv(T(1010)); rtol = 1e-5)
                pressure = particle_pressure(T(1010), Fixed, setup.constants.c₀, setup.constants.ρ₀, SimpleMDBC())
                @test isapprox(Array(support.Pressure)[index], pressure; rtol = 2e-4)
            end
            # A formerly wet jet wall must not retain its last pressure when dry.
            types = Array(gpu.Type)
            types[findall(==(Fluid),types)] .= Frozen
            copyto!(gpu.Type, types)
            CL.enqueue_boundary_correction!(ctx, step, false)
            @test Array(gpu.Density)[index] == setup.constants.ρ₀
            if cache
                @test Array(support.InvDensity)[index] == inv(setup.constants.ρ₀)
                @test iszero(Array(support.Pressure)[index])
            end
        end
    end
    @testset "jet mDBC impact and particle balance" begin
        variants = [(SimpleMDBC,graph,cache) for graph in (false,true) for cache in (false,true)]
        push!(variants,(UpdatedMDBC,true,true))
        for (mode,graph,cache) in variants
            mktempdir() do directory
                setup = impinging_jet_setup(spacing = 0.0025, duration = 0.008, output_interval = 0.008,
                    save_location = directory, use_graph = graph, cache_interaction_fields = cache, boundary_mode = mode)
                @test typeof(setup.metadata).parameters[5] === mode
                result = run_impinging_jet(spacing = 0.0025, duration = 0.008, output_interval = 0.008,
                    save_location = directory, use_graph = graph, cache_interaction_fields = cache, boundary_mode = mode)
                particles, state = result.final_state.particles, result.final_state.open_boundary
                active = findall(==(Fluid),particles.Type)
                impacted_wall = findall(i -> particles.Type[i] == Fixed && abs(particles.Position[i][1]) <= 0.025, eachindex(particles.Type))
                @test result.final_state.complete
                @test state.emitted_particles > 0
                @test state.active_fluid == state.initial_active_fluid+state.emitted_particles-state.retired_particles
                @test all(i -> isfinite(particles.Density[i]) && particles.Density[i] > 0,active)
                @test all(i -> all(isfinite,particles.Velocity[i]),active)
                @test all(i -> isfinite(particles.Density[i]),impacted_wall)
                @test minimum(particles.Position[i][2] for i in active) < 0.02
                @test maximum(abs(particles.Position[i][1]) for i in active) <= 0.21+1e-6
                @test maximum(particles.Density[impacted_wall]) > setup.constants.ρ₀+0.01
            end
        end
    end
end
end
