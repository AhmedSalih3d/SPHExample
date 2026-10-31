using Test
using SPHExampleGPU
using StaticArrays
using StructArrays
using CUDA
using HDF5

const VR = SPHExampleGPU.VariableResolution

module VariableResolutionExample
include(joinpath(@__DIR__,"..","example","VResolutionDamBreak2d","Setup.jl"))
end

@testset "variable-resolution geometry and configuration" begin
    case = VariableResolutionExample.vresolution_dambreak_case(dx = 0.05)
    @test case.region.ratio == 2
    @test case.coarse.constants.dx == 2case.fine.constants.dx
    @test case.coarse.constants.m₀ == 4case.fine.constants.m₀
    @test case.coarse.constants.γ == 1
    @test isempty(case.fine.geometry[2].Particles)
    @test length(case.coarse.geometry[2].Particles) == 800
    for level in (case.coarse,case.fine)
        wall, water = getproperty.(level.geometry,:Particles)
        @test length(unique(wall.Position)) == length(wall)
        @test all(!iszero,wall.GhostNormals)
        @test all(all(isfinite,x) for x in wall.GhostPoints)
        @test all(water.Density .>= level.constants.ρ₀)
    end
    @test_throws ArgumentError VariableResolutionBox((1,0),(0,1))
    @test_throws ArgumentError VariableResolutionBox((0,0),(1,1); ratio = 1)
    @test_throws ArgumentError VariableResolutionBox((0,0),(1,1); buffer_h = 1)
    @test_throws ArgumentError VR.resolution_limits(VariableResolutionBox((0,0),(0.1,0.1)),0.1,0.05)
end

@testset "full Fourtakas diffusion remains active beside a wall" begin
    @test !FourtakasDensityDiffusion().full
    @test FourtakasDensityDiffusion(full = true).full
    if CUDA.functional()
        T = Float64
        constants = SimulationConstants{T}(2; dx = 0.02,g = 0)
        kernel = SPHKernelInstance{2,T}(WendlandC2(); h = 0.04)
        wall = [SVector(0.0,0.0)]
        geometry = [SPHGeometry{2,T}([SVector(0.0,0.02),SVector(0.02,0.02)];
                Density = [1000.,1001.],Type = Fluid,GroupMarker = 2),
            SPHGeometry{2,T}(Particles = particle_struct_array(wall,1000.;
                GhostPoints = [SVector(0.,0.04)],GhostNormals = [SVector(0.,0.04)]),Type = Fixed,GroupMarker = 1)]
        limits = VR.resolution_limits(VariableResolutionBox((10,10),(12,12)),0.04,0.02)
        domain = VR.make_resolution_domain(geometry,constants,kernel,limits,false,1)
        VR.rebuild_domain!(domain)
        p,s,c = domain.particles,domain.support,domain.cells
        rates = map((false,true)) do full
            launch_interactions!(s.dρdtI,p.Acceleration,p.Kernel,p.KernelGradient,s.∇Cᵢ,s.∇◌rᵢ,
                p.Position,p.Density,p.Velocity,p.Type,c.CellStart,p.CellID,c.grid_dev,HostStep(0.,0.),
                FourtakasDensityDiffusion(; full),ArtificialViscosity(),kernel,constants,Val(false),Val(false);
                boundary_forces = Val(false),viscous_dt = s.dual_state.viscous,fastmath = false)
            Array(s.dρdtI)[Array(p.Type) .== Fluid]
        end
        @test all(iszero,rates[1])
        @test all(!iszero,rates[2])
        meta = SimulationMetaData{2,T}(SimulationName="full diffusion",SaveLocation=tempdir(),GPULanesPerParticle=1)
        @test effective_configuration(meta,kernel,FourtakasDensityDiffusion(full=true),ArtificialViscosity();
            particle_count=3).effective.density_diffusion_operator == :fourtakas_full
        fluid_indices = findall(==(Fluid),Array(p.Type))
        invalid_density, invalid_velocity = Array(p.Density), Array(p.Velocity)
        invalid_density[fluid_indices[1]] = 600
        invalid_velocity[fluid_indices[2]] = SVector(NaN,0.)
        copyto!(p.Density,invalid_density)
        copyto!(p.Velocity,invalid_velocity)
        VR.retire_particles!(domain,limits)
        @test Array(domain.events)[3:4] == [1,1]
    end
end

function reconstruction_fixture(; collinear = false)
    constants = SimulationConstants{Float64}(2; dx = 0.1)
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = 0.15)
    grid = CellGrid{2}((Int32(-5),Int32(-5)),(Int32(11),Int32(11)),Int32(121))
    points = [SVector(x,y) for y in (collinear ? (0.0,) : -0.4:0.1:0.4) for x in -0.4:0.1:0.4]
    cells = [SPHExampleGPU.GPUCellGrid.linear_cell(grid,SPHExampleGPU.GPUCellGrid.cell_coords(p,kernel.H⁻¹)) for p in points]
    order = sortperm(cells)
    points = points[order]
    counts = zeros(Int32,grid.ncells)
    for cell in cells
        counts[cell] += 1
    end
    starts = [Int32(0); cumsum(counts)]
    density = [1000 + 2p[1] - 3p[2] for p in points]
    velocity = [SVector(p[1]+2p[2],2-p[1]) for p in points]
    limits = VR.resolution_limits(VariableResolutionBox((10,10),(12,12)),0.15,0.075)
    return (; constants,kernel,grid,points,starts,density,velocity,types = fill(Fluid,length(points)),limits)
end

function reconstruction_sample_kernel!(output, valid, query, points, velocity, density, types, starts, grid, kernel, constants, limits)
    value, found, _, _ = VR.reconstruct_state(query,points,velocity,density,types,starts,grid,kernel,constants,limits,false)
    output[1] = value
    valid[1] = found
    return nothing
end

@testset "variable-resolution reconstruction" begin
    for collinear in (false,true)
        fixture = reconstruction_fixture(; collinear)
        (; points,velocity,density,types,starts,grid,kernel,constants,limits) = fixture
        query = SVector(0.025,0.0)
        value, valid, _, _ = VR.reconstruct_state(query,points,velocity,density,types,starts,grid,kernel,constants,limits,false)
        @test valid
        @test all(isfinite,value)
        if !collinear
            @test value ≈ SVector(1000.05,0.025,1.975) atol=1e-9
        end
        if CUDA.functional()
            result, found = CUDA.zeros(SVector{3,Float64},1), CUDA.zeros(Bool,1)
            @cuda threads=1 reconstruction_sample_kernel!(result,found,query,CuArray(points),CuArray(velocity),
                CuArray(density),CuArray(types),CuArray(starts),grid,kernel,constants,limits)
            @test only(Array(found))
            @test only(Array(result)) ≈ value atol=1e-9
        end
        _, valid, _, _ = VR.reconstruct_state(SVector(100.,100.),points,velocity,density,types,starts,grid,kernel,constants,limits,false)
        @test !valid
    end
end

function moving_resolution_case()
    region = VariableResolutionBox((0.5,-0.3),(1.8,1.5))
    case = VariableResolutionExample.vresolution_dambreak_case(dx = 0.05,region = region,
        duration = 0.04,output_interval = 0.02,save_location = mktempdir())
    for level in (case.coarse,case.fine)
        fluid = level.geometry[2]
        data = StructArray((Position = fluid.Particles.Position, Density = fluid.Particles.Density,
            Velocity = fill(SVector(1f0,0f0),length(fluid.Particles))))
        level.geometry[2] = SPHGeometry{2,Float32}(Particles = data, Type = Fluid,GroupMarker = 2)
    end
    return case
end

@testset "predictor neighbors cross old cell boundaries" begin
    if CUDA.functional()
        constants = SimulationConstants{Float64}(2; dx=0.02,g=0,c₀=20)
        kernel = SPHKernelInstance{2,Float64}(WendlandC2();h=0.04)
        geometry = [SPHGeometry{2,Float64}(Particles = particle_struct_array(
            [SVector(0.039,0.02),SVector(0.121,0.02)],1000.;
            Velocity = [SVector(1.,0.),SVector(-1.,0.)]),Type=Fluid,GroupMarker=2)]
        limits = VR.resolution_limits(VariableResolutionBox((10,10),(12,12)),0.04,0.02)
        domain = VR.make_resolution_domain(geometry,constants,kernel,limits,false,1)
        VR.rebuild_domain!(domain)
        p,s = domain.particles,domain.support
        live = Array(p.Type) .== Fluid
        @test abs(diff(Array(p.CellID)[live])[1]) == 2
        SPHExampleGPU.GPUKernels.launch_dual_predictor!(s.Positionₙ⁺,s.Velocityₙ⁺,s.ρₙ⁺,
            p.Position,p.Velocity,p.Density,p.Acceleration,s.dρdtI,p.Type,0.004,constants)
        VR.predictor_force_stage!(domain,0.002)
        @test all(Array(s.dρdtI)[live] .> 0)
        @test all(!iszero,Array(p.Acceleration)[live])
    end
end

@testset "native variable-resolution transfer and output" begin
    if CUDA.functional()
        case = moving_resolution_case()
        result = run_variable_resolution_simulation(;
            coarse_geometry = case.coarse.geometry,fine_geometry = case.fine.geometry,
            coarse_constants = case.coarse.constants,fine_constants = case.fine.constants,
            coarse_kernel = case.coarse.kernel,fine_kernel = case.fine.kernel,
            region = case.region,simulation_time = case.simulation_time,output_interval = case.output_interval,
            save_location = case.save_location,particle_reserve = 2048)
        @test result.time == 0.04
        @test result.times == [0.0,0.02,0.04]
        @test result.iterations > 1
        for domain in (result.coarse,result.fine)
            p = domain.particles
            live = Array(p.Type) .== Fluid
            @test all(all(isfinite,x) for x in Array(p.Position)[live])
            @test all(all(isfinite,v) for v in Array(p.Velocity)[live])
            @test all(700 .<= Array(p.Density)[live] .<= 1300)
            ids = Array(p.ID)[Array(p.Type) .!= Frozen]
            @test length(unique(ids)) == length(ids)
            owners = Array(p.GhostIndex)
            positions, ghosts, normals = Array(p.Position), Array(p.GhostPoints), Array(p.GhostNormals)
            @test all(Array(p.Type)[owners] .== Fixed)
            @test all(ghosts[i] - positions[i] ≈ normals[i] for i in owners)
        end
        @test isfile(joinpath(result.output,"DamBreak2D_VRes.vtkhdf.series"))
        h5open(joinpath(result.output,last(result.files)),"r") do file
            data = file["VTKHDF/PointData"]
            levels, mass, spacing = read(data["Resolution"]), read(data["Mass"]), read(data["Spacing"])
            @test Set(levels) == Set(Int8[0,1])
            @test all(mass .≈ 1000spacing.^2)
            @test all(==(1),read(data["Owned"]))
            @test all(isfinite,read(data["Pressure"]))
            ids = read(data["ID"])
            @test length(unique(ids)) == length(ids)
        end
        # A deliberately tiny pool must report failed allocation, rather than
        # silently losing incoming flux or overwriting an active particle.
        limits = result.configuration.limits
        small = VR.make_resolution_domain(case.fine.geometry,case.fine.constants,case.fine.kernel,limits,true,1)
        source = VR.make_resolution_domain(case.coarse.geometry,case.coarse.constants,case.coarse.kernel,limits,false,1)
        receiver_geometry = [case.fine.geometry[1],SPHGeometry{2,Float32}(
            Particles = case.fine.geometry[2].Particles,Type = Fluid,GroupMarker = 7)]
        receiver = VR.make_resolution_domain(receiver_geometry,case.fine.constants,case.fine.kernel,limits,true,2048)
        VR.retire_particles!(source,limits)
        VR.rebuild_domain!(source)
        VR.rebuild_domain!(small)
        VR.rebuild_domain!(receiver)
        @test VR.transfer_particles!(receiver,source,limits,0.04)
        @test Array(receiver.events)[1] > 0
        @test all(==(7),Array(receiver.particles.GroupMarker)[Array(receiver.particles.Type) .== Fluid])
        @test maximum(Array(receiver.interface.mass)) < receiver.constants.m₀
        failure = try
            VR.transfer_particles!(small,source,limits,1.0)
            nothing
        catch error
            error
        end
        @test failure isa ErrorException
        @test occursin("reserve exhausted",sprint(showerror,failure))
    else
        @test_skip "CUDA not available"
    end
end

@testset "Float64 native symplectic stages" begin
    if CUDA.functional()
        case = VariableResolutionExample.vresolution_dambreak_case(dx = 0.05,FloatType = Float64)
        result = run_variable_resolution_simulation(;
            coarse_geometry = case.coarse.geometry,fine_geometry = case.fine.geometry,
            coarse_constants = case.coarse.constants,fine_constants = case.fine.constants,
            coarse_kernel = case.coarse.kernel,fine_kernel = case.fine.kernel,
            region = case.region,simulation_time = 0.001,output_interval = 0.001,
            save_location = mktempdir(),particle_reserve = 32)
        @test result.time == 0.001
        @test all(Array(d.events)[3] == 0 for d in (result.coarse,result.fine))
    end
end

if get(ENV,"SPH_VRES_LONG_TEST","0") == "1" && CUDA.functional()
    @testset "dam-break front enters native fine domain" begin
        result = VariableResolutionExample.run_vresolution_dambreak(dx = 0.05,duration = 0.65,
            output_interval = 0.05,save_location = mktempdir())
        @test Array(result.fine.events)[1] > 0
        @test result.time == 0.65
        @test all(Array(d.events)[3] == 0 for d in (result.coarse,result.fine))
        println("Variable-resolution long-run output: ",result.output)
    end
end
