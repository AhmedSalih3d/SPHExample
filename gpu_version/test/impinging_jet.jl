module ImpingingJetTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__,"..","example","ImpingingJet2d.jl"))

@testset "impinging jet geometry and configuration" begin
    @test_throws ArgumentError ImpingingJet{Float32}(Velocity=-20)
    @test_throws ArgumentError ImpingingJet{Float32}(Duration=0)
    @test_throws ArgumentError impinging_jet_geometry(ImpingingJet{Float32}(Spacing=0.003))
    for T in (Float32,Float64)
        flow = ImpingingJet{T}()
        geometry = impinging_jet_geometry(flow)
        @test length(geometry[1].Particles) == 812*4
        @test length(geometry[2].Particles) == 61*10
        @test length(geometry[3].Particles) == 61*4
        @test length(geometry[4].Particles) == length(geometry[5].Particles) == 4*60
        @test all(==(SVector{2,T}(0,-20)),geometry[2].Particles.Velocity)
        @test all(p -> p[2] >= 0.125,geometry[3].Particles.Position)
        @test geometry[6].Type == Frozen
        @test length(geometry[6].Particles) == 61*(ceil(Int,flow.duration*flow.velocity/flow.spacing)+1)
    end
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "inactive inlet reserves stay outside neighbor cells" begin
        for T in (Float32,Float64)
            geometry = [SPHGeometry{2,T}([SVector(0.,0.),SVector(0.01,0.)];Density=1000,Type=Fluid,GroupMarker=1),
                SPHGeometry{2,T}(fill(SVector(0.,1e6),4096);Density=1000,Type=Frozen,GroupMarker=5)]
            host = AllocateDataStructures(geometry)
            gpu = upload_particles(host)
            cl = CellListWorkspace{2,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu,cl,T(50);
                inactive_particles=true,sort_inactive=false)
            types,cells,ids,pos = Array(gpu.Type),Array(gpu.CellID),Array(gpu.ID),Array(gpu.Position)
            @test all(i -> cells[i] == 1,findall(==(Frozen),types))
            @test all(i -> cells[i] > 1,findall(==(Fluid),types))
            @test cl.grid.dims[2] < 10
            @test all(i -> pos[i] == host.Position[ids[i]],eachindex(ids))
        end
    end
    @testset "jet prescribed inlet and extrapolated outlets" begin
        O = SPHExampleGPU.OpenBoundaries
        for T in (Float32,Float64)
            source = [SVector(-0.195,0.01),SVector(0.195,0.01)]
            geometry = [SPHGeometry{2,T}(;Particles=SPHExampleGPU.SimulationGeometry.particle_struct_array(source,T(1010);
                    Velocity=fill(SVector{2,T}(3,-2),2)),Type=Fluid,GroupMarker=1),
                SPHGeometry{2,T}([SVector(0.,0.125)];Density=900,Type=Fluid,GroupMarker=2),
                SPHGeometry{2,T}([SVector(-0.205,0.01)];Density=900,Type=Fluid,GroupMarker=3),
                SPHGeometry{2,T}([SVector(0.205,0.01),SVector(0.205,0.1)];Density=900,Type=Fluid,GroupMarker=4)]
            gpu = upload_particles(AllocateDataStructures(geometry))
            kernel = SPHKernelInstance{2,T}(WendlandC2();h=T(0.01))
            constants = SimulationConstants{T}(2;dx=0.005,c₀=230)
            cl = CellListWorkspace{2,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu,cl,kernel.H⁻¹)
            O.launch_open_boundary_values!((;configuration=ImpingingJet{T}()),gpu.Position,gpu.Velocity,gpu.Density,
                gpu.Type,gpu.GroupMarker,cl.CellStart,cl.grid_dev,kernel,constants,
                SPHExampleGPU.GPUStepState.HostStep(T(0.001),0.))
            groups,pos,rho,vel = Array(gpu.GroupMarker),Array(gpu.Position),Array(gpu.Density),Array(gpu.Velocity)
            for i in eachindex(groups)
                if groups[i] == 2
                    @test rho[i] == 1000
                    @test vel[i] == SVector{2,T}(0,-20)
                elseif groups[i] in (3,4)
                    if pos[i][2] < 0.02
                        @test isapprox(rho[i],1010)
                        @test isapprox(vel[i],SVector{2,T}(3,-2))
                    else
                        @test rho[i] == 1000
                        @test iszero(vel[i])
                    end
                end
            end
        end
    end
    @testset "jet CUDA impact and particle balance" begin
        for graph in (false,true)
            mktempdir() do directory
                result = run_impinging_jet(spacing=0.0025,duration=0.008,output_interval=0.008,
                    save_location=directory,use_graph=graph)
                particles,state = result.final_state.particles,result.final_state.open_boundary
                active = findall(==(Fluid),particles.Type)
                @test result.final_state.complete
                @test state.emitted_particles > 0
                @test state.active_fluid == state.initial_active_fluid+state.emitted_particles-state.retired_particles
                @test all(i -> isfinite(particles.Density[i]) && particles.Density[i] > 0,active)
                @test all(i -> all(isfinite,particles.Velocity[i]),active)
                @test minimum(particles.Position[i][2] for i in active) < 0.02
                @test maximum(abs(particles.Position[i][1]) for i in active) <= 0.21+1e-6
            end
        end
    end
end
end
