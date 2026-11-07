module OpenChannel3DTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra
const O = SPHExampleGPU.OpenBoundaries

function fixture(::Type{T},duration) where T
    positions(xs) = [SVector(x,y,z) for x in xs for y in (0.02,0.04,0.06) for z in (0.02,0.04,0.06)]
    geometry = [SPHGeometry{3,T}(p;Density=1000,Type=Fluid,GroupMarker=marker)
        for (marker,p) in enumerate((positions(0.02:0.02:0.38),positions((0.,)),positions((0.4,))))]
    flow = OpenChannel{T}(Inlet=0,Outlet=0.4,WaterLevel=0.06,Velocity=1,Spacing=0.02,Duration=duration,OutletLayers=5)
    return open_channel_geometry(geometry,flow),flow
end

@testset "3D channel seed planes and reserves" begin
    geometry,flow = fixture(Float32,0.055)
    @test sum(length(g.Particles.Position) for g in geometry if g.Type == Frozen) == 36
    @test all(g -> all(p -> length(p) == 3,g.Particles.Position),geometry)
    @test all(g -> g.Type != Fluid || all(v -> v == SVector{3,Float32}(1,0,0),g.Particles.Velocity),geometry)
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "3D ghost density matches an independent all-pairs sum" begin
        for T in (Float32,Float64), R in (1,2)
            points = [SVector(x,y,z) for x in 0.01:0.02:0.09 for y in 0.01:0.02:0.09 for z in 0.01:0.02:0.09]
            densities = T[1000+5p[1]+2p[2]+3p[3] for p in points]
            geometry = [SPHGeometry{3,T}(points;Density=densities,Type=Fluid,GroupMarker=1),
                SPHGeometry{3,T}([SVector(-0.01,0.05,0.05),SVector(-0.5,0.05,0.05)];Density=1000,Type=Fluid,GroupMarker=2)]
            gpu = SPHExampleGPU.SPHCellList.upload_particles(AllocateDataStructures(geometry))
            kernel = SPHKernelInstance{3,T}(WendlandC2();h=T(0.03))
            constants = SimulationConstants{T}(3;dx=0.02,g=0,c₀=20)
            cl = SPHExampleGPU.GPUCellGrid.CellListWorkspace{3,T}(length(gpu);reach=R)
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu,cl,kernel.H⁻¹)
            positions,density,groups = Array(gpu.Position),Array(gpu.Density),Array(gpu.GroupMarker)
            flow = OpenChannel{T}(Inlet=0,Outlet=1,WaterLevel=0.2,Velocity=1,Spacing=0.02,Duration=0.1)
            O.launch_open_boundary_values!((;configuration=flow),gpu.Position,gpu.Velocity,gpu.Density,gpu.Type,gpu.GroupMarker,
                cl.CellStart,cl.grid_dev,kernel,constants,SPHExampleGPU.GPUStepState.HostStep(T(0.001),0.))
            results = Array(gpu.Density)
            for i in findall(==(2),groups)
                ghost = setindex(positions[i],-positions[i][1],1)
                numerator,denominator = 0.,0.
                for j in findall(==(1),groups)
                    distance = norm(positions[j]-ghost)
                    if distance <= kernel.H
                        w = Wᵢⱼ(kernel,T(distance/kernel.h))*constants.m₀/density[j]
                        numerator += w*density[j]
                        denominator += w
                    end
                end
                @test results[i] ≈ (denominator > 0 ? numerator/denominator : 1000) atol=2e-4
            end
        end
    end
    @testset "hydrostatic background survives truncated corner support" begin
        for T in (Float32,Float64)
            constants = SimulationConstants{T}(3;dx=0.02,g=9.81,c₀=60)
            background(z) = T(1000*(1+7*9.81*max(0,0.2-z)/60^2)^(1/7))
            points = [SVector(x,y,z) for x in 0.01:0.02:0.09 for y in 0.01:0.02:0.09 for z in 0.01:0.02:0.09]
            ghosts = [SVector(-0.01,0.,-0.01),SVector(-0.01,0.01,0.01),SVector(-0.5,0.,-0.01)]
            geometry = [SPHGeometry{3,T}(points;Density=T[background(p[3])+2 for p in points],Type=Fluid,GroupMarker=1),
                SPHGeometry{3,T}(ghosts;Density=1000,Type=Fluid,GroupMarker=2)]
            gpu = SPHExampleGPU.SPHCellList.upload_particles(AllocateDataStructures(geometry))
            kernel = SPHKernelInstance{3,T}(WendlandC2();h=T(0.03))
            cl = SPHExampleGPU.GPUCellGrid.CellListWorkspace{3,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu,cl,kernel.H⁻¹)
            flow = OpenChannel{T}(Inlet=0,Outlet=1,WaterLevel=0.2,Velocity=1,Spacing=0.02,Duration=0.1,HydrostaticCorrection=true)
            positions,groups = Array(gpu.Position),Array(gpu.GroupMarker)
            O.launch_open_boundary_values!((;configuration=flow),gpu.Position,gpu.Velocity,gpu.Density,gpu.Type,gpu.GroupMarker,
                cl.CellStart,cl.grid_dev,kernel,constants,SPHExampleGPU.GPUStepState.HostStep(T(0.001),0.))
            density = Array(gpu.Density)
            for i in findall(==(2),groups)
                expected = background(positions[i][3])+(positions[i][1] > -0.1 ? 2 : 0)
                @test density[i] ≈ expected atol=3e-4 rtol=0
            end
        end
    end
    @testset "3D channel emission and uniform flow, direct and graph" begin
        for T in (Float32,Float64),graph in (false,true)
            geometry,flow = fixture(T,0.055)
            mktempdir() do dir
                meta = SimulationMetaData{3,T}(SimulationName="Open3D",SaveLocation=dir,
                    SimulationTime=0.055,OutputTimes=0.055,VisualizeInParaview=false,OpenLogFile=false,
                    GPUUseGraph=graph,GPUDoublePosition=true,GPUClipTimeStepToOutput=true)
                logger = SimulationLogger(dir)
                result = try
                    redirect_stdout(devnull) do
                        RunSimulation(SimGeometry=geometry,SimMetaData=meta,SimConstants=SimulationConstants{T}(3;dx=0.02,g=0,c₀=20),
                            SimParticles=AllocateDataStructures(geometry,meta),SimKernel=SPHKernelInstance{3,T}(WendlandC2();h=T(0.02)),
                            SimLogger=logger,SimOpenBoundaries=flow,SimViscosity=ArtificialViscosity(),SimDensityDiffusion=LinearDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping())
                    end
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                particles,state = result.final_state.particles,result.final_state.open_boundary
                active = findall(==(Fluid),particles.Type)
                @test state.emitted_particles == 18
                @test state.active_fluid == state.initial_active_fluid+state.emitted_particles-state.retired_particles
                @test maximum(norm(v-SVector{3,T}(1,0,0)) for v in particles.Velocity[active]) < 1e-6
                @test maximum(abs.(particles.Density[active].-1000)) < 1e-3
            end
        end
    end
end
end



