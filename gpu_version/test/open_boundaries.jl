module OpenBoundaryTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra
const O = SPHExampleGPU.OpenBoundaries

@testset "open-channel configuration" begin
    @test_throws ArgumentError OpenChannel{Float32}(Inlet = 0,Outlet = 1,WaterLevel = 0.2,Velocity = -1,Spacing = 0.01,Duration = 1)
    @test_throws ArgumentError OpenChannel{Float32}(Inlet = 1,Outlet = 0,WaterLevel = 0.2,Velocity = 1,Spacing = 0.01,Duration = 1)
    @test_throws ArgumentError OpenChannel{Float32}(Inlet = 0,Outlet = 1,WaterLevel = 0.2,Velocity = 1,Spacing = 0.01,Duration = 1,ReserveMarker = 1)
end

function channel_fixture(::Type{T},duration) where T
    vector(x,y) = SVector(Float64(x),Float64(y))
    water = [vector(x,y) for x in 0.005:0.01:0.995 for y in 0.01:0.01:0.2]
    inlet,outlet = [vector(-0.005,y) for y in 0.01:0.01:0.2],[vector(1.005,y) for y in 0.01:0.01:0.2]
    geometry = [SPHGeometry{2,T}(points;Density = 1000,Type = Fluid,GroupMarker = marker) for (marker,points) in enumerate((water,inlet,outlet))]
    flow = OpenChannel{T}(Inlet = -0.005,Outlet = 1.005,WaterLevel = 0.2,Velocity = 1,Spacing = 0.01,Duration = duration,OutletLayers = 5)
    return open_channel_geometry(geometry,flow),flow
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "open density extrapolation matches independent Shepard sum" begin
        for T in (Float32,Float64)
            points = [SVector(x,y) for x in 0.005:0.01:0.055 for y in 0.02:0.01:0.12]
            density = T[1000+5p[1]+2p[2] for p in points]
            geometry = [SPHGeometry{2,T}(points;Density = density,Type = Fluid,GroupMarker = 1),
                SPHGeometry{2,T}([SVector(-0.015,0.07),SVector(-0.5,0.07)];Density = 1000,Type = Fluid,GroupMarker = 2)]
            host = AllocateDataStructures(geometry)
            gpu = upload_particles(host)
            kernel = SPHKernelInstance{2,T}(WendlandC2();h = T(0.02))
            constants = SimulationConstants{T}(2;dx = 0.01,g = 0,c₀ = 20)
            cl = CellListWorkspace{2,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu,cl,kernel.H⁻¹)
            source_positions,source_density,groups = Array(gpu.Position),Array(gpu.Density),Array(gpu.GroupMarker)
            flow = OpenChannel{T}(Inlet = 0,Outlet = 1,WaterLevel = 0.2,Velocity = 1,Spacing = 0.01,Duration = 0.1)
            O.launch_open_boundary_values!((;configuration = flow),gpu.Position,gpu.Velocity,gpu.Density,gpu.Type,gpu.GroupMarker,
                cl.CellStart,cl.grid_dev,kernel,constants,SPHExampleGPU.GPUStepState.HostStep(T(0.001),0.))
            result = Array(gpu.Density)
            for i in findall(==(2),groups)
                ghost = SVector(-source_positions[i][1],source_positions[i][2])
                numerator,denominator = 0.,0.
                for j in findall(==(1),groups)
                    distance = norm(source_positions[j]-ghost)
                    if distance <= kernel.H
                        weight = Wᵢⱼ(kernel,T(distance/kernel.h))*constants.m₀/source_density[j]
                        numerator += weight*source_density[j]
                        denominator += weight
                    end
                end
                expected = denominator > 0 ? numerator/denominator : 1000.
                @test result[i] ≈ expected atol = 2e-4
            end
        end
    end
    @testset "open channel preserves uniform flow and particle balance" begin
        for T in (Float32,Float64),graph in (false,true),scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            geometry,flow = channel_fixture(T,0.055)
            mktempdir() do directory
                meta = SimulationMetaData{2,T}(SimulationName = "OpenUniform",SaveLocation = directory,
                    SimulationTime = 0.055,OutputTimes = 0.055,VisualizeInParaview = false,OpenLogFile = false,
                    GPUUseGraph = graph,GPUDoublePosition = true,GPUClipTimeStepToOutput = true)
                logger = SimulationLogger(directory)
                result = try
                    redirect_stdout(devnull) do
                        RunSimulation(SimGeometry = geometry,SimMetaData = meta,SimConstants = SimulationConstants{T}(2;dx = 0.01,g = 0,c₀ = 20),
                            SimParticles = AllocateDataStructures(geometry,meta),SimLogger = logger,
                            SimKernel = SPHKernelInstance{2,T}(WendlandC2();h = T(0.02)),SimViscosity = ZeroViscosity(),
                            SimDensityDiffusion = ZeroDensityDiffusion(),SimTimeStepping = scheme,SimOpenBoundaries = flow)
                    end
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                state,particles = result.final_state.open_boundary,result.final_state.particles
                fluid = findall(==(Fluid),particles.Type)
                @test result.final_state.complete
                @test state.emitted_particles == 100
                @test state.active_fluid == state.initial_active_fluid+state.emitted_particles-state.retired_particles
                @test state.retired_particles > 0
                @test maximum(norm(v-SVector{2,T}(1,0)) for v in particles.Velocity[fluid]) < 1e-6
                @test maximum(abs.(particles.Density[fluid].-1000)) < 1e-3
                @test all(i -> particles.Position[i][1] <= flow.outlet+flow.outlet_layers*flow.spacing+1e-8,fluid)
                @test meta.Periodicity === nothing
            end
        end
    end
end
end
