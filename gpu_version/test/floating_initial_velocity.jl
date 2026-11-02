module FloatingInitialVelocityTests
using Test, SPHExampleGPU, CUDA, StaticArrays
const F = SPHExampleGPU.GPUFloating

function free_fall(D,T;constraints=nothing,initial=ntuple(d -> d == D ? -2.0 : 0.0,D))
    center = SVector{D,Float64}(ntuple(d -> d == D ? 1.0 : 0.0,D))
    points = [center+SVector{D,Float64}(ntuple(d -> ((i >> (d-1)) & 1) == 0 ? -0.1 : 0.1,D)) for i in 0:2^D-1]
    constants = SimulationConstants{T}(D;dx=0.1,c₀=10,g=9.81)
    geometry = [SPHGeometry{D,T}(points;Density=constants.ρ₀,Type=Floating,GroupMarker=5,
        Floating=FloatingDetails{T}(;RelativeWeight=1,InitialVelocity=initial,Constraints=constraints))]
    metadata = SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(;
        SimulationName="InitialVelocity",SaveLocation=mktempdir(),SimulationTime=0.01,OutputTimes=0.01,
        GPUDoublePosition=true,GPUUseGraph=true,GPUClipTimeStepToOutput=true,
        VisualizeInParaview=false,OpenLogFile=false)
    particles = AllocateDataStructures(geometry,metadata)
    floating = F.FloatingArrays(geometry,particles,constants;position_type=Float64)
    gpu = upload_particles(particles,metadata)
    F.initialize_floating_velocities!(floating,gpu)
    @test Array(gpu.Velocity) == fill(SVector{D,T}(initial),length(points))
    logger = SimulationLogger(metadata.SaveLocation)
    result = try
        redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=geometry,SimConstants=constants,SimKernel=SPHKernelInstance{D,T}(WendlandC2();h=T(0.1)),
                SimMetaData=metadata,SimParticles=particles,SimLogger=logger,
                SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping())
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    t = result.final_state.time
    expected_velocity = SVector{D,T}(initial)+SVector{D,T}(ntuple(d -> d == D ? -constants.g*t : 0,D))
    expected_center = center+SVector{D,Float64}(initial)*t+SVector{D,Float64}(ntuple(d -> d == D ? -0.5constants.g*t^2 : 0,D))
    @test only(result.final_state.floating.velocity) ≈ expected_velocity atol=3e-6
    @test only(result.final_state.floating.center) ≈ expected_center atol=3e-8
    @test result.final_state.complete
end

@test FloatingDetails{Float64}(1.0,0.0,nothing,()).InitialVelocity === nothing
if CUDA.functional()
    @testset "force-responsive initial body velocity D=$D T=$T" for D in (2,3), T in (Float32,Float64)
        free_fall(D,T)
    end
end
end
