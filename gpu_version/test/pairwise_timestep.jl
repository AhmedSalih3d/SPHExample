module PairwiseTimestepTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
using SPHExampleGPU.GPUStepState
const PAIR = SPHExampleGPU.GPUPairwiseTimesteps
const GKPAIR = SPHExampleGPU.GPUKernels
@testset "pairwise signal criterion" begin
    meta = SimulationMetaData{2,Float32}(SimulationName="pair",SaveLocation=tempdir())
    @test !meta.GPULegacyTimestep
    h,c0 = 0.04,20.
    signal = [3.]
    v = SVector(1.,2.)
    for position in (SVector(0.1,0.2),SVector(100.1,-49.8))
        values = GKPAIR.step_values(position,v,zero(v),position,h,0.0001,signal)
        @test values[1] == norm(v)
        @test PAIR.acoustic_denominator(signal,values[1],c0,h) == max(c0,10norm(v))+h*3
    end
end
function signal_probe!(signal,values)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    value = i <= length(values) ? values[i] : zero(eltype(values))
    PAIR.store_pair_signal!(signal,value)
    return nothing
end
if CUDA.functional()
    @testset "pair signal block reduction and inactive reset" begin
        for T in (Float32,Float64), threads in (32,128,256,512)
            values = CuArray(T.(1:1003))
            signal = CUDA.zeros(T,1)
            @cuda threads=threads blocks=cld(length(values),threads) signal_probe!(signal,values)
            @test only(Array(signal)) == T(1003)
            state = StepState{T}()
            SPHExampleGPU.GPUStepState.set_slot!(state.i,state.ih,I_STOP,STOP_OUTPUT)
            PAIR.reset_pair_signal!(signal,state)
            @test only(Array(signal)) == T(1003)
            PAIR.reset_pair_signal!(signal,HostStep(zero(T),0.))
            @test only(Array(signal)) == 0
        end
    end
    @testset "one maximum per interaction block" begin
        for T in (Float32, Float64), threads in (32, 48, 128, 256)
            values = T.(1:1003)
            blocks = cld(length(values), threads)
            partials = CUDA.fill(T(Inf), blocks)
            wrapped = PAIR.PairSignalPartials(partials, CUDA.zeros(T, 1))
            @cuda threads=threads blocks=blocks signal_probe!(wrapped, CuArray(values))
            expected = [maximum(@view values[first:min(first + threads - 1, length(values))])
                        for first in 1:threads:length(values)]
            @test Array(partials) == expected
            PAIR.compact_pair_signal!(wrapped)
            @test only(Array(wrapped.reduced)) == maximum(values)
        end
    end
    @testset "neighbor oracle and unchanged force operators" begin
        for T in (Float32,Float64), D in (2,3), lanes in (1,4)
            V = SVector{D,T}
            points = [V(ntuple(d -> T(0.02idx[d]),D)) for idx in CartesianIndices(ntuple(_ -> 0:2,D))][:]
            geometry = [SPHGeometry{D,T}(points; Density=T(1000),Type=Fluid,GroupMarker=1)]
            host = AllocateDataStructures(geometry)
            for i in eachindex(host)
                host.Velocity[i] = V(ntuple(d -> T(3sin(i*d)),D))
                host.Density[i] += T(i % 3)
            end
            constants = SimulationConstants{T}(dx=T(0.02),c₀=T(20),g=zero(T))
            kernel = SPHKernelInstance{D,T}(WendlandC2();dx=constants.dx)
            gpu = upload_particles(host)
            cells = CellListWorkspace{D,T}(length(gpu))
            SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu,cells,kernel.H⁻¹)
            support = GPUSupportArrays{D,T}(length(gpu);pairwise_timestep=true)
            function evaluate(signal)
                GKPAIR.launch_interactions!(support.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,
                    support.∇Cᵢ,support.∇◌rᵢ,gpu.Position,gpu.Density,gpu.Velocity,gpu.Type,
                    cells.CellStart,gpu.CellID,cells.grid_dev,HostStep(T(1e-4),0.),
                    ZeroDensityDiffusion(),Laminar(),kernel,constants,Val(false),Val(false);
                    lanes=Val(lanes),pair_signal=signal)
                return Array(gpu.Acceleration),Array(support.dρdtI)
            end
            original = evaluate(nothing)
            measured = evaluate(support.PairSignal)
            x,v = Array(gpu.Position),Array(gpu.Velocity)
            expected = max(zero(T), maximum(dot(v[i]-v[j],x[i]-x[j])/(sum(abs2,x[i]-x[j])+kernel.η²)
                for i in eachindex(x),j in eachindex(x) if i != j && sum(abs2,x[i]-x[j]) <= kernel.H²))
            @test only(Array(support.PairSignal)) ≈ expected
            for (a,b) in zip(original,measured)
                @test a ≈ b rtol=20eps(T)
            end
            fill!(gpu.Acceleration,zero(V))
            red = SPHExampleGPU.GPUReductions.ReductionWorkspace{SVector{3,T}}(length(gpu))
            GKPAIR.launch_step_reduction!(red,gpu.Position,gpu.Velocity,gpu.Acceleration,gpu.Position,kernel;
                pair_signal=support.PairSignal)
            state = StepState{T}()
            set_output_time!(state,1.)
            GKPAIR.launch_finish!(state,red,kernel,constants;pair_signal=support.PairSignal)
            readback!(state)
            expected_dt = constants.CFL*kernel.h/(max(constants.c₀,10maximum(norm,v))+kernel.h*expected)
            @test state.fh[F_DT] ≈ expected_dt rtol=10eps(T)

            gpu.Velocity .= -gpu.Position
            evaluate(support.PairSignal)
            @test only(Array(support.PairSignal)) == zero(T)
        end
    end
    include(joinpath(@__DIR__,"..","validation cases","spheric_test_06","case.jl"))
    states = []
    for graph in (false,true)
        mktempdir() do directory
            case = SphericTest06.build_case(dx=0.1,duration=0.1,interval=0.05,save=directory)
            case.metadata.GPULegacyTimestep = false
            case.metadata.GPUUseGraph = graph
            host = AllocateDataStructures(case.geometry,case.metadata)
            logger = SimulationLogger(directory)
            result = try
                case.metadata.GPUTimestepDiagnosticsEvery = 1
                @test_throws ArgumentError RunSimulation(SimGeometry=case.geometry,SimMetaData=case.metadata,
                    SimConstants=case.constants,SimKernel=case.kernel,SimParticles=host,SimLogger=logger,
                    SimViscosity=Laminar(),SimDensityDiffusion=ZeroGravityLinearDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping())
                case.metadata.GPUTimestepDiagnosticsEvery = 0
                RunSimulation(SimGeometry=case.geometry,SimMetaData=case.metadata,SimConstants=case.constants,
                    SimKernel=case.kernel,SimParticles=host,SimLogger=logger,SimMotion=case.motion,
                    SimViscosity=Laminar(),SimDensityDiffusion=ZeroGravityLinearDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping())
            finally
                close(logger.LoggerIo)
            end
            @test result.final_state.complete
            @test result.configuration.effective.timestep_criterion == :pairwise_signal
            particles = result.final_state.particles
            order = sortperm(particles.ID)
            push!(states,(particles.Position[order],particles.Velocity[order],particles.Density[order]))
        end
    end
    @test states[1] == states[2]
    @testset "two-level default criterion" begin
        mktempdir() do directory
            coarse = [SPHGeometry{2,Float32}([SVector(0.2,0.5),SVector(0.25,0.5)];
                Density=1000f0,Type=Fluid,GroupMarker=1)]
            fine = [SPHGeometry{2,Float32}([SVector(3.,0.3),SVector(3.025,0.3)];
                Density=1000f0,Type=Fluid,GroupMarker=1)]
            constants = SimulationConstants{Float32}(2;dx=0.05,c₀=10,g=0)
            kernel = SPHKernelInstance{2,Float32}(WendlandC2();dx=constants.dx)
            config = VariableResolutionConfig(VariableResolutionBox((2.,-1.),(4.,1.));
                fine_geometry=fine,particle_reserve=32)
            meta = SimulationMetaData{2,Float32}(SimulationName="PairwiseLevels",SaveLocation=directory,
                SimulationTime=0.002,OutputTimes=0.002,VariableResolution=config,GPUUseGraph=true,
                GPUClipTimeStepToOutput=true,VisualizeInParaview=false,OpenLogFile=false)
            host = AllocateDataStructures(coarse,meta)
            logger = SimulationLogger(directory)
            result = try
                redirect_stdout(devnull) do
                    RunSimulation(SimGeometry=coarse,SimMetaData=meta,SimConstants=constants,SimKernel=kernel,
                        SimParticles=host,SimLogger=logger,SimViscosity=ZeroViscosity(),
                        SimDensityDiffusion=ZeroDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping())
                end
            finally
                close(logger.LoggerIo)
            end
            @test result.final_state.complete
            @test result.final_state.time == 0.002
        end
    end
end
end
