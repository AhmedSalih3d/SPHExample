module NestedResolutionTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

function uniform_nested_case(T=Float32;scheme=SymplecticTimeStepping())
    regions = (VariableResolutionBox((-0.8,-0.8),(0.8,0.8)),VariableResolutionBox((-0.3,-0.3),(0.3,0.3)))
    function level(dx,extent)
        points = [SVector(x,y) for x in -extent:dx:extent for y in -extent:dx:extent]
        constants = SimulationConstants{T}(2;dx,c₀=10,g=0,α=0,δᵩ=0)
        geometry = [SPHGeometry{2,T}(;Particles=particle_struct_array(points,constants.ρ₀;
            Velocity=fill(SVector{2,T}(1,0),length(points))),Type=Fluid,GroupMarker=2)]
        return (;geometry,constants,kernel=SPHKernelInstance{2,T}(WendlandC2();h=T(dx)))
    end
    levels = (level(0.08,1.2),level(0.04,1.0),level(0.02,0.4))
    inner = VariableResolutionConfig(regions[2];fine_geometry=levels[3].geometry,particle_reserve=1024)
    config = VariableResolutionConfig(regions[1];fine_geometry=levels[2].geometry,child=inner,particle_reserve=1024)
    directory = mktempdir()
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(;
        SimulationName="NestedUniform",SaveLocation=directory,SimulationTime=0.12,OutputTimes=0.06,
        VariableResolution=config,GPUUseGraph=true,GPUMaxStepsPerSync=8,GPUClipTimeStepToOutput=true,
        ExportSingleVTKHDF=true,VisualizeInParaview=false,OpenLogFile=false,
        OutputVariables=["Velocity","Density","Type","ID","Resolution","Mass","Owned"])
    particles = AllocateDataStructures(levels[1].geometry,meta)
    logger = SimulationLogger(directory)
    result = try
        redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=levels[1].geometry,SimConstants=levels[1].constants,SimKernel=levels[1].kernel,
                SimMetaData=meta,SimParticles=particles,SimLogger=logger,SimTimeStepping=scheme,
                SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion())
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    return result
end

if CUDA.functional()
    @testset "nested interfaces preserve uniform flow and unique creation IDs T=$T scheme=$(typeof(scheme))" for
            (T,scheme) in ((Float32,SymplecticTimeStepping()),(Float32,SingleNeighborTimeStepping()),(Float64,SymplecticTimeStepping()))
        result = uniform_nested_case(T;scheme)
        @test result.final_state.complete
        @test result.final_state.time == 0.12
        levels = result.final_state.resolution.levels
        @test length(levels) == 3 && all(!isempty,levels)
        for level in levels
            @test maximum(abs.(level.Density .- 1000)) < 0.01
            @test maximum(norm.(level.Velocity .- Ref(SVector{2,T}(1,0)))) < 1e-4
            @test all(==(1),level.Owned)
        end
        ids = vcat(map(l -> l.ID,levels)...)
        @test length(ids) == length(unique(ids))
        @test all(e -> e[1] > 0 && e[3] == e[4] == 0,result.diagnostics.resolution.events)
    end
end
end
