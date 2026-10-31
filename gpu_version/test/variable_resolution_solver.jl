module VariableResolutionSolverTests
using Test, SPHExampleGPU, CUDA, StaticArrays, StructArrays, HDF5
const VR = SPHExampleGPU.VariableResolution
include(joinpath(@__DIR__,"..","example","VResolutionDamBreak2d","Setup.jl"))

function moving_case()
    case = vresolution_dambreak_case(dx=0.05,region=VariableResolutionBox((0.5,-0.3),(1.8,1.5)))
    for level in (case.coarse,case.fine)
        fluid = level.geometry[2]
        data = StructArray((;Position=fluid.Particles.Position,Density=fluid.Particles.Density,
            Velocity=fill(SVector(1f0,0f0),length(fluid.Particles))))
        level.geometry[2] = SPHGeometry{2,Float32}(;Particles=data,Type=Fluid,GroupMarker=2)
    end
    return case
end

function run_case(case; single=true,graphs=true,clip=true,duration=0.003,output_times=[0.001,0.002],
        damping=nothing,include_buffers=false,grid=false,logging=false,particle_setup=identity,kwargs...)
    directory = mktempdir()
    config = VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,
        fine_constants=case.fine.constants,fine_kernel=case.fine.kernel,particle_reserve=2048,include_buffers)
    L = logging ? StoreLog : NoLog
    T = typeof(case.coarse.constants.dx)
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,SimpleMDBC,L}(;
        SimulationName="MetadataRefinement",SaveLocation=directory,SimulationTime=duration,OutputTimes=output_times,
        VariableResolution=config,GPUUseGraph=graphs,GPUClipTimeStepToOutput=clip,GPUMaxStepsPerSync=16,
        ExportSingleVTKHDF=single,ExportGridCells=grid,VisualizeInParaview=false,OpenLogFile=false,
        OutputVariables=["Velocity","Density","Pressure","Acceleration","Type","ID","GroupMarker","GhostPoints",
            "Resolution","Spacing","Mass","Owned","IsBuffer"],kwargs...)
    particles = AllocateDataStructures(case.coarse.geometry,meta)
    particle_setup(particles)
    logger = SimulationLogger(directory)
    result = try
        redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=case.coarse.geometry,SimMetaData=meta,SimConstants=case.coarse.constants,
                SimKernel=case.coarse.kernel,SimParticles=particles,SimLogger=logger,
                SimViscosity=ArtificialViscosity(),SimDensityDiffusion=FourtakasDensityDiffusion(full=true),
                SimTimeStepping=DualSPHysicsSymplectic(fast_math=false),SimDamping=damping)
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    return (;result,meta,particles,directory)
end

@testset "metadata refinement configuration" begin
    case = vresolution_dambreak_case(dx=0.05)
    config = VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry)
    @test_throws ArgumentError VariableResolutionConfig(case.region;fine_geometry=[])
    @test_throws ArgumentError VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,particle_reserve=0)
    @test_throws ArgumentError SimulationMetaData{3,Float32}(SimulationName="bad",SaveLocation=".",VariableResolution=config)
    meta = SimulationMetaData{2,Float32}(SimulationName="metadata",SaveLocation=".")
    @test meta.VariableResolution === nothing
    meta.VariableResolution = config
    @test position_float_type(meta) === Float64
    fine,kernel = VR.fine_resolution_parameters(config,case.coarse.constants,case.coarse.kernel)
    @test fine.dx == case.fine.constants.dx
    @test fine.m₀ == case.fine.constants.m₀
    @test fine.c₀ == case.coarse.constants.c₀
    @test kernel.h == case.fine.kernel.h
    @test kernel.η² ≈ case.fine.kernel.η²
    meta.OutputVariables = ["Mass","Resolution"]
    @test resolve_output_variables!(meta) == ["Mass","Resolution"]
    meta.VariableResolution = nothing
    @test_throws ArgumentError resolve_output_variables!(meta)
    @test length(sprint(show,config)) < 500
end

if CUDA.functional()
    @testset "normal solver metadata, schedules, output and result" begin
        case = vresolution_dambreak_case(dx=0.05)
        direct = run_case(case;graphs=false)
        graph = run_case(case;logging=true)
        @test direct.result isa SPHExampleGPU.SimulationResult
        @test graph.result.final_state.particles === graph.particles
        @test graph.result.final_state.complete
        @test graph.result.termination_reason === :completed
        @test graph.meta.TotalTime == 0.003
        @test graph.result.diagnostics.output_frames == 4
        @test graph.result.configuration.execution.path === :cuda_graph
        @test direct.result.configuration.execution.path === :direct
        @test graph.result.configuration.effective.variable_resolution.method === :two_domain_mls
        @test !graph.result.configuration.effective.async_output
        @test !graph.result.configuration.effective.interaction_fastmath
        @test graph.result.configuration.effective.position_type === Float64
        @test graph.result.diagnostics.frozen_particles == 0
        for field in (:Position,:Velocity,:Density,:ID,:Type,:Acceleration)
            @test getproperty(graph.particles,field) == getproperty(direct.particles,field)
        end
        @test graph.meta.Iteration == direct.meta.Iteration
        @test all(all(isfinite,a) for a in graph.particles.Acceleration)
        h5open(joinpath(graph.directory,"MetadataRefinement.vtkhdf"),"r") do file
            root = file["VTKHDF"]
            @test read(root["Steps/Values"]) == [0.,0.001,0.002,0.003]
            counts = read(root["NumberOfPoints"])
            @test length(counts) == 4
            @test size(root["Points"],2) == sum(counts)
            @test Set(keys(root["PointData"])) == Set(graph.meta.OutputVariables)
            @test all(==(1),read(root["PointData/Owned"]))
            @test all(isfinite,read(root["PointData/Pressure"]))
        end
        @test isfile(graph.result.diagnostics.resolution.history_path)
        log = read(joinpath(graph.directory,"SimulationOutput.log"),String)
        @test occursin("Variable resolution:",log)
        @test occursin("Simulation finished",log)
        overshoot = run_case(case;clip=false,duration=0.001,output_times=0.0005)
        @test overshoot.meta.TotalTime > 0.001
        @test overshoot.result.final_state.time == overshoot.meta.TotalTime
        @test_throws ArgumentError run_case(case;GPUCellSubdivision=2)
        @test_throws ArgumentError run_case(case;GPUFixedGrid=true)
        @test_throws ArgumentError run_case(case;GPUTimestepDiagnosticsEvery=1)
        f64 = run_case(vresolution_dambreak_case(dx=0.05,FloatType=Float64);duration=0.0001,output_times=0.0001)
        @test f64.meta.TotalTime == 0.0001
        @test f64.result.configuration.effective.pair_coordinates === :absolute
        edited = run_case(case;duration=0.0001,output_times=0.0001,
            particle_setup=p -> (p.GroupMarker[p.Type .== Fluid] .= 7))
        @test all(==(7),edited.particles.GroupMarker[edited.particles.Type .== Fluid])
    end

    @testset "creation changes normal single-file frame sizes" begin
        run = run_case(vresolution_dambreak_case(dx=0.05);duration=0.65,output_times=[0.2,0.45])
        @test sum(e[1] for e in run.result.diagnostics.resolution.events) > 0
        h5open(joinpath(run.directory,"MetadataRefinement.vtkhdf"),"r") do file
            root = file["VTKHDF"]
            counts = read(root["NumberOfPoints"])
            @test length(unique(counts)) > 1
            @test read(root["Steps/PointOffsets"]) == [0;cumsum(counts)[1:end-1]]
            @test length(root["PointData/ID"]) == sum(counts)
            offsets = read(root["Steps/PointOffsets"])
            ids = read(root["PointData/ID"])
            for i in eachindex(counts)
                @test length(unique(ids[offsets[i]+1:offsets[i]+counts[i]])) == counts[i]
            end
        end
    end

    @testset "normal damping, numbered frames, and current grid export" begin
        damping = DampingZone((-0.5,0.0),(0.0,0.0);overlimit=20,rate=100000)
        case = moving_case()
        run = run_case(case;single=false,duration=0.001,output_times=0.0005,
            damping,include_buffers=true,grid=true)
        @test run.result.diagnostics.output_frames == 3
        @test all(iszero,run.particles.Velocity[run.particles.Type .== Fluid])
        @test isfile(joinpath(run.directory,"MetadataRefinement.vtkhdf.series"))
        h5open(joinpath(run.directory,"MetadataRefinement_000003.vtkhdf"),"r") do file
            @test haskey(file,"VTKHDF/Particles/PointData/Resolution")
            @test haskey(file,"VTKHDF/DampingZones")
        end
        @test isfile(joinpath(run.directory,"CellGrid_MetadataRefinement_000003.vtkhdf"))
        h5open(joinpath(run.directory,"CellGrid_MetadataRefinement_000003.vtkhdf"),"r") do file
            occupied = unique(VR.cell_coords(x,case.coarse.kernel.H⁻¹) for x in run.particles.Position)
            @test only(read(file["VTKHDF/NumberOfCells"])) == length(occupied)
        end
        combined = run_case(case;duration=0.001,output_times=0.0005,damping)
        h5open(joinpath(combined.directory,"MetadataRefinement.vtkhdf"),"r") do file
            times = read(file["VTKHDF/Particles/Steps/Values"])
            @test times == read(file["VTKHDF/DampingZones/Steps/Values"])
            @test times == read(file["VTKHDF/DampingProfiles/Steps/Values"])
        end
    end
end
end
