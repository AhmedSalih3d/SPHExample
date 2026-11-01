module VariableResolutionSolverTests
using Test, SPHExampleGPU, CUDA, StaticArrays, StructArrays, HDF5
const VR = SPHExampleGPU.VariableResolution
include(joinpath(@__DIR__,"..","example","VResolutionDamBreak2d","Setup.jl"))

struct ScaledLinearDiffusion{T} <: SPHDensityDiffusion
    scale::T
end

@inline function SPHExampleGPU.compute_density_diffusion(model::ScaledLinearDiffusion,args...)
    first,second = compute_density_diffusion(LinearDensityDiffusion(),args...)
    return model.scale*first,model.scale*second
end

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
        scheme=SymplecticTimeStepping(),diffusion=LinearDensityDiffusion(),viscosity=ArtificialViscosity(),boundary=SimpleMDBC,
        damping=nothing,include_buffers=false,grid=false,logging=false,particle_setup=identity,kwargs...)
    directory = mktempdir()
    config = VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,
        fine_constants=case.fine.constants,fine_kernel=case.fine.kernel,particle_reserve=2048,include_buffers)
    L = logging ? StoreLog : NoLog
    T = typeof(case.coarse.constants.dx)
    variables = ["Velocity","Density","Pressure","Acceleration","Type","ID","GroupMarker","Resolution","Spacing","Mass","Owned","IsBuffer"]
    boundary === SimpleMDBC && push!(variables,"GhostPoints")
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,boundary,L}(;
        SimulationName="MetadataRefinement",SaveLocation=directory,SimulationTime=duration,OutputTimes=output_times,
        VariableResolution=config,GPUUseGraph=graphs,GPUClipTimeStepToOutput=clip,GPUMaxStepsPerSync=16,
        ExportSingleVTKHDF=single,ExportGridCells=grid,VisualizeInParaview=false,OpenLogFile=false,
        OutputVariables=variables,kwargs...)
    particles = AllocateDataStructures(case.coarse.geometry,meta)
    particle_setup(particles)
    logger = SimulationLogger(directory)
    result = try
        redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=case.coarse.geometry,SimMetaData=meta,SimConstants=case.coarse.constants,
                SimKernel=case.coarse.kernel,SimParticles=particles,SimLogger=logger,
                SimViscosity=viscosity,SimDensityDiffusion=diffusion,
                SimTimeStepping=scheme,SimDamping=damping)
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    return (;result,meta,particles,directory)
end

function isolated_domains(::Type{T}; gravity=9.81) where T
    region = VariableResolutionBox((2,-1),(4,1))
    function level(dx,positions,density)
        constants = SimulationConstants{T}(2;dx,c₀=45,γ=7,g=gravity)
        kernel = SPHKernelInstance{2,T}(WendlandC2();h=T(1.5dx))
        geometry = [SPHGeometry{2,T}(positions;Density=T.(density),Type=Fluid,GroupMarker=2)]
        return (;constants,kernel,geometry)
    end
    coarse = level(0.05,[SVector(0.2,0.5),SVector(0.25,0.5)],[1010,1000])
    fine = level(0.025,[SVector(3.0,0.3),SVector(3.025,0.3)],[1000,1005])
    return (;coarse,fine,region)
end

function uniform_step(level,scheme)
    T = typeof(level.constants.dx)
    directory = mktempdir()
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,SimpleMDBC,NoLog}(
        SimulationName="UniformStep",SaveLocation=directory,SimulationTime=0.0001,OutputTimes=0.0001,
        GPUDoublePosition=true,GPUClipTimeStepToOutput=true,GPULanesPerParticle=1,
        GPUInteractionFastMath=false,GPUBoundaryForces=false,VisualizeInParaview=false,OpenLogFile=false)
    particles = AllocateDataStructures(level.geometry,meta)
    logger = SimulationLogger(directory)
    try
        redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=level.geometry,SimMetaData=meta,SimConstants=level.constants,
                SimKernel=level.kernel,SimParticles=particles,SimLogger=logger,
                SimViscosity=ArtificialViscosity(),SimDensityDiffusion=LinearDensityDiffusion(),SimTimeStepping=scheme)
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    @test meta.Iteration == 1
    return particles
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

@testset "readable variable-resolution example uses the normal API" begin
    mktempdir() do directory
        scope = Module(gensym(:ReadableResolutionExample))
        Core.eval(scope,quote
            captured = Ref{Any}(nothing)
            RunSimulation(;kwargs...) = (captured[] = (;kwargs...))
        end)
        path = joinpath(@__DIR__,"..","example","VResolutionDamBreak2d.jl")
        source = replace(read(path,String),r"SaveLocation\s*=\s*joinpath\([^\n]+" => "SaveLocation = $(repr(directory)),")
        redirect_stdout(devnull) do
            Base.include_string(scope,source,path)
        end
        inputs = Base.invokelatest(getproperty,scope,:captured)[]
        @test inputs.SimDensityDiffusion isa LinearDensityDiffusion
        @test inputs.SimTimeStepping isa SymplecticTimeStepping
        @test inputs.SimConstants.γ == 7
        @test inputs.SimMetaData.VariableResolution isa VariableResolutionConfig
        config = inputs.SimMetaData.VariableResolution
        @test config.fine_constants === nothing
        constants,kernel = VR.fine_resolution_parameters(config,inputs.SimConstants,inputs.SimKernel)
        @test constants.dx == inputs.SimConstants.dx/config.region.ratio
        @test constants.Cb === inputs.SimConstants.Cb
        @test !isopen(inputs.SimLogger.LoggerIo)
        reference = vresolution_dambreak_case(γ=7)
        for (geometry,expected) in ((inputs.SimGeometry,reference.coarse.geometry),
                (inputs.SimMetaData.VariableResolution.fine_geometry,reference.fine.geometry))
            @test getproperty.(geometry,:Type) == [Fixed,Fluid]
            for (actual,wanted) in zip(geometry,expected)
                @test actual.Particles.Position == wanted.Particles.Position
                @test actual.Particles.Density ≈ wanted.Particles.Density rtol=eps(Float32)
            end
        end
        @test isfile(joinpath(directory,"DamBreak2D_VRes_Geometry.vtkhdf"))
    end
end

if CUDA.functional()
    @testset "scripted helper model and timestep selection" begin
        for options in ((;),(;density_diffusion=ZeroDensityDiffusion(),time_stepping=SingleNeighborTimeStepping()))
            result = redirect_stdout(devnull) do
                run_vresolution_dambreak_solver(;dx=0.05,duration=0.0001,output_interval=0.0001,
                    save_location=mktempdir(),options...)
            end
            @test result.final_state.complete
            @test result.configuration.effective.density_diffusion_model === typeof(get(options,:density_diffusion,LinearDensityDiffusion()))
            @test result.configuration.effective.time_stepping === typeof(get(options,:time_stepping,SymplecticTimeStepping()))
        end
    end

    @testset "normal viscosity selection and DBC boundaries" begin
        case = moving_case()
        for (i,level) in enumerate((case.coarse,case.fine))
            particles = level.geometry[2].Particles
            for j in eachindex(particles.Position)
                particles.Velocity[j] = SVector(particles.Position[j][1] < (i == 1 ? 0.25 : 0.8) ? 1f0 : -1f0,0f0)
            end
        end
        states = Dict()
        for viscosity in (ZeroViscosity(),ArtificialViscosity(boundary_factor=0),Laminar(),LaminarSPS())
            direct = run_case(case;viscosity,graphs=false,duration=0.0001,output_times=0.0001)
            graph = run_case(case;viscosity,duration=0.0001,output_times=0.0001)
            @test graph.result.configuration.effective.viscosity_model === typeof(viscosity)
            for field in (:Position,:Velocity,:Density,:Acceleration)
                @test getproperty(direct.particles,field) == getproperty(graph.particles,field)
            end
            states[typeof(viscosity)] = graph.particles.Velocity
        end
        @test states[ZeroViscosity] != states[ArtificialViscosity]
        @test_throws ArgumentError run_case(case;scheme=DualSPHysicsSymplectic(),viscosity=Laminar())

        dbc_case = vresolution_dambreak_case(dx=0.05,γ=7)
        for level in (dbc_case.coarse,dbc_case.fine)
            wall = level.geometry[1]
            level.geometry[1] = SPHGeometry{2,Float32}(wall.Particles.Position;
                Density=wall.Particles.Density,Type=Fixed,GroupMarker=1)
        end
        direct = run_case(dbc_case;boundary=NoMDBC,graphs=false)
        graph = run_case(dbc_case;boundary=NoMDBC)
        @test graph.result.configuration.effective.boundary_mode === NoMDBC
        for field in (:Position,:Velocity,:Density,:Acceleration)
            @test getproperty(direct.particles,field) == getproperty(graph.particles,field)
        end
    end

    @testset "normal timestep schemes reuse ordinary solver numerics" begin
        for T in (Float32,Float64), scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            case = isolated_domains(T)
            run = run_case(case;scheme,duration=0.0001,output_times=0.0001)
            @test run.meta.Iteration == 1
            @test run.result.configuration.effective.time_stepping === typeof(scheme)
            for (actual,level) in zip(run.result.final_state.resolution,(case.coarse,case.fine))
                expected = uniform_step(level,scheme)
                order = sortperm(actual.Position;by=first)
                reference_order = sortperm(expected.Position;by=first)
                for field in (:Position,:Velocity,:Density)
                    @test getproperty(actual,field)[order] == getproperty(expected,field)[reference_order]
                end
            end
        end
        case = vresolution_dambreak_case(dx=0.05,γ=7)
        direct = run_case(case;scheme=SingleNeighborTimeStepping(),graphs=false)
        graph = run_case(case;scheme=SingleNeighborTimeStepping())
        for field in (:Position,:Velocity,:Density,:Acceleration)
            @test getproperty(direct.particles,field) == getproperty(graph.particles,field)
        end
        reference = run_case(case;scheme=DualSPHysicsSymplectic(fast_math=false),duration=0.0001,output_times=0.0001)
        @test reference.meta.TotalTime == 0.0001
        strong_force = isolated_domains(Float32;gravity=1e8)
        single = run_case(strong_force;scheme=SingleNeighborTimeStepping(),duration=1e-5,output_times=1e-5)
        @test single.meta.Iteration > 1
        @test single.meta.TotalTime == 1e-5
    end

    @testset "selectable density diffusion in both resolution domains" begin
        case = moving_case()
        for (i,level) in enumerate((case.coarse,case.fine))
            target = i == 1 ? SVector(0.2,1.0) : SVector(0.8,0.8)
            particles = level.geometry[2].Particles
            particles.Density[argmin(sum(abs2,x-target) for x in particles.Position)] += 30
        end
        states = Dict()
        for diffusion in (ZeroDensityDiffusion(),ZeroGravityLinearDensityDiffusion(),LinearDensityDiffusion(),
                ComplexDensityDiffusion(),
                FourtakasDensityDiffusion(),FourtakasDensityDiffusion(full=true),ScaledLinearDiffusion(0f0),ScaledLinearDiffusion(2f0))
            direct = run_case(case;diffusion,graphs=false,duration=0.0001,output_times=0.0001)
            graph = run_case(case;diffusion,duration=0.0001,output_times=0.0001)
            @test graph.result.configuration.effective.density_diffusion_model === typeof(diffusion)
            @test graph.result.configuration.effective.density_diffusion_operator ===
                (diffusion isa ZeroDensityDiffusion ? :disabled : diffusion isa FourtakasDensityDiffusion ?
                    (diffusion.full ? :fourtakas_full : :fourtakas_inner) : :reference)
            for (a,b) in zip(graph.result.final_state.resolution,direct.result.final_state.resolution)
                for field in (:Position,:Velocity,:Density,:Acceleration)
                    @test getproperty(a,field) == getproperty(b,field)
                end
            end
            states[diffusion] = graph.result.final_state.resolution
        end
        for level in 1:2
            @test states[LinearDensityDiffusion()][level].Density != states[ZeroDensityDiffusion()][level].Density
            @test states[ScaledLinearDiffusion(0f0)][level].Density == states[ZeroDensityDiffusion()][level].Density
            @test states[ScaledLinearDiffusion(2f0)][level].Density != states[LinearDensityDiffusion()][level].Density
        end
    end

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
        @test graph.result.configuration.effective.density_diffusion_model === LinearDensityDiffusion
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
