module MotionExampleTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5, CSV, LinearAlgebra
include(joinpath(@__DIR__,"..","example","Motion","MotionExamples.jl"))
using .MotionExamples

const EXPORT = SPHExampleGPU.ProduceHDFVTK
const PM = SPHExampleGPU.PrescribedMotions
const ROT = SPHExampleGPU.PrescribedRotation
const REFERENCE = joinpath(@__DIR__,"reference","motion_cases")

@testset "distributed Motion01-09 geometry and motion" begin
    cases = [motion_example(number) for number in 1:9]
    @test [case.duration for case in cases] == [6,10,10,2.5,17,35,20,10,18]
    @test [length(case.parts) for case in cases] == [2,2,3,2,5,5,6,5,3]
    @test_throws ArgumentError motion_example(1;dimensions=2)
    @test_throws ArgumentError motion_example(10)
    @test motion_frame_times(17,0.05)[113] == 5.6
    @test motion_frame_times(6,0.01)[271] == 2.7
    for row in CSV.File(joinpath(REFERENCE,"geometry_bounds.csv"))
        case = cases[row.case]
        part = only(p for p in case.parts if p.mk==row.id)
        points,connectivity,offsets,regions = EXPORT.polygonal_regions((part.shape,))
        bounds = [extrema(p[d] for p in points)[side] for d in 1:3 for side in 1:2]
        @test bounds ≈ [row.xmin,row.xmax,row.ymin,row.ymax,row.zmin,row.zmax] atol=1e-5
        @test length(points) == row.points
        @test length(regions) == row.cells
    end
    source_points = Dict{Tuple{Int,Int},Vector{SVector{3,Float64}}}()
    for row in CSV.File(joinpath(REFERENCE,"geometry_vertices.csv"))
        push!(get!(source_points,(row.case,row.id),SVector{3,Float64}[]),SVector(row.x,row.y,row.z))
    end
    for ((number,id),expected) in source_points
        part = only(p for p in cases[number].parts if p.mk==id)
        actual = MotionExamples.surface_points(part.shape)
        # GenCase emits Float32 surfaces; native polygon construction uses
        # Float64. Compare the complete vertex sets, independent of ordering.
        @test maximum(p -> minimum(q -> norm(p-q),actual),expected) < 2e-5
        @test maximum(p -> minimum(q -> norm(p-q),expected),actual) < 2e-5
    end
    for row in CSV.File(joinpath(REFERENCE,"transforms.csv"))
        pose = motion_state(cases[row.case].motion,row.id,row.time)
        expected = permutedims(reshape([row[Symbol("m$i$j")] for i in 1:3 for j in 1:4],4,3))
        @test hcat(pose.rotation,pose.offset) ≈ expected atol=5e-11 rtol=0
    end
end

@testset "full polygon animations retain topology, markers and absolute frames" begin
    for number in 1:9
        mktempdir() do dir
            output = run_motion_example(number;visualize=false,save_location=dir)
            case = output.case
            @test output.simulation === nothing
            @test first(output.times)==0 && last(output.times)==case.duration
            points,connectivity,offsets,regions = EXPORT.polygonal_regions(values(case.regions))
            owners = Int[]
            EXPORT.polygonal_regions(values(case.regions);point_regions=owners)
            h5open(output.motion_path,"r") do file
                root = file["VTKHDF"]
                count = length(output.times)
                @test read(root["Steps/Values"]) == output.times
                @test read(root["Polygons/NumberOfCells"]) == fill(length(regions),count)
                @test all(iszero,read(root["Vertices/NumberOfCells"]))
                @test read(root["Steps/CellDataOffsets/Mk"]) == read(root["Steps/CellDataOffsets/Region"])
                @test read(root["CellData/MkBound"]) == repeat(Int32[case.parts[r].mk for r in regions],count)
                frames = unique([1,2,cld(count,2),count])
                for frame in frames
                    time = output.times[frame]
                    actual = root["Points"][:,((frame-1)*length(points)+1):(frame*length(points))]
                    for (region,part) in enumerate(case.parts)
                        ids = findall(==(region),owners)
                        moving = any(n -> n.group_marker==part.mk+1,case.motion.motions)
                        expected = moving ? [motion_position(case.motion,part.mk,points[i],time) for i in ids] : points[ids]
                        @test actual[:,ids] ≈ stack(expected) atol=1e-12 rtol=0
                    end
                end
            end
        end
    end
end

@testset "reset and file-stop trajectory contracts" begin
    sequence = MotionSequence([KinematicSegment(1;velocity=(1,0,0)),KinematicSegment(1;velocity=(0,0,0))];
        jumps=[(-1,0,0),(0,0,0)])
    @test first(trajectory_state(sequence,prevfloat(1.))) ≈ SVector(1.,0.,0.)
    @test trajectory_state(sequence,1.) == (SVector(0.,0.,0.),SVector(0.,0.,0.),SVector(0.,0.,0.))
    @test_throws DimensionMismatch MotionSequence([KinematicSegment(1;velocity=1)];jumps=[])
    history = SampledTrajectory([0,10],[(0,0,0),(10,20,30)])
    window = TrajectoryWindow(history;end_time=8,finish_offset=(0,0,-0.02))
    @test first(trajectory_state(window,8)) == SVector(8.,16.,24.)
    @test first(trajectory_state(window,8.1)) == SVector(8.,16.,23.98)
    @test iszero(trajectory_state(window,8.1)[2])
    @test_throws ArgumentError TrajectoryWindow(history;start_time=2,end_time=1)
end

@testset "polygon callbacks validate their coordinate results" begin
    mktempdir() do dir
        shape = PolygonSurface([(0,0,0),(1,0,0),(0,1,0)],[[1,2,3]])
        path = joinpath(dir,"invalid.vtkhdf")
        for callback in ((p,t)->nothing,(p,t)->(NaN,0,0),(p,t)->(0,0,0,0))
            @test_throws ArgumentError SavePolygonMotionSequence(path,(;shape);motions=(;shape=callback),times=[0.,1.])
        end
        @test_throws ArgumentError PolygonSurface([(0,0,0),(1,0,0)],[[1,2]])
        @test_throws ArgumentError PolygonSurface([(0,0,0),(1,0,0),(0,1,0)],[[1,2,4]])
    end
end

if CUDA.functional()
    @testset "reference-case device poses include jumps and stopped file histories" begin
        for number in 1:9, T in (Float32,Float64)
            case = motion_example(number)
            parts = [p for p in case.parts if any(n -> n.group_marker==p.mk+1,case.motion.motions)]
            reference = [last(MotionExamples.surface_points(p.shape)) for p in parts]
            host = (Position=reference,ID=collect(1:length(parts)),Type=fill(Moving,length(parts)),GroupMarker=[p.mk+1 for p in parts])
            prepared = PM.prepare_prescribed_motion(case.motion,case.geometry,host)
            positions,velocities = CuArray(reference),CUDA.zeros(SVector{3,T},length(parts))
            ty,gm,ids = CuArray(host.Type),CuArray(host.GroupMarker),CuArray(host.ID)
            for time in (0.,0.2,1.411,2.7,2.71,5.4,5.41,8.,8.02,9.,9.02,case.duration)
                ROT.launch_rotation!(prepared,positions,velocities,ty,gm,ids,HostStep(T(0),T(time)))
                sampled = Float64(T(time))
                p,v = Array(positions),Array(velocities)
                for i in eachindex(parts)
                    @test p[i] ≈ motion_position(case.motion,parts[i].mk,reference[i],sampled) atol=5e-12 rtol=0
                    @test v[i] ≈ SVector{3,T}(motion_velocity(case.motion,parts[i].mk,reference[i],sampled)) atol=2e-6 rtol=1e-6
                end
            end
        end
    end

    function run_transition(number,start_time,end_time,graph,scheme)
        mktempdir() do dir
            case = motion_example(number;dx=0.05)
            metadata = SimulationMetaData{3,Float32}(SimulationName="MotionTransition",SaveLocation=dir,
                TotalTime=start_time,SimulationTime=end_time,OutputTimes=[start_time+0.01,end_time],
                GPUUseGraph=graph,GPUMaxStepsPerSync=4,GPUClipTimeStepToOutput=true,
                GPUDoublePosition=true,VisualizeInParaview=false,OpenLogFile=false,ExportGridCells=false)
            particles = AllocateDataStructures(case.geometry,metadata)
            reference = Dict(id=>p for (id,p) in zip(particles.ID,particles.Position))
            logger = SimulationLogger(dir)
            result = try
                redirect_stdout(devnull) do
                    RunSimulation(SimGeometry=case.geometry,SimMetaData=metadata,SimParticles=particles,
                        SimConstants=case.constants,SimKernel=SPHKernelInstance{3,Float32}(WendlandC2();dx=0.05f0),
                        SimLogger=logger,SimTimeStepping=scheme,SimViscosity=ZeroViscosity(),
                        SimDensityDiffusion=ZeroDensityDiffusion(),SimMotion=case.motion)
                end
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            @test result.final_state.complete
            @test result.configuration.execution.graph_capture_failures == 0
            graph && @test result.configuration.execution.graph_batches > 0
            moving = Set(node.group_marker for node in case.motion.motions if node.group_marker!==nothing)
            time = result.final_state.time
            for i in eachindex(particles.ID)
                ref = reference[particles.ID[i]]
                marker = Int(particles.GroupMarker[i])
                expected = marker in moving ? motion_position(case.motion,marker-1,ref,time) : ref
                @test particles.Position[i] ≈ expected atol=2e-11 rtol=0
            end
            order = sortperm(particles.ID)
            return (particles.Position[order],particles.Velocity[order],particles.Density[order],time)
        end
    end

    @testset "reset and file-stop transitions survive both integrators and graph replay" begin
        for scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            for (number,start,finish) in ((1,2.69,2.72),(8,7.99,8.04))
                @test run_transition(number,start,finish,false,scheme) == run_transition(number,start,finish,true,scheme)
            end
        end
    end
end

end
