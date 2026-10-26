module MotionExampleTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5
include(joinpath(@__DIR__,"..","example","Motion","MotionExamples.jl"))
using .MotionExamples

@testset "all nine generated Motion examples in 2D/3D" begin
    for D in (2,3), number in 1:9
        case = motion_example(number;dimensions=D)
        @test all(g -> g.Type == Moving && !isempty(g.Particles.Position),case.geometry)
        for time in (-1,0,0.5,1,2,3,4,5), g in case.geometry
            p = first(g.Particles.Position)
            @test all(isfinite,motion_position(case.motion,g.GroupMarker,p,time))
            @test all(isfinite,motion_velocity(case.motion,g.GroupMarker,p,time))
        end
        if CUDA.functional()
            mktempdir() do dir
                result = run_motion_example(number;dimensions=D,simulation_time=0.03,output_interval=0.01,
                    visualize=false,open_log_file=false,use_graph=true,save_location=dir)
                @test result.final_state.complete
                @test result.configuration.execution.graph_capture_failures == 0
                @test result.configuration.execution.graph_batches > 0
                p = result.final_state.particles
                initial = AllocateDataStructures(case.geometry;position_type=Float64)
                reference = Dict(id=>x for (id,x) in zip(initial.ID,initial.Position))
                for i in eachindex(p.ID)
                    x = reference[p.ID[i]]
                    g = Int(p.GroupMarker[i])
                    t = result.final_state.time
                    @test p.Position[i] ≈ motion_position(case.motion,g,x,t) atol = 1e-12
                    @test p.Velocity[i] ≈ motion_velocity(case.motion,g,x,t) atol = 2e-6
                end
                path = joinpath(dir,"Motion"*lpad(number,2,'0')*".vtkhdf")
                h5open(path,"r") do file
                    root = file["VTKHDF"]
                    @test all(isfinite,read(root["Points"]))
                    @test length(read(root["Steps/Values"])) == result.diagnostics.output_frames
                    points = read(root["Points"])
                    velocities = read(root["PointData/Velocity"])
                    ids = read(root["PointData/ID"])
                    groups = read(root["PointData/GroupMarker"])
                    offsets = read(root["Steps/PointOffsets"])
                    for (frame,time) in enumerate(read(root["Steps/Values"]))
                        for col in (offsets[frame]+1):(offsets[frame]+length(p))
                            x,g = reference[ids[col]],Int(groups[col])
                            expected = motion_position(case.motion,g,x,time)
                            velocity = motion_velocity(case.motion,g,x,time)
                            @test points[:,col] ≈ (D==2 ? [expected[1],expected[2],0] : expected) atol = 1e-12
                            @test velocities[:,col] ≈ (D==2 ? [velocity[1],velocity[2],0] : velocity) atol = 2e-6
                        end
                    end
                end
            end
        end
    end
end

end
