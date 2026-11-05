module PelamisTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra,SHA,HDF5,CSV
include(joinpath(@__DIR__,"..","benchmark","pelamis_case.jl"))

@testset "Pelamis reference geometry and forcing" begin
    case = prepare_pelamis_2d()
    @test length.(getproperty.(case.geometry,:Particles)) == [1037,300,655,655,655,660,26839]
    @test case.meta.Periodicity === nothing
    @test case.dynamics.contacts === nothing
    @test case.dynamics.max_timestep === nothing
    @test getproperty.(case.dynamics.joints,:kind) == (2,1,1,1)
    lattice = sort!([(Int(g.GroupMarker),round(Int,(p[1]-2.8)/0.025),round(Int,(p[2]+0.1)/0.025))
        for g in case.geometry for p in g.Particles.Position])
    @test bytes2hex(sha256(join((join(row,",")*"\n" for row in lattice)))) ==
        "97eb50325d57d7f13706ba41ed6454477cab615fb29481c26013a321abcd91c2"
    @test_throws ArgumentError prepare_pelamis_2d(;dx = 0.05)
    @test Base.invokelatest(case.piston_state,0.)[1] == 0
    @test Base.invokelatest(case.piston_state,0.)[2] > 0.2
    @test Base.invokelatest(case.piston_state,0.375)[1] ≈ 0.0528809 atol = 1e-7
    @test case.kernel.h ≈ 1.1sqrt(2)*0.025 rtol = 1e-7
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "Pelamis graph/direct coupling and planar force export" begin
        mktempdir() do path
            results = [redirect_stdout(devnull) do
                run_pelamis_2d(;simulation_time = 0.1,save_location = joinpath(path,string(graph)),
                    use_graph = graph,visualize = false,open_log_file = false)
            end for graph in (false,true)]
            @test results[1].performance.steps == results[2].performance.steps
            # Float32 atomic sums of fluid loads can differ in addition order.
            # Bound maximum physical differences instead of requiring identical bits.
            for (field,tolerance) in ((:center,1e-6),(:velocity,3e-5),(:angle,1e-6),(:omega,5e-5))
                x,y = getproperty(results[1].final_state.floating,field),getproperty(results[2].final_state.floating,field)
                @test maximum(norm,x-y) < tolerance
            end
            a,b = results[1].final_state.particles,results[2].final_state.particles
            for (field,tolerance) in ((:Position,1e-5),(:Velocity,5e-4),(:Density,0.01))
                @test maximum(norm,getproperty(a,field)[sortperm(a.ID)]-getproperty(b,field)[sortperm(b.ID)]) < tolerance
            end
            case = prepare_pelamis_2d()
            vtk = Base.invokelatest(case.save_surfaces,joinpath(path,"true"),case.geometry;piston_state = case.piston_state)
            h5open(vtk,"r") do io
                @test first(read(io["VTKHDF/PelamisSurfaces/NumberOfPoints"])) == 43644
                @test first(read(io["VTKHDF/PelamisSurfaces/Polygons/NumberOfCells"])) == 70168
                @test read(io["VTKHDF/PelamisSurfaces/Steps/Values"]) == [0.,0.05,0.1]
                @test haskey(io,"VTKHDF/PistonSurface")
                data = io["VTKHDF/FloatingBodies/PointData"]
                @test size(read(data["ConstraintTorque"]),1) == 3
                @test all(iszero,read(data["ConstraintTorque"])[1:2,:])
                @test any(!iszero,read(data["Force"]))
                @test read(data["TotalForce"]) ≈ read(data["Force"])+read(data["GravityForce"])+read(data["ConstraintForce"])
            end
            @test maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,results[2].final_state.floating)) < 1e-5
        end
    end
end
end # module
