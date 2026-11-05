module PendulumTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, SHA, HDF5
include(joinpath(@__DIR__, "..", "benchmark", "pendulum_case.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "pendulum_validation.jl"))

@testset "Pendulum GenCase geometry and independent reference" begin
    case = prepare_pendulum_3d(; FloatType = Float64)
    @test length.(getproperty.(case.geometry, :Particles)) == [7200,161,161,161,169]
    @test case.meta.Periodicity === nothing
    @test case.dynamics.contacts === nothing
    @test length(case.dynamics.joints) == 4
    @test all(j -> j.kind == 0, case.dynamics.joints)
    @test all(g -> g.Type != Fluid, case.geometry)
    @test_throws ArgumentError prepare_pendulum_3d(; dx = 0.025)
    lattice = sort!([(Int(g.GroupMarker), round(Int,(p[1]-4)/0.05),
        round(Int,(p[2]-0.5)/0.05), round(Int,(p[3]+0.1)/0.05))
        for g in case.geometry for p in g.Particles.Position])
    digest = bytes2hex(sha256(join((join(row,",") * "\n" for row in lattice))))
    @test digest == "c8d2f1085cf7af8e8c1b14044aad1e133468e40e3f20892c05881abc6e7388f1"
    properties = pendulum_properties(case)
    @test properties.mass ≈ [20.125,20.125,20.125,21.125] atol = 1e-12
    @test getindex.(properties.center,2) ≈ [5.478260869565,6.478260869565,7.478260869565,8.50295857988] atol = 1e-11
    @test diag(properties.inertia[1]) ≈ [1.734239130435,0.075,1.734239130435] atol = 1e-10
    @test diag(properties.inertia[4]) ≈ [1.995440088757,0.07875,1.995440088757] atol = 1e-10
    times = collect(0.:0.1:6.)
    reference = pendulum_planar_reference(case,times)
    refined = pendulum_planar_reference(case,times; timestep = 0.000025)
    @test maximum(norm(reference[i][b]-refined[i][b]) for i in eachindex(times), b in 1:4) < 1e-7
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "Pendulum midpoint joint accuracy and default surfaces" begin
        mktempdir() do directory
            case = prepare_pendulum_3d(; FloatType = Float64)
            for scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
                path = joinpath(directory,string(nameof(typeof(scheme))))
                result = redirect_stdout(devnull) do
                    run_pendulum_3d(; FloatType = Float64, save_location = path,
                        time_stepping = scheme, visualize = false, open_log_file = false)
                end
                metrics = pendulum_validation(case,path)
                @test result.final_state.complete
                @test metrics.frames == 601
                @test metrics.max_joint_error_m < 1e-6
                @test metrics.max_plane_error_m < 1e-10
                @test metrics.max_relative_energy_error < 0.00002
                @test metrics.max_planar_reference_error_m < 0.002
                if scheme isa SymplecticTimeStepping
                    vtk = Base.invokelatest(case.save_surfaces,path,case.geometry)
                    h5open(vtk,"r") do io
                        @test first(read(io["VTKHDF/PendulumSurfaces/NumberOfPoints"])) == 196
                        @test length(read(io["VTKHDF/PendulumSurfaces/Steps/Values"])) == 601
                        forces = io["VTKHDF/FloatingBodies/PointData"]
                        @test all(iszero,read(forces["Force"]))
                        @test read(forces["TotalForce"]) ≈ read(forces["GravityForce"]) + read(forces["ConstraintForce"])
                    end
                    script = read(joinpath(path,"Pendulum3D_SingleVTKHDFStateFile.py"),String)
                    @test occursin("Analytical spherical pivot (visual guide)",script)
                    @test occursin("Joint and contact reaction (red)",script)
                    Base.invokelatest(case.save_surfaces,path,case.geometry; write_geometry = false)
                    script = read(joinpath(path,"Pendulum3D_SingleVTKHDFStateFile.py"),String)
                    @test count("# Display CAD surfaces following",script) == 1
                end
            end
            short = [redirect_stdout(devnull) do
                run_pendulum_3d(; FloatType = Float32, save_location = joinpath(directory,"graph_$graph"),
                    simulation_time = 0.04, output_interval = 0.02, use_graph = graph,
                    visualize = false, open_log_file = false)
            end for graph in (false,true)]
            @test short[1].final_state.floating.center == short[2].final_state.floating.center
            @test short[1].final_state.floating.velocity == short[2].final_state.floating.velocity
            @test short[1].final_state.floating.orientation == short[2].final_state.floating.orientation
            a,b = short[1].final_state.particles,short[2].final_state.particles
            for field in (:Position,:Velocity,:Density)
                @test getproperty(a,field)[sortperm(a.ID)] == getproperty(b,field)[sortperm(b.ID)]
            end
        end
    end
end

end # module
