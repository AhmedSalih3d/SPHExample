module PointAbsorberSpringTests
using Test, SPHExampleGPU, CUDA, StaticArrays, CSV, LinearAlgebra, HDF5
let path = joinpath(@__DIR__, "..", "example", "VResolutionPointAbsorberSpring2d.jl")
    source = read(path, String)
    source = replace(source, r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n")
    include_string(@__MODULE__, source, path)
end

@testset "06 Special geometry, link and our solver parameters" begin
    case = point_absorber_spring_case()
    @test case.region.lower == SVector(1.7,0.2)
    @test case.region.upper == SVector(2.3,0.6)
    @test case.region.ratio == 2
    @test length(case.coarse.geometry[1].Particles) == 2832
    @test length(case.coarse.geometry[3].Particles) == 360
    body = case.fine.geometry[3]
    @test body.Type == Floating && body.GroupMarker == 50
    @test length(body.Particles) == 2013
    @test sum(body.Particles.Position)/2013 ≈ SVector(2.0,0.4)
    @test body.Floating.RelativeWeight == 0.5
    @test Tuple(body.Floating.Constraints.translation_free) == (false,true)
    @test Tuple(body.Floating.Constraints.rotation_free) == (false,)
    @test case.spring.Anchor == SVector(2.0,0.0)
    @test case.spring.Attachment == SVector(2.0,0.325)
    @test case.spring.Stiffness == 100 && case.spring.Damping == 500
    @test case.spring.RestLength ≈ 0.325
    @test body.Floating.RelativeWeight*case.fine.constants.m₀*length(body.Particles) ≈ 25.1625
    for level in (case.coarse,case.fine)
        @test level.constants.γ == 7
        @test level.constants.α == 0.05f0
        @test level.constants.δᵩ == 0.1f0
        @test level.constants.CFL == 0.2f0
        @test level.kernel.h ≈ 1.2sqrt(2) * level.constants.dx
        @test level.kernel.η² ≈ (0.1level.kernel.h)^2
        @test all(>=(level.constants.ρ₀),level.geometry[2].Particles.Density)
    end
    theory = collect(CSV.File(joinpath(POINT_ABSORBER_DATA,"PistonTheory.csv")))
    @test maximum(abs(point_absorber_piston_state(r.time)[1]-r.displacement) for r in theory) < 1e-6
    reference = collect(CSV.File(joinpath(POINT_ABSORBER_DATA,"FloatingReference.csv")))
    @test length(reference) == 200
    @test all(r -> r.center_x == 2.0,reference)
    @test maximum(abs(r.spring_force_y-(-100*(r.center_y-0.4)-500r.velocity_y)) for r in reference) < 0.00031
    @test point_absorber_piston_state(0.0) == (0.0,0.0)
end

@testset "bed-following fluid rows have uniform wall clearance" begin
    for spacing in (0.02, 0.01, 0.005)
        water = point_absorber_water(spacing, SVector(3.0, 0.0), SVector(6.0, 0.4))
        @test length(Set(water)) == length(water)
        @test all(x -> 3.0 <= x[1] <= 6.0 && x[2] < 0.4, water)
        columns = Dict{Float64,Vector{SVector{2,Float64}}}()
        for point in water
            push!(get!(columns, point[1], SVector{2,Float64}[]), point)
        end
        clearance = [minimum(last, points) - point_absorber_bed(x) for (x, points) in columns]
        @test maximum(abs.(clearance .- spacing)) < 1e-12
        # Adjacent columns form sheared cells with the original particle area.
        xs = sort!(collect(keys(columns)))
        a, b = first(columns[xs[1]]), first(columns[xs[2]])
        c = columns[xs[1]][2]
        @test abs((b-a)[1]*(c-a)[2] - (b-a)[2]*(c-a)[1]) ≈ spacing^2
    end
    flat = point_absorber_water(0.01, SVector(0.01,0.01), SVector(1.0,0.39))
    @test all(x -> all(isapprox.(x ./ 0.01, round.(x ./ 0.01))), flat)
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "06 Special GPU piston, tracked spring body and output" begin
        runs = []
        for graphs in (false,true)
            directory = mktempdir()
            result = redirect_stdout(devnull) do
                run_point_absorber_spring(;dx=0.02,duration=0.002,output_interval=0.001,save_location=directory,particle_reserve=2048,
                    region=VariableResolutionBox((1.6,0.1),(2.4,0.7)),
                    metadata_options=(GPUUseGraph=graphs,GPUMaxStepsPerSync=16,VisualizeInParaview=false))
            end
            push!(runs,result)
            @test result.final_state.complete
            @test result.configuration.effective.time_stepping === SymplecticTimeStepping
            @test result.configuration.effective.density_diffusion_model === LinearDensityDiffusion
            @test result.configuration.effective.variable_resolution.follow == 50
            @test result.configuration.effective.boundary_mode === NoMDBC
            @test result.configuration.effective.variable_resolution.coarse_smoothing_length ≈ 1.2sqrt(2) * 0.02
            @test result.configuration.effective.variable_resolution.fine_smoothing_length ≈ 1.2sqrt(2) * 0.01
            @test length(result.final_state.springs) == 1
            @test result.final_state.floating.center[1][1] ≈ 2.0 atol=1e-12
            @test result.final_state.floating.angle == [0]
            @test result.final_state.resolution_pose.offset[2] ≈ result.final_state.floating.center[1][2]-0.4 atol=1e-12
            @test !any(==(Floating),result.final_state.resolution.coarse.Type)
            @test count(==(Floating),result.final_state.resolution.fine.Type) == 527
            coarse = result.final_state.resolution.coarse
            piston = findall(==(Moving),coarse.Type)
            @test length(piston) == 90
            trajectory = point_absorber_spring_case(dx=0.02,duration=0.002).motion
            displacement = motion_position(trajectory,SVector(0.0,0.0),0.002)[1]
            @test maximum(first,coarse.Position[piston]) ≈ displacement atol=1e-12
            @test length(readlines(joinpath(directory,"PointAbsorberSpring2D_Springs.csv"))) == 4
            @test filter(f -> endswith(f, ".vtkhdf"), readdir(directory)) == ["PointAbsorberSpring2D.vtkhdf"]
            @test !any(endswith(".series"), readdir(directory))
            h5open(joinpath(directory, "PointAbsorberSpring2D.vtkhdf"), "r") do file
                root = file["VTKHDF"]
                @test read_attribute(root, "Type") == "MultiBlockDataSet"
                @test read(root["Particles/Steps/Values"]) == [0.0, 0.001, 0.002]
                @test read(root["Geometry/Steps/Values"]) == read(root["Particles/Steps/Values"])
                @test haskey(root, "Assembly/Geometry")
                @test all(isfinite, read(root["Geometry/Points"]))
            end
            @test isfile(joinpath(directory,"PointAbsorberSpring2D_Comparison.csv"))
            @test all(d -> d[4] == 0,result.diagnostics.resolution.events)
        end
        @test runs[1].final_state.floating.center ≈ runs[2].final_state.floating.center atol=1e-7
        @test runs[1].final_state.floating.velocity ≈ runs[2].final_state.floating.velocity atol=1e-6
    end
end
end
