module FallingWedgeTests
using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5, CSV
let path = joinpath(@__DIR__,"..","example","VResolutionFallingWedge2d.jl")
    include_string(@__MODULE__,replace(read(path,String),r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n"),path)
end
const REGIONS = (VariableResolutionBox((-0.65,1.15),(0.65,2.0)),VariableResolutionBox((-0.45,1.3),(0.45,1.9)))

@testset "falling wedge inputs and hierarchy validation" begin
    case = falling_wedge_case(;dx=0.02,regions=REGIONS)
    @test map(c -> c.dx,case.constants) == (0.02f0,0.01f0,0.005f0)
    @test all(k.h ≈ 2c.dx for (k,c) in zip(case.kernels,case.constants))
    @test all(c -> c.c₀ == 80 && c.α == 0.01f0,case.constants)
    body = last(last(case.geometry))
    @test body.Type == Floating && body.GroupMarker == 51
    @test body.Floating.InitialVelocity == (0.0,-6.15)
    @test body.Floating.RelativeWeight*case.constants[3].m₀*length(body.Particles) ≈ 241 atol=1e-4
    @test all(g -> all(x -> x.Type != Floating,g),case.geometry[1:2])
    @test all(p -> all(isfinite,p),body.Particles.GhostPoints)
    @test all(p -> !iszero(p),body.Particles.GhostNormals)
    child = VariableResolutionConfig(REGIONS[2];fine_geometry=case.geometry[3],follow=51)
    @test_throws ArgumentError VariableResolutionConfig(REGIONS[1];fine_geometry=case.geometry[2],child)
    @test_throws ArgumentError falling_wedge_case(;initial_velocity=0.0,dx=0.02,regions=REGIONS)

    # Validate production spacings and box clearance without generating or
    # running the production particle set.
    outer = VariableResolutionBox((-0.49,1.31),(0.49,1.90))
    inner = VariableResolutionBox((-0.39,1.40),(0.40,1.798))
    child = VariableResolutionConfig(inner;fine_geometry=case.geometry[3],follow=51)
    config = VariableResolutionConfig(outer;fine_geometry=case.geometry[2],follow=51,child)
    constants = SimulationConstants{Float32}(2;dx=0.002,c₀=80)
    kernel = SPHKernelInstance{2,Float32}(WendlandC2();h=0.004f0)
    plan = SPHExampleGPU.VariableResolution.nested_resolution_plan(config,constants,kernel)
    @test map(c -> c.dx,plan.constants) == (0.002f0,0.001f0,0.0005f0)
    @test length(plan.references) == 2
end

if CUDA.functional()
    @testset "three simultaneous GPU levels and a downward force-responsive wedge" begin
        runs = []
        for graphs in (false,true)
            directory = mktempdir()
            result = redirect_stdout(devnull) do
                run_falling_wedge(;dx=0.02,regions=REGIONS,duration=0.0004,output_interval=0.0002,
                    particle_reserve=2048,visualize=false,graphs,save_location=directory)
            end
            push!(runs,result)
            @test result.final_state.complete
            @test result.configuration.effective.variable_resolution.level_count == 3
            @test result.configuration.execution.path == (graphs ? :cuda_graph : :direct)
            levels = result.final_state.resolution.levels
            @test length(levels) == 3
            @test all(!isempty,levels)
            for i in 1:3
                @test all(==(i-1),levels[i].Resolution)
                @test all(==(1),levels[i].Owned)
                @test all(isfinite,levels[i].Density)
            end
            @test !any(==(Floating),levels[1].Type) && !any(==(Floating),levels[2].Type)
            @test count(==(Floating),levels[3].Type) == length(last(last(falling_wedge_case(;dx=0.02,regions=REGIONS).geometry)).Particles)
            ids = vcat(map(l -> l.ID,levels)...)
            @test length(unique(ids)) == length(ids)
            @test all(e -> e[3] == e[4] == 0,result.diagnostics.resolution.events)
            rows = collect(CSV.File(joinpath(directory,"FallingWedge2D_Floating.csv")))
            @test getproperty(first(rows),Symbol("Velocity:1")) ≈ -6.15 atol=1e-6
            @test getproperty(last(rows),Symbol("Center:1")) < getproperty(first(rows),Symbol("Center:1"))
            @test only(result.final_state.floating.velocity)[2] > -6.15-9.81*0.0004
            @test filter(f -> endswith(f,".vtkhdf"),readdir(directory)) == ["FallingWedge2D.vtkhdf"]
            h5open(joinpath(directory,"FallingWedge2D.vtkhdf"),"r") do io
                root = io["VTKHDF"]
                @test Set(read(root["Particles/PointData/Resolution"])) == Set(Int8[0,1,2])
                @test read(root["Particles/Steps/Values"]) == [0.0,0.0002,0.0004]
                @test read(root["FloatingBodies/Steps/Values"]) == read(root["Particles/Steps/Values"])
                @test size(root["FloatingBodies/PointData/Force"],1) == 3
                @test read(root["Geometry/Steps/Values"]) == read(root["Particles/Steps/Values"])
                @test haskey(root,"Assembly/Geometry")
                @test haskey(root,"DampingZones")
            end
        end
        @test runs[1].final_state.floating.center ≈ runs[2].final_state.floating.center atol=2e-8
        @test runs[1].final_state.floating.velocity ≈ runs[2].final_state.floating.velocity atol=2e-5
    end
end
end
