module StillWedgeSolverTests

using Test, SPHExampleGPU, CUDA, HDF5

function capture_wedge(save,solver;kwargs...)
    scope = Module(gensym(:StillWedgeSolverFixture))
    Core.eval(scope,quote
        captured = Ref{Any}(nothing)
        function RunSimulation(;kwargs...)
            captured[] = (;kwargs...)
            close(kwargs[:SimLogger].LoggerIo)
            return nothing
        end
    end)
    path = joinpath(@__DIR__,"..","example","StillWedgeMDBC.jl")
    source = replace(read(path,String),r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n")
    Base.include_string(scope,source,path)
    run = Base.invokelatest(getproperty,scope,:run_still_wedge)
    Base.invokelatest(run;solver,save_location=save,
        simulation_time=0.02,visualize=false,open_log_file=false,generate_pdf_report=false,kwargs...)
    return Base.invokelatest(getproperty,scope,:captured)[]
end

@testset "StillWedge solver choice preserves the case layout" begin
    mktempdir() do save
        wcsph = capture_wedge(save,WCSPH())
        iisph = capture_wedge(save,IISPH(max_timestep=0.002))
        @test wcsph.SimSolver isa WCSPH
        @test iisph.SimSolver isa IISPH
        @test wcsph.SimTimeStepping isa SymplecticTimeStepping
        @test iisph.SimTimeStepping === nothing
        @test wcsph.SimViscosity isa ArtificialViscosity
        @test wcsph.SimDensityDiffusion isa LinearDensityDiffusion
        @test iisph.SimViscosity isa Laminar
        @test iisph.SimDensityDiffusion isa ZeroDensityDiffusion
        @test wcsph.SimMetaData.GPURadialInteractions && wcsph.SimMetaData.GPUCacheInteractionFields
        @test !iisph.SimMetaData.GPURadialInteractions && !iisph.SimMetaData.GPUCacheInteractionFields
        @test wcsph.ParticleNormalsPath === iisph.ParticleNormalsPath === nothing
        @test length(wcsph.SimParticles) == length(iisph.SimParticles) == 3027
        for field in (:Position,:Density,:Velocity,:ID,:Type,:GroupMarker)
            @test getproperty(wcsph.SimParticles,field) == getproperty(iisph.SimParticles,field)
        end
        for field in (:OutputTimes,:ExportSingleVTKHDF,:ExportGridCells,:OutputVariables,:GPUDoublePosition)
            @test getproperty(wcsph.SimMetaData,field) == getproperty(iisph.SimMetaData,field)
        end
        @test_throws ArgumentError capture_wedge(save,IISPH();time_stepping=SymplecticTimeStepping())
    end
end

if CUDA.functional()
    include(joinpath(@__DIR__,"..","benchmark","still_wedge_iisph.jl"))
    @testset "StillWedge IISPH uses RunSimulation and the shared output layout" begin
        mktempdir() do save
            run = run_still_wedge_iisph_case(save;simulation_time=0.02,generate_pdf_report=false)
            result,statistics = run.result,run.statistics
            @test result.final_state.time == 0.02
            @test statistics.particle_count == 3027
            @test statistics.frames == result.diagnostics.output_frames == 3
            @test result.configuration.effective.solver === :IISPH
            @test result.diagnostics.iisph.max_relative_density_residual <= 1e-3
            @test all(isfinite,result.final_state.particles.Pressure)
            h5open(joinpath(save,"StillWedge.vtkhdf"),"r") do io
                p = result.final_state.particles
                @test haskey(io,"VTKHDF/Steps/Values")
                @test io["VTKHDF/PointData/Pressure"][end-length(p)+1:end] == p.Pressure
            end
            @test isfile(joinpath(save,"StillWedge_GridCells.vtkhdf"))
            @test isfile(result.diagnostics.iisph.history_path)
        end
    end
end

end
