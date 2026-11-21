module Test14VariableResolutionTests

using Test, HDF5
include(joinpath(@__DIR__,"float2_variable_resolution.jl"))

@testset "Test 14 fixed 3D refinement with a fine-only floating body" begin
    mktempdir() do path
        release = last(RELEASES)
        case = build_fixed_refinement_case(release;dx=0.05,half_width=0.8,kernel_h_factor=1.0,
            duration=0.025,pause_time=0.01,snapshot_interval=0.025,
            output_variables=["Type"],batch_steps=4,save_location=path)
        config = case.metadata.VariableResolution
        body = only(filter(g -> g.Type == Floating,config.fine_geometry))
        @test all(g -> g.Type != Floating,case.geometry)
        @test body.Floating.RelativeWeight * config.fine_constants.m₀ * length(body.Particles) ≈ release.mass rtol=2e-7
        @test config.fine_constants.m₀ ≈ case.constants.m₀/8
        @test length(only(filter(g -> g.Type == Fluid,config.fine_geometry)).Particles) > 0

        if CUDA.functional()
            sampling = (;times=collect(0.005:0.005:0.025),floating_history=true)
            logger = SimulationLogger(path)
            host = AllocateDataStructures(case.geometry,case.metadata)
            result = try
                RunSimulation(SimGeometry=case.geometry,SimMetaData=case.metadata,SimConstants=case.constants,
                    SimKernel=case.kernel,SimLogger=logger,SimParticles=host,SimViscosity=case.viscosity,
                    SimDensityDiffusion=case.diffusion,SimTimeStepping=case.stepping,SimDamping=case.damping,
                    SimStateSampling=sampling)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            @test result.final_state.complete
            @test result.diagnostics.output_frames == 2
            @test all(e -> iszero(e[3]) && iszero(e[4]),result.diagnostics.resolution.events)
            (;coarse,fine) = result.final_state.resolution
            @test count(==(Floating),coarse.Type) == 0
            @test count(==(Floating),fine.Type) == length(body.Particles)
            @test length(unique(host.ID)) == length(host)
            @test all(isfinite,host.Density)
            samples = CSV.File(joinpath(path,"FreeHeave_Floating.csv"))
            center = getproperty(samples,Symbol("Center:2"))
            force = getproperty(samples,Symbol("Force:2"))
            @test samples.Time == [0.0; sampling.times]
            @test all(==(first(center)),center[1:3])
            @test center[end] > first(center)
            @test all(isfinite,force) && force[3] > 0
            h5open(joinpath(path,"FreeHeave.vtkhdf"),"r") do file
                @test read(file["VTKHDF/Particles/Steps/Values"]) == [0.0,0.025]
                @test read(file["VTKHDF/FloatingBodies/Steps/Values"]) == [0.0,0.025]
                @test Set(read(file["VTKHDF/Particles/PointData/Resolution"])) == Set(Int8[0,1])
            end
        end
    end
end

end
