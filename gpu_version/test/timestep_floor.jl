using Test, SPHExampleGPU, StaticArrays, CUDA, LinearAlgebra

@testset "reference minimum floor cannot override strict stability bounds" begin
    driver = SPHExampleGPU.SPHCellList
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); dx=0.025)
    constants = SimulationConstants{Float64}(2; dx=0.025,c₀=160,g=0,CFL=0.2)
    reference = DualSPHysicsSymplectic()
    strict = DualSPHysicsSymplectic(respect_stability_limits=true)
    for acceleration in (1e12,1e16)
        criteria = SVector(acceleration^2,0.,0.,0.)
        expected = constants.CFL*sqrt(kernel.h/acceleration)
        @test driver.dual_time_step(criteria,kernel,constants,strict) ≈ expected
        @test driver.dual_time_step(criteria,kernel,constants,reference) > expected
        @test SPHExampleGPU.VariableResolution.resolution_dt([criteria],kernel,constants,true) ≈ expected
        @test SPHExampleGPU.VariableResolution.resolution_dt([criteria],kernel,constants) > expected
    end
    criteria = SVector(1.,1.,1.,0.)
    @test driver.dual_time_step(criteria,kernel,constants,strict) ==
        driver.dual_time_step(criteria,kernel,constants,reference)
    @test SPHExampleGPU.VariableResolution.resolution_dt([criteria],kernel,constants,true) ==
        SPHExampleGPU.VariableResolution.resolution_dt([criteria],kernel,constants)
end

if CUDA.functional()
    @testset "neighbor timestep collection preserves laminar forces" begin
        metadata = SimulationMetaData{2,Float64,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
            SimulationName="LaminarCriteria",SaveLocation=tempdir(),GPUInteractionFastMath=false)
        points = [SVector(0.1,0.1),SVector(0.125,0.1)]
        geometry = [SPHGeometry{2,Float64}(Particles=particle_struct_array(points,1000.;
            Velocity=[SVector(-1.,0.),SVector(1.,0.)]),GroupMarker=1,Type=Fluid)]
        constants = SimulationConstants{Float64}(2; dx=0.025,c₀=160,g=0,ν₀=0.01,α=0)
        kernel = SPHKernelInstance{2,Float64}(WendlandC2(); dx=0.025)
        particles = AllocateDataStructures(geometry,metadata)
        gpu = upload_particles(particles,metadata)
        support = GPUSupportArrays{2,Float64}(gpu,metadata)
        cells = CellListWorkspace{2,Float64}(length(gpu))
        SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu,cells,kernel.H⁻¹; support,kernel,constants)
        function evaluate(rates)
            SPHExampleGPU.GPUKernels.launch_interactions!(support.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,
                support.∇Cᵢ,support.∇◌rᵢ,gpu.Position,gpu.Density,gpu.Velocity,gpu.Type,
                cells.CellStart,gpu.CellID,cells.grid_dev,HostStep(0.,0.),ZeroDensityDiffusion(),Laminar(),
                kernel,constants,Val(false),Val(false); lanes=Val(1),viscous_dt=rates,fastmath=false)
            return Array(gpu.Acceleration)
        end
        reference = evaluate(nothing)
        rates = CUDA.zeros(Float64,length(gpu))
        measured = evaluate(rates)
        @test maximum(norm,reference) > 1e-3
        @test measured ≈ reference rtol=1e-12 atol=1e-12
        @test maximum(Array(rates)) > 0
        @test SPHExampleGPU.SPHCellList.reference_viscous_limit(Laminar(),kernel,constants) ≈ kernel.h^2/(8constants.ν₀)
        @test isinf(SPHExampleGPU.SPHCellList.reference_viscous_limit(ArtificialViscosity(),kernel,constants))
        mktempdir() do directory
            metadata.SaveLocation = directory
            metadata.VariableResolution = VariableResolutionConfig(VariableResolutionBox((0.05,0.05),(0.2,0.2)); fine_geometry=geometry)
            logger = SimulationLogger(directory)
            try
                @test_throws ArgumentError RunSimulation(SimGeometry=geometry,SimMetaData=metadata,SimConstants=constants,
                    SimKernel=kernel,SimParticles=particles,SimLogger=logger,SimViscosity=Laminar(),
                    SimDensityDiffusion=FourtakasDensityDiffusion(),SimTimeStepping=DualSPHysicsSymplectic(respect_stability_limits=true))
            finally
                close(logger.LoggerIo)
            end
        end
    end
    @testset "variable-resolution device timestep respects the optional floor guard" begin
        vr = SPHExampleGPU.VariableResolution
        kernel = SPHKernelInstance{2,Float64}(WendlandC2(); dx=0.025)
        constants = SimulationConstants{Float64}(2; dx=0.025,c₀=160,g=0,CFL=0.2)
        partial = CuArray([SVector(1e24,0.,0.,0.)])
        f = CUDA.zeros(Float64,16)
        state = vr.ResolutionStep(f,CUDA.zeros(Int32,64),CuArray(Int32[1]),nothing,Int32(0),false)
        @cuda threads=1 vr.resolution_dt_kernel!(partial,partial,kernel,kernel,constants,constants,state,true,true)
        @test Array(f)[3] ≈ constants.CFL*sqrt(kernel.h/1e12)
        @cuda threads=1 vr.resolution_dt_kernel!(partial,partial,kernel,kernel,constants,constants,state,false,true)
        @test Array(f)[1] ≈ constants.CFL*sqrt(kernel.h/1e12)
        @test Array(state.i)[1] == 0
    end
end
