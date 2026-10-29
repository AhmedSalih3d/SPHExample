module PeriodicImageTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const CL = SPHExampleGPU.SPHCellList
const GRID = SPHExampleGPU.GPUCellGrid
const GK = SPHExampleGPU.GPUKernels

@testset "periodic copies are the DBC default" begin
    kernel=SPHKernelInstance{2,Float32}(WendlandC2();h=0.04f0)
    function metadata(;kwargs...)
        SimulationMetaData{2,Float32}(SimulationName="DefaultImages",SaveLocation=tempdir(),GPULanesPerParticle=1;kwargs...)
    end
    resolve(meta)=effective_configuration(meta,kernel,ComplexDensityDiffusion(),ArtificialViscosity();particle_count=5)
    periodic=PeriodicBoundary{2}(1;min=0,max=1,shift=(0,0.2))
    plain=metadata()
    @test plain.GPUPeriodicImages
    @test !resolve(plain).effective.periodic_images
    default=metadata(Periodicity=periodic)
    @test resolve(default).effective.periodic_images
    @test resolve(default).effective.time_stepping===typeof(plain.TimeSteppingMode)
    @test resolve(default).effective.density_diffusion_model===ComplexDensityDiffusion
    @test !resolve(metadata(Periodicity=periodic,GPUPeriodicImages=false)).effective.periodic_images
    narrow=metadata(Periodicity=PeriodicBoundary{2}(1;min=0,max=0.1))
    @test only(resolve(narrow).fallbacks).reason===:period_not_larger_than_twice_support
    mdbc=SimulationMetaData{2,Float32,NoShifting,NoKernelOutput,SimpleMDBC}(
        SimulationName="Fallback",SaveLocation=tempdir(),Periodicity=periodic,GPULanesPerParticle=1)
    @test !resolve(mdbc).effective.periodic_images
    @test only(resolve(mdbc).fallbacks).reason===:unsupported_boundary_mode
    @test DualSPHysicsSymplectic().periodic_images
end

if CUDA.functional()
    @testset "standard diffusion keeps reference pair orientation" begin
        for D in (2,3), T in (Float32,Float64), cache in (false,true)
            point(a,b)=SVector{D,Float64}(ntuple(d->d==1 ? a : d==2 ? b : 0.2,D))
            periodic=PeriodicBoundary{D}(1;min=0,max=1,shift=ntuple(d->d==2 ? 0.2 : 0.0,D))
            positions=[point(0.01,0.4),point(0.99,0.2),point(0.98,0.22),point(0.025,0.425),point(0.5,0.3)]
            geometry=[SPHGeometry{D,T}(Particles=particle_struct_array(positions,T(1000);
                Density=T[1002,1003,1005,1001,1004],Velocity=[SVector{D,T}(ntuple(d->T(0.02*(i-d)),D)) for i in 1:5]),GroupMarker=1,Type=Fluid)]
            constants=SimulationConstants{T}(D;dx=0.02,c₀=20,g=9.81,ν₀=1e-6)
            kernel=SPHKernelInstance{D,T}(WendlandC2();h=T(0.04))
            for diffusion in (LinearDensityDiffusion(),ComplexDensityDiffusion(),FourtakasDensityDiffusion()),
                viscosity in (ArtificialViscosity(),LaminarSPS())
                results=map((false,true)) do images
                    meta=SimulationMetaData{D,T}(SimulationName="OrderedImages",SaveLocation=tempdir(),
                        Periodicity=periodic,GPUDoublePosition=true,GPUCellSubdivision=2,
                        GPUPeriodicImages=images,GPUCacheInteractionFields=cache)
                    gpu=upload_particles(AllocateDataStructures(geometry,meta),meta)
                    sup=GPUSupportArrays{D,T}(gpu,meta;kernel)
                    cl=CellListWorkspace{D,Float64}(5;reach=2,periodicity=periodic)
                    CL.rebuild_cell_list!(gpu,cl,kernel.H⁻¹;support=sup,constants,kernel)
                    images && CL.configure_periodic_images!(sup.PeriodicImages,periodic,cl.grid,kernel)
                    ctx=(;gpu,sup,cl,SimKernel=kernel,SimConstants=constants,SimDensityDiffusion=diffusion,
                        SimViscosity=viscosity,FlagKernel=Val(false),FlagShift=Val(false),threads=128,lanes=Val(1),bforces=Val(true))
                    snapshots=Any[]
                    for fraction in (0.0,1.0)
                        # Keep old cell anchors while the evaluated positions cross
                        # cells and a shifted seam; density and velocity change too.
                        x=CuArray(Array(gpu.Position) .+ Ref(point(-0.02fraction,0.003fraction)))
                        v=CuArray(Array(gpu.Velocity) .* T(1+fraction))
                        rho=CuArray(Array(gpu.Density) .+ T(2fraction))
                        step=SPHExampleGPU.GPUStepState.HostStep(0.001,0.0)
                        GK.launch_pos_cells!(sup.PosCells,x,gpu.CellID,cl.grid_dev,step,kernel)
                        CL.prepare_periodic_bounds!(ctx,x,step)
                        GK.launch_prepare_interactions!(x,step;Density=rho,ParticleType=gpu.Type,SimConstants=constants,
                            inverse_density=sup.InvDensity,pressure=sup.Pressure)
                        CL.launch_solver_interactions!(ctx,x,rho,v,sup.PosCells,step)
                        push!(snapshots,(;rate=Array(sup.dρdtI),acceleration=Array(gpu.Acceleration)))
                    end
                    snapshots
                end
                tolerance=T===Float32 ? 3e-4 : 2e-10
                for (a,b) in zip(results...)
                    @test b.rate ≈ a.rate rtol=tolerance atol=tolerance
                    @test b.acceleration ≈ a.acceleration rtol=tolerance atol=tolerance
                end
            end
        end
    end
    @testset "dense periodic sources use the evaluated state" begin
        for D in (2,3), T in (Float32,Float64), R in (1,2), axis in 1:D
            transverse=mod1(axis+1,D)
            point(a,b)=SVector{D,Float64}(ntuple(d->d==axis ? a : d==transverse ? b : 0.2,D))
            periodic=PeriodicBoundary{D}(axis;min=0,max=1,shift=ntuple(d->d==transverse ? 0.2 : 0.0,D))
            positions=[point(0.01,0.4),point(0.99,0.2),point(0.98,0.22),point(0.025,0.425),point(0.5,0.3)]
            velocity=[SVector{D,T}(ntuple(d->T(0.02*(i-d)),D)) for i in 1:5]
            geometry=[SPHGeometry{D,T}(Particles=particle_struct_array(positions,T(1000);
                Density=T[1002,1003,1005,1001,1004],Velocity=velocity),GroupMarker=1,Type=Fluid)]
            constants=SimulationConstants{T}(D;dx=0.02,c₀=20,g=0)
            kernel=SPHKernelInstance{D,T}(WendlandC2();h=T(0.04))
            results=map((false,true)) do images
                meta=SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
                    SimulationName="ImageSources",SaveLocation=tempdir(),Periodicity=periodic,
                    GPUDoublePosition=true,GPUCellSubdivision=R,GPUBoundaryForces=false)
                scheme=DualSPHysicsSymplectic(;domain_min=(-1.,-1.,-1.),domain_max=(2.,2.,2.),periodic_images=images)
                meta.TimeSteppingMode=scheme
                gpu=upload_particles(AllocateDataStructures(geometry,meta),meta)
                copyto!(gpu.Type,[Fluid,Fluid,Moving,Fixed,Frozen])
                sup=GPUSupportArrays{D,T}(gpu,meta)
                cl=CellListWorkspace{D,Float64}(5;reach=R,periodicity=periodic)
                CL.rebuild_cell_list!(gpu,cl,kernel.H⁻¹;support=sup,constants,kernel)
                function evaluate(x,v,rho)
                    dt=CL.dual_force_stage!(x,v,rho,sup.PosCells,gpu,cl,sup,kernel,constants,
                        FourtakasDensityDiffusion(),ArtificialViscosity(),meta,SPHExampleGPU.GPUStepState.HostStep(0.001,0.0))
                    order=sortperm(Array(gpu.ID))
                    (;dt,acceleration=Array(gpu.Acceleration)[order],rate=Array(sup.dρdtI)[order],
                        viscous=Array(sup.dual_state.viscous)[order])
                end
                initial=evaluate(gpu.Position,gpu.Velocity,gpu.Density)
                # A continuous predictor crosses the seam. Its velocity and density
                # differ from the original state, including the moving wall.
                xp=CuArray(Array(gpu.Position) .+ Ref(point(-0.02,0.0)))
                vp=CuArray(Array(gpu.Velocity) .* T(1.7))
                rp=CuArray(Array(gpu.Density) .+ T(2))
                predictor=evaluate(xp,vp,rp)
                if images
                    workspace=sup.dual_state.images
                    @test only(Array(workspace.count))==9
                    @test only(Array(workspace.invalid))==0
                    @test length(unique(Array(workspace.targets)))==5
                    @test length(gpu)==5
                    dense=workspace.sorted
                    targets=Array(workspace.targets)
                    @test Array(dense.Density)[targets] == Array(rp)
                    @test Array(dense.Velocity)[targets] == Array(vp)
                    @test Array(dense.Type)[targets] == Array(gpu.Type)
                    broken=copy(xp)
                    copyto!(broken,fill(point(0.2,10.0),5))
                    @test_throws ArgumentError evaluate(broken,vp,rp)
                end
                (;initial,predictor)
            end
            for state in (:initial,:predictor)
                a,b=getproperty.(results,state)
                tolerance=T===Float32 ? 2e-4 : 2e-11
                @test b.dt ≈ a.dt rtol=tolerance
                @test b.acceleration ≈ a.acceleration rtol=tolerance atol=tolerance
                @test b.rate ≈ a.rate rtol=tolerance atol=tolerance
                @test b.viscous ≈ a.viscous rtol=tolerance atol=tolerance
            end
        end
    end
    @testset "periodic image allocation contracts" begin
        @test_throws ArgumentError GPUSupportArrays{3,Float32}(2;periodic_images=true)
        @test_throws ArgumentError GPUSupportArrays{3,Float32}(2;periodic=true,dual_time_stepping=true,
            periodic_images=true,cache_periodic_queries=true)
        images=CL.PeriodicImageWorkspace{3,Float32,Float64}(2)
        periodic=PeriodicBoundary{3}(2;min=0,max=1)
        kernel=SPHKernelInstance{3,Float32}(WendlandC2();h=0.04f0)
        @test_throws ArgumentError CL.initialize_periodic_images!(images,periodic,DualSPHysicsSymplectic(),kernel)
        narrow=PeriodicBoundary{3}(2;min=0,max=0.1)
        @test_throws ArgumentError CL.initialize_periodic_images!(images,narrow,
            DualSPHysicsSymplectic(domain_min=(-1.,-1.,-1.),domain_max=(1.,1.,1.)),kernel)
    end
    @testset "source failure stops integration and survives batch grid checks" begin
        periodic=PeriodicBoundary{2}(1;min=0,max=1,shift=(0,0.2))
        geometry=[SPHGeometry{2,Float32}([SVector(0.99,0.2),SVector(0.02,0.4)];Density=1000,GroupMarker=1,Type=Fluid)]
        meta=SimulationMetaData{2,Float32}(SimulationName="InvalidSources",SaveLocation=tempdir(),Periodicity=periodic)
        constants=SimulationConstants{Float32}(2;dx=0.02,c₀=20)
        kernel=SPHKernelInstance{2,Float32}(WendlandC2();h=0.04f0)
        gpu=upload_particles(AllocateDataStructures(geometry,meta),meta)
        sup=GPUSupportArrays{2,Float32}(gpu,meta;kernel)
        cl=CellListWorkspace{2,Float32}(2;periodicity=periodic)
        CL.rebuild_cell_list!(gpu,cl,kernel.H⁻¹;support=sup,constants,kernel)
        CL.configure_periodic_images!(sup.PeriodicImages,periodic,cl.grid,kernel)
        state=SPHExampleGPU.GPUStepState.StepState{Float32}()
        ctx=(;gpu,sup,cl,SimKernel=kernel,SimConstants=constants,SimDensityDiffusion=ComplexDensityDiffusion(),
            SimViscosity=ArtificialViscosity(),FlagKernel=Val(false),FlagShift=Val(false),threads=128,lanes=Val(1),bforces=Val(true))
        fill!(gpu.Acceleration,SVector(3f0,4f0))
        broken=CuArray([SVector(0.5f0,100f0),SVector(0.5f0,100f0)])
        CL.launch_solver_interactions!(ctx,broken,gpu.Density,gpu.Velocity,sup.PosCells,state)
        SPHExampleGPU.GPUStepState.readback!(state)
        @test state.ih[SPHExampleGPU.GPUStepState.I_STOP] == SPHExampleGPU.GPUStepState.STOP_PERIODIC_IMAGES
        @test Array(gpu.Acceleration) == fill(SVector(3f0,4f0),2)
        copyto!(gpu.Position,[SVector(1.01f0,0.2f0),SVector(0.02f0,0.4f0)])
        GRID.prepare_cell_grid!(cl,gpu.Position,kernel.H⁻¹;step=state,periodic_state=CL.periodic_particle_state(gpu,nothing))
        SPHExampleGPU.GPUStepState.readback!(state)
        @test state.ih[SPHExampleGPU.GPUStepState.I_STOP] == SPHExampleGPU.GPUStepState.STOP_PERIODIC_IMAGES
        @test only(Array(sup.PeriodicImages.builds)) == 1
    end
end
end
