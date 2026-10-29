module PeriodicImageTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const CL = SPHExampleGPU.SPHCellList
const GRID = SPHExampleGPU.GPUCellGrid
const GK = SPHExampleGPU.GPUKernels

if CUDA.functional()
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
end
end
