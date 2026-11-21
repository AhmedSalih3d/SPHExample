module FloatingMDBCTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra
const K = SPHExampleGPU.GPUKernels
const CL = SPHExampleGPU.SPHCellList

if CUDA.functional()
    @testset "floating mDBC follows translated and rotated stage geometry" begin
        for D in (2,3), T in (Float32,Float64), compact in (false,true), half in (false,true)
            V = SVector{D,Float64}
            normal = SVector{D,T}(ntuple(d -> d == 1 ? T(0.04) : zero(T),D))
            x = V(ntuple(_ -> 1.,D))
            angle = half ? 0.5 : 0.3
            offset = V(ntuple(d -> d == 1 ? 0.04cos(angle) : d == 2 ? 0.04sin(angle) : 0.,D))
            expected_ghost = x+offset
            fluid = [expected_ghost + V(ntuple(d -> 0.02index[d],D))
                for index in CartesianIndices(ntuple(_ -> -2:2,D))][:]
            rho(p) = T(1000+sum(d*p[d] for d in 1:D))
            stale = V(ntuple(_ -> -2.,D))
            geometry = [SPHGeometry{D,T}(Particles=particle_struct_array([x],T(777);
                GhostPoints=[stale],GhostNormals=[normal]),Type=Floating,GroupMarker=3,
                Floating=FloatingDetails{T}(RelativeWeight=1)),
                SPHGeometry{D,T}(fluid;Density=rho.(fluid),Type=Fluid,GroupMarker=2)]
            host = AllocateDataStructures(geometry;position_type=Float64)
            gpu = upload_particles(host;position_type=Float64,boundary_only=compact)
            constants = SimulationConstants{T}(D;dx=0.02,c₀=25,g=0)
            kernel = SPHKernelInstance{D,T}(WendlandC2();h=T(0.08))
            cells = CellListWorkspace{D,Float64}(length(gpu))
            CL.rebuild_cell_list!(gpu,cells,kernel.H⁻¹)
            body = CuArray(Int32[0,0,1])
            if D == 2
                f = (;active=true,body,angle=CuArray(T[0.3]),turn=CuArray(T[0.2]))
            else
                q(a) = SVector{4,T}(cos(a/2),0,0,sin(a/2))
                f = (;active=true,body,orientation=CuArray([q(0.3)]),orientation_half=CuArray([q(0.5)]))
            end
            pose = CL.floating_mdbc_pose((;gpu,floating=f),half)
            step = HostStep(T(0.001),0.)
            evaluate!() = K.launch_mdbc!(gpu.Density,gpu.Position,gpu.GhostPoints,gpu.GhostIndex,gpu.Type,
                cells.CellStart,cells.grid_dev,step,kernel,constants;ghost_normals=gpu.GhostNormals,
                floating_pose=pose,compact_ghosts=Val(compact))
            evaluate!()
            ids = Array(gpu.Type)
            owner = only(findall(==(Floating),ids))
            slot = compact ? 1 : owner
            @test Array(gpu.GhostPoints)[slot] ≈ expected_ghost atol=2e-8
            @test Array(gpu.Density)[owner] ≈ rho(x) atol=(T === Float32 ? 0.05 : 1e-8)
            graph = CUDA.instantiate(CUDA.capture(evaluate!))
            CUDA.launch(graph)
            @test Array(gpu.GhostPoints)[slot] ≈ expected_ghost atol=2e-8
            K.launch_refresh_floating_ghosts!(gpu,CL.floating_mdbc_pose((;gpu,floating=f),false),step)
            accepted = x+V(ntuple(d -> d == 1 ? 0.04cos(0.3) : d == 2 ? 0.04sin(0.3) : 0.,D))
            @test Array(gpu.GhostPoints)[slot] ≈ accepted atol=2e-8
        end
    end
end
end
