module FloatingInteractionMassTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra
const F = SPHExampleGPU.GPUFloating
const K = SPHExampleGPU.GPUKernels
const CL = SPHExampleGPU.SPHCellList
const S = SPHExampleGPU.GPUStepState

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "custom interaction masses preserve pair momentum and continuity" begin
        for D in (2,3),T in (Float32,Float64)
            V = SVector{D,T}
            vector(x) = V(ntuple(d -> d == 1 ? T(x) : zero(T),D))
            geometry = [SPHGeometry{D,T}(Particles = particle_struct_array([vector(0)],T(1001);Velocity = [vector(1)]),
                GroupMarker = 1,Type = Floating,Floating = FloatingDetails{T}(RelativeWeight = 2)),
                SPHGeometry{D,T}(Particles = particle_struct_array([vector(0.05)],T(1005);Velocity = [vector(-1)]),GroupMarker = 2,Type = Fluid)]
            host = AllocateDataStructures(geometry)
            constants = SimulationConstants{T}(D;dx = 0.05,g = 0,c₀ = 20,α = 0.02)
            kernel = SPHKernelInstance{D,T}(WendlandC2();h = T(0.1))
            gpu = upload_particles(host)
            cl = CellListWorkspace{D,T}(length(gpu))
            CL.rebuild_cell_list!(gpu,cl,kernel.H⁻¹)
            groups = Array(gpu.GroupMarker)
            mass = T[0.3constants.m₀,constants.m₀]
            function evaluate(custom)
                rate,acceleration = CUDA.zeros(T,2),CUDA.zeros(V,2)
                K.launch_interactions!(rate,acceleration,nothing,nothing,nothing,nothing,
                    gpu.Position,gpu.Density,gpu.Velocity,gpu.Type,cl.CellStart,gpu.CellID,cl.grid_dev,
                    S.HostStep(T(0.001),zero(T)),ZeroDensityDiffusion(),ArtificialViscosity(),kernel,constants,
                    Val(false),Val(false);native_rigid = Val(true),fastmath = false,
                    interaction_mass = custom ? (;values = CuArray(mass),groups = gpu.GroupMarker) : nothing)
                return Array(rate),Array(acceleration)
            end
            uniform,custom = evaluate(false),evaluate(true)
            body,fluid = findfirst(==(1),groups),findfirst(==(2),groups)
            @test custom[2][body] ≈ uniform[2][body]
            @test custom[2][fluid] ≈ T(0.3)*uniform[2][fluid] rtol = 5e-6
            @test custom[1][body] ≈ uniform[1][body]
            @test custom[1][fluid] ≈ T(0.3)*uniform[1][fluid] rtol = 5e-6
            force = sum(mass[groups[i]]*custom[2][i] for i in eachindex(groups))
            scale = maximum(norm,mass[groups].*custom[2])
            @test norm(force) < 1e-5scale
        end
    end

    @testset "body load reduction uses interaction mass without changing rigid mass" begin
        for D in (2,3),T in (Float32,Float64)
            points = [SVector{D,T}(ntuple(d -> T(0.05*((i >> (d-1)) & 1)),D)) for i in 0:2^D-1]
            constants = SimulationConstants{T}(D;dx = 0.05)
            properties = FloatingDetails{T}(RelativeWeight = 2,InteractionMass = 0.3constants.m₀)
            geometry = [SPHGeometry{D,T}(points;Density = 1000,GroupMarker = 7,Type = Floating,Floating = properties)]
            host = AllocateDataStructures(geometry)
            f = F.FloatingArrays(geometry,host,constants)
            gpu = upload_particles(host)
            acceleration = fill(SVector{D,T}(ntuple(_ -> T(2),D)),length(host))
            copyto!(gpu.Acceleration,acceleration)
            F.launch_floating_forces!(f,gpu.Acceleration,gpu.Position,gpu.Type,gpu.GroupMarker,f.center,constants.m₀,S.HostStep(T(0.001),zero(T)))
            @test Array(f.force)[1:D] ≈ 0.3constants.m₀*sum(acceleration)
            @test only(Array(f.mass)) ≈ 2constants.m₀*length(host)
            for value in (0,-1,Inf,NaN)
                invalid = [SPHGeometry{D,T}(points;Density = 1000,GroupMarker = 7,Type = Floating,
                    Floating = FloatingDetails{T}(RelativeWeight = 2,InteractionMass = value))]
                @test_throws ArgumentError F.FloatingArrays(invalid,host,constants)
            end
        end
    end
end
end
