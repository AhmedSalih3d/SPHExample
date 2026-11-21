module FloatingUpdatedMDBCTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const K = SPHExampleGPU.GPUKernels
const CL = SPHExampleGPU.SPHCellList

if CUDA.functional()
    @testset "floating cloned pressure and rotating no-slip surface" begin
        for D in (2,3), T in (Float32,Float64), compact in (false,true), half in (false,true), cached in (false,true)
            V = SVector{D,Float64}
            W = SVector{D,T}
            x = V(ntuple(_ -> 1., D))
            angle = half ? 0.5 : 0.3
            normal = W(ntuple(d -> d == 1 ? 0.04 : 0., D))
            offset = V(ntuple(d -> d == 1 ? 0.04cos(angle) : d == 2 ? 0.04sin(angle) : 0., D))
            ghost = x+offset
            center = x-V(ntuple(d -> d == 1 ? 0.1 : 0., D))
            spin(r) = D == 2 ? W(-2r[2], 2r[1]) : cross(W(0,0,2),W(r))
            velocity(p) = W(ntuple(_ -> 0.2, D))+spin(p-center)
            fluid = [ghost+V(ntuple(d -> 0.02index[d], D)) for index in CartesianIndices(ntuple(_ -> -2:2, D))][:]
            geometry = [SPHGeometry{D,T}(Particles=particle_struct_array([x],T(1000);
                    GhostPoints=[V(ntuple(_ -> -2., D))], GhostNormals=[normal], Velocity=[velocity(x)]),
                    Type=Floating,GroupMarker=3,Floating=FloatingDetails{T}(RelativeWeight=1)),
                SPHGeometry{D,T}(Particles=particle_struct_array(fluid,T(1000);Velocity=velocity.(fluid)),Type=Fluid,GroupMarker=2)]
            host = AllocateDataStructures(geometry;position_type=Float64)
            gpu = upload_particles(host;position_type=Float64,boundary_only=compact)
            c = SimulationConstants{T}(D;dx=0.02,c₀=25,g=0)
            kernel = SPHKernelInstance{D,T}(WendlandC2();h=T(0.08))
            cells = CellListWorkspace{D,Float64}(length(gpu))
            CL.rebuild_cell_list!(gpu,cells,kernel.H⁻¹)
            body = CuArray(Int32[0,0,1])
            pose = if D == 2
                (;groups=gpu.GroupMarker,body,angle=CuArray(T[0.3]),turn=half ? CuArray(T[0.2]) : nothing)
            else
                (;groups=gpu.GroupMarker,body,orientation=CuArray([SVector{4,T}(cos(angle/2),0,0,sin(angle/2))]))
            end
            state = (;pose,center=CuArray([center]),omega=CuArray(D == 2 ? T[2] : [W(0,0,2)]),
                acceleration=CuArray([W(ntuple(_ -> 0.3,D))]),
                angular_acceleration=CuArray(D == 2 ? T[0.4] : [W(0,0,0.4)]),pause=CuArray(T[0]),fraction=half ? 0.5 : 0.)
            support = CL.GPUSupportArrays{D,T}(length(gpu);position_type=Float64,updated_mdbc=true,
                cache_interaction_fields=cached,boundary_slots=gpu.BoundarySlot,
                boundary_count=compact ? length(gpu.GhostIndex) : length(gpu))
            step = HostStep(T(0.001),0.)
            K.launch_prepare_interactions!(gpu.Position,step;inverse_density=support.InvDensity,pressure=support.Pressure,
                Density=gpu.Density,ParticleType=gpu.Type,SimConstants=c,boundary_mode=UpdatedMDBC())
            evaluate!() = K.launch_updated_mdbc!(gpu.Density,gpu.Position,gpu.Velocity,
                gpu.GhostPoints,gpu.GhostNormals,gpu.GhostIndex,gpu.Type,cells.CellStart,cells.grid_dev,
                step,kernel,c,support.boundary_data;compact_ghosts=Val(compact),floating_state=state,
                inverse_density=support.InvDensity,pressure=support.Pressure)
            owner = only(findall(==(Floating),Array(gpu.Type)))
            slot = compact ? 1 : owner
            evaluate!()
            r = W(x-center)
            acceleration = W(ntuple(_ -> 0.3,D)) + (D == 2 ? W(-0.4r[2],0.4r[1]) : cross(W(0,0,0.4),r)) + spin(spin(r))
            n = W(offset/norm(offset))
            pressure = T(1000)*dot(-acceleration,n)*dot(-W(offset),n)
            @test Array(gpu.GhostPoints)[slot] ≈ ghost atol=2e-8
            @test Array(support.boundary_data.active)[slot]
            @test Array(support.boundary_data.velocity)[slot] ≈ velocity(x) atol=1e-5
            @test Array(support.boundary_data.tangent_velocity)[slot] ≈ velocity(x) atol=1e-5
            @test Array(gpu.Density)[owner] ≈ T(1000)+pressure/c.c₀^2 atol=(T === Float32 ? 0.003 : 1e-9)
            @test particle_pressure(Array(gpu.Density)[owner],Floating,c.c₀,c.ρ₀,UpdatedMDBC()) ≈ pressure atol=(T === Float32 ? 2. : 1e-6)
            K.launch_interactions!(support.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,support.∇Cᵢ,support.∇◌rᵢ,
                gpu.Position,gpu.Density,gpu.Velocity,gpu.Type,cells.CellStart,gpu.CellID,cells.grid_dev,step,
                ZeroDensityDiffusion(),Laminar(),kernel,c,Val(false),Val(false);boundary_data=support.boundary_data,
                inverse_density=support.InvDensity,pressure=support.Pressure)
            # Every fluid/body pair contributes equal and opposite momentum.
            acceleration_values = Array(gpu.Acceleration)
            @test norm(sum(acceleration_values)) <= (T === Float32 ? 1e-4 : 1e-10)*max(1,sum(norm,acceleration_values))
            graph = CUDA.instantiate(CUDA.capture(evaluate!))
            CUDA.launch(graph)
            @test Array(gpu.GhostPoints)[slot] ≈ ghost atol=2e-8
            # Holding the body removes translational, angular and centripetal acceleration.
            copyto!(state.pause,T[1])
            evaluate!()
            @test Array(gpu.Density)[owner] ≈ T(1000) atol=(T === Float32 ? 0.003 : 1e-9)
        end
    end

    @testset "predictor acceleration cache is opt-in" begin
        for (D, heave_only) in ((2,false), (3,false), (2,true))
            V = SVector{D,Float64}
            x = V(ntuple(_ -> 0.5,D))
            locks = heave_only ? FloatingConstraints{2,Float64}(TranslationLocks=(true,false), RotationLocks=(true,)) : nothing
            geometry = [SPHGeometry{D,Float64}([x];Density=1000.,Type=Floating,GroupMarker=1,
                Floating=FloatingDetails{Float64}(RelativeWeight=1,Center=Tuple(x),Constraints=locks,
                    Inertia=D == 2 ? 0.01 : (0.01,0.01,0.01)))]
            constants = SimulationConstants{Float64}(D;dx=0.02,g=0)
            host = AllocateDataStructures(geometry)
            floating = SPHExampleGPU.GPUFloating.FloatingArrays(geometry,host,constants;position_type=Float64)
            force = [0.001i for i in 1:(D == 2 ? 3 : 6)]
            for enabled in (false,true)
                fill!(floating.acceleration,zero(V))
                fill!(floating.angular_acceleration,zero(eltype(floating.angular_acceleration)))
                copyto!(floating.force,force)
                SPHExampleGPU.GPUFloating.launch_floating_update!(floating,HostStep(0.001,0.),0.,false;cache_derivative=enabled)
                expected = enabled ? V(force[1:D])/only(Array(floating.mass)) : zero(V)
                heave_only && (expected = V(0.0,expected[2]))
                @test only(Array(floating.acceleration)) ≈ expected
                alpha = only(Array(floating.angular_acceleration))
                @test enabled && !heave_only ? norm(alpha) > 0 : iszero(alpha)
            end
        end
    end

    @testset "floating UpdatedMDBC stages preserve uniform translation" begin
        for D in (2,3), scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            T = Float64
            V = SVector{D,T}
            x = V(ntuple(_ -> 0.5,D))
            normal = V(ntuple(d -> d == 1 ? 0.04 : 0.,D))
            cloud = [x+normal+V(ntuple(d -> 0.02index[d],D)) for index in CartesianIndices(ntuple(_ -> -2:2,D))][:]
            speed = V(ntuple(_ -> 0.2,D))
            states = []
            for graph in (false,true)
                geometry = [SPHGeometry{D,T}(Particles=particle_struct_array([x],T(1000);
                    GhostPoints=[x+normal],GhostNormals=[normal],Velocity=[speed]),Type=Floating,GroupMarker=1,
                    Floating=FloatingDetails{T}(RelativeWeight=1,Center=Tuple(x),InitialVelocity=Tuple(speed),
                        Inertia=D == 2 ? 0.01 : (0.01,0.01,0.01))),
                    SPHGeometry{D,T}(Particles=particle_struct_array(cloud,T(1000);Velocity=fill(speed,length(cloud))),Type=Fluid,GroupMarker=2)]
                save = mktempdir()
                meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,UpdatedMDBC,NoLog}(
                    SimulationName="FloatingUpdated",SaveLocation=save,SimulationTime=0.001,OutputTimes=0.0005,
                    VisualizeInParaview=false,OpenLogFile=false,GPUUseGraph=graph,GPUMaxStepsPerSync=graph ? 32 : 1,
                    GPUDoublePosition=true,GPUCacheInteractionFields=true,GPUClipTimeStepToOutput=true,
                    OutputVariables=["Type","GhostPoints","Pressure"])
                constants = SimulationConstants{T}(D;dx=0.02,c₀=25,g=0,ν₀=1e-6)
                kernel = SPHKernelInstance{D,T}(WendlandC2();h=0.08)
                particles = AllocateDataStructures(geometry,meta)
                logger = SimulationLogger(save)
                result = try
                    RunSimulation(SimGeometry=geometry,SimMetaData=meta,SimConstants=constants,SimKernel=kernel,
                        SimParticles=particles,SimLogger=logger,SimViscosity=Laminar(),SimDensityDiffusion=ZeroDensityDiffusion(),SimTimeStepping=scheme)
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                @test result.final_state.complete && result.diagnostics.frozen_particles == 0
                @test maximum(norm.(particles.Velocity .- Ref(speed))) < 1e-10
                owner = only(findall(==(Floating),particles.Type))
                @test particles.Position[owner] ≈ x+0.001speed atol=1e-10
                @test particles.GhostPoints[owner] ≈ particles.Position[owner]+normal atol=1e-10
                push!(states,particles[sortperm(particles.ID)])
            end
            @test states[1].Position == states[2].Position
            @test states[1].Density == states[2].Density
            @test states[1].Velocity == states[2].Velocity
        end
    end
end
end
