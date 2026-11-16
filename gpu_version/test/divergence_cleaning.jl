module DivergenceCleaningTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const DC = SPHExampleGPU.GPUDivergenceCleaning
const DR = SPHExampleGPU.SPHCellList
const GK = SPHExampleGPU.GPUKernels
const STEP = SPHExampleGPU.GPUStepState

@testset "divergence cleaning parameters and decay sign" begin
    @test DivergenceCleaning().sigma == 1
    @test DC.local_speed(DivergenceCleaning(),-100.,1000.) ≈ sqrt(10.)
    @test DC.local_speed(DivergenceCleaning(pressure_mode=:positive),-100.,1000.) == 0
    @test_throws ArgumentError DivergenceCleaning(pressure_mode=:unknown)
    @test DC.local_speed(DivergenceCleaning(),100.,1000.) ≈ sqrt(10.)
    @test DC.CleaningWorkspace(nothing,10,Float32) === nothing
    for parameters in ((; kappa=0),(; kappa=NaN),(; sigma=-1),(; sigma=1.1),(; courant=0),(; courant=Inf))
        @test_throws ArgumentError DivergenceCleaning(; parameters...)
    end
    metadata = SimulationMetaData{2,Float64}(SimulationName="CleaningValidation",SaveLocation=tempdir())
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); dx=0.05)
    model = DivergenceCleaning()
    @test DC.validate_cleaning(model,metadata,kernel,SymplecticTimeStepping()) === nothing
    @test_throws ArgumentError DC.validate_cleaning(true,metadata,kernel,SymplecticTimeStepping())
    @test_throws ArgumentError DC.validate_cleaning(model,metadata,kernel,SingleNeighborTimeStepping())
    @test_throws ArgumentError DC.validate_cleaning(model,metadata,kernel,SymplecticTimeStepping(); floating=true)
    @test_throws ArgumentError DC.validate_cleaning(model,metadata,kernel,SymplecticTimeStepping(); open_boundary=true)
    metadata.TotalTime = 1
    @test_throws ArgumentError DC.validate_cleaning(model,metadata,kernel,SymplecticTimeStepping())
    metadata.TotalTime = 0
    metadata.Periodicity = PeriodicBoundary{2,Float64}(1; min=0,max=1)
    @test_throws ArgumentError DC.validate_cleaning(model,metadata,kernel,SymplecticTimeStepping())
    metadata.GPUPeriodicImages = false
    @test DC.validate_cleaning(model,metadata,kernel,SymplecticTimeStepping()) === nothing
end

function fixture(D,T; double_position = true,cache = false,fuse = false,background_pressure = 0)
    dx = T(0.05)
    positions = reverse(vec([SVector{D,T}(ntuple(d -> T(Tuple(index)[d])*dx,D))
        for index in CartesianIndices(ntuple(_ -> 1:4,D))]))
    wall(x) = SVector{D,T}(ntuple(d -> d == 1 ? T(x) : T(0.1),D))
    geometry = [SPHGeometry{D,T}(positions; Density=1000,GroupMarker=1,Type=Fluid),
        SPHGeometry{D,T}([wall(0.),wall(0.25)]; Density=1000,GroupMarker=2,Type=Fixed)]
    metadata = SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
        SimulationName="CleaningTest",SaveLocation=tempdir(),GPUDoublePosition=double_position,
        GPUCacheInteractionFields=cache,GPUInteractionFastMath=false)
    host = AllocateDataStructures(geometry,metadata)
    for i in eachindex(host)
        host.Density[i] = T(1000+(isodd(i) ? -1 : 1)*0.015i)
        host.Velocity[i] = SVector{D,T}(ntuple(d -> T(0.1sin(i+0.3d)),D))
    end
    host.Type[2] = Frozen
    host.Type[3] = Moving
    host.Type[4] = Floating
    constants = SimulationConstants{T}(D; dx,c₀=20,g=0,α=0,CFL=0.2,BackgroundPressure=background_pressure)
    kernel = SPHKernelInstance{D,T}(WendlandC2(); dx)
    gpu = upload_particles(host,metadata)
    support = GPUSupportArrays{D,T}(gpu,metadata; fuse_predictor=fuse)
    cl = CellListWorkspace{D,position_type(gpu)}(length(gpu))
    cleaning = DC.CleaningWorkspace(DivergenceCleaning(),length(gpu),T)
    seed = [host.Type[i] == Fluid ? T(0.3+0.1sin(i)) : zero(T) for i in eachindex(host)]
    copyto!(cleaning.psi,seed)
    DR.rebuild_cell_list!(gpu,cl,kernel.H⁻¹; support,kernel,divergence_cleaning=cleaning)
    return (; geometry,metadata,host,constants,kernel,gpu,support,cl,cleaning,seed)
end

function dense_reference(f)
    (; gpu,kernel,constants,cleaning) = f
    x,v,rho,types,psi = Array(gpu.Position),Array(gpu.Velocity),Array(gpu.Density),Array(gpu.Type),Array(cleaning.psi)
    acceleration = zero.(v)
    rate = zero.(rho)
    for i in eachindex(x)
        types[i] == Fluid || continue
        divergence = zero(eltype(rho))
        for j in eachindex(x)
            (i == j || types[j] == Frozen) && continue
            separation = eltype(v)(x[i]-x[j])
            r2 = dot(separation,separation)
            r2 <= kernel.H² || continue
            gradient = ∇Wᵢⱼ(kernel,sqrt(r2)*kernel.h⁻¹,separation)
            volume = constants.m₀/rho[j]
            divergence -= volume*dot(v[i]-v[j],gradient)
            psi_j = types[j] == Fluid ? psi[j] : zero(eltype(rho))
            acceleration[i] += (psi[i]-psi_j)*volume*gradient
        end
        p = EquationOfStateGamma7(rho[i],constants.c₀,constants.ρ₀,inv(constants.ρ₀),constants.BackgroundPressure)
        decay_pressure = cleaning.model.absolute_pressure ? abs(p) : max(p,zero(p))
        decay = cleaning.model.sigma*cleaning.model.kappa*sqrt(decay_pressure/rho[i])/kernel.h
        rate[i] = -constants.c₀^2*divergence-decay*psi[i]
    end
    return acceleration,rate
end

function context(f,lanes)
    (; gpu,support,cl,kernel,constants,cleaning) = f
    return (; gpu,sup=support,cl,SimKernel=kernel,SimConstants=constants,SimDensityDiffusion=ZeroGravityLinearDensityDiffusion(),
        SimViscosity=ZeroViscosity(),FlagKernel=Val(false),FlagShift=Val(false),threads=128,lanes=Val(lanes),
        bforces=Val(true),divergence_cleaning=cleaning,interaction_fastmath=false,frozen_particles=Val(true))
end

if CUDA.functional()
    @testset "cleaning differences match independent dense operators" begin
        for D in (2,3), T in (Float32,Float64), mixed in (false,true), cache in (false,true), background_pressure in (0,101325)
            f = fixture(D,T; double_position=mixed,cache,background_pressure)
            (; gpu,support,cl,kernel,constants,cleaning) = f
            @test Array(cleaning.psi) == f.seed[Array(gpu.ID)]
            expected_acceleration,expected_rate = dense_reference(f)
            fields = DC.cleaning_fields(cleaning)
            for lanes in (1,4)
                ctx = context(f,lanes)
                step = HostStep(zero(T),0.)
                launch_prepare_interactions!(gpu.Position,step; inverse_density=support.InvDensity,
                    pressure=support.Pressure,Density=gpu.Density,ParticleType=gpu.Type,SimConstants=constants,boundary_mode=NoMDBC())
                DR.launch_solver_interactions!(merge(ctx,(; divergence_cleaning=nothing)),gpu.Position,gpu.Density,gpu.Velocity,support.PosCells,step)
                original_acceleration,original_density_rate = Array(gpu.Acceleration),Array(support.dρdtI)
                DR.launch_solver_interactions!(ctx,gpu.Position,gpu.Density,gpu.Velocity,support.PosCells,step)
                tolerance = T === Float32 ? 2e-4 : 2e-11
                # A large offset makes this subtraction cancellation-limited in Float32.
                acceleration_atol = max(tolerance,8eps(T)*maximum(norm,original_acceleration))
                @test Array(gpu.Acceleration)-original_acceleration ≈ expected_acceleration rtol=tolerance atol=acceleration_atol
                @test Array(cleaning.derivative) ≈ expected_rate rtol=tolerance atol=tolerance
                @test Array(support.dρdtI) ≈ original_density_rate rtol=tolerance atol=tolerance
            end
            # Force another permutation while psi must continue following IDs.
            positions = Array(gpu.Position)
            positions .= reverse(positions)
            copyto!(gpu.Position,positions)
            DR.rebuild_cell_list!(gpu,cl,kernel.H⁻¹; support,kernel,divergence_cleaning=cleaning)
            @test Array(cleaning.psi) == f.seed[Array(gpu.ID)]
        end
    end

    @testset "scalar integration, wall protection and device cleaning CFL" begin
        for T in (Float32,Float64), background_pressure in (0,101325)
            f = fixture(2,T; background_pressure)
            (; gpu,support,cl,kernel,constants,cleaning) = f
            n = length(gpu)
            fill!(gpu.Density,T(1020))
            fill!(gpu.Velocity,zero(SVector{2,T}))
            fill!(gpu.Acceleration,zero(SVector{2,T}))
            fill!(support.dρdtI,zero(T))
            types = Array(gpu.Type)
            psi = [type == Fluid ? T(2) : zero(T) for type in types]
            copyto!(cleaning.psi,psi)
            p = EquationOfStateGamma7(T(1020),constants.c₀,constants.ρ₀,inv(constants.ρ₀),constants.BackgroundPressure)
            decay = cleaning.model.sigma*DC.local_speed(cleaning.model,p,T(1020))/kernel.h
            dt = T(0.01)/decay
            copyto!(cleaning.derivative,-decay .* psi)
            position_half,velocity_half = DR.predictor_arrays(support,gpu)
            motion = MotionArrays(f.geometry,f.host)
            step = HostStep(dt,0.)
            launch_half_step!(position_half,velocity_half,support.ρₙ⁺,gpu.Position,gpu.Velocity,gpu.Acceleration,gpu.Density,
                support.dρdtI,gpu.Type,gpu.GroupMarker,motion,step,constants; SimKernel=kernel,
                pos_cells=support.PosCellsₙ⁺,CellID=gpu.CellID,grid=cl.grid_dev,reuse_predictors=support.ReusePredictors,
                boundary_mode=NoMDBC(),cleaning=DC.cleaning_fields(cleaning))
            half = psi .* (1-dt*decay/2)
            @test Array(cleaning.half) ≈ half
            copyto!(cleaning.derivative,-decay .* half)
            red = ReductionWorkspace{SVector{3,T}}(n)
            launch_final_step!(gpu.Position,gpu.Velocity,gpu.Acceleration,gpu.Density,support.dρdtI,support.ρₙ⁺,
                position_half,velocity_half,gpu.Type,support.∇Cᵢ,support.∇◌rᵢ,step,kernel,constants,red,Val(false);
                boundary_mode=NoMDBC(),cleaning=DC.cleaning_fields(cleaning))
            expected = psi .* (1-dt*decay+(dt*decay)^2/2)
            @test Array(cleaning.psi) ≈ expected
            @test maximum(Array(cleaning.psi)) < maximum(psi)
            state = StepState{T}()
            STEP.set_output_time!(state,1.; clip_timestep=true)
            GK.launch_step_reduction!(red,gpu.Position,gpu.Velocity,gpu.Acceleration,gpu.Position,kernel;
                cleaning_state=cleaning,density=gpu.Density,types=gpu.Type,constants)
            launch_finish!(state,red,kernel,constants; cleaning=DC.cleaning_fields(cleaning))
            readback!(state)
            expected_dt = cleaning.model.courant*kernel.h/max(constants.c₀,DC.local_speed(cleaning.model,p,T(1020)))
            @test state.fh[STEP.F_DT] ≈ expected_dt rtol=1e-6
        end
    end

    @testset "fused prediction uses the same psi half state" begin
        for T in (Float32,Float64)
            f = fixture(2,T; cache=true,fuse=true)
            (; gpu,support,cl,kernel,constants,cleaning) = f
            ctx = context(f,4)
            step = HostStep(T(1e-5),0.)
            launch_prepare_interactions!(gpu.Position,step; inverse_density=support.InvDensity,pressure=support.Pressure,
                Density=gpu.Density,ParticleType=gpu.Type,SimConstants=constants,boundary_mode=NoMDBC())
            DR.launch_solver_interactions!(ctx,gpu.Position,gpu.Density,gpu.Velocity,support.PosCells,step;
                predictor_output=DR.predictor_outputs(ctx))
            position_half,velocity_half = DR.predictor_arrays(support,gpu)
            fused = (Array(position_half),Array(velocity_half),Array(support.ρₙ⁺),Array(cleaning.half))
            launch_half_step!(position_half,velocity_half,support.ρₙ⁺,gpu.Position,gpu.Velocity,gpu.Acceleration,gpu.Density,
                support.dρdtI,gpu.Type,gpu.GroupMarker,MotionArrays(f.geometry,f.host),step,constants;
                SimKernel=kernel,pos_cells=support.PosCellsₙ⁺,CellID=gpu.CellID,grid=cl.grid_dev,
                reuse_predictors=support.ReusePredictors,boundary_mode=NoMDBC(),inverse_density=support.InvDensityₙ⁺,
                pressure=support.Pressureₙ⁺,cleaning=DC.cleaning_fields(cleaning))
            separate = (Array(position_half),Array(velocity_half),Array(support.ρₙ⁺),Array(cleaning.half))
            @test all(a ≈ b for (a,b) in zip(fused,separate))
        end
    end

    @testset "coupled cleaning follows a damped discrete Fourier wave" begin
        # Isolate the cleaning subsystem on fixed periodic positions. Its
        # discrete SPH derivative has a computable Fourier symbol beta, so
        # velocity and psi have an analytic damped-oscillator solution.
        T = Float64
        n,dx,c0,rho = 64,1/64,10.,1001.
        points = [SVector((i+0.5)*dx,0.5) for i in 0:n-1]
        amplitude,k = 0.01,2π
        velocity = [SVector(amplitude*sin(k*x[1]),0.) for x in points]
        geometry = [SPHGeometry{2,T}(Particles=particle_struct_array(points,rho; Velocity=velocity),GroupMarker=1,Type=Fluid)]
        metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
            SimulationName="CleaningWave",SaveLocation=tempdir(),GPUPeriodicImages=false,
            Periodicity=PeriodicBoundary{2,T}(1; min=0,max=1))
        host = AllocateDataStructures(geometry,metadata)
        gpu = upload_particles(host,metadata)
        sup = GPUSupportArrays{2,T}(n; periodic=true,cache_interaction_fields=true,shifting=false)
        cl = CellListWorkspace{2,T}(n; periodicity=metadata.Periodicity)
        kernel = SPHKernelInstance{2,T}(WendlandC2(); dx)
        constants = SimulationConstants{T}(2; dx,c₀=c0,g=0,α=0)
        scalar = DC.CleaningWorkspace(DivergenceCleaning(sigma=0.05),n,T)
        DR.rebuild_cell_list!(gpu,cl,kernel.H⁻¹; support=sup,kernel)
        fill!(sup.InvDensity,1/rho)
        fill!(sup.Pressure,0.) # Remove physical pressure forces in this subsystem test.
        ctx = (; gpu,sup,cl,SimKernel=kernel,SimConstants=constants,SimDensityDiffusion=ZeroDensityDiffusion(),
            SimViscosity=ZeroViscosity(),FlagKernel=Val(false),FlagShift=Val(false),threads=128,lanes=Val(1),
            bforces=Val(false),divergence_cleaning=scalar,interaction_fastmath=false)
        dt,steps = 0.0001,600
        step = HostStep(dt,0.)
        DR.prepare_periodic_bounds!(ctx,gpu.Position,step)
        half_velocity = similar(gpu.Velocity)
        for _ in 1:steps
            DR.launch_solver_interactions!(ctx,gpu.Position,gpu.Density,gpu.Velocity,nothing,step)
            half_velocity .= gpu.Velocity .+ (dt/2) .* gpu.Acceleration
            scalar.half .= scalar.psi .+ (dt/2) .* scalar.derivative
            DR.launch_solver_interactions!(ctx,gpu.Position,gpu.Density,half_velocity,nothing,step; cleaning_half=true)
            gpu.Velocity .+= dt .* gpu.Acceleration
            scalar.psi .+= dt .* scalar.derivative
        end
        beta = 0.
        origin = points[1]
        for image in -1:1, point in points
            separation = origin-point-SVector(Float64(image),0.)
            r2 = dot(separation,separation)
            if 0 < r2 <= kernel.H²
                gradient = ∇Wᵢⱼ(kernel,sqrt(r2)*kernel.h⁻¹,separation)
                beta += sin(-k*separation[1])*gradient[1]*(constants.m₀/rho)
            end
        end
        @test beta > 0
        pressure = EquationOfStateGamma7(rho,c0,constants.ρ₀)
        decay = scalar.model.sigma*scalar.model.kappa*sqrt(pressure/rho)/kernel.h
        frequency = sqrt((c0*beta)^2-(decay/2)^2)
        time = steps*dt
        factor = exp(-decay*time/2)
        u = amplitude*factor*(cos(frequency*time)+decay/(2frequency)*sin(frequency*time))
        psi = -amplitude*c0^2*beta/frequency*factor*sin(frequency*time)
        positions = Array(gpu.Position)
        expected_u = [SVector(u*sin(k*x[1]),0.) for x in positions]
        expected_psi = [psi*cos(k*x[1]) for x in positions]
        @test norm(Array(gpu.Velocity)-expected_u)/norm(expected_u) < 5e-4
        @test norm(Array(scalar.psi)-expected_psi)/norm(expected_psi) < 5e-4
    end

    include(joinpath(@__DIR__,"..","validation cases","spheric_test_06","case.jl"))
    function smoke_run(directory; graph=true,cleaning=DivergenceCleaning(),fuse=false)
        case = SphericTest06.build_case(; dx=0.1,duration=0.12,interval=0.04,save=directory,graph,cfl=0.1)
        case.metadata.GPUInteractionFastMath = fuse
        case.metadata.GPUFusePredictor = fuse
        host = AllocateDataStructures(case.geometry,case.metadata)
        logger = SimulationLogger(directory)
        result = open(joinpath(directory,"console.log"),"w") do io
            redirect_stdout(io) do
                try
                    RunSimulation(SimGeometry=case.geometry,SimMetaData=case.metadata,SimConstants=case.constants,
                        SimKernel=case.kernel,SimLogger=logger,SimParticles=host,SimViscosity=Laminar(),
                        SimDensityDiffusion=ZeroGravityLinearDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping(),
                        SimMotion=case.motion,SimDivergenceCleaning=cleaning)
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
            end
        end
        return result
    end

    @testset "moving-square graph replay and optional-mode output" begin
        mktempdir() do directory
            direct = smoke_run(mkpath(joinpath(directory,"direct")); graph=false)
            graph = smoke_run(mkpath(joinpath(directory,"graph")); graph=true)
            off = smoke_run(mkpath(joinpath(directory,"off")); cleaning=nothing)
            @test direct.final_state.complete && graph.final_state.complete
            @test graph.diagnostics.output_frames == direct.diagnostics.output_frames == 4
            @test graph.configuration.execution.graph_batches > 0
            @test graph.configuration.execution.graph_capture_failures == 0
            @test graph.performance.steps == direct.performance.steps
            a,b = direct.final_state.particles,graph.final_state.particles
            ia,ib = sortperm(a.ID),sortperm(b.ID)
            @test a.Position[ia] == b.Position[ib]
            @test a.Velocity[ia] == b.Velocity[ib]
            @test a.Density[ia] == b.Density[ib]
            @test direct.final_state.divergence_cleaning.psi[ia] == graph.final_state.divergence_cleaning.psi[ib]
            psi = graph.final_state.divergence_cleaning.psi
            @test all(isfinite,psi)
            @test maximum(abs,psi) > 0
            @test all(==(0),psi[b.Type .!= Fluid])
            @test !hasproperty(off.final_state,:divergence_cleaning)
            @test !hasproperty(off.configuration.effective,:divergence_cleaning)
            @test graph.configuration.effective.divergence_cleaning.scalar_integration == :explicit_midpoint
        end
    end

    @testset "cleaning with stationary mDBC walls in 2D and 3D" begin
        for D in (2,3), T in (Float32,Float64), B in (SimpleMDBC,UpdatedMDBC)
            mktempdir() do directory
                V = SVector{D,T}
                wall = V(ntuple(d -> d == D ? 0 : 0.5,D))
                normal = V(ntuple(d -> d == D ? 0.1 : 0,D))
                fluid = vec([wall+normal+V(ntuple(d -> T(Tuple(i)[d])*T(0.04),D))
                    for i in CartesianIndices(ntuple(_ -> -2:2,D))])
                geometry = [SPHGeometry{D,T}(Particles=particle_struct_array([wall],T(1000);
                    GhostPoints=[wall+normal],GhostNormals=[normal]),GroupMarker=1,Type=Fixed),
                    SPHGeometry{D,T}(Particles=particle_struct_array(fluid,T(1000);
                        Velocity=fill(V(ntuple(_ -> T(0.1),D)),length(fluid))),GroupMarker=2,Type=Fluid)]
                metadata = SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,NoLog}(
                    SimulationName="CleaningWall",SaveLocation=directory,SimulationTime=0.0002,
                    OutputTimes=[0.0002],VisualizeInParaview=false,OpenLogFile=false,GPUUseGraph=true)
                constants = SimulationConstants{T}(D; dx=0.04,c₀=20,g=0,α=0)
                kernel = SPHKernelInstance{D,T}(WendlandC2(); dx=constants.dx)
                host = AllocateDataStructures(geometry,metadata)
                logger = SimulationLogger(directory)
                result = open(joinpath(directory,"console.log"),"w") do io
                    redirect_stdout(io) do
                        try
                            RunSimulation(SimGeometry=geometry,SimMetaData=metadata,SimConstants=constants,
                                SimKernel=kernel,SimLogger=logger,SimParticles=host,SimViscosity=ZeroViscosity(),
                                SimDensityDiffusion=ZeroDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping(),
                                SimDivergenceCleaning=DivergenceCleaning())
                        finally
                            isopen(logger.LoggerIo) && close(logger.LoggerIo)
                        end
                    end
                end
                @test result.final_state.complete && result.diagnostics.output_frames == 2
                @test all(isfinite,result.final_state.divergence_cleaning.psi)
                @test all(==(0),result.final_state.divergence_cleaning.psi[host.Type .!= Fluid])
            end
        end
    end

    @testset "cleaning follows translated periodic crossings" begin
        mktempdir() do directory
            points = [SVector((i+0.5)*0.05,(j+0.5)*0.05) for j in 0:5 for i in 0:19]
            velocity = [SVector{2,Float32}(2+0.01sin(2π*x[1]),0) for x in points]
            geometry = [SPHGeometry{2,Float32}(Particles=particle_struct_array(points,1000f0; Velocity=velocity),
                GroupMarker=1,Type=Fluid)]
            metadata = SimulationMetaData{2,Float32,NoShifting,NoKernelOutput,NoMDBC,NoLog}(
                SimulationName="CleaningPeriodic",SaveLocation=directory,SimulationTime=0.02,OutputTimes=[0.02],
                VisualizeInParaview=false,OpenLogFile=false,GPUDoublePosition=true,GPUUseGraph=true,
                GPUClipTimeStepToOutput=true,GPUPeriodicImages=false,Periodicity=PeriodicBoundary{2,Float64}(1; min=0,max=1))
            constants = SimulationConstants{Float32}(2; dx=0.05,c₀=20,g=0,α=0)
            kernel = SPHKernelInstance{2,Float32}(WendlandC2(); dx=constants.dx)
            host = AllocateDataStructures(geometry,metadata)
            logger = SimulationLogger(directory)
            result = open(joinpath(directory,"console.log"),"w") do io
                redirect_stdout(io) do
                    try
                        RunSimulation(SimGeometry=geometry,SimMetaData=metadata,SimConstants=constants,SimKernel=kernel,
                            SimLogger=logger,SimParticles=host,SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion(),
                            SimTimeStepping=SymplecticTimeStepping(),SimDivergenceCleaning=DivergenceCleaning())
                    finally
                        isopen(logger.LoggerIo) && close(logger.LoggerIo)
                    end
                end
            end
            @test result.final_state.complete
            @test all(x -> 0 <= x[1] < 1,host.Position)
            @test maximum(first.(host.Position)) < 0.99
            @test all(isfinite,result.final_state.divergence_cleaning.psi)
            @test maximum(abs,result.final_state.divergence_cleaning.psi) > 0
        end
    end
end

end
