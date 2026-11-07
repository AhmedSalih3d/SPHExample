module IISPHOperatorTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const DRIVER = SPHExampleGPU.SPHCellList
const OPS = SPHExampleGPU.GPUIISPH
const Workspace = SPHExampleGPU.IISPHSupport.IISPHWorkspace

@testset "IISPH configuration" begin
    @test WCSPH() isa SPHSolver
    @test IISPH() isa SPHSolver
    for kwargs in ((;relative_density_tolerance=0),(;relative_density_tolerance=NaN),
                   (;min_iterations=0),(;max_iterations=1),(;relaxation=0),(;cfl=Inf),
                   (;max_timestep=0),(;warm_start=1.1))
        @test_throws ArgumentError IISPH(;kwargs...)
    end
    meta = SimulationMetaData{2,Float32}(SimulationName="IISPHConfig",SaveLocation=tempdir())
    kernel = SPHKernelInstance{2,Float32}(WendlandC2();dx=0.02f0)
    config = effective_configuration(meta,kernel,ZeroDensityDiffusion(),Laminar();particle_count=0,solver=IISPH())
    @test config.effective.solver === :IISPH
    @test config.effective.pressure_source === :solved
    @test config.effective.time_stepping === :symplectic_euler
    @test !config.effective.use_graph && !config.effective.async_output
    @test !config.effective.interaction_fastmath
    @test any(f -> f.option === :GPUUseGraph && f.reason === :iisph_foundation_backend,config.fallbacks)
    criteria = SVector(2f0,4f0,0f0,0f0)
    slow_sound = SimulationConstants{Float32}(2;dx=0.02f0,c₀=1f0)
    fast_sound = SimulationConstants{Float32}(2;dx=0.02f0,c₀=10000f0)
    dt1 = DRIVER.iisph_timestep(criteria,kernel,slow_sound,IISPH(),1.0,ZeroViscosity())
    dt2 = DRIVER.iisph_timestep(criteria,kernel,fast_sound,IISPH(),1.0,ZeroViscosity())
    @test dt1 === dt2
    @test dt1 ≈ 0.25kernel.h/2
    @test DRIVER.iisph_timestep(criteria,kernel,slow_sound,IISPH(max_timestep=0.001),1.0,ZeroViscosity()) ≈ 0.001
    viscous = SimulationConstants{Float32}(2;dx=0.02f0,ν₀=1f0)
    @test DRIVER.iisph_timestep(criteria,kernel,viscous,IISPH(),1.0,Laminar()) ≈ 0.125kernel.h^2
    @test_throws ErrorException DRIVER.iisph_timestep(SVector(0f0,0f0,1f0,0f0),kernel,viscous,IISPH(),1.0,Laminar())
end

function operator_test(D,T;compact=true,double_position=false)
    constants = SimulationConstants{T}(D;dx=T(0.08),g=0)
    kernel = SPHKernelInstance{D,T}(WendlandC2();dx=constants.dx)
    # Deliberately irregular particles exercise an asymmetric neighbour stencil.
    positions = [SVector{D,T}(ntuple(d -> T(0.08)*T(((i >> (d-1)) & 1)+0.05sin(i+d)),D)) for i in 0:(2^D-1)]
    wall = [positions[1]-SVector{D,T}(ntuple(d -> d == D ? T(0.07) : zero(T),D))]
    geometry = [SPHGeometry{D,T}(positions;Density=constants.ρ₀,Type=Fluid,GroupMarker=2),
                SPHGeometry{D,T}(wall;Density=constants.ρ₀,Type=Fixed,GroupMarker=1)]
    meta = SimulationMetaData{D,T}(SimulationName="IISPHOperator",SaveLocation=tempdir(),
        GPUCompactBuffers=compact,GPUDoublePosition=double_position,VisualizeInParaview=false)
    particles = AllocateDataStructures(geometry,meta)
    gpu = upload_particles(particles;position_type=position_float_type(meta),ghost_data=false,
        kernel_output=false,reconstruct_fields=compact)
    ws = Workspace{D,T}(length(gpu))
    cl = CellListWorkspace{D,position_type(gpu)}(length(gpu))
    DRIVER.rebuild_iisph!(gpu,cl,ws,kernel,constants)
    x,types = Array(gpu.Position),Array(gpu.Type)
    n = length(x)
    fluid = findall(==(Fluid),types)
    # Independent all-pairs host construction of density and the acceleration map B.
    grad = [SVector{D,T}(i == j || norm(x[i]-x[j]) > kernel.H ? zero(SVector{D,T}) :
        ∇Wᵢⱼ(kernel,T(norm(x[i]-x[j]))*kernel.h⁻¹,SVector{D,T}(x[i]-x[j]))) for i in 1:n,j in 1:n]
    wall_mass = [types[j] == Fixed ? constants.m₀ : zero(T) for j in 1:n]
    mass = [types[j] == Fluid ? constants.m₀ : wall_mass[j] for j in 1:n]
    rho = [types[i] == Fluid ? sum(mass[j]*Wᵢⱼ(kernel,T(norm(x[i]-x[j]))*kernel.h⁻¹)
        for j in 1:n if norm(x[i]-x[j]) <= kernel.H) : constants.ρ₀ for i in 1:n]
    tol = T === Float32 ? 4e-5 : 2e-12
    @test Array(ws.boundary_mass) ≈ wall_mass rtol=tol
    @test Array(gpu.Density) ≈ rho rtol=tol
    B = zeros(T,D*n,n)
    for i in fluid,j in 1:n
        B[D*(i-1)+1:D*i,i] .-= mass[j]*grad[i,j]/rho[i]^2
        if types[j] == Fluid
            B[D*(i-1)+1:D*i,j] .-= mass[j]*grad[i,j]/rho[j]^2
        end
    end
    # Independently compose continuity C with B. This detects missing wall terms
    # and a wrong neighbour response in the analytic Jacobi diagonal.
    C = zeros(T,n,D*n)
    for i in fluid,j in 1:n
        C[i,D*(i-1)+1:D*i] .+= mass[j]*grad[i,j]
        C[i,D*(j-1)+1:D*j] .-= mass[j]*grad[i,j]
    end
    A = C*B
    velocity = [SVector{D,T}(ntuple(d -> T(0.01sin(i+d)),D)) for i in 1:n]
    velocity[findall(==(Fixed),types)] .= Ref(zero(SVector{D,T}))
    copyto!(gpu.Velocity,velocity)
    fill!(ws.nonpressure_acceleration,zero(SVector{D,T}))
    dt = T(0.003)
    solver = IISPH(warm_start=1)
    OPS.launch_predict!(ws,gpu,dt,solver)
    OPS.launch_source!(ws,gpu,cl,kernel,constants,dt)
    @test Array(ws.diagonal) ≈ diag(A) rtol=tol atol=tol
    @test Array(ws.density_adv) ≈ rho+dt*C*reduce(vcat,velocity) rtol=tol
    p = [types[i] == Fluid ? T(10+i) : zero(T) for i in 1:n]
    copyto!(ws.pressure,p)
    OPS.launch_pressure_acceleration!(ws,gpu,cl,kernel,constants)
    @test reduce(vcat,Array(ws.pressure_acceleration)) ≈ B*p rtol=tol atol=tol
    OPS.launch_pressure_update!(ws,gpu,cl,kernel,constants,dt,solver)
    residual = Array(ws.density_adv)+dt^2*A*p.-constants.ρ₀
    expected = [types[i] == Fluid ? abs(residual[i])/constants.ρ₀ : zero(T) for i in 1:n]
    @test Array(ws.residual) ≈ expected rtol=tol atol=tol
    # Manufacture a positive-pressure solution, then compare the converged
    # GPU solve to a direct dense solve rather than another Jacobi loop.
    # Keep the manufactured density signal well above Float32 subtraction
    # roundoff, so a density-residual tolerance also resolves pressure values.
    solve_dt = T(0.1)
    manufactured = zeros(T,n)
    manufactured[fluid] .= T.(100 .+ 10 .* (1:length(fluid)))
    source = fill(constants.ρ₀,n)-solve_dt^2*A*manufactured
    copyto!(ws.density_adv,source)
    fill!(ws.pressure,zero(T))
    solve_options = IISPH(relative_density_tolerance=T === Float32 ? 1e-6 : 1e-10,
        max_iterations=5000)
    info = DRIVER.solve_iisph_pressure!(ws,gpu,cl,kernel,constants,solve_dt,solve_options)
    direct = A[fluid,fluid] \ ((fill(constants.ρ₀,length(fluid))-source[fluid])/solve_dt^2)
    @test info.relative_density_residual <= solve_options.relative_density_tolerance
    @test Array(ws.pressure)[fluid] ≈ direct rtol=T === Float32 ? 0.002 : 1e-6
    # A reorder must preserve accepted pressure by stable input identity.
    copyto!(ws.pressure,p)
    pressure_by_id = Dict(zip(Array(gpu.ID),p))
    copyto!(gpu.Position,reverse(x))
    DRIVER.rebuild_iisph!(gpu,cl,ws,kernel,constants)
    @test Array(ws.pressure) == [pressure_by_id[id] for id in Array(gpu.ID)]
end

if CUDA.functional()
    @testset "IISPH operator against independent dense CPU matrices" begin
        for D in (2,3),T in (Float32,Float64)
            operator_test(D,T;compact=T === Float32,double_position=T === Float32)
        end
    end
else
    @testset "IISPH operator requires CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
