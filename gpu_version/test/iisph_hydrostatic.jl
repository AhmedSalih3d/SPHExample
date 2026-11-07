module IISPHHydrostaticTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, Statistics, HDF5
include(joinpath(@__DIR__,"..","example","IISPH","cases.jl"))
using .IISPHCases
const DRIVER = SPHExampleGPU.SPHCellList

function run_case(case;kwargs...)
    try
        return RunSimulation(;case...,kwargs...)
    finally
        isopen(case.SimLogger.LoggerIo) && close(case.SimLogger.LoggerIo)
    end
end

@testset "IISPH rejects unsupported combinations before allocation" begin
    mktempdir() do save
        case = tank_case(Float32,save;nx=4,ny=4,duration=0.01)
        m,g,p,c,k = case.SimMetaData,case.SimGeometry,case.SimParticles,case.SimConstants,case.SimKernel
        validate(v,d,s;kwargs...) = DRIVER.validate_iisph(m,g,p,c,k,v,d,s;kwargs...)
        try
            @test validate(Laminar(),ZeroDensityDiffusion(),nothing) === nothing
            @test_throws ArgumentError validate(ArtificialViscosity(),ZeroDensityDiffusion(),nothing)
            @test_throws ArgumentError validate(Laminar(),LinearDensityDiffusion(),nothing)
            @test_throws ArgumentError validate(Laminar(),ZeroDensityDiffusion(),SymplecticTimeStepping())
            @test_throws ArgumentError validate(Laminar(),ZeroDensityDiffusion(),nothing;motion=:unsupported)
            m.GPUCacheInteractionFields = true
            @test_throws ArgumentError validate(Laminar(),ZeroDensityDiffusion(),nothing)
            m.GPUCacheInteractionFields = false
            m.GPUCellSubdivision = 2
            @test_throws ArgumentError validate(Laminar(),ZeroDensityDiffusion(),nothing)
            m.GPUCellSubdivision = 1
            m.TotalTime = 0.01
            @test_throws ArgumentError validate(Laminar(),ZeroDensityDiffusion(),nothing)
            m.TotalTime = 0
            fixed = findfirst(==(Fixed),p.Type)
            p.Velocity[fixed] = SVector(1f0,0f0)
            @test_throws ArgumentError validate(Laminar(),ZeroDensityDiffusion(),nothing)
        finally
            close(case.SimLogger.LoggerIo)
        end
    end
end

if CUDA.functional()
    @testset "IISPH hydrostatic support, pressure output and fixed walls" begin
        for (T,numbered,compact,double_position) in ((Float32,false,true,true),(Float64,true,false,false))
            mktempdir() do save
                # Allow the initially unpressurized column to settle. Viscosity
                # damps the startup transient without changing hydrostatic balance.
                case = tank_case(T,save;duration=1.0,output_interval=0.5,nu=0.01)
                meta = case.SimMetaData
                meta.ExportSingleVTKHDF = !numbered
                meta.GPUCompactBuffers = compact
                meta.GPUDoublePosition = double_position
                # Request deferred optimizations to check that the result states
                # what actually ran. The numerical path must still be direct.
                meta.GPUUseGraph = true
                meta.GPUAsyncOutput = true
                particles = AllocateDataStructures(case.SimGeometry,meta)
                case = merge(case,(;SimParticles=particles))
                initial_walls = Dict(particles.ID[i] => particles.Position[i] for i in eachindex(particles.Type) if particles.Type[i] == Fixed)
                result = run_case(case)
                p = result.final_state.particles
                fluid = p.Type .== Fluid
                @test result.termination_reason === :completed
                @test result.final_state.time == 1.0
                @test p.Position === particles.Position
                @test hasproperty(p,:Pressure)
                @test all(isfinite,p.Pressure) && all(>=(0),p.Pressure)
                @test result.configuration.effective.solver === :IISPH
                @test !result.configuration.effective.use_graph && !result.configuration.effective.async_output
                @test result.configuration.execution.path === :direct
                @test result.diagnostics.iisph.max_relative_density_residual <= case.SimSolver.relative_density_tolerance
                @test result.diagnostics.iisph.total_pressure_iterations >= 2result.performance.steps
                @test result.diagnostics.cell_list_rebuilds > result.performance.steps
                @test maximum(p.Density[fluid]) < 1.01case.SimConstants.ρ₀
                @test maximum(norm,p.Velocity[fluid]) < 0.1
                for i in eachindex(p.Type)
                    if p.Type[i] == Fixed
                        @test p.Position[i] == initial_walls[p.ID[i]]
                        @test iszero(p.Velocity[i]) && iszero(p.Pressure[i])
                    end
                end
                interior = findall(i -> p.Type[i] == Fluid && 0.1 < p.Position[i][1] < 0.5 && p.Position[i][2] < 0.25,eachindex(p.Type))
                @test !isempty(interior)
                # Compare physical probe values to the analytical hydrostatic
                # profile. Nodal pressure contains high-frequency modes of the
                # collocated SPH operator, even at tighter iteration tolerances.
                for y in (0.1,0.2,0.3)
                    ids = findall(i -> p.Type[i] == Fluid && norm(p.Position[i]-SVector(0.3,y)) <= case.SimKernel.H,eachindex(p.Type))
                    weights = [(case.SimConstants.m₀/p.Density[i])*Wᵢⱼ(case.SimKernel,
                        T(norm(p.Position[i]-SVector(0.3,y)))*case.SimKernel.h⁻¹) for i in ids]
                    measured = sum(weights.*p.Pressure[ids])/sum(weights)
                    expected = case.SimConstants.ρ₀*case.SimConstants.g*(0.4125-y)
                    @test measured ≈ expected rtol=0.05
                end
                @test !isapprox(p.Pressure[interior],EquationOfStateGamma7.(p.Density[interior],case.SimConstants.c₀,case.SimConstants.ρ₀);rtol=0.1)
                path = joinpath(save,numbered ? "IISPHTank_000003.vtkhdf" : "IISPHTank.vtkhdf")
                h5open(path,"r+") do io
                    stored = read(io["VTKHDF/PointData/Pressure"])
                    @test stored[end-length(p)+1:end] == p.Pressure
                    numbered || @test all(iszero,stored[1:length(p)])
                end
                history = readlines(result.diagnostics.iisph.history_path)
                @test length(history) == result.performance.steps+1
                @test startswith(first(history),"Iteration,Time,TimeStep,PressureIterations")
            end
        end
    end

    @testset "IISPH pressure probes use solved pressure even when omitted from particle output" begin
        mktempdir() do save
            case = tank_case(Float32,save;duration=0.1)
            case.SimMetaData.OutputVariables = ["Density","ID"]
            measurements = MeasurementConfig(pressure_probes=[MeasurementProbe("bottom",(0.3,0.1))])
            result = run_case(case;SimMeasurements=measurements)
            h5open(joinpath(save,"IISPHTank.vtkhdf"),"r") do io
                @test !haskey(io,"VTKHDF/Particles/PointData/Pressure")
                values = read(io["VTKHDF/PressureProbes/PointData/Pressure"])
                @test first(values) == 0
                @test 1500 < last(values) < 4500
            end
            @test maximum(result.final_state.particles.Pressure) > 1000
        end
    end

    @testset "IISPH failed solve does not accept an integration step" begin
        mktempdir() do save
            case = tank_case(Float32,save;nx=8,ny=8,duration=0.002,
                solver=IISPH(relative_density_tolerance=1e-10,min_iterations=1,max_iterations=1,max_timestep=0.002))
            particles = case.SimParticles
            initial = Dict(particles.ID[i] => particles.Position[i] for i in eachindex(particles.ID))
            @test_throws ErrorException run_case(case)
            @test case.SimMetaData.Iteration == 0 && case.SimMetaData.TotalTime == 0
            @test all(i -> particles.Position[i] == initial[particles.ID[i]],eachindex(particles.ID))
            h5open(joinpath(save,"IISPHTank.vtkhdf"),"r+") do io
                @test read(io["VTKHDF/Steps/Values"]) == [0.0]
            end
        end
    end

    @testset "IISPH fluid-only free fall in 2D and 3D" begin
        for D in (2,3),T in (Float32,Float64)
            mktempdir() do save
                position = SVector{D,T}(ntuple(_ -> one(T),D))
                geometry = [SPHGeometry{D,T}([position];Density=T(1000),Type=Fluid,GroupMarker=1)]
                meta = SimulationMetaData{D,T}(SimulationName="FreeFall",SaveLocation=save,
                    SimulationTime=0.01,OutputTimes=0.01,GPUUseGraph=false,VisualizeInParaview=false,OpenLogFile=false)
                constants = SimulationConstants{T}(D;dx=T(0.05))
                case = (;SimGeometry=geometry,SimMetaData=meta,SimConstants=constants,
                    SimKernel=SPHKernelInstance{D,T}(WendlandC2();dx=constants.dx),
                    SimParticles=AllocateDataStructures(geometry,meta),SimLogger=SimulationLogger(save),
                    SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion(),
                    SimSolver=IISPH(max_timestep=0.002))
                result = run_case(case)
                p = result.final_state.particles
                @test p.Velocity[1][D] ≈ -constants.g*T(0.01) rtol=10eps(T)
                @test all(iszero,p.Velocity[1][1:D-1])
                @test iszero(p.Pressure[1])
                @test p.Position[1][D] < position[D]
            end
        end
    end

    @testset "IISPH 3D fixed-wall projection" begin
        mktempdir() do save
            T,dx,n = Float32,0.04f0,6
            fluid = [SVector{3,T}(i*dx,j*dx,k*dx) for k in 1:n for j in 1:n for i in 1:n]
            walls = [SVector{3,T}(i*dx,j*dx,k*dx) for k in -2:(n+3) for j in -2:(n+3) for i in -2:(n+3)
                     if k <= 0 || i <= 0 || i > n || j <= 0 || j > n]
            constants = SimulationConstants{T}(3;dx)
            geometry = [SPHGeometry{3,T}(walls;Density=constants.ρ₀,Type=Fixed,GroupMarker=1),
                        SPHGeometry{3,T}(fluid;Density=constants.ρ₀,Type=Fluid,GroupMarker=2)]
            meta = SimulationMetaData{3,T}(SimulationName="Tank3D",SaveLocation=save,
                SimulationTime=0.02,OutputTimes=0.02,GPUUseGraph=false,VisualizeInParaview=false,OpenLogFile=false)
            case = (;SimGeometry=geometry,SimMetaData=meta,SimConstants=constants,
                SimKernel=SPHKernelInstance{3,T}(WendlandC2();dx),
                SimParticles=AllocateDataStructures(geometry,meta),SimLogger=SimulationLogger(save),
                SimViscosity=Laminar(),SimDensityDiffusion=ZeroDensityDiffusion(),
                SimSolver=IISPH(max_timestep=0.002))
            result = run_case(case)
            p = result.final_state.particles
            active = p.Type .== Fluid
            @test result.final_state.time == 0.02
            @test all(x -> all(isfinite,x),p.Velocity)
            @test maximum(p.Density[active]) < 1.01constants.ρ₀
            @test maximum(norm,p.Velocity[active]) < 0.3
            @test result.diagnostics.iisph.max_relative_density_residual <= 1e-3
        end
    end

    @testset "IISPH external acceleration includes gravity override" begin
        mktempdir() do save
            geometry = [SPHGeometry{2,Float32}([SVector(1f0,1f0)];Density=1000f0,Type=Fluid,GroupMarker=7)]
            meta = SimulationMetaData{2,Float32}(SimulationName="Forced",SaveLocation=save,
                SimulationTime=0.01,OutputTimes=0.01,GPUUseGraph=false,VisualizeInParaview=false,OpenLogFile=false)
            constants = SimulationConstants{Float32}(2;dx=0.05f0)
            force = ExternalForce{2,Float32}(7;times=[0,1],linear_acceleration=[(2,0),(2,0)],global_gravity=false)
            case = (;SimGeometry=geometry,SimMetaData=meta,SimConstants=constants,
                SimKernel=SPHKernelInstance{2,Float32}(WendlandC2();dx=constants.dx),
                SimParticles=AllocateDataStructures(geometry,meta),SimLogger=SimulationLogger(save),
                SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion(),
                SimSolver=IISPH(max_timestep=0.002))
            result = run_case(case;SimExternalForces=force)
            @test result.final_state.particles.Velocity[1] ≈ SVector(0.02f0,0f0) atol=1e-7
            @test iszero(result.final_state.particles.Pressure[1])
        end
    end

    @testset "default WCSPH selection preserves the explicit symplectic path" begin
        snapshots = []
        for explicit in (false,true)
            mktempdir() do save
                case = tank_case(Float32,save;nx=6,ny=6,duration=0.002,solver=WCSPH())
                result = explicit ? run_case(case;SimTimeStepping=SymplecticTimeStepping()) : run_case(case)
                p = result.final_state.particles
                order = sortperm(p.ID)
                push!(snapshots,(;x=p.Position[order],v=p.Velocity[order],rho=p.Density[order],steps=result.performance.steps))
                @test result.configuration.effective.solver === :WCSPH
            end
        end
        @test snapshots[1] == snapshots[2]
    end
else
    @testset "IISPH physical tests require CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
