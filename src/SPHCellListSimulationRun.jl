module SPHCellListSimulationRun

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
using ..SimulationEquations
using ..SimulationGeometry
using ..AuxiliaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..OpenExternalPrograms
using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SPHCellListComputation
using ..SPHCellListHelperFunctions
using ..SPHCellListNeighborSearch

import StructArrays: StructArray, foreachfield
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using UnicodePlots
using LinearAlgebra
using Bumper

@inbounds function SimulationLoop(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData::SimulationMetaData{Dimensions, FloatType}, SimConstants, SimParticles, Stencil,  ParticleRanges, UniqueCells, SortingScratchSpace, SimThreadedArrays, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ, MotionDefinition) where {Dimensions, FloatType}
    Position       = SimParticles.Position
    Density        = SimParticles.Density
    Pressure       = SimParticles.Pressure
    Velocity       = SimParticles.Velocity
    Acceleration   = SimParticles.Acceleration
    MotionLimiter  = SimParticles.MotionLimiter
    ParticleType   = SimParticles.Type
    ParticleMarker = SimParticles.GroupMarker
    Kernel         = SimParticles.Kernel
    KernelGradient = SimParticles.KernelGradient
    GhostPoints    = SimParticles.GhostPoints
    GhostNormals   = SimParticles.GhostNormals

    ###
    DimensionsPlus = Dimensions + 1
    Δx = one(eltype(Density)) + SimKernel.h
    UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
    EnumeratedIndices = enumerate(index_chunks(UniqueCellsView; n=nthreads()))
    @no_escape begin
        while SimMetaData.TotalTime <= SimMetaData.OutputEach * SimMetaData.OutputIterationCounter

            # Δx = update_delta_x!(Δx, Positionₙ⁺, SimParticles.Position)

            # println("Δx: ", Δx, "h: ", SimKernel.h," dt: ", SimMetaData.CurrentTimeStep, " Iteration: ", SimMetaData.Iteration, " TotalTime: ", SimMetaData.TotalTime, " OutputIterationCounter: ", SimMetaData.OutputIterationCounter)

            @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants, SimKernel)
            dt₂ = dt * 0.5

            @timeit SimMetaData.HourGlass "02 Calculate IndexCounter"  begin
                # Note: If particles are not inside of the neighbor list visualiation, try setting this if statement to always true, since UniqueCells will be updated always then
                # In theory, the maximal speed is the speed of sound, this should give a safe guard
                # and ensure it is always updated in a reasonable manner. This only works well, assuming that
                # c₀ >= maximum(norm.(Velocity))
                # Remove if statement logic if you want to update each iteration
                if mod(SimMetaData.Iteration, ceil(Int, SimKernel.H / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
                # if 1.25 * Δx >= SimKernel.h
                    @timeit SimMetaData.HourGlass "02a Actual Calculate IndexCounter" SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace,  ParticleRanges, UniqueCells)
                    Δx = zero(eltype(Density))
                    UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                    EnumeratedIndices = enumerate(index_chunks(UniqueCellsView; n=nthreads()))
                end
            end

            @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
        
            if !SimMetaData.FlagSingleStepTimeStepping
                ###=== First step of resetting arrays
                @timeit SimMetaData.HourGlass "ResetArrays"                          ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
                ###===
            
                @timeit SimMetaData.HourGlass "03 Pressure"                          Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
                if SimMetaData.FlagMDBCSimple
                    bᵧ = @alloc(SVector{DimensionsPlus, FloatType}, length(Position))
                    Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus * DimensionsPlus}, length(Position))
                    @timeit SimMetaData.HourGlass "04 First NeighborLoopMDBC"        NeighborLoopMDBC!(SimKernel, SimMetaData, SimConstants, ParticleRanges, Position, Density, UniqueCellsView, GhostPoints, GhostNormals, ParticleType, bᵧ, Aᵧ)
                end
                @timeit SimMetaData.HourGlass "04 First NeighborLoop"                NeighborLoop!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, ParticleRanges, Stencil, Position, Density, Pressure, Velocity, MotionLimiter, UniqueCellsView, EnumeratedIndices)
                @timeit SimMetaData.HourGlass "Reduction"                            ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
            end

            if SimMetaData.FlagMDBCSimple
                @timeit SimMetaData.HourGlass "05a Apply MDBC before Half TimeStep"  ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)
            end
            
            @timeit SimMetaData.HourGlass "05b Update To Half TimeStep"              HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)


            @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)
        
            ###=== Second step of resetting arrays
            @timeit SimMetaData.HourGlass "ResetArrays"                              ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
            ###===

            @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
        
            @timeit SimMetaData.HourGlass "03 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
            @timeit SimMetaData.HourGlass "08 Second NeighborLoop"                   NeighborLoop!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, ParticleRanges, Stencil, Positionₙ⁺, ρₙ⁺, Pressure, Velocityₙ⁺, MotionLimiter, UniqueCellsView, EnumeratedIndices)
            @timeit SimMetaData.HourGlass "Reduction"                                ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)

        
            @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary"          LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)
        
            @timeit SimMetaData.HourGlass "10 Final Density"                         DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)
        
            @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"              FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
        
            @timeit SimMetaData.HourGlass "12 Update MetaData"                       UpdateMetaData!(SimMetaData, dt)

        end
    end
    
    return nothing
end

###===
function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}}, #Don't further specify type for now
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    SimKernel::SPHKernelInstance,
    SimLogger::SimulationLogger,
    SimParticles::StructArray,
    SimViscosity::SPHViscosity,
    SimDensityDiffusion::SPHDensityDiffusion,
    ParticleNormalsPath::Union{Nothing,String} = nothing
    ) where {Dimensions,FloatType}

    # Unpack the relevant simulation meta data
    @unpack HourGlass = SimMetaData;

    # Vector of time steps
    TimeSteps = Vector{FloatType}()
    
    dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimParticles.Position)

    # Implement this properly later in shaa Allah
    # if !isnothing(ParticleNormalsPath)
        if SimMetaData.FlagMDBCSimple
            _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(Dimensions), FloatType, ParticleNormalsPath)
        

            #TODO: In the future decide on one of the two in shaa Allah
            for gi ∈ eachindex(GhostPoints)
                SimParticles.GhostPoints[gi]  = GhostPoints[gi]
                SimParticles.GhostNormals[gi] = GhostNormals[gi]
            end
        end
    # end

    if !SimMetaData.FlagShifting
        resize!(∇Cᵢ , 0)
        resize!(∇◌rᵢ, 0)
    end

    if SimMetaData.FlagLog
        InitializeLogger(SimLogger, SimConstants, SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
    end

    # To generate first line
    if SimMetaData.FlagLog
        LogStep(SimLogger, SimMetaData, HourGlass)
        SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
    end
    
    NumberOfPoints = length(SimParticles)::Int
    Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)

    SimThreadedArrays = AllocateThreadedArrays(SimMetaData, SimParticles, dρdtI, ∇Cᵢ, ∇◌rᵢ)

    # Produce sorting related variables
    ParticleRanges         = zeros(Int, NumberOfPoints + 1)
    UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
    Stencil                = ConstructStencil(Val(Dimensions))
    _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)

    output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    # Save initial state, use 1 else this cannot be used to index fid vector
    SimMetaData.OutputIterationCounter = 1
    output.save_particles(SimMetaData.OutputIterationCounter)
    output.save_grid(SimMetaData.OutputIterationCounter, UniqueCells, SimParticles)


    # Assuming group markers are sequential
    MotionDefinition = Vector{Union{Nothing, MotionDetails{Dimensions, FloatType}}}(undef, maximum(SimParticles.GroupMarker))

    for geom in SimGeometry
        group_marker = geom.GroupMarker
        if geom.Motion !== nothing
            MotionDefinition[group_marker] = geom.Motion
        else
            MotionDefinition[group_marker] = nothing
        end
    end

    # Normal run and save data
    generate_showvalues(Iteration, TotalTime, TimeLeftInSeconds) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime)), (:(TimeLeftInSeconds),format(FormatExpr("{1:3.1f} [s]"), TimeLeftInSeconds))]
    

    if !SimLogger.ToConsole
        @timeit HourGlass "14 Next TimeStep" next!(
            SimMetaData.ProgressSpecification;
            showvalues = generate_showvalues(
                SimMetaData.Iteration,
                SimMetaData.TotalTime,
                1e6,
            ),
        )
    end

    @inbounds while true

        @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoop(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, Stencil, ParticleRanges, UniqueCells, SortingScratchSpace, SimThreadedArrays, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ, MotionDefinition)
        push!(TimeSteps, SimMetaData.CurrentTimeStep)

        if SimMetaData.FlagLog
            LogStep(SimLogger, SimMetaData, HourGlass)
            SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        end

        SimMetaData.OutputIterationCounter += 1

        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
        @sync Threads.@spawn begin
            @timeit SimMetaData.HourGlass "13A Save Particle Data" output.save_particles(SimMetaData.OutputIterationCounter)
            @timeit SimMetaData.HourGlass "13A Save CellGrid Data" output.save_grid(SimMetaData.OutputIterationCounter, UniqueCellsView, SimParticles)
        end

        if !SimLogger.ToConsole
            TimeLeftInSeconds = (SimMetaData.SimulationTime - SimMetaData.TotalTime) *
                                (TimerOutputs.tottime(HourGlass) / 1e9 / SimMetaData.TotalTime)
            @timeit HourGlass "14 Next TimeStep" next!(
                SimMetaData.ProgressSpecification;
                showvalues = generate_showvalues(
                    SimMetaData.Iteration,
                    SimMetaData.TotalTime,
                    TimeLeftInSeconds,
                ),
            )
        end

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            
            # At end of simulation
            @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

            if !SimLogger.ToConsole
                finish!(SimMetaData.ProgressSpecification)
            end
            show(HourGlass,sortby=:name)
            show(HourGlass)

            AutoOpenParaview(SimMetaData, output.variable_names)

            # Time steps line plot
            UnicodeTimeStepsGraph = lineplot(1:length(TimeSteps), TimeSteps, title="Time Steps [s] as a function of iteration", name="Time Steps", xlabel="Iterations [-]", ylabel="Time Step Size [s]")

            if SimMetaData.FlagLog
                LogFinal(SimLogger, HourGlass)
         
                with_logger(SimLogger.Logger) do
                    @info ""
                    show(SimLogger.LoggerIo, UnicodeTimeStepsGraph)
                end

                close(SimLogger.LoggerIo)
                AutoOpenLogFile(SimLogger, SimMetaData)
            end

            break
        end
    end
end


end
