"""
GPU time stepping driver.

`RunSimulation` has the same signature as the CPU version. Particles are
loaded on the host (`AllocateDataStructures`), uploaded once, integrated on
the GPU and copied back to the host `StructArray` only when an output is
written, so the host arrays always hold the most recently written state.
"""
module SPHCellList

export GPUParticles, GPUSupportArrays, MotionArrays, upload_particles, download_particles!,
       RunSimulation, SimulationLoop, StepReduction

using CUDA
using StaticArrays
using LinearAlgebra
using Printf
using TimerOutputs
using Logging, LoggingExtras
using UnicodePlots

using ..SimulationEquations
using ..SimulationGeometry
using ..AuxiliaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..OpenExternalPrograms
using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..GPUReductions
using ..GPUCellGrid
using ..GPUKernels

import StructArrays: StructArray
import ProgressMeter: next!, finish!

#---------------------------------------------------------------
# Device data containers
#---------------------------------------------------------------

"""
Device copy of the particle `StructArray`. The first group of fields is the
persistent state, which is physically reordered whenever the cell list is
rebuilt; the second group is recomputed from scratch every step and therefore
never needs reordering. `scratch` holds a second set of the persistent arrays
used as the target of the reordering (the two sets are swapped afterwards).
"""
mutable struct GPUParticles{D, T, S}
    Position::CuVector{SVector{D, T}}
    Velocity::CuVector{SVector{D, T}}
    Density::CuVector{T}
    GravityFactor::CuVector{T}
    MotionLimiter::CuVector{T}
    BoundaryBool::CuVector{UInt8}
    ID::CuVector{Int}
    Type::CuVector{ParticleType}
    GroupMarker::CuVector{UInt}
    GhostPoints::CuVector{SVector{D, T}}
    GhostNormals::CuVector{SVector{D, T}}
    Acceleration::CuVector{SVector{D, T}}
    Pressure::CuVector{T}
    CellID::CuVector{Int32}

    Kernel::CuVector{T}
    KernelGradient::CuVector{SVector{D, T}}
    ChunkID::CuVector{Int}

    scratch::S
end

Base.length(p::GPUParticles) = length(p.Position)

const PERSISTENT_FIELDS = (:Position, :Velocity, :Density, :GravityFactor, :MotionLimiter,
                           :BoundaryBool, :ID, :Type, :GroupMarker, :GhostPoints, :GhostNormals,
                           :Acceleration, :Pressure)

"""
    upload_particles(SimParticles::StructArray) -> GPUParticles

Copy every field of the host particle array to the GPU.
"""
function upload_particles(SimParticles::StructArray)
    Position = CuArray(SimParticles.Position)
    D = length(eltype(Position))
    T = eltype(eltype(Position))
    n = length(Position)

    scratch = (
        Position      = similar(Position),
        Velocity      = CuVector{SVector{D, T}}(undef, n),
        Density       = CuVector{T}(undef, n),
        GravityFactor = CuVector{T}(undef, n),
        MotionLimiter = CuVector{T}(undef, n),
        BoundaryBool  = CuVector{UInt8}(undef, n),
        ID            = CuVector{Int}(undef, n),
        Type          = CuVector{ParticleType}(undef, n),
        GroupMarker   = CuVector{UInt}(undef, n),
        GhostPoints   = CuVector{SVector{D, T}}(undef, n),
        GhostNormals  = CuVector{SVector{D, T}}(undef, n),
        Acceleration  = CuVector{SVector{D, T}}(undef, n),
        Pressure      = CuVector{T}(undef, n),
    )

    return GPUParticles{D, T, typeof(scratch)}(
        Position,
        CuArray(SimParticles.Velocity),
        CuArray(SimParticles.Density),
        CuArray(SimParticles.GravityFactor),
        CuArray(SimParticles.MotionLimiter),
        CuArray(SimParticles.BoundaryBool),
        CuArray(SimParticles.ID),
        CuArray(SimParticles.Type),
        CuArray(SimParticles.GroupMarker),
        CuArray(SimParticles.GhostPoints),
        CuArray(SimParticles.GhostNormals),
        CuArray(SimParticles.Acceleration),
        CuArray(SimParticles.Pressure),
        CUDA.zeros(Int32, n),
        CuArray(SimParticles.Kernel),
        CuArray(SimParticles.KernelGradient),
        CuArray(SimParticles.ChunkID),
        scratch,
    )
end

"""
    download_particles!(SimParticles, gpu, grid)

Copy the device state back into the host `StructArray`, including the cell of
every particle as a `CartesianIndex`.
"""
function download_particles!(SimParticles::StructArray, gpu::GPUParticles{D, T}, grid::CellGrid{D}) where {D, T}
    copyto!(SimParticles.Position,       gpu.Position)
    copyto!(SimParticles.Velocity,       gpu.Velocity)
    copyto!(SimParticles.Density,        gpu.Density)
    copyto!(SimParticles.GravityFactor,  gpu.GravityFactor)
    copyto!(SimParticles.MotionLimiter,  gpu.MotionLimiter)
    copyto!(SimParticles.BoundaryBool,   gpu.BoundaryBool)
    copyto!(SimParticles.ID,             gpu.ID)
    copyto!(SimParticles.Type,           gpu.Type)
    copyto!(SimParticles.GroupMarker,    gpu.GroupMarker)
    copyto!(SimParticles.GhostPoints,    gpu.GhostPoints)
    copyto!(SimParticles.GhostNormals,   gpu.GhostNormals)
    copyto!(SimParticles.Acceleration,   gpu.Acceleration)
    copyto!(SimParticles.Pressure,       gpu.Pressure)
    copyto!(SimParticles.Kernel,         gpu.Kernel)
    copyto!(SimParticles.KernelGradient, gpu.KernelGradient)
    copyto!(SimParticles.ChunkID,        gpu.ChunkID)

    cid = Array(gpu.CellID)
    @inbounds for i in eachindex(cid)
        l = local_coords(grid, cid[i])
        SimParticles.Cells[i] = CartesianIndex(ntuple(d -> Int(l[d] + grid.origin[d]), Val(D)))
    end
    return cid
end

"""
Device arrays that are recomputed every step (half step state and shifting
terms). They are never reordered because they are overwritten after every
cell list update.
"""
struct GPUSupportArrays{D, T}
    dρdtI::CuVector{T}
    Velocityₙ⁺::CuVector{SVector{D, T}}
    Positionₙ⁺::CuVector{SVector{D, T}}
    ρₙ⁺::CuVector{T}
    ∇Cᵢ::CuVector{SVector{D, T}}
    ∇◌rᵢ::CuVector{T}
end

function GPUSupportArrays{D, T}(n::Integer) where {D, T}
    return GPUSupportArrays{D, T}(
        CUDA.zeros(T, n),
        CUDA.zeros(SVector{D, T}, n),
        CUDA.zeros(SVector{D, T}, n),
        CUDA.zeros(T, n),
        CUDA.zeros(SVector{D, T}, n),
        CUDA.zeros(T, n),
    )
end

"""
    MotionArrays(SimGeometry, SimParticles) -> NamedTuple of device arrays

Prescribed motion parameters indexed by group marker, for use in kernels.
"""
function MotionArrays(SimGeometry::Vector{Geometry{D, T}}, SimParticles) where {D, T}
    ngroups = max(1, Int(maximum(SimParticles.GroupMarker; init = 0)))
    has       = zeros(Bool, ngroups)
    velocity  = zeros(T, ngroups)
    start     = zeros(T, ngroups)
    duration  = zeros(T, ngroups)
    direction = zeros(SVector{D, T}, ngroups)
    for geom in SimGeometry
        m = geom.Motion
        if m !== nothing
            g = geom.GroupMarker
            has[g]       = true
            velocity[g]  = m.Velocity
            start[g]     = m.StartTime
            duration[g]  = m.Duration
            direction[g] = m.Direction
        end
    end
    active = any(has) && any(==(Moving), SimParticles.Type)
    return (active = active, has = CuArray(has), velocity = CuArray(velocity), start = CuArray(start),
            duration = CuArray(duration), direction = CuArray(direction))
end

#---------------------------------------------------------------
# Cell list rebuild with reordering of the persistent fields
#---------------------------------------------------------------

function rebuild_cell_list!(gpu::GPUParticles, cl::CellListWorkspace, InverseCutOff)
    s = gpu.scratch
    srcs = (gpu.Position, gpu.Velocity, gpu.Density, gpu.GravityFactor, gpu.MotionLimiter,
            gpu.BoundaryBool, gpu.ID, gpu.Type, gpu.GroupMarker, gpu.GhostPoints, gpu.GhostNormals,
            gpu.Acceleration, gpu.Pressure, cl.CellIDScratch)
    dsts = (s.Position, s.Velocity, s.Density, s.GravityFactor, s.MotionLimiter,
            s.BoundaryBool, s.ID, s.Type, s.GroupMarker, s.GhostPoints, s.GhostNormals,
            s.Acceleration, s.Pressure, gpu.CellID)

    grid = update_cell_list!(cl, gpu.Position, InverseCutOff, srcs, dsts)

    # Swap the two sets of persistent arrays.
    gpu.scratch = (
        Position = gpu.Position, Velocity = gpu.Velocity, Density = gpu.Density,
        GravityFactor = gpu.GravityFactor, MotionLimiter = gpu.MotionLimiter,
        BoundaryBool = gpu.BoundaryBool, ID = gpu.ID, Type = gpu.Type,
        GroupMarker = gpu.GroupMarker, GhostPoints = gpu.GhostPoints, GhostNormals = gpu.GhostNormals,
        Acceleration = gpu.Acceleration, Pressure = gpu.Pressure,
    )
    gpu.Position      = s.Position
    gpu.Velocity      = s.Velocity
    gpu.Density       = s.Density
    gpu.GravityFactor = s.GravityFactor
    gpu.MotionLimiter = s.MotionLimiter
    gpu.BoundaryBool  = s.BoundaryBool
    gpu.ID            = s.ID
    gpu.Type          = s.Type
    gpu.GroupMarker   = s.GroupMarker
    gpu.GhostPoints   = s.GhostPoints
    gpu.GhostNormals  = s.GhostNormals
    gpu.Acceleration  = s.Acceleration
    gpu.Pressure      = s.Pressure

    return grid
end

#---------------------------------------------------------------
# Fused per step reduction: time step limits and maximum displacement
#---------------------------------------------------------------

"""
    StepReduction(ws, gpu, sup, SimKernel; fused) -> (visc, dt1, maxdisp)

Time step limits and maximum displacement of all particles. After the first
step the values are produced by the final step kernel of the previous step
(`fused = true`) and only the per block partial results need to be combined.
"""
function StepReduction(ws::ReductionWorkspace{SVector{3, T}}, gpu::GPUParticles{D, T},
                       sup::GPUSupportArrays{D, T}, SimKernel; fused::Bool = false) where {D, T}
    init = SVector{3, T}(zero(T), T(Inf), zero(T))
    if fused
        return finish_reduction(ws, step_reduce, init)
    else
        return reduce_svector(ws, step_map, step_reduce, init, length(gpu), gpu.Position, gpu.Velocity,
                              gpu.Acceleration, sup.Positionₙ⁺, T(SimKernel.h), T(SimKernel.η²))
    end
end

#---------------------------------------------------------------
# Time loop
#---------------------------------------------------------------

function UpdateMetaData!(SimMetaData, dt)
    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt
    return nothing
end

@inline next_output_time(SimMetaData) = next_output_time(SimMetaData.OutputTimes, SimMetaData)
@inline next_output_time(interval::Real, SimMetaData) = interval * SimMetaData.OutputIterationCounter
@inline function next_output_time(times::AbstractVector, SimMetaData)
    idx = SimMetaData.OutputIterationCounter
    if idx < length(times)
        return times[idx]
    else
        return SimMetaData.SimulationTime
    end
end

@inline maybe_sync(SimMetaData) = (SimMetaData.GPUSyncTimers && CUDA.synchronize(); nothing)

"""
Advance the simulation on the GPU until the next output time. Mirrors the CPU
`SimulationLoop` step for step; see `GPUKernels` for the fused kernels.
"""
function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                        SimMetaData::SimulationMetaData{Dimensions, FloatType},
                        SimConstants, gpu::GPUParticles{Dimensions, FloatType},
                        cl::CellListWorkspace, sup::GPUSupportArrays, red::ReductionWorkspace,
                        motion) where {Dimensions, FloatType, SDD <: SPHDensityDiffusion, SV <: SPHViscosity}
    HourGlass = SimMetaData.HourGlass
    (; CFL, c₀) = SimConstants
    h = SimKernel.h

    FlagKernel = Val(SimMetaData.FlagOutputKernelValues)
    FlagShift  = Val(SimMetaData.FlagShifting)
    threads    = SimMetaData.GPUInteractionThreads
    nlanes     = SimMetaData.GPULanesPerParticle
    lanes      = Val(nlanes <= 0 ? choose_lanes(length(gpu)) : nlanes)
    bforces    = Val(SimMetaData.GPUBoundaryForces)

    Δx = one(FloatType) + h

    # The reduction of the previous step's final kernel is only available
    # after at least one step has been taken.
    fused_reduction = SimMetaData.Iteration > 0

    while SimMetaData.TotalTime <= next_output_time(SimMetaData)

        @timeit HourGlass "01 Update TimeStep" begin
            r    = StepReduction(red, gpu, sup, SimKernel; fused = fused_reduction)
            Δx  += 4 * r[3]
            visc = r[1]
            dt1  = r[2]
            dt   = CFL * min(dt1, h / (c₀ + visc))
        end
        dt₂ = dt / 2

        @timeit HourGlass "02 Calculate IndexCounter" begin
            if Δx >= h
                @timeit HourGlass "02a Actual Calculate IndexCounter" begin
                    rebuild_cell_list!(gpu, cl, SimKernel.H⁻¹)
                    maybe_sync(SimMetaData)
                end
                Δx = zero(FloatType)
            end
        end
        grid      = cl.grid
        CellStart = cl.CellStart
        # Arrays read by the viscosity and density diffusion models (same
        # convention as the CPU code: always the state at the start of the step)
        SimParticlesNT = (Density = gpu.Density, Velocity = gpu.Velocity)

        # The pressure of the start-of-step density was already computed by the
        # final kernel of the previous step (and on the host before the first).
        if motion.active
            @timeit HourGlass "Motion" begin
                launch_motion!(gpu.Position, gpu.Velocity, gpu.Type, gpu.GroupMarker, motion, dt₂,
                               SimMetaData.TotalTime)
                maybe_sync(SimMetaData)
            end
        end

        if !SimMetaData.FlagSingleStepTimeStepping
            if SimMetaData.FlagMDBCSimple
                @timeit HourGlass "04a First NeighborLoopMDBC" begin
                    launch_mdbc!(gpu.Density, gpu.Position, gpu.GhostPoints, gpu.Type, CellStart, grid,
                                 SimKernel, SimConstants; threads = threads, lanes = lanes)
                    maybe_sync(SimMetaData)
                end
            end

            @timeit HourGlass "04 First NeighborLoop" begin
                launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                                     gpu.ChunkID, gpu.Position, gpu.Density, gpu.Pressure, gpu.Velocity,
                                     gpu.MotionLimiter, SimParticlesNT, CellStart, gpu.CellID, grid,
                                     SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                                     FlagKernel, FlagShift; threads = threads, lanes = lanes,
                                     boundary_forces = bforces)
                maybe_sync(SimMetaData)
            end
        end

        @timeit HourGlass "05b Update To Half TimeStep" begin
            launch_half_step!(sup.Positionₙ⁺, sup.Velocityₙ⁺, sup.ρₙ⁺, gpu.Pressure,
                              gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, sup.dρdtI,
                              gpu.GravityFactor, gpu.MotionLimiter, gpu.Type, gpu.GroupMarker, motion,
                              dt₂, SimMetaData.TotalTime, SimConstants)
            maybe_sync(SimMetaData)
        end

        @timeit HourGlass "08 Second NeighborLoop" begin
            launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                                 gpu.ChunkID, sup.Positionₙ⁺, sup.ρₙ⁺, gpu.Pressure, sup.Velocityₙ⁺,
                                 gpu.MotionLimiter, SimParticlesNT, CellStart, gpu.CellID, grid,
                                 SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                                 FlagKernel, FlagShift; threads = threads, lanes = lanes,
                                 boundary_forces = bforces)
            maybe_sync(SimMetaData)
        end

        @timeit HourGlass "11 Update To Final TimeStep" begin
            launch_final_step!(gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, gpu.Pressure,
                               sup.dρdtI, sup.ρₙ⁺, sup.Positionₙ⁺, gpu.GravityFactor, gpu.MotionLimiter,
                               sup.∇Cᵢ, sup.∇◌rᵢ, dt, SimKernel, SimConstants, red, FlagShift)
            maybe_sync(SimMetaData)
        end
        fused_reduction = true

        @timeit HourGlass "12 Update MetaData" UpdateMetaData!(SimMetaData, dt)
    end

    return nothing
end

#---------------------------------------------------------------
# Driver
#---------------------------------------------------------------

"""
    RunSimulation(; SimGeometry, SimMetaData, SimConstants, SimKernel, SimLogger,
                    SimParticles, SimViscosity, SimDensityDiffusion, ParticleNormalsPath)

Run a complete simulation on the GPU. Same interface as the CPU version. On
return the host `SimParticles` hold the final state (reordered by cell, like
the CPU version).
"""
function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}},
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    SimKernel::SPHKernelInstance,
    SimLogger::SimulationLogger,
    SimParticles::StructArray,
    SimViscosity::SV,
    SimDensityDiffusion::SDD,
    ParticleNormalsPath::Union{Nothing, String} = nothing
    ) where {Dimensions, FloatType, SV <: SPHViscosity, SDD <: SPHDensityDiffusion}

    CUDA.functional() || error("CUDA is not functional on this machine; use the CPU package SPHExample instead.")

    (; HourGlass) = SimMetaData

    TimeSteps = Vector{FloatType}()

    if SimMetaData.FlagMDBCSimple
        ParticleNormalsPath === nothing && error("FlagMDBCSimple requires `ParticleNormalsPath`.")
        _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(Dimensions), FloatType, ParticleNormalsPath)
        for gi ∈ eachindex(GhostPoints)
            SimParticles.GhostPoints[gi]  = GhostPoints[gi]
            SimParticles.GhostNormals[gi] = GhostNormals[gi]
        end
    end

    if SimMetaData.FlagLog
        InitializeLogger(SimLogger, SimConstants, SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        with_logger(SimLogger.Logger) do
            dev = CUDA.device()
            @info "GPU: $(CUDA.name(dev)), compute capability $(CUDA.capability(dev)), " *
                  "$(round(CUDA.totalmem(dev) / 2^30; digits = 1)) GiB, CUDA.jl $(pkgversion(CUDA))"
            @info "GPU float type: $(FloatType)"
        end
    end

    if SimMetaData.FlagLog
        LogStep(SimLogger, SimMetaData, HourGlass)
        SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
    end

    NumberOfPoints = length(SimParticles)::Int
    Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)

    # Device side data
    @timeit HourGlass "00a Upload To GPU" begin
        gpu    = upload_particles(SimParticles)
        sup    = GPUSupportArrays{Dimensions, FloatType}(NumberOfPoints)
        red    = ReductionWorkspace{SVector{3, FloatType}}(NumberOfPoints)
        cl     = CellListWorkspace{Dimensions, FloatType}(NumberOfPoints;
                     max_cells = SimMetaData.GPUMaxCells, deterministic = SimMetaData.GPUDeterministicSort)
        motion = MotionArrays(SimGeometry, SimParticles)
        CUDA.synchronize()
    end

    output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    # Save initial state, use 1 else this cannot be used to index fid vector
    SimMetaData.OutputIterationCounter = 1
    output.save_particles(SimMetaData.OutputIterationCounter)
    output.save_grid(SimMetaData.OutputIterationCounter, CartesianIndex{Dimensions}[], SimParticles)

    generate_showvalues(Iteration, TotalTime, TimeLeftInSeconds) = () -> [
        (:(Iteration), string(Iteration)),
        (:(TotalTime), @sprintf("%3.3f", TotalTime)),
        (:(TimeLeftInSeconds), @sprintf("%3.1f [s]", TimeLeftInSeconds)),
    ]

    if !SimLogger.ToConsole
        @timeit HourGlass "14 Next TimeStep" next!(
            SimMetaData.ProgressSpecification;
            showvalues = generate_showvalues(SimMetaData.Iteration, SimMetaData.TotalTime, 1e6),
        )
    end

    # Output is written by a task so that the GPU can already continue with
    # the next output interval. The host arrays are only overwritten after
    # the previous write finished.
    write_task     = nothing
    async_write_time = 0.0
    function wait_for_output()
        if write_task !== nothing
            async_write_time += fetch(write_task)::Float64
            write_task = nothing
        end
        return nothing
    end

    while true
        @timeit HourGlass "00 SimulationLoop" SimulationLoop(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                             SimConstants, gpu, cl, sup, red, motion)
        push!(TimeSteps, SimMetaData.CurrentTimeStep)

        if SimMetaData.FlagLog
            LogStep(SimLogger, SimMetaData, HourGlass)
            SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        end

        SimMetaData.OutputIterationCounter += 1

        @timeit HourGlass "13 Download From GPU" begin
            wait_for_output()
            cid = download_particles!(SimParticles, gpu, cl.grid)
        end

        UniqueCells = SimMetaData.ExportGridCells ? unique_cells_host(cl.grid, cid) : CartesianIndex{Dimensions}[]
        SimMetaData.IndexCounter = length(UniqueCells)

        counter = SimMetaData.OutputIterationCounter
        t_out   = SimMetaData.TotalTime
        if SimMetaData.GPUAsyncOutput
            write_task = Threads.@spawn begin
                t0 = time()
                output.save_particles(counter, t_out)
                output.save_grid(counter, UniqueCells, SimParticles, t_out)
                time() - t0
            end
        else
            @timeit HourGlass "13A Save Particle Data" output.save_particles(counter, t_out)
            @timeit HourGlass "13A Save CellGrid Data" output.save_grid(counter, UniqueCells, SimParticles, t_out)
        end

        if !SimLogger.ToConsole
            TimeLeftInSeconds = (SimMetaData.SimulationTime - SimMetaData.TotalTime) *
                                (TimerOutputs.tottime(HourGlass) / 1e9 / SimMetaData.TotalTime)
            @timeit HourGlass "14 Next TimeStep" next!(
                SimMetaData.ProgressSpecification;
                showvalues = generate_showvalues(SimMetaData.Iteration, SimMetaData.TotalTime, TimeLeftInSeconds),
            )
        end

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            @timeit HourGlass "13B Close Data Streams" begin
                wait_for_output()
                output.close_files()
            end

            if !SimLogger.ToConsole
                finish!(SimMetaData.ProgressSpecification)
            end
            show(HourGlass, sortby = :name)
            show(HourGlass)

            AutoOpenParaview(SimMetaData, output.variable_names)

            UnicodeTimeStepsGraph = lineplot(1:length(TimeSteps), TimeSteps, title = "Time Steps [s] as a function of iteration",
                                             name = "Time Steps", xlabel = "Iterations [-]", ylabel = "Time Step Size [s]")

            if SimMetaData.FlagLog
                with_logger(SimLogger.Logger) do
                    @info "Cell list rebuilds: $(cl.nrebuilds), grid dims: $(cl.grid.dims)"
                    @info @sprintf("Asynchronous output write time (overlapped with GPU work): %.2f [s]", async_write_time)
                end
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

    return nothing
end

end # module
