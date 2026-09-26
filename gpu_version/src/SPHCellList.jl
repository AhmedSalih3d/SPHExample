"""
GPU time stepping driver.

`RunSimulation` has the same signature as the CPU version. Particles are
loaded on the host (`AllocateDataStructures`), uploaded once, integrated on
the GPU and copied back to the host `StructArray` only when an output is
written, so the host arrays always hold the most recently written state.
"""
module SPHCellList

export GPUParticles, GPUSupportArrays, MotionArrays, upload_particles, download_particles!,
       RunSimulation, SimulationLoop, enqueue_step!, batch_size

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
using ..GPUStepState
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
`GhostIndex` lists the particles that own a ghost node (non-zero
`GhostPoints`) in cell order; it is derived from the reordered `GhostPoints`
after every rebuild and the mDBC kernel is launched over it alone.
"""
mutable struct GPUParticles{D, T, S}
    Position::CuVector{SVector{D, T}}
    Velocity::CuVector{SVector{D, T}}
    Density::CuVector{T}
    ID::CuVector{Int}
    Type::CuVector{ParticleType}
    GroupMarker::CuVector{UInt}
    GhostPoints::CuVector{SVector{D, T}}
    GhostNormals::CuVector{SVector{D, T}}
    Acceleration::CuVector{SVector{D, T}}
    Pressure::CuVector{T}
    CellID::CuVector{Int32}
    GhostIndex::CuVector{Int32}

    Kernel::CuVector{T}
    KernelGradient::CuVector{SVector{D, T}}

    scratch::S
end

Base.length(p::GPUParticles) = length(p.Position)

const PERSISTENT_FIELDS = (:Position, :Velocity, :Density,
                           :ID, :Type, :GroupMarker, :GhostPoints, :GhostNormals,
                           :Acceleration, :Pressure)

# Device fields that can be copied back into the host `StructArray`.
const DOWNLOADABLE_FIELDS = (:Velocity, :Density, :ID, :Type, :GroupMarker, :GhostPoints, :GhostNormals,
                             :Acceleration, :Pressure, :Kernel, :KernelGradient)

"""
    upload_particles(SimParticles::StructArray) -> GPUParticles

Copy every stored field of the host particle array to the GPU. The ghost
node owners (`GhostIndex`) are listed from the host `GhostPoints`, so set
those before uploading.
"""
function upload_particles(SimParticles::StructArray)
    Position = CuArray(SimParticles.Position)
    D = length(eltype(Position))
    T = eltype(eltype(Position))
    n = length(Position)

    GhostIndex = CuArray(Int32.(findall(!iszero, SimParticles.GhostPoints)))
    scratch = (
        Position      = similar(Position),
        Velocity      = CuVector{SVector{D, T}}(undef, n),
        Density       = CuVector{T}(undef, n),
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
        CuArray(SimParticles.ID),
        CuArray(SimParticles.Type),
        CuArray(SimParticles.GroupMarker),
        CuArray(SimParticles.GhostPoints),
        CuArray(SimParticles.GhostNormals),
        CuArray(SimParticles.Acceleration),
        CuArray(SimParticles.Pressure),
        CUDA.zeros(Int32, n),
        GhostIndex,
        CuArray(SimParticles.Kernel),
        CuArray(SimParticles.KernelGradient),
        scratch,
    )
end

"""
    download_particles!(SimParticles, gpu, grid, fields = DOWNLOADABLE_FIELDS; cells = false)

Copy `Position` and the device fields named in `fields` back into the host
`StructArray`. Host fields that are not listed are left untouched, so pass
exactly the fields that are written to the output. With `cells = true` the
cell of every particle is also stored as a `CartesianIndex`. Returns the cell
ids as a host vector (empty unless `cells = true`).
"""
function download_particles!(SimParticles::StructArray, gpu::GPUParticles{D, T}, grid::CellGrid{D},
                             fields = DOWNLOADABLE_FIELDS; cells::Bool = false) where {D, T}
    copyto!(SimParticles.Position, gpu.Position)
    for f in fields
        copyto!(getproperty(SimParticles, f), getproperty(gpu, f))
    end
    cells || return Int32[]

    cid = Array(gpu.CellID)
    @inbounds for i in eachindex(cid)
        l = local_coords(grid, cid[i])
        SimParticles.Cells[i] = CartesianIndex(ntuple(d -> Int(l[d] + grid.origin[d]), Val(D)))
    end
    return cid
end

"""
Device arrays that are recomputed every step (half step state, reciprocal
densities and shifting terms). They are never reordered because they are
overwritten after every cell list update. `InvDensity` holds `1 / Density`
of the start-of-step state and `InvDensityₙ⁺` that of the predictor state;
the interaction kernel multiplies by them instead of dividing per pair
(same as `FillInverseDensity!` of the CPU code).
"""
struct GPUSupportArrays{D, T}
    dρdtI::CuVector{T}
    Velocityₙ⁺::CuVector{SVector{D, T}}
    Positionₙ⁺::CuVector{SVector{D, T}}
    ρₙ⁺::CuVector{T}
    InvDensity::CuVector{T}
    InvDensityₙ⁺::CuVector{T}
    ∇Cᵢ::CuVector{SVector{D, T}}
    ∇◌rᵢ::CuVector{T}
end

function GPUSupportArrays{D, T}(n::Integer) where {D, T}
    return GPUSupportArrays{D, T}(
        CUDA.zeros(T, n),
        CUDA.zeros(SVector{D, T}, n),
        CUDA.zeros(SVector{D, T}, n),
        CUDA.zeros(T, n),
        CUDA.zeros(T, n),
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
    srcs = (gpu.Position, gpu.Velocity, gpu.Density,
            gpu.ID, gpu.Type, gpu.GroupMarker, gpu.GhostPoints, gpu.GhostNormals,
            gpu.Acceleration, gpu.Pressure, cl.CellIDScratch)
    dsts = (s.Position, s.Velocity, s.Density,
            s.ID, s.Type, s.GroupMarker, s.GhostPoints, s.GhostNormals,
            s.Acceleration, s.Pressure, gpu.CellID)

    grid = update_cell_list!(cl, gpu.Position, InverseCutOff, srcs, dsts)

    # Swap the two sets of persistent arrays.
    gpu.scratch = (
        Position = gpu.Position, Velocity = gpu.Velocity, Density = gpu.Density,
        ID = gpu.ID, Type = gpu.Type,
        GroupMarker = gpu.GroupMarker, GhostPoints = gpu.GhostPoints, GhostNormals = gpu.GhostNormals,
        Acceleration = gpu.Acceleration, Pressure = gpu.Pressure,
    )
    gpu.Position      = s.Position
    gpu.Velocity      = s.Velocity
    gpu.Density       = s.Density
    gpu.ID            = s.ID
    gpu.Type          = s.Type
    gpu.GroupMarker   = s.GroupMarker
    gpu.GhostPoints   = s.GhostPoints
    gpu.GhostNormals  = s.GhostNormals
    gpu.Acceleration  = s.Acceleration
    gpu.Pressure      = s.Pressure

    # The reorder moved the ghost node owners: relist them in the new (cell)
    # order. Their number never changes, so the list keeps its length and
    # device pointer, and captured graphs stay valid.
    isempty(gpu.GhostIndex) || compact_nonzero!(cl, gpu.GhostPoints, gpu.GhostIndex)

    return grid
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
# Deadline for the current output interval. The GPU frame counter is one
# based because it indexes the writer's file handle vector directly, so
# `interval * counter` here is the CPU's `interval * (counter + 1)` with its
# zero based counter. The deadline is clamped to the simulation end so the
# last interval never overshoots `SimulationTime` by up to one interval.
@inline function next_output_time(interval::Real, SimMetaData)
    return min(interval * SimMetaData.OutputIterationCounter, SimMetaData.SimulationTime)
end
@inline function next_output_time(times::AbstractVector, SimMetaData)
    idx = SimMetaData.OutputIterationCounter
    if idx <= length(times)
        return min(times[idx], SimMetaData.SimulationTime)
    else
        return SimMetaData.SimulationTime
    end
end

@inline maybe_sync(SimMetaData) = (SimMetaData.GPUSyncTimers && CUDA.synchronize(); nothing)

#---------------------------------------------------------------
# Launch sequence of one time step
#---------------------------------------------------------------

# Time a phase (and wait for the GPU) only when `timed` is set; otherwise
# just enqueue it, so that the same code can be captured into a graph.
macro phase(hg, name, timed, ex)
    quote
        if $(esc(timed))
            TimerOutputs.@timeit $(esc(hg)) $(esc(name)) begin
                $(esc(ex))
                CUDA.synchronize()
            end
        else
            $(esc(ex))
        end
    end
end

"""
    enqueue_state_derivative!(ctx, step, timed, mdbc_name, loop_name)

Enqueue the evaluation of the start-of-step state: the mDBC correction of
the boundary densities together with the pressure of the corrected density
(`launch_mdbc!`, with `SimpleMDBC`), the reciprocal densities and the
neighbour loop that produces `dρdtI` and the acceleration. The symplectic
scheme runs this at every step as its first neighbour loop. The single
neighbour scheme carries the corrector derivative of the previous step into
the predictor instead and only runs this to start that derivative: before
the first step and after a cell list rebuild (`SimulationLoop`). `step`
gates every kernel; `mdbc_name` and `loop_name` are the timer labels.
"""
function enqueue_state_derivative!(ctx, step, timed::Bool, mdbc_name::AbstractString, loop_name::AbstractString)
    (; gpu, cl, sup, SimKernel, SimConstants, SimDensityDiffusion, SimViscosity,
       FlagKernel, FlagShift, UseMDBC, threads, lanes, bforces, HourGlass) = ctx
    grid      = cl.grid_dev
    CellStart = cl.CellStart

    if UseMDBC
        @phase HourGlass mdbc_name timed launch_mdbc!(
            gpu.Density, gpu.Pressure, gpu.Position, gpu.GhostPoints, gpu.GhostIndex, gpu.Type, CellStart,
            grid, step, SimKernel, SimConstants; threads = threads, lanes = lanes)
    end

    @phase HourGlass loop_name timed begin
        launch_inv_density!(sup.InvDensity, gpu.Density, step)
        launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                             gpu.Position, gpu.Density, sup.InvDensity, gpu.Pressure,
                             gpu.Velocity, gpu.Type, CellStart, gpu.CellID, grid, step,
                             SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                             FlagKernel, FlagShift; threads = threads, lanes = lanes,
                             boundary_forces = bforces)
    end
    return nothing
end

"""
    enqueue_step!(ctx, timed)

Enqueue every kernel of one time step. Nothing in here reads the device:
the time step and the time are taken from `ctx.state` by the kernels, the
grid from `ctx.cl.grid_dev`, and every kernel exits at once when the device
side stop flag is set. The sequence is therefore identical from step to step
and can be captured as a CUDA graph (`launch_step_graph!`). With `timed`
every phase is followed by a synchronization and timed separately.

The step follows the CPU `SimulationLoop` of the selected scheme. Symplectic:
motion, mDBC, first neighbour loop, half step, second neighbour loop, final
step. Single neighbour: motion, mDBC correction of the densities the
predictor starts from, half step, neighbour loop, final step. (The CPU
additionally re-evaluates the carried derivative every 20 steps; the GPU
does not, the carried derivative is only re-evaluated after a cell list
rebuild, see `SimulationLoop`.)
"""
function enqueue_step!(ctx, timed::Bool)
    (; gpu, cl, sup, red, motion, state, SimKernel, SimConstants, SimDensityDiffusion, SimViscosity,
       FlagKernel, FlagShift, UseMDBC, SingleNeighbor, threads, lanes, bforces, HourGlass) = ctx
    grid      = cl.grid_dev
    CellStart = cl.CellStart

    # dt of this step from the reduction of the previous one; also decides
    # whether the step may run at all (cell list rebuild, output time).
    @phase HourGlass "01 Update TimeStep" timed launch_finish!(state, red, SimKernel, SimConstants)

    # The pressure of the start-of-step density was already computed by the
    # final kernel of the previous step (and on the host before the first).
    if motion.active
        @phase HourGlass "Motion" timed launch_motion!(gpu.Position, gpu.Velocity, gpu.Type, gpu.GroupMarker,
                                                       motion, state)
    end

    if SingleNeighbor
        # CPU "02 Apply MDBC before Half TimeStep": the boundary densities the
        # predictor and the final density update start from are corrected
        # every step, the carried derivative is kept.
        if UseMDBC
            @phase HourGlass "04a NeighborLoopMDBC before Half TimeStep" timed launch_mdbc!(
                gpu.Density, gpu.Pressure, gpu.Position, gpu.GhostPoints, gpu.GhostIndex, gpu.Type, CellStart,
                grid, state, SimKernel, SimConstants; threads = threads, lanes = lanes)
        end
    else
        enqueue_state_derivative!(ctx, state, timed, "04a First NeighborLoopMDBC", "04 First NeighborLoop")
    end

    @phase HourGlass "05b Update To Half TimeStep" timed launch_half_step!(
        sup.Positionₙ⁺, sup.Velocityₙ⁺, sup.ρₙ⁺, sup.InvDensityₙ⁺, gpu.Pressure,
        gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, sup.dρdtI,
        gpu.Type, gpu.GroupMarker, motion, state, SimConstants)

    # Corrector: every term, including the viscosity and density diffusion
    # models, is evaluated at the predictor state.
    @phase HourGlass "08 Second NeighborLoop" timed launch_interactions!(
        sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
        sup.Positionₙ⁺, sup.ρₙ⁺, sup.InvDensityₙ⁺, gpu.Pressure,
        sup.Velocityₙ⁺, gpu.Type, CellStart, gpu.CellID, grid, state,
        SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
        FlagKernel, FlagShift; threads = threads, lanes = lanes,
        boundary_forces = bforces)

    @phase HourGlass "11 Update To Final TimeStep" timed launch_final_step!(
        gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, gpu.Pressure,
        sup.dρdtI, sup.ρₙ⁺, sup.Positionₙ⁺, sup.Velocityₙ⁺, gpu.Type,
        sup.∇Cᵢ, sup.∇◌rᵢ, state, SimKernel, SimConstants, red, FlagShift)

    @phase HourGlass "12 Update MetaData" timed launch_commit!(state)
    return nothing
end

"""
    launch_step_graph!(ctx)

Replay the launch sequence of a step as a CUDA graph. The graph bakes in the
device pointers of its arguments, and the persistent particle arrays are
swapped with their scratch copies at every cell list rebuild, so one graph
is kept per set of pointers (normally two, keyed by the position array and
the reallocation generation of the cell list). The first call for a key
captures and instantiates the graph; a capture that fails (kernels that
still have to be compiled) falls back to direct launches for that step.
"""
function launch_step_graph!(ctx)
    state = ctx.state
    key   = (UInt(pointer(ctx.gpu.Position)), ctx.cl.generation)
    exec  = get(state.graphs, key, nothing)
    if exec === nothing
        graph = CUDA.capture(() -> enqueue_step!(ctx, false); throw_error = false)
        if graph === nothing
            enqueue_step!(ctx, false)
            return nothing
        end
        exec = CUDA.instantiate(graph)
        state.graphs[key] = exec
    end
    CUDA.launch(exec)
    return nothing
end

"""
    batch_size(state, h, t_out, kmax) -> K

Number of steps to enqueue before the next host read back: the estimated
number of steps until the cell list must be rebuilt or the output time is
reached (from the last read back), plus one so that the device rather than
the host makes the final decision, capped by `kmax`. Steps enqueued beyond
the device side stop are no-ops.
"""
function batch_size(state::StepState{T}, h, t_out, kmax::Int) where {T}
    kmax <= 1 && return 1
    fh = state.fh
    dx = fh[F_DX]
    dx >= h && return 1                        # a rebuild is already due
    d4 = fh[F_DISP]
    dt = fh[F_DT]
    est_rebuild = d4 > zero(T) ? (h - dx) / d4 : T(Inf)
    est_output  = dt > zero(T) ? (T(t_out) - fh[F_TIME]) / dt : T(Inf)
    est = min(est_rebuild, est_output)
    isfinite(est) || return kmax
    return clamp(floor(Int, est) + 1, 1, kmax)
end

# Mirror the device state into the meta data after a read back.
function sync_meta_data!(SimMetaData, state::StepState)
    SimMetaData.Iteration = Int(state.ih[I_ITER])
    SimMetaData.TotalTime = state.fh[F_TIME]
    if state.ih[I_PHASE] == PHASE_NEED_DT
        # otherwise `dt` belongs to the step that still has to run
        SimMetaData.CurrentTimeStep = state.fh[F_DT]
    end
    return nothing
end

#---------------------------------------------------------------
# Time loop
#---------------------------------------------------------------

"""
Advance the simulation on the GPU until the next output time. Mirrors the CPU
`SimulationLoop` step for step; see `GPUKernels` for the fused kernels.

The host enqueues batches of steps (`batch_size`) and reads the device
resident step state back once per batch (`GPUStepState`). The device decides
when the cell list has to be rebuilt and when the output time is reached; the
host reacts to the read back state by rebuilding (`rebuild_cell_list!`) or
returning. With `GPUUseGraph` a step is replayed as a CUDA graph.
`GPUSyncTimers` forces one step per batch with a synchronization after every
phase so that the timer output is meaningful.
"""
function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                        SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                        SimConstants, gpu::GPUParticles{Dimensions, FloatType},
                        cl::CellListWorkspace, sup::GPUSupportArrays, red::ReductionWorkspace,
                        motion, state::StepState{FloatType}) where {Dimensions, FloatType, SMode, KMode, BMode, LMode,
                                                                    SDD <: SPHDensityDiffusion, SV <: SPHViscosity}
    HourGlass = SimMetaData.HourGlass
    h = SimKernel.h

    # The mode type parameters of the meta data select the kernel variants at
    # compile time; the time stepping scheme is a run time field set by
    # `RunSimulation`.
    FlagKernel     = Val(KMode === StoreKernelOutput)
    FlagShift      = Val(SMode === PlanarShifting)
    UseMDBC        = BMode === SimpleMDBC
    SingleNeighbor = SimMetaData.TimeSteppingMode isa SingleNeighborTimeStepping
    threads    = SimMetaData.GPUInteractionThreads
    nlanes     = SimMetaData.GPULanesPerParticle
    lanes      = Val(nlanes <= 0 ? choose_lanes(length(gpu)) : nlanes)
    bforces    = Val(SimMetaData.GPUBoundaryForces)

    timed     = SimMetaData.GPUSyncTimers
    use_graph = SimMetaData.GPUUseGraph && !timed
    kmax      = timed ? 1 : max(1, SimMetaData.GPUMaxStepsPerSync)

    ctx = (; gpu, cl, sup, red, motion, state, SimKernel, SimConstants, SimDensityDiffusion, SimViscosity,
             FlagKernel, FlagShift, UseMDBC, SingleNeighbor, threads, lanes, bforces, HourGlass)

    t_out = next_output_time(SimMetaData)
    set_output_time!(state, t_out)

    if !state.primed
        # No final step kernel has run yet: reduce the initial state for the
        # first time step. (`Positionₙ⁺` is still zero, so the displacement
        # term is meaningless, but the initial displacement bound already
        # forces a cell list rebuild before the first step.)
        launch_step_reduction!(red, gpu.Position, gpu.Velocity, gpu.Acceleration, sup.Positionₙ⁺, SimKernel)
        state.primed = true
    end

    generation = cl.generation
    while true
        K = batch_size(state, h, t_out, kmax)
        @timeit HourGlass "03 Launch Steps" for _ in 1:K
            if use_graph
                launch_step_graph!(ctx)
            else
                enqueue_step!(ctx, timed)
            end
        end

        @timeit HourGlass "13 Read Back State" begin
            readback!(state)
            sync_meta_data!(SimMetaData, state)
        end

        stop = state.ih[I_STOP]
        if stop == STOP_REBUILD
            @timeit HourGlass "02a Actual Calculate IndexCounter" begin
                rebuild_cell_list!(gpu, cl, SimKernel.H⁻¹)
                if cl.generation != generation
                    # the cell start buffer was reallocated: cached graphs point at the old one
                    invalidate_graphs!(state)
                    generation = cl.generation
                end
                resume_after_rebuild!(state)

                # The single neighbour scheme carries `dρdtI` and the
                # acceleration from the previous step, but the rebuild has
                # reordered the particles (and `dρdtI` is not permuted with
                # them): re-evaluate both at the accepted full state, as the
                # CPU does ("03a Rebuild MDBC" .. "03c Rebuild NeighborLoop").
                # Before the first step this is the CPU "00 Init" evaluation,
                # because the initial displacement bound forces a rebuild.
                if SingleNeighbor
                    enqueue_state_derivative!(ctx, state, timed, "02b Rebuild MDBC", "02c Rebuild NeighborLoop")
                end
            end
        elseif stop == STOP_OUTPUT || SimMetaData.TotalTime > t_out
            break
        end
    end

    return nothing
end

#---------------------------------------------------------------
# Driver
#---------------------------------------------------------------

"""
    RunSimulation(; SimGeometry, SimMetaData, SimConstants, SimKernel, SimLogger,
                    SimParticles, SimViscosity, SimDensityDiffusion, SimTimeStepping,
                    ParticleNormalsPath)

Run a complete simulation on the GPU. Same interface as the CPU version: the
shifting, kernel output, mDBC and log modes are the type parameters of
`SimMetaData`, the time stepping scheme is `SimTimeStepping`
(`SymplecticTimeStepping()` or `SingleNeighborTimeStepping()`). On return the
host `SimParticles` hold the final state (reordered by cell, like the CPU
version).
"""
function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}},
    SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
    SimConstants::SimulationConstants,
    SimKernel::SPHKernelInstance,
    SimLogger::SimulationLogger,
    SimParticles::StructArray,
    SimViscosity::SV,
    SimDensityDiffusion::SDD,
    SimTimeStepping::TimeSteppingMode,
    ParticleNormalsPath::Union{Nothing, String} = nothing
    ) where {Dimensions, FloatType, SMode, KMode, BMode, LMode, SV <: SPHViscosity, SDD <: SPHDensityDiffusion}

    CUDA.functional() || error("CUDA is not functional on this machine; use the CPU package SPHExample instead.")

    (; HourGlass) = SimMetaData

    SimMetaData.TimeSteppingMode = SimTimeStepping
    StoreLogOutput = LMode === StoreLog

    # Only the fields that end up in the output files are copied back from the
    # GPU at every output; the host arrays of the other fields stay untouched.
    output_vars     = resolve_output_variables!(SimMetaData)
    download_fields = Tuple(Symbol.(output_vars))

    TimeSteps = Vector{FloatType}()

    if BMode === SimpleMDBC
        ParticleNormalsPath === nothing && error("SimpleMDBC requires `ParticleNormalsPath`.")
        _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(Dimensions), FloatType, ParticleNormalsPath)
        for gi ∈ eachindex(GhostPoints)
            SimParticles.GhostPoints[gi]  = GhostPoints[gi]
            SimParticles.GhostNormals[gi] = GhostNormals[gi]
        end
    end

    if StoreLogOutput
        InitializeLogger(SimLogger, SimConstants, SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        with_logger(SimLogger.Logger) do
            dev = CUDA.device()
            @info "GPU: $(CUDA.name(dev)), compute capability $(CUDA.capability(dev)), " *
                  "$(round(CUDA.totalmem(dev) / 2^30; digits = 1)) GiB, CUDA.jl $(pkgversion(CUDA))"
            @info "GPU float type: $(FloatType)"
        end
    end

    if StoreLogOutput
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
                     reach = SimMetaData.GPUCellSubdivision, max_cells = SimMetaData.GPUMaxCells,
                     deterministic = SimMetaData.GPUDeterministicSort)
        motion = MotionArrays(SimGeometry, SimParticles)
        # Device resident loop state. The displacement bound starts above `h`
        # so that the cell list is built before the first step.
        state  = StepState{FloatType}(; time = SimMetaData.TotalTime, iteration = SimMetaData.Iteration,
                                        dx = one(FloatType) + SimKernel.h)
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
                                                             SimConstants, gpu, cl, sup, red, motion, state)
        push!(TimeSteps, SimMetaData.CurrentTimeStep)

        if StoreLogOutput
            LogStep(SimLogger, SimMetaData, HourGlass)
            SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        end

        SimMetaData.OutputIterationCounter += 1

        @timeit HourGlass "13 Download From GPU" begin
            wait_for_output()
            cid = download_particles!(SimParticles, gpu, cl.grid, download_fields;
                                      cells = SimMetaData.ExportGridCells)
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

            # Leave the complete final state on the host, not only the output
            # fields, so that callers can inspect every particle field.
            @timeit HourGlass "13 Download From GPU" download_particles!(SimParticles, gpu, cl.grid; cells = true)

            if !SimLogger.ToConsole
                finish!(SimMetaData.ProgressSpecification)
            end
            show(HourGlass, sortby = :name)
            show(HourGlass)

            AutoOpenParaview(SimMetaData, output.variable_names)

            UnicodeTimeStepsGraph = lineplot(1:length(TimeSteps), TimeSteps, title = "Time Steps [s] as a function of iteration",
                                             name = "Time Steps", xlabel = "Iterations [-]", ylabel = "Time Step Size [s]")

            if StoreLogOutput
                with_logger(SimLogger.Logger) do
                    @info "Cell list rebuilds: $(cl.nrebuilds), grid dims: $(cl.grid.dims)"
                    @info "Host read backs of the step state: $(state.readbacks) for $(SimMetaData.Iteration) steps, " *
                          "captured step graphs: $(length(state.graphs))"
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
