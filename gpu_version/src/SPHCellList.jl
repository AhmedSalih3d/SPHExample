"""
GPU time stepping driver.

`RunSimulation` has the same signature as the CPU version. Particles are
loaded on the host (`AllocateDataStructures`), uploaded once, integrated on
the GPU and copied back to the host `StructArray` only when an output is
written, so the host arrays always hold the most recently written state.
"""
module SPHCellList

export GPUParticles, GPUSupportArrays, MotionArrays, upload_particles, download_particles!,
       RunSimulation, SimulationLoop, enqueue_step!, batch_size, position_type, uses_pos_cells

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
using ..SPHMeasurements: MeasurementConfig, has_measurements,
    required_particle_fields
using ..OpenExternalPrograms
using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..GPUReductions
using ..GPUStepState
using ..GPUCellGrid
using ..GPUKernels
using ..GPUFloating

import StructArrays: StructArray

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

`T` is the working precision, `TP` the precision of the positions and ghost
node positions (`Float64` with `GPUDoublePosition`, otherwise `T`). Nothing
else is stored in `TP`: velocities, densities, accelerations and the kernel
sums stay in `T`, as in DualSPHysics.

Compact buffers omit acceleration/pressure reorder destinations: pressure is
reconstructed from sorted density, and acceleration is either overwritten by
the next derivative evaluation or reordered using idle velocity scratch.
Unused ghost and kernel-output fields may be empty device vectors. IDs and
group markers may use lossless 32-bit device storage; host and output fields
retain their original types.
"""
mutable struct GPUParticles{D, T, TP, S, I <: Integer, G <: Unsigned}
    Position::CuVector{SVector{D, TP}}
    Velocity::CuVector{SVector{D, T}}
    Density::CuVector{T}
    ID::CuVector{I}
    Type::CuVector{ParticleType}
    GroupMarker::CuVector{G}
    GhostPoints::CuVector{SVector{D, TP}}
    GhostNormals::CuVector{SVector{D, T}}
    Acceleration::CuVector{SVector{D, T}}
    Pressure::CuVector{T}
    CellID::CuVector{Int32}
    GhostIndex::CuVector{Int32}

    Kernel::CuVector{T}
    KernelGradient::CuVector{SVector{D, T}}

    scratch::S
end

function GPUParticles{D, T, TP, S}(Position, Velocity, Density, ID::CuVector{I}, Type,
                                  GroupMarker::CuVector{G}, GhostPoints, GhostNormals,
                                  Acceleration, Pressure, CellID, GhostIndex, Kernel,
                                  KernelGradient, scratch::S) where {D, T, TP, S, I <: Integer, G <: Unsigned}
    return GPUParticles{D, T, TP, S, I, G}(Position, Velocity, Density, ID, Type,
        GroupMarker, GhostPoints, GhostNormals, Acceleration, Pressure, CellID,
        GhostIndex, Kernel, KernelGradient, scratch)
end

Base.length(p::GPUParticles) = length(p.Position)

"""
    position_type(gpu) -> TP

Element precision of the positions stored on the device.
"""
position_type(::GPUParticles{D, T, TP}) where {D, T, TP} = TP

const PERSISTENT_FIELDS = (:Position, :Velocity, :Density,
                           :ID, :Type, :GroupMarker, :GhostPoints, :GhostNormals,
                           :Acceleration, :Pressure)

# Device fields that can be copied back into the host `StructArray`.
const DOWNLOADABLE_FIELDS = (:Velocity, :Density, :ID, :Type, :GroupMarker, :GhostPoints, :GhostNormals,
                             :Acceleration, :Pressure, :Kernel, :KernelGradient)
const OPTIONAL_PARTICLE_FIELDS = (:GhostPoints, :GhostNormals, :Kernel, :KernelGradient)

allocated_download_fields(gpu::GPUParticles) =
    filter(f -> !(f in OPTIONAL_PARTICLE_FIELDS && isempty(getproperty(gpu, f))), DOWNLOADABLE_FIELDS)

# Device copy of a host vector, converted to element type `E` when it differs.
_upload(::Type{E}, x::AbstractVector) where {E} = CuArray(convert(Vector{E}, x))

function metadata_storage_type(::Type{N}, ::Type{W}, values) where {N <: Integer, W <: Integer}
    return sizeof(N) < sizeof(W) && all(x -> typemin(N) <= x <= typemax(N), values) ? N : W
end

"""
    upload_particles(SimParticles::StructArray; position_type = eltype of the host positions,
                     compact_metadata = false, kernel_output = true, ghost_data = true,
                     reconstruct_fields = false) -> GPUParticles
    upload_particles(SimParticles::StructArray, SimMetaData) -> GPUParticles

Copy every stored field of the host particle array to the GPU. The ghost
node owners (`GhostIndex`) are listed from the host `GhostPoints`, so set
those before uploading. `position_type` is the element precision of the
device positions and ghost node positions (`Float64` with
`GPUDoublePosition`); the host arrays are converted when they differ. The
working precision is that of the host densities.

The one-argument form retains full buffers and host-width metadata by default.
`compact_metadata` selects `Int32` IDs and `UInt32` group markers only when
every value fits; otherwise each field independently retains its host width.
`kernel_output = false` or `ghost_data = false` omits those fields and their
reorder scratch. The metadata form selects these options from the simulation
modes and `GPUCompactBuffers`, and uses `position_float_type(SimMetaData)`.
`reconstruct_fields = true` omits acceleration and pressure reorder scratch;
rebuilds then require simulation constants to reconstruct pressure.
"""
function upload_particles(SimParticles::StructArray;
                          position_type::Type{TP} = eltype(eltype(SimParticles.Position)),
                          compact_metadata::Bool = false, kernel_output::Bool = true,
                          ghost_data::Bool = true, reconstruct_fields::Bool = false) where {TP}
    D = length(eltype(SimParticles.Position))
    T = eltype(SimParticles.Density)
    n = length(SimParticles)

    Position   = _upload(SVector{D, TP}, SimParticles.Position)
    I = compact_metadata ? metadata_storage_type(Int32, Int, SimParticles.ID) : Int
    G = compact_metadata ? metadata_storage_type(UInt32, UInt, SimParticles.GroupMarker) : UInt
    GhostIndex = ghost_data ? CuArray(Int32.(findall(!iszero, SimParticles.GhostPoints))) :
                             CuVector{Int32}(undef, 0)
    GhostPoints = ghost_data ? _upload(SVector{D, TP}, SimParticles.GhostPoints) :
                               CuVector{SVector{D, TP}}(undef, 0)
    GhostNormals = ghost_data ? CuArray(SimParticles.GhostNormals) :
                                CuVector{SVector{D, T}}(undef, 0)
    scratch = (
        Position      = similar(Position),
        Velocity      = CuVector{SVector{D, T}}(undef, n),
        Density       = CuVector{T}(undef, n),
        ID            = CuVector{I}(undef, n),
        Type          = CuVector{ParticleType}(undef, n),
        GroupMarker   = CuVector{G}(undef, n),
        GhostPoints   = similar(GhostPoints),
        GhostNormals  = similar(GhostNormals),
        Acceleration  = CuVector{SVector{D, T}}(undef, reconstruct_fields ? 0 : n),
        Pressure      = CuVector{T}(undef, reconstruct_fields ? 0 : n),
    )

    return GPUParticles{D, T, TP, typeof(scratch)}(
        Position,
        CuArray(SimParticles.Velocity),
        CuArray(SimParticles.Density),
        _upload(I, SimParticles.ID),
        CuArray(SimParticles.Type),
        _upload(G, SimParticles.GroupMarker),
        GhostPoints,
        GhostNormals,
        CuArray(SimParticles.Acceleration),
        CuArray(SimParticles.Pressure),
        CUDA.zeros(Int32, n),
        GhostIndex,
        kernel_output ? CuArray(SimParticles.Kernel) : CuVector{T}(undef, 0),
        kernel_output ? CuArray(SimParticles.KernelGradient) : CuVector{SVector{D, T}}(undef, 0),
        scratch,
    )
end

function upload_particles(SimParticles::StructArray, meta::SimulationMetaData{D, T, S, K, B}) where {D, T, S, K, B}
    compact = meta.GPUCompactBuffers
    return upload_particles(SimParticles; position_type = position_float_type(meta),
        compact_metadata = compact, kernel_output = !compact || K === StoreKernelOutput,
        ghost_data = !compact || B !== NoMDBC, reconstruct_fields = compact)
end

# Host copy of a device vector; converted element-wise when the host element
# type differs (positions of a different precision than the host arrays).
function _download!(dst::AbstractVector, src::CuVector)
    if eltype(dst) === eltype(src)
        copyto!(dst, src)
    else
        copyto!(dst, Array(src))
    end
    return dst
end

"""
    download_particles!(SimParticles, gpu, grid, fields = allocated downloadable fields; cells = false)

Copy `Position` and the device fields named in `fields` back into the host
`StructArray`. Host fields that are not listed are left untouched, so pass
exactly the fields that are written to the output. With `cells = true` the
cell of every particle is also stored as a `CartesianIndex`. Returns the cell
ids as a host vector (empty unless `cells = true`). Fields whose host element
type differs from the device type (positions when the host array was not
allocated for `GPUDoublePosition`) are converted.
Omitted device fields are excluded from the default field list; requesting
one explicitly throws `ArgumentError`.
"""
function download_particles!(SimParticles::StructArray, gpu::GPUParticles{D, T}, grid::CellGrid{D},
                             fields = allocated_download_fields(gpu);
                             cells::Bool = false) where {D, T}
    foreach(f -> require_particle_field(gpu, f), fields)
    _download!(SimParticles.Position, gpu.Position)
    for f in fields
        _download!(getproperty(SimParticles, f), getproperty(gpu, f))
    end
    cells || return Int32[]

    cid = Array(gpu.CellID)
    store_cells!(SimParticles, grid, cid)
    return cid
end

function require_particle_field(gpu::GPUParticles, field::Symbol)
    array = getproperty(gpu, field)
    length(array) == length(gpu) ||
        throw(ArgumentError("GPU particle field $field was not allocated; enable its buffer when uploading particles."))
    return array
end

# Store the cell of every particle in the host `StructArray` from a host copy
# of the device cell ids and return the distinct cells in cell order.
function store_cells!(SimParticles::StructArray, grid::CellGrid{D}, cid::AbstractVector{Int32}) where {D}
    @inbounds for i in eachindex(cid)
        l = local_coords(grid, cid[i])
        SimParticles.Cells[i] = CartesianIndex(ntuple(d -> Int(l[d] + grid.origin[d]), Val(D)))
    end
    return unique_cells_host(grid, cid)
end

#---------------------------------------------------------------
# Asynchronous download of an output frame
#---------------------------------------------------------------

"""
Page-locked host staging buffers for the fields of an output frame and the
event that marks their arrival.

`copyto!(::Array, ::CuArray)` waits for the device before it copies, so a
download after every output interval drained the GPU pipeline and left the
GPU idle for the whole host side of an output (copy, cell bookkeeping, log
line, task spawn), several milliseconds per frame. Instead,
`enqueue_download!` issues asynchronous device to host copies into these
buffers on the stream of the step kernels: they read the state at the output
time, the kernels of the next interval (which overwrite the device arrays)
queue up behind them and the host continues at once. `finish_download!`,
called by the output writer, waits on the event and moves the buffers into
the host `StructArray`, converting fields whose host element type differs
(`Float32` host positions of a `GPUDoublePosition` run).

Julia arrays are not registered as pinned memory: small arrays share memory
pages with other objects and CUDA cannot register a page twice. The buffers
are separate page-locked allocations; release them with `free!`.
"""
struct OutputDownload{D}
    fields::Vector{Symbol}          # `Position` and the downloaded fields, in the order of `buffers`
    buffers::Vector{Vector}         # pinned staging buffer per field, in the device element type
    cid::Vector{Int32}              # pinned cell ids (empty unless the cells are exported)
    memory::Vector{CUDA.HostMemory} # the page-locked allocations behind the buffers
    event::CuEvent                  # recorded behind the copies of the current frame
end

function OutputDownload(gpu::GPUParticles{D}, fields; cells::Bool) where {D}
    foreach(f -> require_particle_field(gpu, f), fields)
    memory = CUDA.HostMemory[]
    function pinned(::Type{T}, n) where {T}
        n == 0 && return T[]
        mem = CUDA.alloc(CUDA.HostMemory, n * sizeof(T))
        push!(memory, mem)
        return unsafe_wrap(Array, convert(Ptr{T}, mem), n)
    end
    names   = Symbol[:Position, fields...]
    buffers = Vector[pinned(eltype(getproperty(gpu, f)), length(gpu)) for f in names]
    cid     = cells ? pinned(Int32, length(gpu)) : Int32[]
    return OutputDownload{D}(names, buffers, cid, memory, CuEvent(CUDA.EVENT_DISABLE_TIMING))
end

# Asynchronous device to host copy into page-locked memory on the current stream.
function copy_async!(dst::Vector{T}, src::CuVector{T}) where {T}
    length(dst) == length(src) || throw(DimensionMismatch("host buffer of length $(length(dst)) for $(length(src)) elements"))
    GC.@preserve dst src unsafe_copyto!(pointer(dst), pointer(src), length(src); async = true)
    return dst
end

"""
    enqueue_download!(download, gpu)

Enqueue the asynchronous copies of the current device state into the staging
buffers and record the event behind them. Returns at once; the copies complete
in stream order, before any kernel enqueued afterwards runs.
"""
function enqueue_download!(dl::OutputDownload, gpu::GPUParticles)
    for (f, buf) in zip(dl.fields, dl.buffers)
        copy_async!(buf, getproperty(gpu, f))
    end
    isempty(dl.cid) || copy_async!(dl.cid, gpu.CellID)
    CUDA.record(dl.event)
    return dl
end

"""
    finish_download!(download, SimParticles, grid) -> UniqueCells

Wait for the copies of the frame and move them into the host `StructArray`
(fields with a different host element type are converted). With exported
cells the particle cells are stored as well and the distinct cells are
returned; `grid` must be the cell grid at the time of `enqueue_download!`.
Runs on the writer task.
"""
function finish_download!(dl::OutputDownload{D}, SimParticles::StructArray, grid::CellGrid{D}) where {D}
    CUDA.synchronize(dl.event)
    for (f, buf) in zip(dl.fields, dl.buffers)
        copyto!(getproperty(SimParticles, f), buf)
    end
    isempty(dl.cid) && return CartesianIndex{D}[]
    return store_cells!(SimParticles, grid, dl.cid)
end

"""
    free!(download)

Release the page-locked staging buffers; the download must not be used afterwards.
"""
function free!(dl::OutputDownload)
    foreach(CUDA.free, dl.memory)
    empty!(dl.memory)
    return nothing
end

"""
Sets of staging buffers (`OutputDownload`) that rotate between the
simulation thread and the output writer. With one set the simulation thread
had to wait for the previous frame to be written before it could enqueue the
next download; with two, a frame whose write takes longer than an output
interval (a flush of the buffered file writer, a slow disk) no longer stalls
the GPU unless the writer falls a whole ring behind.
"""
const OUTPUT_STAGING_FRAMES = 2

"""
One output frame handed from the simulation thread to the writer task: the
staging buffers that receive its copies, the output counter and time, the
cell grid at the time of the download (the simulation thread may rebuild the
cell list while the frame is written).
"""
struct OutputFrameJob{D}
    download::OutputDownload{D}
    counter::Int
    time::Float64
    grid::CellGrid{D}
end

# Keep the lightweight collector with the simulation pool so a blocking disk
# write on the other pool cannot prevent downloads from being collected.
function spawn_collector(f)
    if Threads.threadpool() === :interactive
        return Threads.@spawn :interactive f()
    end
    return Threads.@spawn :default f()
end

# Spawn the output writer on a thread pool other than the one of the calling
# thread, so that the file write never shares a thread with the kernel
# launches. With the default `-t 1,1` of Julia 1.12 the script runs on the
# interactive thread (thread 1) and the writer goes to the default thread;
# with `-t N,0` there is no other pool and the writer competes with the
# simulation for the default threads.
function spawn_writer(f)
    here = Threads.threadpool()
    if here === :interactive && Threads.nthreads(:default) >= 1
        return Threads.@spawn :default f()
    elseif here === :default && Threads.nthreads(:interactive) >= 1
        return Threads.@spawn :interactive f()
    else
        return Threads.@spawn f()
    end
end

"""
Device arrays that are recomputed every step (half step state and shifting
terms). They are never reordered because they are overwritten after every
cell list update. Reciprocal densities are computed on demand by the
interaction kernel, without cached vectors. `Positionₙ⁺` is in the
position precision `TP`. When that differs from the working precision `T`,
`PosCells` and `PosCellsₙ⁺` hold the cell relative form (`PosCell`) of the
start-of-step and half step positions for the pair loops; otherwise both are
`nothing` and the pair loops read the positions directly.
With `ReusePredictors`, the half-step position/velocity arrays are omitted.
The driver instead uses the existing reorder scratch, which is idle between
rebuilds. Fixed entries are initialized after sorting and left untouched during
steps. Neighbor reads remain dense, without an index map or extra allocation.
"""
struct GPUSupportArrays{D, T, TP, PC, BD}
    dρdtI::CuVector{T}
    Velocityₙ⁺::CuVector{SVector{D, T}}
    Positionₙ⁺::CuVector{SVector{D, TP}}
    ρₙ⁺::CuVector{T}
    ∇Cᵢ::CuVector{SVector{D, T}}
    ∇◌rᵢ::CuVector{T}
    PosCells::PC
    PosCellsₙ⁺::PC
    boundary_data::BD
    ReusePredictors::Bool
end

"""
    GPUSupportArrays{D, T}(n; position_type = T, updated_mdbc = false, shifting = true)
    GPUSupportArrays{D, T}(n, SimMetaData)
    GPUSupportArrays{D, T}(gpu, SimMetaData)

Support arrays for `n` particles in the working precision `T`, with positions
of precision `position_type`. The cell relative positions are allocated
exactly when `position_type !== T`. `updated_mdbc` allocates the boundary
activation mask and mirrored/tangential interaction velocities; these are
scratch fields recomputed after sorting and before each interaction stage.
`shifting = false` omits the two shifting work arrays. The metadata form
selects the position precision, updated-mDBC and shifting buffers from the
simulation modes and `GPUCompactBuffers`.
The device-particle form omits separate position/velocity predictors when compact
buffers are enabled. Pass `support` to `rebuild_cell_list!` to initialize fixed
entries in the reorder scratch after sorting. Density predictors remain allocated
for all particles.
"""
function GPUSupportArrays{D, T}(n::Integer; position_type::Type{TP} = T, updated_mdbc::Bool = false,
                              shifting::Bool = true, reuse_predictors::Bool = false) where {D, T, TP}
    pc = TP === T ? nothing : CuVector{PosCell{D, T}}(undef, n)
    pcₙ⁺ = TP === T ? nothing : CuVector{PosCell{D, T}}(undef, n)
    boundary_data = updated_mdbc ? (; active = CUDA.ones(Bool, n), velocity = CUDA.zeros(SVector{D, T}, n),
                                    tangent_velocity = CUDA.zeros(SVector{D, T}, n)) : nothing
    return GPUSupportArrays{D, T, TP, typeof(pc), typeof(boundary_data)}(
        CUDA.zeros(T, n),
        CUDA.zeros(SVector{D, T}, reuse_predictors ? 0 : n),
        CUDA.zeros(SVector{D, TP}, reuse_predictors ? 0 : n),
        CUDA.zeros(T, n),
        CUDA.zeros(SVector{D, T}, shifting ? n : 0),
        CUDA.zeros(T, shifting ? n : 0),
        pc,
        pcₙ⁺,
        boundary_data,
        reuse_predictors,
    )
end

function GPUSupportArrays{D, T}(gpu::GPUParticles{D, T}, meta::SimulationMetaData{D, T, S, K, B}) where {D, T, S, K, B}
    position_type(gpu) === position_float_type(meta) ||
        throw(ArgumentError("GPU positions and support-array metadata must use the same precision"))
    return GPUSupportArrays{D, T}(length(gpu); position_type = position_type(gpu),
        updated_mdbc = B === UpdatedMDBC, shifting = !meta.GPUCompactBuffers || S === PlanarShifting,
        reuse_predictors = meta.GPUCompactBuffers)
end

@inline function predictor_arrays(sup::GPUSupportArrays, gpu::GPUParticles)
    return sup.ReusePredictors ? (gpu.scratch.Position, gpu.scratch.Velocity) :
                                (sup.Positionₙ⁺, sup.Velocityₙ⁺)
end

function fixed_predictors_kernel!(Positionₙ⁺, Velocityₙ⁺, Position, Velocity, types, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds if types[i] == Fixed
        Positionₙ⁺[i] = Position[i]
        Velocityₙ⁺[i] = Velocity[i]
    end
    return nothing
end

function initialize_fixed_predictors!(sup::GPUSupportArrays, gpu::GPUParticles)
    sup.ReusePredictors || return nothing
    n = length(gpu)
    n == 0 && return nothing
    position_half, velocity_half = predictor_arrays(sup, gpu)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) fixed_predictors_kernel!(
        position_half, velocity_half, gpu.Position, gpu.Velocity, gpu.Type, Int32(n))
    return nothing
end

function GPUSupportArrays{D, T}(n::Integer, meta::SimulationMetaData{D, T, S, K, B}) where {D, T, S, K, B}
    return GPUSupportArrays{D, T}(n; position_type = position_float_type(meta),
        updated_mdbc = B === UpdatedMDBC, shifting = !meta.GPUCompactBuffers || S === PlanarShifting)
end

"""
    uses_pos_cells(sup) -> Bool

Whether the pair loops read cell relative positions (positions of a higher
precision than the working precision).
"""
uses_pos_cells(sup::GPUSupportArrays) = sup.PosCells !== nothing

"""
    MotionArrays(SimGeometry, SimParticles) -> NamedTuple of device arrays

Prescribed motion parameters indexed by group marker, for use in kernels.
"""
function MotionArrays(SimGeometry::Vector{SPHGeometry{D, T}}, SimParticles) where {D, T}
    ngroups = max(1, Int(maximum(SimParticles.GroupMarker; init = 0)))
    has       = zeros(Bool, ngroups)
    velocity  = zeros(T, ngroups)
    start     = zeros(T, ngroups)
    duration  = zeros(T, ngroups)
    direction = zeros(SVector{D, T}, ngroups)
    move_particles = ones(Bool, ngroups)
    for geom in SimGeometry
        m = geom.Motion
        if m !== nothing
            g = geom.GroupMarker
            has[g]       = true
            velocity[g]  = m.Velocity
            start[g]     = m.StartTime
            duration[g]  = m.Duration
            direction[g] = m.Direction
            move_particles[g] = m.MoveParticles
        end
    end
    active = any(has) && any(==(Moving), SimParticles.Type)
    return (active = active, has = CuArray(has), velocity = CuArray(velocity), start = CuArray(start),
            duration = CuArray(duration), direction = CuArray(direction),
            move_particles = CuArray(move_particles))
end

#---------------------------------------------------------------
# Cell list rebuild with reordering of the persistent fields
#---------------------------------------------------------------

function reconstruct_fields_kernel!(Pressure, Acceleration, acceleration_sorted, pressure_sorted, Density, types,
                                    constants, boundary_mode, ::Val{PreserveAcceleration},
                                    ::Val{PreservePressure}, n::Int32) where {PreserveAcceleration, PreservePressure}
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        Pressure[i] = PreservePressure ? pressure_sorted[i] :
            GPUKernels.particle_pressure(Density[i], types[i], constants.c₀, constants.ρ₀, boundary_mode)
        if PreserveAcceleration
            Acceleration[i] = acceleration_sorted[i]
        end
    end
    return nothing
end

"""
    rebuild_cell_list!(gpu, cl, InverseCutOff; constants = nothing, boundary_mode = NoMDBC(),
                       discard_acceleration = false, preserve_pressure = false,
                       prepared = false, support = nothing, floating = nothing)

Sort persistent particle state. When pressure reorder scratch is omitted,
`constants` is required and `boundary_mode` selects the EOS (linear for
UpdatedMDBC walls, gamma-seven otherwise). Acceleration is preserved by default,
using idle velocity reorder scratch when its own destination is omitted.
`discard_acceleration = true` leaves acceleration in the old order: only use it
when a derivative evaluation will overwrite every entry before any reader,
including timestep reductions, floating forces and downloads. Single-neighbor
carried derivatives and body acceleration caches must also be refreshed.
`preserve_pressure = true` instead reorders pressure using idle density scratch,
as used at startup to preserve the host EOS rounding before the first step.
"""
function rebuild_cell_list!(gpu::GPUParticles, cl::CellListWorkspace, InverseCutOff;
                            prepared::Bool = false, support = nothing, floating = nothing,
                            constants = nothing, boundary_mode = NoMDBC(), discard_acceleration::Bool = false,
                            preserve_pressure::Bool = false)
    s = gpu.scratch
    reconstruct = isempty(s.Pressure) && !isempty(gpu.Pressure)
    reconstruct && constants === nothing &&
        throw(ArgumentError("Rebuilding without pressure reorder scratch requires simulation constants."))
    fields = filter(f -> !isempty(getproperty(s, f)), PERSISTENT_FIELDS)
    srcs = (map(f -> getproperty(gpu, f), fields)..., cl.CellIDScratch)
    dsts = (map(f -> getproperty(s, f), fields)..., gpu.CellID)

    grid = update_cell_list!(cl, gpu.Position, InverseCutOff, srcs, dsts; prepared)

    # Swap the two sets of persistent arrays.
    gpu.scratch = (
        Position = gpu.Position, Velocity = gpu.Velocity, Density = gpu.Density,
        ID = gpu.ID, Type = gpu.Type,
        GroupMarker = gpu.GroupMarker, GhostPoints = gpu.GhostPoints, GhostNormals = gpu.GhostNormals,
        Acceleration = reconstruct ? s.Acceleration : gpu.Acceleration,
        Pressure = reconstruct ? s.Pressure : gpu.Pressure,
    )
    gpu.Position      = s.Position
    gpu.Velocity      = s.Velocity
    gpu.Density       = s.Density
    gpu.ID            = s.ID
    gpu.Type          = s.Type
    gpu.GroupMarker   = s.GroupMarker
    gpu.GhostPoints   = s.GhostPoints
    gpu.GhostNormals  = s.GhostNormals
    if reconstruct
        # The old velocity buffer is idle until fixed predictors are initialized.
        preserve = !discard_acceleration
        n = Int32(length(gpu))
        sources = ((preserve ? (gpu.Acceleration,) : ())..., (preserve_pressure ? (gpu.Pressure,) : ())...)
        destinations = ((preserve ? (gpu.scratch.Velocity,) : ())..., (preserve_pressure ? (gpu.scratch.Density,) : ())...)
        if preserve || preserve_pressure
            @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) gather_kernel!(
                cl.Perm, sources, destinations, n)
        end
        @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) reconstruct_fields_kernel!(
            gpu.Pressure, gpu.Acceleration, gpu.scratch.Velocity, gpu.scratch.Density, gpu.Density, gpu.Type,
            constants, boundary_mode, Val(preserve), Val(preserve_pressure), n)
    else
        gpu.Acceleration = s.Acceleration
        gpu.Pressure = s.Pressure
    end

    # The reorder moved the ghost node owners: relist them in the new (cell)
    # order. Their number never changes, so the list keeps its length and
    # device pointer, and captured graphs stay valid.
    isempty(gpu.GhostIndex) || compact_nonzero!(cl, gpu.GhostPoints, gpu.GhostIndex)
    floating === nothing || update_floating_indices!(floating, gpu.Type, cl)
    support === nothing || initialize_fixed_predictors!(support, gpu)

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
    enqueue_pos_cells!(ctx, step, timed)

Enqueue the cell relative form of the start-of-step positions
(`launch_pos_cells!`) when the pair loops read it; nothing otherwise. Must
precede every kernel that scans neighbours of the start-of-step positions
(the mDBC correction and the first neighbour loop) because the positions
change between those kernels and the previous ones (final step, motion,
rebuild).
"""
function enqueue_pos_cells!(ctx, step, timed::Bool)
    (; gpu, cl, sup, SimKernel, HourGlass) = ctx
    uses_pos_cells(sup) || return nothing
    @phase HourGlass "03d Cell Relative Positions" timed launch_pos_cells!(
        sup.PosCells, gpu.Position, gpu.CellID, cl.grid_dev, step, SimKernel)
    return nothing
end

# Recompute scratch boundary data after every reorder and at both evaluated states.
function enqueue_boundary_correction!(ctx, step, half::Bool)
    (; gpu, sup, cl, SimKernel, SimConstants, threads, lanes) = ctx
    density = half ? sup.ρₙ⁺ : gpu.Density
    position, velocity = half ? predictor_arrays(sup, gpu) : (gpu.Position, gpu.Velocity)
    pairs = half ? sup.PosCellsₙ⁺ : sup.PosCells
    if get(ctx, :boundary_mode, SimpleMDBC()) isa UpdatedMDBC
        GPUKernels.launch_updated_mdbc!(density, gpu.Pressure, position, velocity,
            gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart, cl.grid_dev,
            step, SimKernel, SimConstants, sup.boundary_data; threads, lanes, pos_cells = pairs,
            motion = get(ctx, :motion, nothing), groups = gpu.GroupMarker)
    else
        launch_mdbc!(density, gpu.Pressure, position, gpu.GhostPoints, gpu.GhostIndex, gpu.Type,
            cl.CellStart, cl.grid_dev, step, SimKernel, SimConstants; threads, lanes, pos_cells = pairs)
    end
    return nothing
end

"""
    enqueue_state_derivative!(ctx, step, timed, mdbc_name, loop_name)

Enqueue the evaluation of the start-of-step state: the cell relative
positions (with `GPUDoublePosition`), the mDBC correction of the boundary
densities together with the pressure of the corrected density
(`launch_mdbc!`, with `SimpleMDBC`) and the
neighbour loop that produces `dρdtI` and the acceleration. The symplectic
scheme runs this at every step as its first neighbour loop. The single
neighbour scheme carries the corrector derivative of the previous step into
the predictor instead and only runs this to start that derivative: before
the first step and after a cell list rebuild (`SimulationLoop`). `step`
gates every kernel; `mdbc_name` and `loop_name` are the timer labels.
"""
function enqueue_state_derivative!(ctx, step, timed::Bool, mdbc_name::AbstractString, loop_name::AbstractString; prepare_motion::Bool = false)
    (; gpu, cl, sup, SimKernel, SimConstants, SimDensityDiffusion, SimViscosity,
       FlagKernel, FlagShift, UseMDBC, threads, lanes, bforces, HourGlass) = ctx
    grid      = cl.grid_dev
    CellStart = cl.CellStart

    if UseMDBC
        enqueue_pos_cells!(ctx, step, timed)
        @phase HourGlass mdbc_name timed enqueue_boundary_correction!(ctx, step, false)
    end

    @phase HourGlass loop_name timed begin
        if !UseMDBC
            launch_prepare_interactions!(gpu.Position, step;
                pos_cells = sup.PosCells,
                CellID = gpu.CellID, grid = grid, H = SimKernel.H,
                Velocity = gpu.Velocity, ParticleType = gpu.Type, GroupMarker = gpu.GroupMarker,
                motion = prepare_motion ? ctx.motion : nothing)
        end
        launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                             gpu.Position, gpu.Density, gpu.Pressure,
                             gpu.Velocity, gpu.Type, CellStart, gpu.CellID, grid, step,
                             SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                             FlagKernel, FlagShift; threads = threads, lanes = lanes,
                             boundary_forces = bforces, pos_cells = sup.PosCells, boundary_data = sup.boundary_data,
                             radial = get(ctx, :radial, Val(false)),
                             scalar_sps = get(ctx, :scalar_sps, Val(false)))
    end
    floating = get(ctx, :floating, nothing)
    if get(ctx, :SingleNeighbor, false) && floating !== nothing && floating.active
        @phase HourGlass "02d Rebuild Floating Derivative" timed begin
            launch_floating_forces!(floating, gpu.Acceleration, gpu.Position, gpu.Type, gpu.GroupMarker,
                                    floating.center, SimConstants.m₀, step)
            launch_floating_derivative!(floating, step, SimConstants.g)
        end
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
rebuild, see `SimulationLoop`.) With floating bodies, each evaluated
neighbour loop is followed by the force on the bodies and their predictor or
corrector update, and the half step and final step by the rigid placement of
the body particles (see `GPUFloating`). The single-neighbor predictor uses
cached body linear/angular acceleration from the previous corrector; startup
and rebuild evaluations refresh that cache at the accepted full state.
"""
function enqueue_step!(ctx, timed::Bool)
    (; gpu, cl, sup, red, motion, state, SimKernel, SimConstants, SimDensityDiffusion, SimViscosity,
       FlagKernel, FlagShift, UseMDBC, SingleNeighbor, threads, lanes, bforces, HourGlass) = ctx
    grid      = cl.grid_dev
    CellStart = cl.CellStart
    floating  = get(ctx, :floating, nothing)
    floats    = floating !== nothing && floating.active
    position_half, velocity_half = predictor_arrays(sup, gpu)

    # dt of this step from the reduction of the previous one; also decides
    # whether the step may run at all (cell list rebuild, output time).
    @phase HourGlass "01 Update TimeStep" timed launch_finish!(state, red, SimKernel, SimConstants)

    # The pressure of the start-of-step density was already computed by the
    # final kernel of the previous step (and on the host before the first).
    # DBC has no neighbor evaluation between motion and per-particle preparation.
    # mDBC needs the moved geometry before its separate boundary correction.
    fused_motion = !UseMDBC
    if motion.active && !fused_motion
        @phase HourGlass "Motion" timed launch_motion!(gpu.Position, gpu.Velocity, gpu.Type, gpu.GroupMarker,
                                                       motion, state)
    end

    if SingleNeighbor
        # CPU "02 Apply MDBC before Half TimeStep": the boundary densities the
        # predictor and the final density update start from are corrected
        # every step, the carried derivative is kept.
        if UseMDBC
            enqueue_pos_cells!(ctx, state, timed)
            @phase HourGlass "04a NeighborLoopMDBC before Half TimeStep" timed enqueue_boundary_correction!(ctx, state, false)
        end
    else
        enqueue_state_derivative!(ctx, state, timed, "04a First NeighborLoopMDBC", "04 First NeighborLoop";
            prepare_motion = fused_motion && motion.active)
    end

    if floats
        @phase HourGlass "05a Floating Predictor" timed begin
            if !SingleNeighbor
                launch_floating_forces!(floating, gpu.Acceleration, gpu.Position, gpu.Type, gpu.GroupMarker,
                                        floating.center, SimConstants.m₀, state)
            end
            launch_floating_update!(floating, state, SimConstants.g, false; carried = SingleNeighbor)
        end
    end

    @phase HourGlass "05b Update To Half TimeStep" timed launch_half_step!(
        position_half, velocity_half, sup.ρₙ⁺, gpu.Pressure,
        gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, sup.dρdtI,
        gpu.Type, gpu.GroupMarker, motion, state, SimConstants;
        pos_cells = sup.PosCellsₙ⁺, CellID = gpu.CellID, grid = grid, SimKernel = SimKernel,
        boundary_mode = get(ctx, :boundary_mode, SimpleMDBC()),
        reuse_predictors = sup.ReusePredictors,
        first_motion = SingleNeighbor && fused_motion && motion.active)

    if floats
        @phase HourGlass "05c Floating Half TimeStep" timed launch_floating_particles!(
            floating, position_half, velocity_half, gpu.Position, gpu.Type, gpu.GroupMarker, state, false;
            pos_cells = sup.PosCellsₙ⁺, CellID = gpu.CellID, grid = grid, H = SimKernel.H)
    end

    if get(ctx, :boundary_mode, SimpleMDBC()) isa UpdatedMDBC
        @phase HourGlass "07 Updated mDBC corrector" timed begin
            enqueue_boundary_correction!(ctx, state, true)
        end
    end

    # Corrector: every term, including the viscosity and density diffusion
    # models, is evaluated at the predictor state.
    @phase HourGlass "08 Second NeighborLoop" timed launch_interactions!(
        sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, sup.∇Cᵢ, sup.∇◌rᵢ,
        position_half, sup.ρₙ⁺, gpu.Pressure,
        velocity_half, gpu.Type, CellStart, gpu.CellID, grid, state,
        SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
        FlagKernel, FlagShift; threads = threads, lanes = lanes,
        boundary_forces = bforces, pos_cells = sup.PosCellsₙ⁺, boundary_data = sup.boundary_data,
        radial = get(ctx, :radial, Val(false)),
        scalar_sps = get(ctx, :scalar_sps, Val(false)))

    if floats
        @phase HourGlass "10 Floating Corrector" timed begin
            launch_floating_forces!(floating, gpu.Acceleration, position_half, gpu.Type, gpu.GroupMarker,
                                    floating.center_half, SimConstants.m₀, state)
            launch_floating_update!(floating, state, SimConstants.g, true)
        end
    end

    @phase HourGlass "11 Update To Final TimeStep" timed launch_final_step!(
        gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density, gpu.Pressure,
        sup.dρdtI, sup.ρₙ⁺, position_half, velocity_half, gpu.Type,
        sup.∇Cᵢ, sup.∇◌rᵢ, state, SimKernel, SimConstants, red, FlagShift;
        boundary_mode = get(ctx, :boundary_mode, SimpleMDBC()),
        ghost_points = gpu.GhostPoints, ghost_normals = gpu.GhostNormals,
        commit_step = red.nblocks == 1)

    # Rigid final placement reads neither time nor phase, so a one-block
    # integration may commit before this launch without changing body placement.
    if floats
        @phase HourGlass "11b Floating Final TimeStep" timed launch_floating_particles!(
            floating, gpu.Position, gpu.Velocity, gpu.Position, gpu.Type, gpu.GroupMarker, state, true)
    end

    if red.nblocks != 1
        @phase HourGlass "12 Update MetaData" timed launch_commit!(state)
    end
    return nothing
end

"""
    launch_step_graph!(ctx, steps = 1)

Replay the launch sequence of a step batch as a CUDA graph. The graph bakes in the
device pointers of its arguments, and the persistent particle arrays are
swapped with their scratch copies at every cell list rebuild, so one graph
is kept per set of pointers and batch length, keyed by the position array,
the reallocation generation of the cell list and the number of steps. The first call
captures and instantiates the graph; a capture that fails (kernels that
still have to be compiled) falls back to direct launches for that batch.
"""
function launch_step_graph!(ctx, steps::Int = 1)
    state = ctx.state
    key   = (UInt(pointer(ctx.gpu.Position)), ctx.cl.generation, steps)
    exec  = get(state.graphs, key, nothing)
    if exec === nothing
        # Output workers query events recorded outside this capture on their
        # own streams. Global capture mode would reject those concurrent calls.
        graph = CUDA.capture(; flags = CUDA.STREAM_CAPTURE_MODE_RELAXED,
                               throw_error = false) do
            for _ in 1:steps
                enqueue_step!(ctx, false)
            end
        end
        if graph === nothing
            for _ in 1:steps
                enqueue_step!(ctx, false)
            end
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
phase so that the timer output is meaningful. `floating` is the state of the
floating bodies (`FloatingArrays`), or `nothing` for a case without them.
"""
function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                        SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                        SimConstants, gpu::GPUParticles{Dimensions, FloatType},
                        cl::CellListWorkspace, sup::GPUSupportArrays, red::ReductionWorkspace,
                        motion, state::StepState{FloatType};
                        floating = nothing) where {Dimensions, FloatType, SMode, KMode, BMode, LMode,
                                                   SDD <: SPHDensityDiffusion, SV <: SPHViscosity}
    HourGlass = SimMetaData.HourGlass
    h = SimKernel.h

    # The mode type parameters of the meta data select the kernel variants at
    # compile time; the time stepping scheme is a run time field set by
    # `RunSimulation`.
    FlagKernel     = Val(KMode === StoreKernelOutput)
    FlagShift      = Val(SMode === PlanarShifting)
    UseMDBC        = BMode !== NoMDBC
    if KMode === StoreKernelOutput
        require_particle_field(gpu, :Kernel)
        require_particle_field(gpu, :KernelGradient)
    end
    if UseMDBC
        require_particle_field(gpu, :GhostPoints)
        require_particle_field(gpu, :GhostNormals)
    end
    if SMode === PlanarShifting
        length(sup.∇Cᵢ) == length(gpu) && length(sup.∇◌rᵢ) == length(gpu) ||
            throw(ArgumentError("GPU shifting buffers were not allocated; enable shifting when allocating support arrays."))
    end
    boundary_mode = BMode()
    SingleNeighbor = SimMetaData.TimeSteppingMode isa SingleNeighborTimeStepping
    threads    = SimMetaData.GPUInteractionThreads
    nlanes     = SimMetaData.GPULanesPerParticle
    lanes      = Val(nlanes <= 0 ? choose_lanes(length(gpu), Val(Dimensions)) : nlanes)
    bforces    = Val(SimMetaData.GPUBoundaryForces)
    use_radial = SimMetaData.GPURadialInteractions && radial_interactions_supported(SimKernel, SimDensityDiffusion, SimViscosity)
    use_scalar_sps = SimMetaData.GPUScalarSPSViscosity && scalar_sps_supported(SimKernel, SimViscosity)
    interaction_mode_changed = state.radial_interactions != use_radial || state.scalar_sps_viscosity != use_scalar_sps
    if interaction_mode_changed
        invalidate_graphs!(state)
        state.radial_interactions = use_radial
        state.scalar_sps_viscosity = use_scalar_sps
    end
    radial = Val(use_radial)
    scalar_sps = Val(use_scalar_sps)

    timed     = SimMetaData.GPUSyncTimers
    use_graph = SimMetaData.GPUUseGraph && !timed
    kmax      = timed ? 1 : max(1, SimMetaData.GPUMaxStepsPerSync)

    ctx = (; gpu, cl, sup, red, motion, floating, state, SimKernel, SimConstants, SimDensityDiffusion,
             SimViscosity, FlagKernel, FlagShift, UseMDBC, SingleNeighbor, threads, lanes, bforces, radial, scalar_sps, HourGlass, boundary_mode)

    t_out = next_output_time(SimMetaData)
    set_output_time!(state, t_out)

    if SingleNeighbor && interaction_mode_changed && state.primed && cl.nrebuilds > 0
        # The carried derivative belongs to the previous interaction operator.
        step = HostStep(state.fh[F_DT], state.fh[F_TIME])
        enqueue_state_derivative!(ctx, step, timed, "00a Refresh Interaction MDBC", "00b Refresh Interaction Derivative")
    end

    if !state.primed
        # No final step kernel has run yet: reduce the initial state for the
        # first time step. There is no half state yet; the initial displacement
        # bound already forces a cell list rebuild before the first step.
        position_half = sup.ReusePredictors ? gpu.Position : sup.Positionₙ⁺
        launch_step_reduction!(red, gpu.Position, gpu.Velocity, gpu.Acceleration, position_half, SimKernel)
        state.primed = true
    end

    generation = cl.generation
    while true
        K = batch_size(state, h, t_out, kmax)
        @timeit HourGlass "03 Launch Steps" begin
            if use_graph
                # Reuse a small set of graph sizes instead of launching one graph
                # per step. Device stop flags still guard every individual step.
                remaining = K
                while remaining > 0
                    steps = prevpow(2, min(remaining, 32))
                    launch_step_graph!(ctx, steps)
                    remaining -= steps
                end
            else
                for _ in 1:K
                    enqueue_step!(ctx, timed)
                end
            end
        end

        # A stopped batch has stable positions. Prepare only for STOP_REBUILD
        # and publish its grid header in the integer timestep state, which is
        # about to be read anyway. No extra header transfer or wait is needed.
        @timeit HourGlass "02a Prepare Cell Grid" prepare_cell_grid!(cl, gpu.Position, SimKernel.H⁻¹; step = state)

        @timeit HourGlass "13 Read Back State" begin
            readback!(state)
            sync_meta_data!(SimMetaData, state)
        end

        stop = state.ih[I_STOP]
        if stop == STOP_REBUILD
            @timeit HourGlass "02a Actual Calculate IndexCounter" begin
                consume_cell_grid!(cl, state.ih) || error("A requested cell-list rebuild has no prepared grid.")
                if cl.generation != generation
                    # Capacity growth changed device pointers. Drop cached
                    # graphs before any subsequent launches or replay.
                    invalidate_graphs!(state)
                    generation = cl.generation
                end
                rebuild_cell_list!(gpu, cl, SimKernel.H⁻¹; prepared = true, support = sup, floating,
                    constants = SimConstants, boundary_mode, discard_acceleration = true,
                    preserve_pressure = cl.nrebuilds == 0)
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
                    ParticleNormalsPath, SimMeasurements = nothing,
                    ParaviewPressureRange = nothing)

Run a complete simulation on the GPU. Same interface as the CPU version: the
shifting, kernel output, mDBC and log modes are the type parameters of
`SimMetaData`, the time stepping scheme is `SimTimeStepping`
(`SymplecticTimeStepping()` or `SingleNeighborTimeStepping()`). On return the
host `SimParticles` hold the final state (reordered by cell, like the CPU
version).

With `SimMetaData.GPUDoublePosition` the positions (and ghost node
positions) are integrated in `Float64` while everything else stays in
`FloatType`; the pair loops then work on cell relative positions in
`FloatType` (see `GPUCellGrid`). Host particle arrays allocated with
`AllocateDataStructures(SimGeometry, SimMetaData)` already hold `Float64`
positions; other host arrays are converted on upload and download.

`SimMeasurements` optionally records selected pressure, velocity, water-column
and free-surface series at output times. With asynchronous output, samples are
computed by the output worker rather than the timestep kernels.

`ParaviewPressureRange = (min, max)` fixes the generated ParaView pressure color
scale in Pa. By default use zero through the hydrostatic pressure of the initial
fluid height plus one particle spacing. This scale stays fixed across frames;
use an explicit range for impacts or negative pressures.
"""
function RunSimulation(;SimGeometry::Vector{SPHGeometry{Dimensions, FloatType}},
    SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
    SimConstants::SimulationConstants,
    SimKernel::SPHKernelInstance,
    SimLogger::SimulationLogger,
    SimParticles::StructArray,
    SimViscosity::SV,
    SimDensityDiffusion::SDD,
    SimTimeStepping::TimeSteppingMode,
    ParticleNormalsPath::Union{Nothing, String} = nothing,
    SimMeasurements::Union{Nothing, MeasurementConfig} = nothing,
    ParaviewPressureRange = nothing,
    ) where {Dimensions, FloatType, SMode, KMode, BMode, LMode, SV <: SPHViscosity, SDD <: SPHDensityDiffusion}

    CUDA.functional() || error("CUDA is not functional on this machine; use the CPU package SPHExample instead.")

    if SimMetaData.GPURadialInteractions
        if radial_interactions_supported(SimKernel, SimDensityDiffusion, SimViscosity)
            @warn "GPURadialInteractions uses symmetric reciprocal-density weighting for linear diffusion, changing the reference diffusion operator as well as rounding. Validate accuracy and convergence for this case."
        else
            @warn "GPURadialInteractions does not support this kernel/model combination; using reference interactions." kernel = typeof(SimKernel) diffusion = typeof(SimDensityDiffusion) viscosity = typeof(SimViscosity)
        end
    end
    if SimMetaData.GPUScalarSPSViscosity
        if scalar_sps_supported(SimKernel, SimViscosity)
            @warn "GPUScalarSPSViscosity uses experimental reassociated SPS stress arithmetic; validate accuracy and convergence for this case."
        else
            @warn "GPUScalarSPSViscosity requires a supported LaminarSPS configuration; using reference viscosity." kernel = typeof(SimKernel) viscosity = typeof(SimViscosity)
        end
    end

    (; HourGlass) = SimMetaData

    SimMetaData.TimeSteppingMode = SimTimeStepping
    StoreLogOutput = LMode === StoreLog
    PositionType   = position_float_type(SimMetaData)
    paraview_pressure_range = ParaviewPressureRange === nothing ?
        OpenExternalPrograms.hydrostatic_pressure_range(SimParticles, SimConstants) :
        ParaviewPressureRange

    # Only the fields that end up in the output files are copied back from the
    # GPU at every output, plus any fields needed by the optional measurements.
    # The host arrays of all other fields stay untouched.
    output_vars     = resolve_output_variables!(SimMetaData)
    download_fields = if SimMeasurements === nothing ||
                         !has_measurements(SimMeasurements)
        Tuple(Symbol.(output_vars))
    else
        Tuple(unique(vcat(
            Symbol.(output_vars), required_particle_fields(SimMeasurements),
        )))
    end

    TimeSteps = Vector{FloatType}()

    if BMode !== NoMDBC
        if ParticleNormalsPath === nothing
            all(geom -> !is_wall(geom.Type) ||
                (hasproperty(geom.Particles, :GhostPoints) &&
                 hasproperty(geom.Particles, :GhostNormals)), SimGeometry) ||
                error("$(BMode) requires ParticleNormalsPath or geometry ghost fields.")
        else
            # Directions use working precision; ghost positions use position precision.
            points, ghosts, normals = LoadBoundaryNormals(
                Val(Dimensions), PositionType, ParticleNormalsPath)
            if all(geom -> isempty(geom.CSVFile), SimGeometry)
                # Generated particles can have a different order from the CSV normals.
                tolerance = Float64(SimConstants.dx) * 1e-3
                key(x) = ntuple(d -> round(Int, x[d] / tolerance), Dimensions)
                lookup = Dict(key(x) => i for (i, x) in enumerate(points))
                for i in eachindex(SimParticles.Position)
                    is_wall(SimParticles.Type[i]) || continue
                    j = get(lookup, key(SimParticles.Position[i]), 0)
                    j > 0 || error("Boundary normals do not match generated particles.")
                    SimParticles.GhostPoints[i] = ghosts[j]
                    SimParticles.GhostNormals[i] = normals[j]
                end
            else
                for i in eachindex(ghosts)
                    SimParticles.GhostPoints[i] = ghosts[i]
                    SimParticles.GhostNormals[i] = normals[i]
                end
            end
        end
    end

    if BMode === UpdatedMDBC
        any(==(Floating), SimParticles.Type) && error("UpdatedMDBC does not support floating bodies (future work in the 2024 presentation).")
        for i in eachindex(SimParticles.Type)
            is_wall(SimParticles.Type[i]) || continue
            normal = SimParticles.GhostNormals[i]
            ghost = SimParticles.GhostPoints[i]
            all(isfinite, normal) && norm(normal) > zero(FloatType) && all(isfinite, ghost) && !iszero(ghost) ||
                error("UpdatedMDBC requires a finite nonzero normal and ghost point for every wall particle (particle $i).")
        end
    end

    if StoreLogOutput
        InitializeLogger(SimLogger, SimConstants, SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        with_logger(SimLogger.Logger) do
            dev = CUDA.device()
            @info "GPU: $(CUDA.name(dev)), compute capability $(CUDA.capability(dev)), " *
                  "$(round(CUDA.totalmem(dev) / 2^30; digits = 1)) GiB, CUDA.jl $(pkgversion(CUDA))"
            @info "GPU float type: $(FloatType)"
            if PositionType !== FloatType
                @info "GPU position type: $(PositionType) (pair loops use cell relative positions in $(FloatType))"
            end
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
        gpu    = upload_particles(SimParticles, SimMetaData)
        red    = ReductionWorkspace{SVector{3, FloatType}}(NumberOfPoints)
        cl     = CellListWorkspace{Dimensions, PositionType}(NumberOfPoints;
                     reach = SimMetaData.GPUCellSubdivision, max_cells = SimMetaData.GPUMaxCells,
                     deterministic = SimMetaData.GPUDeterministicSort)
        sup    = GPUSupportArrays{Dimensions, FloatType}(gpu, SimMetaData)
        motion = MotionArrays(SimGeometry, SimParticles)
        floating = FloatingArrays(SimGeometry, SimParticles, SimConstants; position_type = PositionType)
        # Device resident loop state. The displacement bound starts above `h`
        # so that the cell list is built before the first step.
        state  = StepState{FloatType}(; time = SimMetaData.TotalTime, iteration = SimMetaData.Iteration,
                                        dx = one(FloatType) + SimKernel.h)
        CUDA.synchronize()
    end

    output = SetupVTKOutput(
        SimMetaData, SimParticles, SimKernel, Dimensions;
        measurements = SimMeasurements, fluid_type = Fluid,
    )

    # Motion of the floating bodies at every output time, one row per body.
    floating_log = nothing
    if floating.active
        floating_log = open(joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName * "_Floating.csv"), "w")
        if Dimensions == 2
            println(floating_log, "Time,Body,GroupMarker,Center:0,Center:1,Velocity:0,Velocity:1,Angle,Omega")
        else
            println(floating_log, "Time,Body,GroupMarker,Center:0,Center:1,Center:2," *
                                "Velocity:0,Velocity:1,Velocity:2,Orientation:0,Orientation:1," *
                                "Orientation:2,Orientation:3,Omega:0,Omega:1,Omega:2")
        end
    end
    function log_floating(t)
        floating_log === nothing && return nothing
        s = floating_state(floating)
        markers = [geom.GroupMarker for geom in SimGeometry if geom.Type == Floating]
        for b in 1:Int(floating.nbodies)
            if Dimensions == 2
                println(floating_log, join((t, b, markers[b], s.center[b]..., s.velocity[b]...,
                                            s.angle[b], s.omega[b]), ","))
            else
                println(floating_log, join((t, b, markers[b], s.center[b]..., s.velocity[b]...,
                                            s.orientation[b]..., s.omega[b]...), ","))
            end
        end
        flush(floating_log)
        return nothing
    end
    log_floating(SimMetaData.TotalTime)

    # Save initial state, use 1 else this cannot be used to index fid vector
    SimMetaData.OutputIterationCounter = 1
    output.save_particles(SimMetaData.OutputIterationCounter)
    output.save_grid(SimMetaData.OutputIterationCounter, CartesianIndex{Dimensions}[], SimParticles)

    generate_showvalues(Iteration, TotalTime, TimeLeftInSeconds) = () -> [
        (:(Iteration), string(Iteration)),
        (:(TotalTime), @sprintf("%3.3f", TotalTime)),
        (:(TimeLeftInSeconds), @sprintf("%3.1f [s]", TimeLeftInSeconds)),
    ]

    # Two pinned download buffers feed a collector, which copies completed
    # frames into a reusable, memory-budgeted host queue. The disk writer owns
    # SimParticles and consumes queued frames in order. Staging buffers return
    # before any HDF5 work; backpressure occurs only when the host queue fills.
    staging = OutputDownload{Dimensions}[OutputDownload(gpu, download_fields; cells = SimMetaData.ExportGridCells)
                                         for _ in 1:(SimMetaData.GPUAsyncOutput ? OUTPUT_STAGING_FRAMES : 1)]
    free    = Channel{OutputDownload{Dimensions}}(length(staging))
    jobs    = Channel{OutputFrameJob{Dimensions}}(length(staging))
    foreach(dl -> put!(free, dl), staging)
    write_time = Ref(0.0)
    # Host side of one frame: wait for its copies, move them into the host
    # arrays and write the files. Runs on the writer task (or inline without
    # `GPUAsyncOutput`).
    function write_frame(job::OutputFrameJob)
        t0 = time()
        if SimMetaData.GPUAsyncOutput
            for (f, buf) in zip(job.download.fields, job.download.buffers)
                copyto!(getproperty(SimParticles, f), buf)
            end
            UniqueCells = isempty(job.download.cid) ? CartesianIndex{Dimensions}[] :
                store_cells!(SimParticles, job.grid, job.download.cid)
        else
            UniqueCells = finish_download!(job.download, SimParticles, job.grid)
            put!(free, job.download)
        end
        SimMetaData.IndexCounter = length(UniqueCells)
        output.save_particles(job.counter, job.time)
        if SimMetaData.ExportGridCells
            # Bin the downloaded positions, not the cells from the last neighbor rebuild.
            grid_cells = occupied_cells_host(first(job.download.buffers), bin_scale(job.grid, SimKernel.H⁻¹))
            output.save_grid(job.counter, grid_cells, SimParticles, job.time)
        end
        write_time[] += time() - t0
        return nothing
    end
    writer = nothing
    collector = nothing
    if SimMetaData.GPUAsyncOutput
        prototype = first(staging)
        frame_bytes = sum(sizeof, prototype.buffers) + sizeof(prototype.cid)
        queue_frames = max(1, div(SimMetaData.GPUOutputQueueBytes, max(frame_bytes, 1)))
        host_free = Channel{OutputDownload{Dimensions}}(queue_frames)
        ready = Channel{OutputFrameJob{Dimensions}}(queue_frames)
        for _ in 1:queue_frames
            # Ordinary host memory; the event is unused for collected frames.
            frame = OutputDownload{Dimensions}(prototype.fields,
                Vector[similar(buf) for buf in prototype.buffers],
                similar(prototype.cid), CUDA.HostMemory[], prototype.event)
            put!(host_free, frame)
        end
        writer = spawn_writer() do
            for job in ready
                write_frame(job)
                put!(host_free, job.download)
            end
        end
        bind(host_free, writer)
        bind(ready, writer)
        collector = spawn_collector() do
            try
                for job in jobs
                    frame = take!(host_free)
                    CUDA.synchronize(job.download.event)
                    for (dst, src) in zip(frame.buffers, job.download.buffers)
                        copyto!(dst, src)
                    end
                    copyto!(frame.cid, job.download.cid)
                    put!(free, job.download)
                    put!(ready, OutputFrameJob{Dimensions}(frame, job.counter,
                        job.time, job.grid))
                end
            finally
                close(ready)
            end
        end
        bind(free, collector)
        bind(jobs, collector)
    end

    output_workers_stopped = Ref(false)
    output_streams_closed = Ref(false)
    output_finalized = Ref(false)
    exit_finalizer = Ref{Union{Nothing, Function}}(nothing)

    function stop_output_workers!()
        output_workers_stopped[] && return nothing
        output_workers_stopped[] = true
        close(jobs)
        try
            collector === nothing || wait(collector)
        finally
            try
                writer === nothing || wait(writer)
            finally
                try
                    CUDA.synchronize()
                    foreach(free!, staging)
                finally
                    floating_log === nothing || close(floating_log)
                end
            end
        end
        return nothing
    end

    function close_output_streams!()
        output_streams_closed[] && return nothing
        @timeit HourGlass "13c Close Data Streams" begin
            try
                stop_output_workers!()
            finally
                output.close_files()
            end
        end
        output_streams_closed[] = true
        return nothing
    end

    function finalize_output!(status::String)
        output_finalized[] && return nothing
        close_output_streams!()

        show(HourGlass, sortby = :name)
        AutoOpenParaview(SimMetaData, SimConstants, output.variable_names;
            pressure_range = paraview_pressure_range)

        if StoreLogOutput
            with_logger(SimLogger.Logger) do
                @info "Cell list rebuilds: $(cl.nrebuilds), grid dims: $(cl.grid.dims)"
                @info "Device grid status reads: $(cl.grid_status_readbacks), capacity growths: $(cl.capacity_growths), " *
                      "bounding-box host partial reads: $(cl.bbox_host_readbacks)"
                @info "Host read backs of the step state: $(state.readbacks) for $(SimMetaData.Iteration) steps, " *
                      "captured step graphs: $(length(state.graphs))"
                @info @sprintf("Output frame write time (%s): %.2f [s], %d frames buffered per file flush",
                               SimMetaData.GPUAsyncOutput ? "writer task, overlapped with GPU work" : "simulation thread",
                               write_time[], output.frames_per_flush)
            end
            LogFinal(SimLogger, HourGlass; status = status)

            if !isempty(TimeSteps)
                time_steps_graph = lineplot(1:length(TimeSteps), TimeSteps,
                                            title = "Time Steps [s] as a function of iteration",
                                            name = "Time Steps", xlabel = "Iterations [-]",
                                            ylabel = "Time Step Size [s]")
                with_logger(SimLogger.Logger) do
                    @info ""
                    show(SimLogger.LoggerIo, time_steps_graph)
                end
            end

            close(SimLogger.LoggerIo)
            AutoOpenLogFile(SimLogger, SimMetaData)
        end

        output_finalized[] = true
        exit_finalizer[] = nothing
        return nothing
    end

    exit_finalizer[] = () -> begin
        if !output_finalized[]
            @warn "Julia is exiting before the simulation completed; closing VTKHDF output, " *
                  "finalizing the log, and opening ParaView for the data written so far."
            finalize_output!("stopped as Julia exited")
        end
    end
    atexit() do
        finalizer = exit_finalizer[]
        finalizer === nothing || finalizer()
    end
    Base.exit_on_sigint(false)

    run_status = "finished"
    try
        while true
            @timeit HourGlass "00 SimulationLoop" SimulationLoop(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                                 SimConstants, gpu, cl, sup, red, motion, state;
                                                                 floating = floating)
            push!(TimeSteps, SimMetaData.CurrentTimeStep)
            log_floating(SimMetaData.TotalTime)

            # Publish progress immediately, independently of the queued disk writes.
            if StoreLogOutput
                LogStep(SimLogger, SimMetaData, HourGlass)
                SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
            end

            SimMetaData.OutputIterationCounter += 1
            counter = SimMetaData.OutputIterationCounter
            t_out   = SimMetaData.TotalTime
            grid    = cl.grid

            @timeit HourGlass "13 Output Frame" begin
                dl = @timeit HourGlass "13a Wait For Staging Buffers" take!(free)
                enqueue_download!(dl, gpu)
                job = OutputFrameJob{Dimensions}(dl, counter, Float64(t_out), grid)
                if SimMetaData.GPUAsyncOutput
                    put!(jobs, job)
                else
                    @timeit HourGlass "13b Write Frame" write_frame(job)
                end
            end

            if SimMetaData.TotalTime > SimMetaData.SimulationTime
                close_output_streams!()

                # Leave the complete final state on the host, not only the output
                # fields, so that callers can inspect every particle field.
                @timeit HourGlass "13d Final Download From GPU" download_particles!(SimParticles, gpu, cl.grid; cells = true)
                
                break
            end
        end
    catch e
        if e isa InterruptException
            @warn "Simulation interrupted; closing VTKHDF output, finalizing the log, " *
                  "and opening ParaView for the data written so far."
            run_status = "interrupted"
        else
            run_status = "failed"
            rethrow()
        end
    finally
        finalize_output!(run_status)
    end

    return nothing
end

end # module
