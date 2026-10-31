module SimulationMetaDataConfiguration
using ..PeriodicBoundaries: PeriodicBoundary
using ..VariableResolutionConfiguration: VariableResolutionConfig

using TimerOutputs

export SimulationMetaData, UpdateMetaData!, ShiftingMode, NoShifting, PlanarShifting,
       KernelOutputMode, NoKernelOutput, StoreKernelOutput,
       MDBCMode, NoMDBC, SimpleMDBC, UpdatedMDBC,
       LogMode, NoLog, StoreLog,
       TimeSteppingMode, SymplecticTimeStepping, SingleNeighborTimeStepping, DualSPHysicsSymplectic,
       OUTPUT_VARIABLES, DEFAULT_OUTPUT_VARIABLES, resolve_output_variables!, position_float_type

# CPU-compatible mode types mirror the CPU package. They select code paths at compile
# time (as type parameters of `SimulationMetaData`) instead of run time flags,
# so that one case definition constructs against both packages.
abstract type ShiftingMode end
struct NoShifting     <: ShiftingMode end
struct PlanarShifting <: ShiftingMode end

abstract type KernelOutputMode end
struct NoKernelOutput    <: KernelOutputMode end
struct StoreKernelOutput <: KernelOutputMode end

abstract type MDBCMode end
struct NoMDBC     <: MDBCMode end
struct SimpleMDBC <: MDBCMode end
"The 2024 mDBC update: numerical checks, pressure cloning and mirrored no-slip velocity."
struct UpdatedMDBC <: MDBCMode end

abstract type LogMode end
struct NoLog    <: LogMode end
struct StoreLog <: LogMode end

abstract type TimeSteppingMode end
struct SymplecticTimeStepping     <: TimeSteppingMode end
struct SingleNeighborTimeStepping <: TimeSteppingMode end
"""
    DualSPHysicsSymplectic(; cache_periodic_queries = false, ...)

Reference DBC predictor/corrector, pair-based timestep limits and Float64 clock.
For periodic runs, caching image search limits preserves row/pair order and
adds 48 bytes per particle in 2D or 72 in 3D to device support storage. Limits
are refreshed for each evaluated state; nonperiodic runs cannot enable it.
`periodic_images = true` (the default) instead builds dense, cell-sorted originals plus
periodic copies for each force evaluation. Only originals are integrated,
reduced and exported. The period must exceed twice the kernel support radius.
Finite reference domain bounds initialize the source grid when supplied;
otherwise it follows the original particle grid. Selecting query caching uses
the query backend. `periodic_images = false` disables copies for this scheme.
It uses additional device storage and changes neighbor summation order.
"""
Base.@kwdef struct DualSPHysicsSymplectic <: TimeSteppingMode
    smoothing_length::Float64 = NaN
    sound_speed::Float64 = NaN
    domain_min::NTuple{3,Float64} = (-Inf,-Inf,-Inf)
    domain_max::NTuple{3,Float64} = (Inf,Inf,Inf)
    density_min::Float64 = -Inf
    density_max::Float64 = Inf
    fast_math::Bool = true
    cache_periodic_queries::Bool = false
    periodic_images::Bool = true
end

"""
    SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}(; SimulationName, SaveLocation, kwargs...)

Run time meta data of a simulation. Same type parameters, fields and defaults
as the CPU version plus a few GPU specific options (all prefixed `GPU`).

The mode parameters select code paths at compile time:

* `SMode <: ShiftingMode`: `NoShifting` (default) or `PlanarShifting`
* `KMode <: KernelOutputMode`: `NoKernelOutput` (default) or `StoreKernelOutput`
  (also store the kernel sum and kernel gradient sum of every particle)
* `BMode <: MDBCMode`: `NoMDBC` (default), `SimpleMDBC`, or `UpdatedMDBC` (requires
  `ParticleNormalsPath` in `RunSimulation`)
* `LMode <: LogMode`: `NoLog` (default) or `StoreLog`

Trailing mode parameters may be omitted, `SimulationMetaData{D, T}(...)`
selects all defaults. The time stepping scheme is not a type parameter; it is
passed to `RunSimulation` as `SimTimeStepping` and stored in the
`TimeSteppingMode` field.

The accumulated clock (`TotalTime`), simulation end (`SimulationTime`) and
output schedule (`OutputEach`, `OutputTimes`) use Float64 independently of
`FloatType`. `CurrentTimeStep` and particle arithmetic retain `FloatType`.
The keyword constructor and later assignments accept scalar or vector output
times in either precision.

`VariableResolution = VariableResolutionConfig(region; fine_geometry, ...)`
enables two-domain refinement in the normal `RunSimulation` entry point.
The usual geometry, constants and kernel describe the coarse domain; the
configuration supplies the fine geometry and optional fine constants/kernel.
This backend requires 2D SimpleMDBC, NoShifting, NoKernelOutput, reference
symplectic stepping, artificial viscosity and full Fourtakas diffusion.
It supports `SimDamping`, normal output fields/names, single-file or numbered
VTKHDF, scalar/vector schedules and `GPUClipTimeStepToOutput`. Graph execution,
batch size, interaction threads, sort determinism and cell limits use the
corresponding metadata fields. Positions use Float64; conservative interaction
arithmetic and synchronous variable-count output are reported in the effective
configuration. Periodic/fixed grids, subdivision other than one, moving/floating
walls, external forces and measurements are currently unsupported and fail
explicitly. `nothing` (the default) selects the ordinary uniform-resolution path.

`GPUClipTimeStepToOutput = true` clips each adaptive step to the next output
deadline (including `SimulationTime`). Frames then contain the integrated state
at that Float64 deadline, with the timestep converted to `FloatType` for
particle integration, without interpolation. The default `false`
keeps full adaptive steps and writes the first state strictly beyond each
deadline. Clipping changes the integration trajectory and can change the step
count; with it enabled, changing `OutputTimes` can also change the trajectory.

`GPUDoublePosition = true` stores and integrates the particle positions in
`Float64` while velocities, densities, forces and all constants stay in
`FloatType` (the `posdouble` mode of DualSPHysics). The pair loops then read
cell relative positions in `FloatType`, see `GPUCellGrid.PosCell`. Allocate
the host particles with `AllocateDataStructures(SimGeometry, SimMetaData)` so
that they hold `Float64` positions from the input files onwards.

`GPUCompactBuffers = true` omits unused ghost, kernel-output and shifting
buffers (and omits inactive `Cells`, `Kernel`, `KernelGradient`, `GhostPoints`
and `GhostNormals` columns from host `SimParticles` when allocated with
`AllocateDataStructures(SimGeometry, SimMetaData)`), shares position/velocity
predictors with reorder scratch, omits acceleration sorting shadows,
and narrows host and GPU IDs (`Int32`) and group markers (`UInt8`, `UInt16` or
`UInt32`) losslessly when every value fits. The driver
overwrites acceleration before any post-rebuild reader. Fixed predictor entries
are initialized after sorting and not rewritten each step. Boundary density
updates and requested boundary-force calculations are unchanged.
Out-of-range metadata retains its original width. `false` keeps the previous
full host and GPU allocation layout.

`GPUCacheInteractionFields = true` stores reciprocal density and pressure per
particle for reuse by the interaction loops, using an additional
`2 * sizeof(FloatType)` bytes per particle. Cell-relative positions are already
cached when needed; with `GPUCacheInteractionFields = false` (default), both
reciprocal density and pressure are computed on demand from density. The
reciprocal and pressure buffers are shared between evaluated states and
refreshed after density prediction and boundary corrections. This option is
independent of `GPUCompactBuffers`, defaults to `false`, and trades device
memory for fewer neighbor-loop divisions and EOS evaluations; elapsed speed
depends on the case and GPU.
With periodic copies, the dense source capacity also holds its own two cache
fields. Capacity is twice the original count, adding another
`4 * sizeof(FloatType)` bytes per original beyond the original-state cache.
The live-buffer report includes these arrays.

`GPURadialInteractions = true` opts into experimental scalar Wendland C2 pair contractions
for supported linear diffusion and built-in viscosity models. Linear diffusion
uses the symmetric reciprocal-density average `(ρᵢ⁻¹ + ρⱼ⁻¹) / 2`, changing
the reference diffusion operator, not just rounding. SPS keeps its original
viscosity arithmetic and pair ordering. The default is `false`, preserving
the reference arithmetic. Accuracy and convergence must be validated for the
intended simulation; unsupported models use the reference path.

`GPUScalarSPSViscosity = true` separately opts into a rank-one contraction of
the existing SPS stress tensors. Pair orientation and the closure are unchanged,
but rounding can differ. It defaults to `false` and does not require radial
continuity/diffusion. Other viscosity models retain their existing path.

`GPUInteractionAlwaysInline = true` forces helper-function inlining in the
interaction kernel without requesting relaxed floating-point semantics.
`GPUInteractionFastMath = true` enables CUDA fast math in that kernel and its
callees, including custom models: approximate arithmetic, reassociation,
Float32 subnormal flushing and assumptions about NaN/Inf may change results.
Both default to `true`, can be disabled independently, and do not change the compilation of
boundary-correction, integration, reduction or interaction-cache preparation
kernels. Validate accuracy and elapsed solver time for the intended case.
Changing either option between output intervals invalidates captured graphs
and refreshes the carried single-neighbor derivative.

`GPUFixedGrid = true` retains the neighbour-grid bounds from the initial
particle geometry. `GPUGridExpansion` selects which sides may grow, as one
`(negative, positive)` Boolean pair per axis, for example
`((false, false), (false, true))` allows only upward growth in 2D. All sides
are locked by default. Bounds never shrink in this mode; particles crossing
a locked side become permanently `Frozen` at the next batch check, with zero
velocity/acceleration and no SPH interactions. Their IDs, positions, time and
crossed limits are logged, even with `NoLog`. Floating bodies freeze as a whole.
Cell size and particle rebinning are unchanged. Expansion requires `GPUFixedGrid = true`.

`Periodicity = PeriodicBoundary{Dimensions}(axis; min, max, shift)` connects
opposite faces of a physical interval, with an optional transverse translation
when crossing `max`. The default `nothing` disables periodicity. Interactions
and mDBC interpolation include neighbors across the seam. Accepted positions
are wrapped before grid escape checks and output, triggering a reorder when
needed. The periodic axis is exempt from freezing; `GPUGridExpansion` still
controls the other axes, including height changes from `shift`. One periodic
axis is supported, with interval length greater than twice the support radius.
Floating bodies wrap as a whole, keeping all centers, torque arms and hinge
anchors in the same periodic image. Their diameter plus twice the support
radius must fit inside the interval. Fixed boundaries remain stationary.

`GPUTimestepDiagnosticsEvery = N` records the last accepted step's limiting
criterion and stable particle ID every `N` output intervals in
`<SimulationName>_Timestep.csv`. The default `0` disables diagnostics. Positive
values retain integer IDs alongside the existing timestep reduction without
changing its arithmetic. Sampling follows `OutputTimes` and works with `NoLog`.
"""
mutable struct SimulationMetaData{Dimensions,
                                  FloatType <: AbstractFloat,
                                  SMode <: ShiftingMode,
                                  KMode <: KernelOutputMode,
                                  BMode <: MDBCMode,
                                  LMode <: LogMode}
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput
    Iteration::Int
    OutputEach::Float64
    OutputTimes::Union{Float64, Vector{Float64}}
    OutputIterationCounter::Int
    StepsTakenForLastOutput::Int
    CurrentTimeStep::FloatType
    TotalTime::Float64
    SimulationTime::Float64
    IndexCounter::Int
    VisualizeInParaview::Bool
    ExportSingleVTKHDF::Bool
    ExportGridCells::Bool
    OutputVariables::Vector{String}
    OpenLogFile::Bool
    TimeSteppingMode::TimeSteppingMode
    VariableResolution::Union{Nothing,VariableResolutionConfig}
    # GPU specific options
    GPUSyncTimers::Bool          # synchronize after every phase so the timer output is meaningful
    GPUDeterministicSort::Bool   # sort particles inside each cell for bitwise reproducible runs
    GPUMaxCells::Int             # safety limit for the size of the neighbour grid
    GPUInteractionThreads::Int   # threads per block for the interaction kernels
    GPULanesPerParticle::Int     # warp lanes per particle in the gather kernels (0 = automatic)
    GPUInteractionAlwaysInline::Bool # force inlining only in the interaction kernel
    GPUInteractionFastMath::Bool # relaxed arithmetic only in the interaction kernel
    GPURadialInteractions::Bool  # opt-in scalar Wendland arithmetic and symmetric linear diffusion
    GPUScalarSPSViscosity::Bool   # separate opt-in rank-one SPS stress arithmetic (changes rounding)
    GPUBoundaryForces::Bool      # also evaluate the momentum equation for boundary particles (CPU parity)
    GPUAsyncOutput::Bool         # write output files on a Julia task while the GPU continues
    GPUMaxStepsPerSync::Int      # upper bound on the time steps enqueued between two host read backs
    GPUUseGraph::Bool            # replay the launch sequence of a step as a CUDA graph
    GPUClipTimeStepToOutput::Bool # shorten adaptive steps to land exactly on output deadlines
    GPUCellSubdivision::Int      # cells per support radius H per axis (1: edge H, 3^D stencil; 2: edge H/2, 5^D stencil)
    GPUFixedGrid::Bool           # retain initial grid bounds, growing only on explicitly allowed sides
    GPUGridExpansion::NTuple{Dimensions, Tuple{Bool, Bool}} # (negative, positive) growth per axis
    Periodicity::Union{Nothing, PeriodicBoundary{Dimensions, Float64}}
    GPUDoublePosition::Bool      # integrate positions in Float64 (pair loops use cell relative positions in FloatType)
    GPUPeriodicCellCoordinates::Bool # opt-in cell-relative arithmetic for periodic force pairs
    GPUPeriodicImages::Bool      # dense periodic force sources for supported DBC runs
    GPUCacheInteractionFields::Bool # cache reciprocal density for interaction loops
    GPUCompactBuffers::Bool      # omit unused buffers and narrow GPU-only metadata when lossless
    GPUOutputQueueBytes::Int     # reusable host frames between collector and disk writer
    GPUOutputBufferBytes::Int    # host memory that holds output frames before they are written to the file
    GPUTimestepDiagnosticsEvery::Int # record the timestep limiter every N output intervals (0 disables)
end

"""
    position_float_type(SimMetaData) -> Type

Element precision of the particle positions: `Float64` with
`GPUDoublePosition` or variable resolution, otherwise the `FloatType` of the meta data. With
`FloatType = Float64` the option changes nothing.
"""
position_float_type(m::SimulationMetaData{D, T}) where {D, T} = m.GPUDoublePosition || m.VariableResolution !== nothing ? Float64 : T

# Particle fields that can be written to the output files. `Position` is always
# written as the point coordinates and is not listed. `Kernel` and
# `KernelGradient` are only computed with `KMode = StoreKernelOutput`, the mDBC
# ghost data only exists with `BMode = SimpleMDBC` or `UpdatedMDBC`; in every other mode these
# fields would be written as zeros, so `resolve_output_variables!` drops them.
const OUTPUT_VARIABLES = [
    "Velocity",
    "Density",
    "Pressure",
    "Acceleration",
    "ID",
    "Type",
    "GroupMarker",
    "Kernel",
    "KernelGradient",
    "GhostPoints",
    "GhostNormals",
    "Resolution", "Spacing", "Mass", "Owned", "IsBuffer",
]

# The state and identity of every particle. Diagnostics (`Acceleration`, the
# kernel sums and the ghost data) are opt in through `OutputVariables`.
const DEFAULT_OUTPUT_VARIABLES = [
    "Velocity",
    "Density",
    "Pressure",
    "ID",
    "Type",
    "GroupMarker",
]

"""
    resolve_output_variables!(SimMetaData) -> Vector{String}

Validate `SimMetaData.OutputVariables` against the mode type parameters and
drop, with a warning, the variables this run never fills with data (the kernel
sums without `StoreKernelOutput`, the ghost data without an mDBC mode). Unknown
names are an error. The trimmed list is stored back into the meta data so that
the writer, the GPU download and the ParaView state file all agree on it.
"""
function resolve_output_variables!(SimMetaData::SimulationMetaData{D, T, SMode, KMode, BMode}) where {D, T, SMode, KMode, BMode}
    requested = SimMetaData.OutputVariables
    SimMetaData.VariableResolution !== nothing ||
        isempty(intersect(requested,["Resolution","Spacing","Mass","Owned","IsBuffer"])) ||
        throw(ArgumentError("resolution output fields require VariableResolution in SimulationMetaData"))
    unknown   = setdiff(requested, OUTPUT_VARIABLES)
    if !isempty(unknown)
        hint = any(in(("ChunkID", "BoundaryBool")), unknown) ?
               " `ChunkID` and `BoundaryBool` are no longer written: the first carried no information " *
               "on the GPU and the second equals `Type != Fluid`." : ""
        error("Unknown output variable(s) $(unknown). Available: $(OUTPUT_VARIABLES)." * hint)
    end
    keep = String[]
    for name in unique(requested)
        if name in ("Kernel", "KernelGradient") && KMode !== StoreKernelOutput
            @warn "Output variable `$name` is only computed with the `StoreKernelOutput` kernel mode and is not written."
        elseif name in ("GhostPoints", "GhostNormals") && BMode === NoMDBC
            @warn "Output variable `$name` only exists with an mDBC boundary mode and is not written."
        else
            push!(keep, name)
        end
    end
    SimMetaData.OutputVariables = keep
    return keep
end

_output_times(::Type{T}, x::Real) where {T} = T(x)
_output_times(::Type{T}, x::AbstractVector) where {T} = Vector{T}(x)

function SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}(;
        SimulationName::String,
        SaveLocation::String,
        HourGlass::TimerOutput                  = TimerOutput(),
        Iteration::Int                          = 0,
        OutputEach                              = 0.02, # seconds
        OutputTimes                             = OutputEach,
        OutputIterationCounter::Int             = 0,
        StepsTakenForLastOutput::Int            = 0,
        CurrentTimeStep                         = 0,
        TotalTime                               = 0,
        SimulationTime                          = 0,
        IndexCounter::Int                       = 0,
        VisualizeInParaview::Bool               = true,
        ExportSingleVTKHDF::Bool                = true,
        ExportGridCells::Bool                   = false,
        OutputVariables::Vector{String}         = copy(DEFAULT_OUTPUT_VARIABLES),
        OpenLogFile::Bool                       = true,
        TimeSteppingMode::TimeSteppingMode      = SingleNeighborTimeStepping(),
        VariableResolution::Union{Nothing,VariableResolutionConfig} = nothing,
        GPUSyncTimers::Bool                     = false,
        GPUDeterministicSort::Bool              = true,
        GPUMaxCells::Int                        = 50_000_000,
        GPUInteractionThreads::Int              = 128,
        GPULanesPerParticle::Int                = 0,
        GPUInteractionAlwaysInline::Bool        = true,
        GPUInteractionFastMath::Bool            = true,
        GPURadialInteractions::Bool             = false,
        GPUScalarSPSViscosity::Bool              = false,
        GPUBoundaryForces::Bool                 = true,
        GPUAsyncOutput::Bool                    = true,
        GPUMaxStepsPerSync::Int                 = 32,
        GPUUseGraph::Bool                       = true,
        GPUClipTimeStepToOutput::Bool            = false,
        GPUCellSubdivision::Int                 = 1,
        GPUFixedGrid::Bool                      = false,
        GPUGridExpansion::NTuple{Dimensions, Tuple{Bool, Bool}} = ntuple(_ -> (false, false), Dimensions),
        Periodicity::Union{Nothing, PeriodicBoundary{Dimensions}} = nothing,
        GPUDoublePosition::Bool                 = false,
        GPUPeriodicCellCoordinates::Bool        = false,
        GPUPeriodicImages::Bool                 = true,
        GPUCacheInteractionFields::Bool         = false,
        GPUCompactBuffers::Bool                 = true,
        GPUOutputQueueBytes::Int                = 256 * 2^20,
        GPUOutputBufferBytes::Int               = 256 * 2^20,
        GPUTimestepDiagnosticsEvery::Int        = 0,
    ) where {Dimensions, FloatType <: AbstractFloat, SMode <: ShiftingMode, KMode <: KernelOutputMode,
             BMode <: MDBCMode, LMode <: LogMode}
    GPUCellSubdivision >= 1 || throw(ArgumentError("GPUCellSubdivision must be at least 1, got $GPUCellSubdivision"))
    GPUFixedGrid || !any(sides -> any(sides), GPUGridExpansion) ||
        throw(ArgumentError("GPUGridExpansion requires GPUFixedGrid = true"))
    GPUOutputQueueBytes >= 0 || throw(ArgumentError("GPUOutputQueueBytes must be nonnegative"))
    GPUOutputBufferBytes >= 0 || throw(ArgumentError("GPUOutputBufferBytes must not be negative, got $GPUOutputBufferBytes"))
    GPUTimestepDiagnosticsEvery >= 0 || throw(ArgumentError("GPUTimestepDiagnosticsEvery must be nonnegative"))
    VariableResolution === nothing || Dimensions == 2 || throw(ArgumentError("variable resolution currently supports only 2D"))
    return SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}(
        SimulationName, SaveLocation, HourGlass, Iteration,
        Float64(OutputEach), _output_times(Float64, OutputTimes),
        OutputIterationCounter, StepsTakenForLastOutput,
        FloatType(CurrentTimeStep), Float64(TotalTime), Float64(SimulationTime),
        IndexCounter, VisualizeInParaview, ExportSingleVTKHDF, ExportGridCells,
        OutputVariables, OpenLogFile, TimeSteppingMode, VariableResolution,
        GPUSyncTimers, GPUDeterministicSort, GPUMaxCells, GPUInteractionThreads, GPULanesPerParticle,
        GPUInteractionAlwaysInline, GPUInteractionFastMath, GPURadialInteractions, GPUScalarSPSViscosity,
        GPUBoundaryForces, GPUAsyncOutput, GPUMaxStepsPerSync, GPUUseGraph, GPUClipTimeStepToOutput, GPUCellSubdivision,
        GPUFixedGrid, GPUGridExpansion, Periodicity,
        GPUDoublePosition, GPUPeriodicCellCoordinates, GPUPeriodicImages, GPUCacheInteractionFields, GPUCompactBuffers, GPUOutputQueueBytes, GPUOutputBufferBytes,
        GPUTimestepDiagnosticsEvery,
    )
end

# Legacy query selection will be deprecated after compatibility
# and trajectory validation; the opt-out remains supported in this release.
function periodic_images_enabled(meta::SimulationMetaData{D,T,S,K,B}, kernel=nothing) where {D,T,S,K,B}
    mode = meta.TimeSteppingMode
    return meta.GPUPeriodicImages && meta.Periodicity !== nothing && B === NoMDBC &&
        (!(mode isa DualSPHysicsSymplectic) || (mode.periodic_images && !mode.cache_periodic_queries)) &&
        (kernel === nothing || meta.Periodicity.upper-meta.Periodicity.lower > 2kernel.H)
end

# Trailing mode parameters default like in the CPU package.
SimulationMetaData{D,T,S,K,B}(; kwargs...) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode} =
    SimulationMetaData{D,T,S,K,B,NoLog}(; kwargs...)
SimulationMetaData{D,T,S,K}(; kwargs...) where {D,T,S<:ShiftingMode,K<:KernelOutputMode} =
    SimulationMetaData{D,T,S,K,NoMDBC,NoLog}(; kwargs...)
SimulationMetaData{D,T,S}(; kwargs...) where {D,T,S<:ShiftingMode} =
    SimulationMetaData{D,T,S,NoKernelOutput,NoMDBC,NoLog}(; kwargs...)
SimulationMetaData{D,T}(; kwargs...) where {D,T} =
    SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(; kwargs...)

# Accept scalar and vector schedules without narrowing the clock precision.
function Base.setproperty!(m::SimulationMetaData{D, T}, name::Symbol, x) where {D, T}
    if name === :OutputTimes
        return setfield!(m, name, _output_times(Float64, x))
    else
        return setfield!(m, name, convert(fieldtype(typeof(m), name), x))
    end
end

function UpdateMetaData!(SimMetaData, dt)
    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt
    return nothing
end

end
