module SimulationMetaDataConfiguration

using TimerOutputs
using ProgressMeter

export SimulationMetaData

"""
    SimulationMetaData{Dimensions, FloatType}(; SimulationName, SaveLocation, kwargs...)

Run time meta data of a simulation. Same fields and defaults as the CPU
version plus a few GPU specific options (all prefixed `GPU`). Unlike the CPU
version the keyword constructor converts `OutputTimes` (a number or a vector
of numbers) to `FloatType`, so `OutputTimes = 0.01` also works for
`FloatType = Float32`.
"""
mutable struct SimulationMetaData{Dimensions, FloatType <: AbstractFloat}
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput
    Iteration::Int
    OutputEach::FloatType
    OutputTimes::Union{FloatType, Vector{FloatType}}
    OutputIterationCounter::Int
    StepsTakenForLastOutput::Int
    CurrentTimeStep::FloatType
    TotalTime::FloatType
    SimulationTime::FloatType
    IndexCounter::Int
    ProgressSpecification::ProgressUnknown
    VisualizeInParaview::Bool
    ExportSingleVTKHDF::Bool
    ExportGridCells::Bool
    OutputVariables::Vector{String}
    OpenLogFile::Bool
    FlagOutputKernelValues::Bool
    FlagLog::Bool
    FlagShifting::Bool
    FlagSingleStepTimeStepping::Bool
    ChunkMultiplier::Int
    FlagMDBCSimple::Bool
    # GPU specific options
    GPUSyncTimers::Bool          # synchronize after every phase so the timer output is meaningful
    GPUDeterministicSort::Bool   # sort particles inside each cell for bitwise reproducible runs
    GPUMaxCells::Int             # safety limit for the size of the neighbour grid
    GPUInteractionThreads::Int   # threads per block for the interaction kernels
    GPULanesPerParticle::Int     # warp lanes per particle in the gather kernels (0 = automatic)
    GPUBoundaryForces::Bool      # also evaluate the momentum equation for boundary particles (CPU parity)
    GPUAsyncOutput::Bool         # write output files on a Julia task while the GPU continues
end

const DEFAULT_OUTPUT_VARIABLES = [
    "ChunkID",
    "Kernel",
    "KernelGradient",
    "Density",
    "Pressure",
    "Velocity",
    "Acceleration",
    "BoundaryBool",
    "ID",
    "Type",
    "GroupMarker",
    "GhostPoints",
    "GhostNormals",
]

_output_times(::Type{T}, x::Real) where {T} = T(x)
_output_times(::Type{T}, x::AbstractVector) where {T} = Vector{T}(x)

function SimulationMetaData{Dimensions, FloatType}(;
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
        ProgressSpecification::ProgressUnknown  = ProgressUnknown(desc = "Simulation time per output each:", spinner = true, showspeed = true),
        VisualizeInParaview::Bool               = true,
        ExportSingleVTKHDF::Bool                = true,
        ExportGridCells::Bool                   = false,
        OutputVariables::Vector{String}         = copy(DEFAULT_OUTPUT_VARIABLES),
        OpenLogFile::Bool                       = true,
        FlagOutputKernelValues::Bool            = false,
        FlagLog::Bool                           = false,
        FlagShifting::Bool                      = false,
        FlagSingleStepTimeStepping::Bool        = false,
        ChunkMultiplier::Int                    = 1,
        FlagMDBCSimple::Bool                    = false,
        GPUSyncTimers::Bool                     = false,
        GPUDeterministicSort::Bool              = true,
        GPUMaxCells::Int                        = 50_000_000,
        GPUInteractionThreads::Int              = 128,
        GPULanesPerParticle::Int                = 0,
        GPUBoundaryForces::Bool                 = true,
        GPUAsyncOutput::Bool                    = true,
    ) where {Dimensions, FloatType <: AbstractFloat}
    return SimulationMetaData{Dimensions, FloatType}(
        SimulationName, SaveLocation, HourGlass, Iteration,
        FloatType(OutputEach), _output_times(FloatType, OutputTimes),
        OutputIterationCounter, StepsTakenForLastOutput,
        FloatType(CurrentTimeStep), FloatType(TotalTime), FloatType(SimulationTime),
        IndexCounter, ProgressSpecification, VisualizeInParaview, ExportSingleVTKHDF, ExportGridCells,
        OutputVariables, OpenLogFile, FlagOutputKernelValues, FlagLog, FlagShifting,
        FlagSingleStepTimeStepping, ChunkMultiplier, FlagMDBCSimple,
        GPUSyncTimers, GPUDeterministicSort, GPUMaxCells, GPUInteractionThreads, GPULanesPerParticle,
        GPUBoundaryForces, GPUAsyncOutput,
    )
end

# Allow `meta.OutputTimes = 0.01` for Float32 meta data as well.
function Base.setproperty!(m::SimulationMetaData{D, T}, name::Symbol, x) where {D, T}
    if name === :OutputTimes
        return setfield!(m, name, _output_times(T, x))
    else
        return setfield!(m, name, convert(fieldtype(typeof(m), name), x))
    end
end

end
