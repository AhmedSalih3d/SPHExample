module SimulationMetaDataConfiguration

using Parameters
using TimerOutputs
using ProgressMeter

export SimulationMetaData, ShiftingMode, NoShifting, PlanarShifting,
       KernelOutputMode, NoKernelOutput, StoreKernelOutput

abstract type ShiftingMode end
struct NoShifting    <: ShiftingMode end
struct PlanarShifting <: ShiftingMode end

abstract type KernelOutputMode end
struct NoKernelOutput    <: KernelOutputMode end
struct StoreKernelOutput <: KernelOutputMode end

@with_kw mutable struct SimulationMetaData{Dimensions,
                                           FloatType <: AbstractFloat,
                                           SMode <: ShiftingMode,
                                           KMode <: KernelOutputMode}
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput                  = TimerOutput()
    Iteration::Int                          = 0
    OutputEach::FloatType                   = 0.02 #seconds
    OutputTimes::Union{FloatType,Vector{FloatType}} = OutputEach
    OutputIterationCounter::Int             = 0
    StepsTakenForLastOutput::Int            = 0
    CurrentTimeStep::FloatType              = 0
    TotalTime::FloatType                    = 0
    SimulationTime::FloatType               = 0
    IndexCounter::Int                       = 0
    ProgressSpecification::ProgressUnknown  = ProgressUnknown(desc="Simulation time per output each:", spinner=true, showspeed=true)
    VisualizeInParaview::Bool               = true
    ExportSingleVTKHDF::Bool                = true
    ExportGridCells::Bool                   = false
    OutputVariables::Vector{String}         = [
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
    OpenLogFile::Bool                       = true
    FlagLog::Bool                           = false
    FlagSingleStepTimeStepping::Bool        = false
    ChunkMultiplier::Int                    = 1
    FlagMDBCSimple::Bool                    = false
end
SimulationMetaData{D,T,S}(; kwargs...) where {D,T,S<:ShiftingMode} =
    SimulationMetaData{D,T,S,NoKernelOutput}(; kwargs...)
SimulationMetaData{D,T}(; kwargs...) where {D,T} =
    SimulationMetaData{D,T,NoShifting,NoKernelOutput}(; kwargs...)

end

