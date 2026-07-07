module SimulationMetaDataConfiguration

using Parameters
using TimerOutputs


export SimulationMetaData, UpdateMetaData!, ShiftingMode, NoShifting, PlanarShifting,
       KernelOutputMode, NoKernelOutput, StoreKernelOutput,
       MDBCMode, NoMDBC, SimpleMDBC,
       LogMode, NoLog, StoreLog,
       TimeSteppingMode, SymplecticTimeStepping, SingleNeighborTimeStepping

abstract type ShiftingMode end
struct NoShifting    <: ShiftingMode end
struct PlanarShifting <: ShiftingMode end

abstract type KernelOutputMode end
struct NoKernelOutput    <: KernelOutputMode end
struct StoreKernelOutput <: KernelOutputMode end

abstract type MDBCMode end
struct NoMDBC    <: MDBCMode end
struct SimpleMDBC <: MDBCMode end

abstract type LogMode end
struct NoLog    <: LogMode end
struct StoreLog <: LogMode end

abstract type TimeSteppingMode end
struct SymplecticTimeStepping    <: TimeSteppingMode end
struct SingleNeighborTimeStepping <: TimeSteppingMode end

@with_kw mutable struct SimulationMetaData{Dimensions,
                                           FloatType <: AbstractFloat,
                                           SMode <: ShiftingMode,
                                           KMode <: KernelOutputMode,
                                           BMode <: MDBCMode,
                                           LMode <: LogMode}
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput                  = TimerOutput()
    Iteration::Int                          = 0
    OutputEach::FloatType                   = FloatType(0.02) #seconds
    OutputTimes::Union{FloatType,Vector{FloatType}} = OutputEach
    OutputIterationCounter::Int             = 0
    StepsTakenForLastOutput::Int            = 0
    CurrentTimeStep::FloatType              = zero(FloatType)
    TotalTime::FloatType                    = zero(FloatType)
    SimulationTime::FloatType               = zero(FloatType)
    TimeSteps                               = FloatType[]
    IndexCounter::Int                       = 0
    VisualizeInParaview::Bool               = true
    ExportSingleVTKHDF::Bool                = true
    ExportGridCells::Bool                   = false
    ExportGridCellParticleCounts::Bool      = false
    OpenLogFile::Bool                       = true
    Δx::FloatType                           = zero(FloatType)
    TimeSteppingMode::TimeSteppingMode      = SingleNeighborTimeStepping()
end

const SimulationMetaDataKeywordNames = (:SimulationName, :SaveLocation, :HourGlass, :Iteration, :OutputEach,
                                        :OutputTimes, :OutputIterationCounter, :StepsTakenForLastOutput,
                                        :CurrentTimeStep, :TotalTime, :SimulationTime, :TimeSteps,
                                        :IndexCounter, :VisualizeInParaview, :ExportSingleVTKHDF,
                                        :ExportGridCells, :ExportGridCellParticleCounts, :OpenLogFile,
                                        :Δx, :TimeSteppingMode)

@inline ConvertMetaDataFloat(::Type{T}, Value::Real) where {T <: AbstractFloat} = T(Value)
@inline ConvertMetaDataFloat(::Type{T}, Value::AbstractVector{<:Real}) where {T <: AbstractFloat} = T.(Value)
@inline ConvertMetaDataFloat(::Type{T}, Value) where {T <: AbstractFloat} = Value

function SimulationMetaData{D,T,S,K,B,L}(; kwargs...) where {D,T<:AbstractFloat,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode,L<:LogMode}
    for Key in keys(kwargs)
        Key in SimulationMetaDataKeywordNames || throw(ArgumentError("unsupported SimulationMetaData keyword: $Key"))
    end

    haskey(kwargs, :SimulationName) || throw(UndefKeywordError(:SimulationName))
    haskey(kwargs, :SaveLocation) || throw(UndefKeywordError(:SaveLocation))

    SimulationName = kwargs[:SimulationName]
    SaveLocation = kwargs[:SaveLocation]
    HourGlass = get(kwargs, :HourGlass, TimerOutput())
    Iteration = get(kwargs, :Iteration, 0)
    OutputEach = ConvertMetaDataFloat(T, get(kwargs, :OutputEach, T(0.02)))
    OutputTimes = ConvertMetaDataFloat(T, get(kwargs, :OutputTimes, OutputEach))
    OutputIterationCounter = get(kwargs, :OutputIterationCounter, 0)
    StepsTakenForLastOutput = get(kwargs, :StepsTakenForLastOutput, 0)
    CurrentTimeStep = ConvertMetaDataFloat(T, get(kwargs, :CurrentTimeStep, zero(T)))
    TotalTime = ConvertMetaDataFloat(T, get(kwargs, :TotalTime, zero(T)))
    SimulationTime = ConvertMetaDataFloat(T, get(kwargs, :SimulationTime, zero(T)))
    TimeSteps = ConvertMetaDataFloat(T, get(kwargs, :TimeSteps, T[]))
    IndexCounter = get(kwargs, :IndexCounter, 0)
    VisualizeInParaview = get(kwargs, :VisualizeInParaview, true)
    ExportSingleVTKHDF = get(kwargs, :ExportSingleVTKHDF, true)
    ExportGridCells = get(kwargs, :ExportGridCells, false)
    ExportGridCellParticleCounts = get(kwargs, :ExportGridCellParticleCounts, false)
    OpenLogFile = get(kwargs, :OpenLogFile, true)
    Δx = ConvertMetaDataFloat(T, get(kwargs, :Δx, zero(T)))
    TimeSteppingMode = get(kwargs, :TimeSteppingMode, SingleNeighborTimeStepping())

    return SimulationMetaData{D,T,S,K,B,L}(SimulationName, SaveLocation, HourGlass, Iteration, OutputEach,
                                          OutputTimes, OutputIterationCounter, StepsTakenForLastOutput,
                                          CurrentTimeStep, TotalTime, SimulationTime, TimeSteps,
                                          IndexCounter, VisualizeInParaview, ExportSingleVTKHDF,
                                          ExportGridCells, ExportGridCellParticleCounts, OpenLogFile,
                                          Δx, TimeSteppingMode)
end
SimulationMetaData{D,T,S,K,B}(; kwargs...) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode} =
    SimulationMetaData{D,T,S,K,B,NoLog}(; kwargs...)
SimulationMetaData{D,T,S,K}(; kwargs...) where {D,T,S<:ShiftingMode,K<:KernelOutputMode} =
    SimulationMetaData{D,T,S,K,NoMDBC,NoLog}(; kwargs...)
SimulationMetaData{D,T,S}(; kwargs...) where {D,T,S<:ShiftingMode} =
    SimulationMetaData{D,T,S,NoKernelOutput,NoMDBC,NoLog}(; kwargs...)
SimulationMetaData{D,T}(; kwargs...) where {D,T} =
    SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(; kwargs...)

function UpdateMetaData!(SimMetaData, dt)
    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt

    return nothing
end

end
