module SimulationMetaDataConfiguration

using Parameters
using TimerOutputs
using ProgressMeter

export SimulationMetaData


@with_kw mutable struct SimulationMetaData{Dimensions, FloatType <: AbstractFloat}
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput                  = TimerOutput()
    Iteration::Int                          = 0
    OutputEach::FloatType                   = 0.02 #seconds
    OutputIterationCounter::Int             = 0
    StepsTakenForLastOutput::Int            = 0
    CurrentTimeStep::FloatType              = 0
    TotalTime::FloatType                    = 0
    SimulationTime::FloatType               = 0
    IndexCounter::Int                       = 0
    ProgressSpecification::ProgressUnknown  =  ProgressUnknown(desc="Simulation time per output each:", spinner=true, showspeed=true) 
    VisualizeInParaview::Bool               = true
    ExportSingleVTKHDF::Bool                = true
    ExportGridCells::Bool                   = false    
    OpenLogFile::Bool                       = true
    FlagOutputKernelValues::Bool            = false     
    FlagLog::Bool                           = false
    FlagShifting::Bool                      = false
    FlagSingleStepTimeStepping::Bool        = false
    FlagMDBCSimple::Bool                    = false
end

end