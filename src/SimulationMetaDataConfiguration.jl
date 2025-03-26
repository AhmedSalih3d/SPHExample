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
    ProgressSpecification::ProgressUnknown  =  ProgressUnknown(desc="Burning the midnight oil:", spinner=true, showspeed=true) 
    FlagViscosityTreatment::Symbol          = :ArtificialViscosity; @assert in(FlagViscosityTreatment, Set((:None, :ArtificialViscosity, :Laminar, :LaminarSPS))) == true "ViscosityTreatment must be either :None, :ArtificialViscosity, :Laminar, :LaminarSPS"
    VisualizeInParaview::Bool               = true
    ExportSingleVTKHDF::Bool                = true
    ExportGridCells::Bool                   = false    
    OpenLogFile::Bool                       = true
    FlagDensityDiffusion::Bool              = false
    FlagLinearizedDDT::Bool                 = false
    FlagOutputKernelValues::Bool            = false     
    FlagLog::Bool                           = false
    FlagShifting::Bool                      = false 
end

end