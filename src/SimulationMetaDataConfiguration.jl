module SimulationMetaDataConfiguration

using Parameters
using TimerOutputs

export SimulationMetaData

@with_kw mutable struct SimulationMetaData
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput           = TimerOutput()
    Iteration::Int                   = 0
    MaxIterations::Int               = 1000
    OutputIteration::Int             = 50
    SilentOutput::Bool               = false
    ThreadsCPU::Int                  = Threads.nthreads()
    FloatType::DataType              = Float64
    IntType::DataType                = Int64        
end

end