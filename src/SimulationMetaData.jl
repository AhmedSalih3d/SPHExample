module SimulationMetaData

using Parameters
using TimerOutputs

export SimMetaData

@with_kw mutable struct SimMetaData
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput           = TimerOutput()
    Iteration::Int                   = 0
    MaxIterations::Int               = Inf
    OutputIteration::Int             = 50
    SilentOutput::Bool               = false
    ThreadsCPU::Int                  = Threads.nthreads()
    FloatType::DataType              = Float64
    IntType::DataType                = Int64        
end

end