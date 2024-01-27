module SimulationMetaDataConfiguration

using Parameters
using TimerOutputs

export SimulationMetaData

"""
    mutable struct SimulationMetaData

SimulationMetaData is a mutable struct representing the metadata and configuration settings for a simulation. These settings include the simulation name, save location, timing information, and various parameters that control the simulation behavior.

# Fields
- `SimulationName::String`: A string representing the name of the simulation.
- `SaveLocation::String`: A string representing the location where simulation results will be saved.
- `HourGlass::TimerOutput`: A `TimerOutput` object used for timing the simulation. Default is a new `TimerOutput`.
- `Iteration::Int`: The current iteration of the simulation. Default is 0.
- `MaxIterations::Int`: The maximum number of iterations for the simulation. Default is 1000.
- `OutputIteration::Int`: The iteration interval at which simulation results are saved. Default is 50.
- `SilentOutput::Bool`: A boolean indicating whether to suppress output during the simulation. Default is false.
- `ThreadsCPU::Int`: The number of CPU threads to use for the simulation. Default is the number of available threads.
- `FloatType::DataType`: The data type to use for floating-point values. Default is `Float64`.
- `IntType::DataType`: The data type to use for integer values. Default is `Int64`.

# Example
```julia
using SimulationMetaDataConfiguration
using TimerOutputs

# Create a SimulationMetaData instance with custom parameters
metadata = SimulationMetaData(
    SimulationName="MySimulation",
    SaveLocation="./results",
    MaxIterations=500,
    ThreadsCPU=4,
    FloatType=Float32
)
```
"""
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