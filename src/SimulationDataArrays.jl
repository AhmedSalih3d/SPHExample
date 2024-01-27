module SimulationDataArrays

export SimulationDataResults

using Parameters
using StaticArrays

@with_kw mutable struct SimulationDataResults{D,T}
    NumberOfParticles ::Int                                 
    Kernel            ::Vector{T}            = zeros(T,NumberOfParticles)
    KernelGradient    ::Vector{SVector{D,T}} = zeros(SVector{D,T},NumberOfParticles)
    Density           ::Vector{T}            = zeros(T,NumberOfParticles)
    Position          ::Vector{SVector{D,T}} = zeros(SVector{D,T},NumberOfParticles)
    Acceleration      ::Vector{SVector{D,T}} = zeros(SVector{D,T},NumberOfParticles)
    Velocity          ::Vector{SVector{D,T}} = zeros(SVector{D,T},NumberOfParticles)
end

end