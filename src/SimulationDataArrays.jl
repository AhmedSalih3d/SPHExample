module SimulationDataArrays

export SimulationDataResults, ResetArrays!, ResizeBuffers!

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

function ResetArrays!(arrays...)
    @inbounds for array in arrays
        fill!(array,zero(eltype(array)))
    end
end

function ResizeBuffers!(args...; N::Int = 0)
    for a in args
        if length(a) != N resize!(a, N) end
    end
    args
end

end