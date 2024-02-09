module SimulationDataArrays

export ResetArrays!, ResizeBuffers!

using Parameters
using StaticArrays

ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

function ResizeBuffers!(args...; N::Int = 0)
    for a in args
        if length(a) != N resize!(a, N) end
    end
    args
end

end