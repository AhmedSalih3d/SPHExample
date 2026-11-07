module IISPHSupport

using CUDA, StaticArrays
using ..GPUReductions: ReductionWorkspace

export IISPHWorkspace

"""Solver-owned dense buffers; pressure survives sorting, other fields are recomputed."""
mutable struct IISPHWorkspace{D,T}
    pressure::CuVector{T}
    next_pressure::CuVector{T}
    boundary_mass::CuVector{T}
    diagonal::CuVector{T}
    density_adv::CuVector{T}
    residual::CuVector{T}
    nonpressure_acceleration::CuVector{SVector{D,T}}
    velocity_adv::CuVector{SVector{D,T}}
    pressure_acceleration::CuVector{SVector{D,T}}
    reduction::ReductionWorkspace{SVector{4,T}}
end

function IISPHWorkspace{D,T}(n::Integer) where {D,T}
    n > 0 || throw(ArgumentError("IISPH requires particles"))
    scalars = ntuple(_ -> CUDA.zeros(T,n), 6)
    vectors = ntuple(_ -> CUDA.zeros(SVector{D,T},n), 3)
    return IISPHWorkspace{D,T}(scalars..., vectors..., ReductionWorkspace{SVector{4,T}}(n))
end

end
