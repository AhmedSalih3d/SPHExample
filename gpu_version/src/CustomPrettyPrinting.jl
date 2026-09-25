module CustomPrettyPrinting

using TimerOutputs
using ProgressMeter
import Base: show

using ..SimulationConstantsConfiguration: SimulationConstants
using ..SimulationMetaDataConfiguration: SimulationMetaData
using ..SPHKernels: SPHKernelInstance

"""Return a concise representation of a value for pretty printing."""
_val_repr(val) = string(val)
_val_repr(val::AbstractString) = string('"', val, '"')
_val_repr(val::AbstractArray) = string("Array{", eltype(val), "}(", size(val), ")")
_val_repr(::TimerOutput) = ""
_val_repr(::ProgressMeter.AbstractProgress) = ""

function show(io::IO, sc::SimulationConstants{T}) where {T}
    println(io, "SimulationConstants{$T}")
    for field in fieldnames(typeof(sc))
        val = getfield(sc, field)
        repr = _val_repr(val)
        suffix = isempty(repr) ? "" : " $(repr)"
        println(io, "  $(field): $(typeof(val))$(suffix)")
    end
end

function show(io::IO, meta::SimulationMetaData{D, T}) where {D, T}
    println(io, "SimulationMetaData{$D, $T}")
    for field in fieldnames(typeof(meta))
        val = getfield(meta, field)
        repr = _val_repr(val)
        suffix = isempty(repr) ? "" : " $(repr)"
        println(io, "  $(field): $(typeof(val))$(suffix)")
    end
end

function show(io::IO, ker::SPHKernelInstance{K, D, T}) where {K, D, T}
    println(io, "SPHKernelInstance{$K, $D, $T}")
    for field in fieldnames(typeof(ker))
        val = getfield(ker, field)
        repr = _val_repr(val)
        suffix = isempty(repr) ? "" : " $(repr)"
        println(io, "  $(field): $(typeof(val))$(suffix)")
    end
end

end # module
