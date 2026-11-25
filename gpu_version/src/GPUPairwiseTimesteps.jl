module GPUPairwiseTimesteps
using CUDA, StaticArrays, Adapt
using ..GPUStepState: step_active
using ..GPUReductions: warp_reduce, block_reduce, REDUCE_THREADS

@inline pair_max(a,b) = max.(a,b)

struct PairSignalPartials{A}
    values::A
    reduced::A
end
Adapt.@adapt_structure PairSignalPartials

reset_pair_signal!(::PairSignalPartials, step) = nothing
@inline predictor_reset_signal(signal) = signal
@inline predictor_reset_signal(::PairSignalPartials) = nothing

@inline function write_pair_max!(signal, value)
    CUDA.@atomic signal[1] = max(signal[1],value)
    return nothing
end

@inline function write_pair_max!(signal::PairSignalPartials, value)
    @inbounds signal.values[blockIdx().x] = value
    return nothing
end

@inline function store_partial_warp_signal!(signal, rate)
    write_pair_max!(signal, rate)
    return nothing
end

function reset_kernel!(signal,step)
    # Inactive graph nodes must retain the last completed corrector's signal.
    step_active(step) && (@inbounds signal[1] = zero(eltype(signal)))
    return nothing
end
reset_pair_signal!(::Nothing,step) = nothing
function reset_pair_signal!(signal,step)
    @cuda threads=1 blocks=1 reset_kernel!(signal,step)
    return nothing
end

@inline store_pair_signal!(::Nothing,rate) = nothing
@inline function store_pair_signal!(signal,rate::T) where {T}
    if !(signal isa PairSignalPartials) && blockDim().x % Int32(32) != 0
        store_partial_warp_signal!(signal,rate)
        return nothing
    end
    tid = threadIdx().x-Int32(1)
    lane,warp = tid % Int32(32)+Int32(1),tid ÷ Int32(32)+Int32(1)
    nwarps = cld(blockDim().x, Int32(32))
    shared = CuStaticSharedArray(SVector{1,T},32)
    warp_size = min(Int32(32), blockDim().x - (warp - Int32(1)) * Int32(32))
    value = if warp_size == Int32(32)
        warp_reduce(SVector(rate),pair_max,lane,Int32(32))
    else
        mask = (UInt32(1) << warp_size) - UInt32(1)
        offset = Int32(16)
        while offset > Int32(0)
            other = CUDA.shfl_down_sync(mask, rate, offset)
            lane + offset <= warp_size && (rate = max(rate, other))
            offset ÷= Int32(2)
        end
        SVector(rate)
    end
    lane == 1 && (@inbounds shared[warp] = value)
    sync_threads()
    if warp == 1
        value = lane <= nwarps ? (@inbounds shared[lane]) : SVector(zero(T))
        nwarps > Int32(1) && (value = warp_reduce(value,pair_max,lane,Int32(32)))
        if lane == 1
            write_pair_max!(signal,value[1])
        end
    end
    return nothing
end

@inline acoustic_denominator(::Nothing,speed,c0,h) = c0+speed
# Keep this separate from reference viscous_dt: that argument also selects
# reference pressure/viscosity arithmetic in the interaction kernel.
@inline acoustic_denominator(signal,speed,c0,h) = max(c0,10speed)+h*(@inbounds signal[1])
@inline acoustic_denominator(signal::AbstractFloat,speed,c0,h) = max(c0,10speed)+h*signal

@inline function scan_pair_maxima(values, first, stride)
    value = zero(eltype(values))
    @inbounds for i in first:stride:length(values)
        value = max(value, values[i])
    end
    return value
end

@inline function pair_signal_accumulator(signal::PairSignalPartials, acc, lane, stride)
    value = scan_pair_maxima(signal.reduced, lane, stride)
    return SVector(acc[1], acc[2], acc[3], value)
end

@inline pair_signal_accumulator(signal, acc, lane, stride) = acc
@inline reduced_pair_signal(signal, result) = signal
@inline reduced_pair_signal(::PairSignalPartials, result) = result[4]

@inline function integration_pair_maximum(signal)
    first = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    return scan_pair_maxima(signal.values, first, blockDim().x * gridDim().x)
end

function compact_pair_signal_kernel!(signal)
    value = block_reduce(SVector(integration_pair_maximum(signal)), pair_max)
    if threadIdx().x == Int32(1)
        @inbounds signal.reduced[blockIdx().x] = value[1]
    end
    return nothing
end

compact_pair_signal!(signal) = nothing
function compact_pair_signal!(signal::PairSignalPartials)
    @cuda threads=REDUCE_THREADS blocks=length(signal.reduced) compact_pair_signal_kernel!(signal)
    return nothing
end
end
