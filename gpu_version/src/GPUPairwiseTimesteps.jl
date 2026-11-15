module GPUPairwiseTimesteps
using CUDA, StaticArrays
using ..GPUStepState: step_active
using ..GPUReductions: warp_reduce

@inline pair_max(a,b) = max.(a,b)

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
    # One atomic maximum per block, rather than one per particle or pair.
    if blockDim().x % Int32(32) != 0
        CUDA.@atomic signal[1] = max(signal[1],rate)
        return nothing
    end
    tid = threadIdx().x-Int32(1)
    lane,warp = tid % Int32(32)+Int32(1),tid ÷ Int32(32)+Int32(1)
    nwarps = blockDim().x ÷ Int32(32)
    shared = CuStaticSharedArray(SVector{1,T},32)
    value = warp_reduce(SVector(rate),pair_max,lane,Int32(32))
    lane == 1 && (@inbounds shared[warp] = value)
    sync_threads()
    if warp == 1
        value = lane <= nwarps ? (@inbounds shared[lane]) : SVector(zero(T))
        value = warp_reduce(value,pair_max,lane,Int32(32))
        if lane == 1
            CUDA.@atomic signal[1] = max(signal[1],value[1])
        end
    end
    return nothing
end

@inline acoustic_denominator(::Nothing,speed,c0,h) = c0+speed
# Keep this separate from reference viscous_dt: that argument also selects
# reference pressure/viscosity arithmetic in the interaction kernel.
@inline acoustic_denominator(signal,speed,c0,h) = max(c0,10speed)+h*(@inbounds signal[1])
end
