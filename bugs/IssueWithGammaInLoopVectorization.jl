using BenchmarkTools
using StaticArrays
using LoopVectorization


# READ: https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/17
# Generate the needed constants
N     = 6000
Cb    = inv(Float64(100000))
xᵢⱼᶻ  = rand(Float64,N*4)
xᵢⱼ   = rand(SVector{3,Float64},N*4)
ρ₀    = Float64(1000)
g     = Float64(9.81)
drhopLp = zeros(Float64, length(xᵢⱼ))
drhopLn = zeros(Float64, length(xᵢⱼ))

function loopvectorization_approach(Cb,xᵢⱼᶻ,ρ₀,γ⁻¹,g,drhopLp,drhopLn)
    @tturbo for iter in eachindex(xᵢⱼᶻ)
        # IF I COMMENT '* -xᵢⱼᶻ[iter]' NO ALLOCATIONS, WHY?
        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼᶻ[iter]
        ρᵢⱼᴴ  = ρ₀ * ( ^( 1 + (Pᵢⱼᴴ * Cb), 0.14285714285714285) - 1)  
        Pⱼᵢᴴ  = -Pᵢⱼᴴ
        ρⱼᵢᴴ  = ρ₀ * ( ^( 1 + (Pⱼᵢᴴ * Cb), 0.14285714285714285) - 1)
        
        drhopLp[iter] = ρᵢⱼᴴ
        drhopLn[iter] = ρⱼᵢᴴ
    end
end

l  = @benchmark loopvectorization_approach($Cb,$xᵢⱼᶻ,$ρ₀,$γ⁻¹,$g,$drhopLp,$drhopLn)

display(l)

# julia> l  = @benchmark loopvectorization_approach($Cb,$xᵢⱼᶻ,$ρ₀,$γ⁻¹,$g,$drhopLp,$drhopLn)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  339.900 μs …  4.994 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     358.600 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   382.900 μs ± 97.110 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

#    ▅█▇▅▄▃▂▂▁▁▁▁▁          ▁▁▁▁                                 ▂
#   █████████████████▇▇█▇█████████▇█▇▇▇▇▇▆▄▆▅▆▆▆▅▅▆▄▆▅▁▅▅▅▄▄▆▆▇▇ █
#   340 μs        Histogram: log(frequency) by time       718 μs <

#  Memory estimate: 0 bytes, allocs estimate: 0.