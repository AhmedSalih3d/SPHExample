using BenchmarkTools
using StaticArrays
using LoopVectorization


# READ: https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/17
# Generate the needed constants
N     = 6000
Cb    = inv(Float64(100000))
invCb = inv(Cb)
xᵢⱼᶻ  = rand(Float64,N*4)
xᵢⱼ   = rand(SVector{3,Float64},N*4)
γ⁻¹   = Float64(1/7)
ρ₀    = Float64(1000)
g     = Float64(9.81)
drhopLp = zeros(Float64, length(xᵢⱼ))
drhopLn = zeros(Float64, length(xᵢⱼ))


function fancy7th(x::Float64)
    # todo tune the magic constant
    # initial guess based on fast inverse sqrt trick but adjusted to compute x^(1/7)
    # t = copysign(reinterpret(Float64, 0x36cd000000000000 + reinterpret(UInt64,abs(x))÷7), x)
    # @fastmath for _ in 1:2
    #     # newton's method for t^3 - x/t^4 = 0
    #     t2 = t*t
    #     t3 = t2*t
    #     t4 = t2*t2
    #     xot4 = x/t4
    #     t = t - t*(t3 - xot4)/(4*t3 + 3*xot4)
    # end
    # t
    return x^0.14
end

faux(ρ₀, P, Cb, γ⁻¹)  = ρ₀ * ( ^( 1 + (P * Cb), γ⁻¹)   - 1)
faux_fancy(ρ₀, P, Cb) = ρ₀ * ( fancy7th( 1 + (P * Cb)) - 1)


function loopvectorization_approach3(invCb,xᵢⱼᶻ,ρ₀,γ⁻¹,g,drhopLp,drhopLn)
        @tturbo for iter in eachindex(xᵢⱼᶻ)
               # IF I COMMENT '* -xᵢⱼᶻ[iter]' NO ALLOCATIONS, WHY?
               Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼᶻ[iter]
               Pⱼᵢᴴ  = -Pᵢⱼᴴ
               ρᵢⱼᴴ  = faux(ρ₀, Pᵢⱼᴴ, invCb, γ⁻¹)
               ρⱼᵢᴴ  = faux(ρ₀, Pⱼᵢᴴ, invCb, γ⁻¹)
               drhopLp[iter] = ρᵢⱼᴴ
               drhopLn[iter] = ρⱼᵢᴴ
        end
    end

function fancy_loopvectorization_approach3(invCb,xᵢⱼᶻ,ρ₀,g,drhopLp,drhopLn)
    @tturbo for iter in eachindex(xᵢⱼᶻ)
           # IF I COMMENT '* -xᵢⱼᶻ[iter]' NO ALLOCATIONS, WHY?
           Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼᶻ[iter]
           Pⱼᵢᴴ  = -Pᵢⱼᴴ
           ρᵢⱼᴴ  = faux_fancy(ρ₀, Pᵢⱼᴴ, invCb)
           ρⱼᵢᴴ  = faux_fancy(ρ₀, Pⱼᵢᴴ, invCb)
           drhopLp[iter] = ρᵢⱼᴴ
           drhopLn[iter] = ρⱼᵢᴴ
    end
end



# l3 = @benchmark loopvectorization_approach3($invCb,$xᵢⱼᶻ,$ρ₀, $γ⁻¹, $g,$drhopLp,$drhopLn)
f3 = @benchmark fancy_loopvectorization_approach3($invCb,$xᵢⱼᶻ,$ρ₀, $g,$drhopLp,$drhopLn)

display(l3)
display(f3)
