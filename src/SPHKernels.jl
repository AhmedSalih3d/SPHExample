module SPHKernels

using Parameters
using LinearAlgebra
using FastPow

export SPHKernel, SPHKernelInstance, WendlandC2, CubicSpline, Wᵢⱼ, ∇Wᵢⱼ, tensile_correction

# Abstract type for SPH Kernels
abstract type SPHKernel end

# Specific kernel types
struct WendlandC2  <: SPHKernel end

struct CubicSpline{T<:AbstractFloat} <: SPHKernel 
    eps::T
end
CubicSpline{T}() where {T} = CubicSpline{T}(one(T))

# Internal normalization constant functions
# For this Wendland C2 function, there is not a 1D constant
@inline _αD(::Type{WendlandC2}, ::Val{2}, h) = 7 / (4 * π * h^2)
@inline _αD(::Type{WendlandC2}, ::Val{3}, h) = 21 / (16 * π * h^3)

@inline _αD(::Type{CubicSpline{T}}, ::Val{1}, h) where {T} = 2 / (3 * h)
@inline _αD(::Type{CubicSpline{T}}, ::Val{2}, h) where {T} = 10 / (7 * π * h^2)
@inline _αD(::Type{CubicSpline{T}}, ::Val{3}, h) where {T} = 1 / (π * h^3)

# General SPH Kernel Type
@with_kw struct SPHKernelInstance{KernelType, Dimensions, FloatType}
    kernel::KernelType
    k::FloatType   = 2.0          ; @assert k   > 0 "Scaling factor k must be positive"
    h::FloatType                  ; @assert h   > 0 "Smoothing length h must be positive"
    h⁻¹::FloatType = 1 / h        ; @assert h⁻¹ > 0 "Inverse smoothing length h⁻¹ must be positive"
    H::FloatType   = k * h        ; @assert H   > 0 "Support radius H must be positive"
    H⁻¹::FloatType = 1/H          ; @assert H⁻¹ > 0 "InverseCutOff must be greater than zero"
    H²::FloatType  = H * H        ; @assert H²  > 0 "Support radius squared H² must be positive"
    αD::FloatType                 ; @assert αD  > 0 "Normalization constant αD must be positive"
    η²::FloatType  = (0.01 * h)^2 ; @assert η²  ≥ 0 "η² must be non-negative"
end

function SPHKernelInstance{D, T}(kernel::KernelType, dx::T, k::T=T(2.0)) where {KernelType<:SPHKernel, D, T}
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H⁻¹ = 1/H
    H² = H * H
    αD = _αD(KernelType, Val(D), h)
    η² = (0.01 * h)^2
    return SPHKernelInstance{KernelType, D, T}(kernel=kernel, k=k, h=h, h⁻¹=h⁻¹, H=H, H⁻¹ = H⁻¹, H²=H², αD=αD, η²=η²)
end

# Kernel Evaluation Functions
@inline function Wᵢⱼ(kernel::SPHKernelInstance{<:WendlandC2}, q::T) where {T}
    @unpack αD = kernel
    return αD * (1 - q/2)^4 * (2q + 1)
end

@inline function ∇Wᵢⱼ(kernel::SPHKernelInstance{<:WendlandC2}, q::T, xᵢⱼ) where {T}
    @unpack h, αD, η² = kernel
    # Subhan Allah, if this math is correct, then η² can be avoided
    # denom = (q * h + η²)
    # factor = αD * 5 * (q - 2)^3 * q / (8 * h * denom)
    factor = αD * 5 * (q - 2)^3 / (8 * h * h)
    return factor * xᵢⱼ
end

@inline function Wᵢⱼ(kernel::SPHKernelInstance{<:CubicSpline}, q::T) where {T}
    @unpack αD = kernel
    return αD * (((1 - (3/2)*q^2 + (3/4)*q^3) * (0 <= q <= 1)) + ((1/4)*(2 - q)^3 * (1 < q <= 2)))
end

@inline function ∇Wᵢⱼ(kernel::SPHKernelInstance{<:CubicSpline}, q::T, xᵢⱼ) where {T}
    @unpack h, h⁻¹, αD, η² = kernel
    # r = norm(xᵢⱼ)
    # inv_r_h = 1/(r + η²)  # η² is a small regularization to avoid division by zero
    
    if 0 <= q <= 1
        dWdq = αD * (-3*q + (9/4)*q^2)
    elseif 1 < q <= 2
        dWdq = αD * (-3/4)*(2 - q)^2
    else
        dWdq = zero(T)
    end
    
    # Chain rule: ∇W = (dW/dq) * (∇q)
    # Where ∇q = xᵢⱼ/(r*h)
    return dWdq * h⁻¹ * xᵢⱼ / (norm(xᵢⱼ) + η²)
end

#---------------------------------------------------------------
# Tensile Corrections for specific kernels
#---------------------------------------------------------------
@inline function tensile_correction(instance::SPHKernelInstance{<:WendlandC2}, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
     return zero(eltype(q))
end

@inline function tensile_correction(instance::SPHKernelInstance{<:CubicSpline}, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx; n = 4)
    eps_val = instance.kernel.eps

    Wᵢⱼ(instance, q)

    Wij_q  = Wᵢⱼ(instance, q)
    Wij_dx = Wᵢⱼ(instance, dx)

    return eps_val * ( ((Pᵢ/ρᵢ^2) + (Pⱼ/ρⱼ^2)) * (Wij_q / Wij_dx)^n )
end


end # module
