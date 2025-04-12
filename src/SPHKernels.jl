module SPHKernels

using Parameters

export SPHKernel, SPHKernelInstance, WendlandC2, CubicSpline, Gaussian, Wᵢⱼ, ∇Wᵢⱼ

# Abstract type for SPH Kernels
abstract type SPHKernel end

# Specific kernel types
struct WendlandC2  <: SPHKernel end
struct CubicSpline <: SPHKernel end
struct Gaussian    <: SPHKernel  end

# General SPH Kernel Type
@with_kw struct SPHKernelInstance{KernelType<:SPHKernel, Dimensions, FloatType}
    k::FloatType  = 2.0                                ; @assert k > 0 "Scaling factor k must be positive"
    h::FloatType  = k * 0.01                           ; @assert h > 0 "Smoothing length h must be positive"
    h⁻¹::FloatType = 1 / h                             ; @assert h⁻¹ > 0 "Inverse smoothing length h⁻¹ must be positive"
    H::FloatType  = k * h                              ; @assert H > 0 "Support radius H must be positive"
    H²::FloatType = H * H                              ; @assert H² > 0 "Support radius squared H² must be positive"
    αD::FloatType                                      ; @assert αD > 0 "Normalization constant αD must be positive"
    η²::FloatType = (0.01 * h)^2                       ; @assert η² ≥ 0 "η² must be non-negative"
end

# Internal normalization constant functions
@inline _αD(::Type{WendlandC2}, ::Val{1}, h) = 5 / (8 * h)
@inline _αD(::Type{WendlandC2}, ::Val{2}, h) = 7 / (4 * π * h^2)
@inline _αD(::Type{WendlandC2}, ::Val{3}, h) = 21 / (16 * π * h^3)

@inline _αD(::Type{CubicSpline}, ::Val{1}, h) = 2 / (3 * h)
@inline _αD(::Type{CubicSpline}, ::Val{2}, h) = 10 / (7 * π * h^2)
@inline _αD(::Type{CubicSpline}, ::Val{3}, h) = 1 / (π * h^3)

@inline _αD(::Type{Gaussian}, ::Val{1}, h) = 1 / (sqrt(π) * h)
@inline _αD(::Type{Gaussian}, ::Val{2}, h) = 1 / (π * h^2)
@inline _αD(::Type{Gaussian}, ::Val{3}, h) = 1 / (π^(3/2) * h^3)

# Constructor for SPHKernelInstance
function SPHKernelInstance{KernelType, D, T}(dx::T) where {KernelType<:SPHKernel, D, T}
    k = 2.0
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H² = H * H
    αD = _αD(KernelType, Val(D), h)
    η² = (0.01 * h)^2
    return SPHKernelInstance{KernelType, D, T}(k=k, h=h, h⁻¹=h⁻¹, H=H, H²=H², αD=αD, η²=η²)
end

# Kernel Evaluation Functions
@inline function Wᵢⱼ(kernel::SPHKernelInstance{WendlandC2,D,T}, q::T) where {D,T}
    @unpack αD = kernel
    return αD * (1 - q/2)^4 * (2q + 1)
end

@inline function ∇Wᵢⱼ(kernel::SPHKernelInstance{WendlandC2,D,T}, q::T, xᵢⱼ) where {D,T}
    @unpack h, αD, η² = kernel
    denom = (q * h + η²)
    factor = αD * 5 * (q - 2)^3 * q / (8 * h * denom)
    return factor * (-xᵢⱼ)
end

@inline function Wᵢⱼ(kernel::SPHKernelInstance{CubicSpline,D,T}, q::T) where {D,T}
    @unpack αD = kernel
    return αD * (1 - 1.5q^2 + 0.75q^3)
end

@inline function ∇Wᵢⱼ(kernel::SPHKernelInstance{CubicSpline,D,T}, q::T, xᵢⱼ) where {D,T}
    @unpack h, h⁻¹, αD = kernel
    factor = (q > 0) * ((-3q + 2.25q^2) * αD * h⁻¹)
    return factor * (-xᵢⱼ / (q * h + (q == 0.0)))
end

@inline function Wᵢⱼ(kernel::SPHKernelInstance{Gaussian,D,T}, q::T) where {D,T}
    @unpack αD = kernel
    return αD * exp(-q^2)
end

@inline function ∇Wᵢⱼ(kernel::SPHKernelInstance{Gaussian,D,T}, q::T, xᵢⱼ) where {D,T}
    @unpack h, h⁻¹, αD = kernel
    factor = -2 * αD * q * h⁻¹ * exp(-q^2)
    return factor * (-xᵢⱼ / (q * h + (q == 0.0)))
end

end # module
