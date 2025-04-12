module SPHKernels

using Parameters

export AbstractSPHKernel, WendlandC2Kernel, CubicSplineKernel, GaussianKernel, Wᵢⱼ, ∇Wᵢⱼ

# Abstract type for SPH Kernels
abstract type AbstractSPHKernel end

# General guidance for kernel parameters:
# k   : Scaling factor of kernel support (k > 0)
# h   : Smoothing length (h = k * dx)
# H   : Kernel support domain (CutOff) (H = k * h)
# H²  : CutOffSquared (H² = H^2)
# h⁻¹ : Inverse smoothing length (h⁻¹ = 1/h)

#####################
# Wendland C2 Kernel
#####################
@with_kw struct WendlandC2Kernel{Dimensions,FloatType} <: AbstractSPHKernel
    k::FloatType  = 2.0
    h::FloatType  = k * 0.01                     ; @assert h > 0 "h must be positive"
    h⁻¹::FloatType = 1 / h                       ; @assert h⁻¹ > 0 "h⁻¹ must be positive"
    H::FloatType  = k * h                        ; @assert H > 0 "H must be positive"
    H²::FloatType = H * H                        ; @assert H² > 0 "H² must be positive"
    αD::FloatType = 7 / (4 * π * h^2)             ; @assert αD > 0 "αD must be positive"
    η²::FloatType = (0.01 * h)^2                 ; @assert η² ≥ 0 "η² must be non-negative"
end

#######################
# Cubic Spline Kernel
#######################
@with_kw struct CubicSplineKernel{Dimensions,FloatType} <: AbstractSPHKernel
    k::FloatType  = 2.0
    h::FloatType  = k * 0.01                     ; @assert h > 0 "h must be positive"
    h⁻¹::FloatType = 1 / h                       ; @assert h⁻¹ > 0 "h⁻¹ must be positive"
    H::FloatType  = k * h                        ; @assert H > 0 "H must be positive"
    H²::FloatType = H * H                        ; @assert H² > 0 "H² must be positive"
    αD::FloatType = 10 / (7 * π * h^2)            ; @assert αD > 0 "αD must be positive"
end

###################
# Gaussian Kernel
###################
@with_kw struct GaussianKernel{Dimensions,FloatType} <: AbstractSPHKernel
    k::FloatType  = 3.0
    h::FloatType  = k * 0.01                     ; @assert h > 0 "h must be positive"
    h⁻¹::FloatType = 1 / h                       ; @assert h⁻¹ > 0 "h⁻¹ must be positive"
    H::FloatType  = k * h                        ; @assert H > 0 "H must be positive"
    H²::FloatType = H * H                        ; @assert H² > 0 "H² must be positive"
    αD::FloatType = 1 / (π * h^2)                 ; @assert αD > 0 "αD must be positive"
end

# Internal helper functions to compute normalization constants based on kernel types
@inline _αD(::Type{WendlandC2Kernel{1,T}}, h::T) where {T} = 5 / (8 * h)
@inline _αD(::Type{WendlandC2Kernel{2,T}}, h::T) where {T} = 7 / (4 * π * h^2)
@inline _αD(::Type{WendlandC2Kernel{3,T}}, h::T) where {T} = 21 / (16 * π * h^3)

@inline _αD(::Type{CubicSplineKernel{1,T}}, h::T) where {T} = 2 / (3 * h)
@inline _αD(::Type{CubicSplineKernel{2,T}}, h::T) where {T} = 10 / (7 * π * h^2)
@inline _αD(::Type{CubicSplineKernel{3,T}}, h::T) where {T} = 1 / (π * h^3)

@inline _αD(::Type{GaussianKernel{1,T}}, h::T) where {T} = 1 / (sqrt(π) * h)
@inline _αD(::Type{GaussianKernel{2,T}}, h::T) where {T} = 1 / (π * h^2)
@inline _αD(::Type{GaussianKernel{3,T}}, h::T) where {T} = 1 / (π^(3/2) * h^3)

# Outer constructors to create kernels from dx
function WendlandC2Kernel{D,T}(dx::T) where {D, T}
    k = 2.0
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H² = H * H
    αD = _αD(WendlandC2Kernel{D, T}, h)
    η² = (0.01 * h)^2
    return WendlandC2Kernel{D, T}(k=k, h=h, h⁻¹=h⁻¹, H=H, H²=H², αD=αD, η²=η²)
end

function CubicSplineKernel{D,T}(dx::T) where {D, T}
    k = 2.0
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H² = H * H
    αD = _αD(CubicSplineKernel{D, T}, h)
    return CubicSplineKernel{D, T}(k=k, h=h, h⁻¹=h⁻¹, H=H, H²=H², αD=αD)
end

function GaussianKernel{D,T}(dx::T) where {D, T}
    k = 3.0
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H² = H * H
    αD = _αD(GaussianKernel{D, T}, h)
    return GaussianKernel{D, T}(k=k, h=h, h⁻¹=h⁻¹, H=H, H²=H², αD=αD)
end

# Kernel Operations
@inline function Wᵢⱼ(kernel::WendlandC2Kernel{D, T}, q::T) where {D, T}
    @unpack αD = kernel
    return αD * (1 - q / 2)^4 * (2 * q + 1)
end

@inline function ∇Wᵢⱼ(kernel::WendlandC2Kernel{D, T}, q::T, xᵢⱼ) where {D, T}
    @unpack h, αD, η² = kernel
    denom = (q * h + η²)
    factor = αD * 5 * (q - 2)^3 * q / (8 * h * denom)
    return factor * (-xᵢⱼ)
end

@inline function Wᵢⱼ(kernel::CubicSplineKernel{D, T}, q::T) where {D, T}
    @unpack αD = kernel
    return αD * (1 - 1.5q^2 + 0.75q^3)
end

@inline function ∇Wᵢⱼ(kernel::CubicSplineKernel{D, T}, q::T, xᵢⱼ) where {D, T}
    @unpack h, h⁻¹, αD = kernel
    factor = (q > 0) * ((-3q + 2.25q^2) * αD * h⁻¹)
    return factor * (-xᵢⱼ / (q * h + (q == 0.0)))
end

@inline function Wᵢⱼ(kernel::GaussianKernel{D, T}, q::T) where {D, T}
    @unpack αD = kernel
    return αD * exp(-q^2)
end

@inline function ∇Wᵢⱼ(kernel::GaussianKernel{D, T}, q::T, xᵢⱼ) where {D, T}
    @unpack h, h⁻¹, αD = kernel
    factor = -2 * αD * q * h⁻¹ * exp(-q^2)
    return factor * (-xᵢⱼ / (q * h + (q == 0.0)))
end

end # module SPHKernels