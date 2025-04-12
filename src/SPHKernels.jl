module SPHKernels

export AbstractSPHKernel, KernelParameters, WendlandC2Kernel, CubicSplineKernel, GaussianKernel, Wᵢⱼ, ∇Wᵢⱼ

# Abstract type for SPH Kernels
abstract type AbstractSPHKernel end

# General guidance for kernel parameters:
# k   : Scaling factor of kernel support (k > 0)
# h   : Smoothing length (h = k * dx)
# H   : Kernel support domain (CutOff) (H = k * h)
# H²  : CutOffSquared (H² = H^2)
# h⁻¹ : Inverse smoothing length (h⁻¹ = 1/h)

# Common struct to hold shared kernel parameters
struct KernelParameters{T}
    k::T
    h::T
    h⁻¹::T
    H::T
    H²::T
    αD::T
    function KernelParameters(k::T, h::T, h⁻¹::T, H::T, H²::T, αD::T) where {T}
        @assert k > 0 "Scaling factor k must be positive"
        @assert h > 0 "Smoothing length h must be positive"
        @assert h⁻¹ > 0 "Inverse smoothing length h⁻¹ must be positive"
        @assert H > 0 "Support domain H must be positive"
        @assert H² > 0 "Support domain squared H² must be positive"
        @assert αD > 0 "Normalization factor αD must be positive"
        new{T}(k, h, h⁻¹, H, H², αD)
    end
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

#####################
# Wendland C2 Kernel
#####################
struct WendlandC2Kernel{Dimensions,FloatType} <: AbstractSPHKernel
    params::KernelParameters{FloatType}
    η²::FloatType
end

function WendlandC2Kernel{D, T}(dx::T, η²::T=1e-12) where {D, T}
    @assert dx > 0 "dx must be positive"
    @assert η² ≥ 0 "η² must be non-negative"
    k = 2.0
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H² = H * H
    αD = _αD(WendlandC2Kernel{D, T}, h)
    params = KernelParameters(k, h, h⁻¹, H, H², αD)
    return WendlandC2Kernel{D, T}(params, η²)
end

@inline function Wᵢⱼ(kernel::WendlandC2Kernel{D, T}, d::T) where {D, T}
    q = d * kernel.params.h⁻¹
    return kernel.params.αD * (1 - q / 2)^4 * (2 * q + 1)
end

@inline function ∇Wᵢⱼ(kernel::WendlandC2Kernel{D, T}, d::T, xᵢⱼ) where {D, T}
    q = d * kernel.params.h⁻¹
    denom = (q * kernel.params.h + kernel.η²)
    factor = kernel.params.αD * 5 * (q - 2)^3 * q / (8 * kernel.params.h * denom)
    return factor * (-xᵢⱼ)
end

#######################
# Cubic Spline Kernel
#######################
struct CubicSplineKernel{Dimensions,FloatType} <: AbstractSPHKernel
    params::KernelParameters{FloatType}
end

function CubicSplineKernel{D, T}(dx::T) where {D, T}
    @assert dx > 0 "dx must be positive"
    k = 2.0
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H² = H * H
    αD = _αD(CubicSplineKernel{D, T}, h)
    params = KernelParameters(k, h, h⁻¹, H, H², αD)
    return CubicSplineKernel{D, T}(params)
end

@inline function Wᵢⱼ(kernel::CubicSplineKernel{D, T}, d::T) where {D, T}
    q = d * kernel.params.h⁻¹
    return kernel.params.αD * (1 - 1.5q^2 + 0.75q^3)
end

@inline function ∇Wᵢⱼ(kernel::CubicSplineKernel{D, T}, d::T, xᵢⱼ) where {D, T}
    q = d * kernel.params.h⁻¹
    factor = (d > 0) * ((-3q + 2.25q^2) * kernel.params.αD * kernel.params.h⁻¹)
    return factor * (-xᵢⱼ / (d + (d == 0.0)))
end

###################
# Gaussian Kernel
###################
struct GaussianKernel{Dimensions,FloatType} <: AbstractSPHKernel
    params::KernelParameters{FloatType}
end

function GaussianKernel{D, T}(dx::T) where {D, T}
    @assert dx > 0 "dx must be positive"
    k = 3.0
    h = k * dx
    h⁻¹ = 1 / h
    H = k * h
    H² = H * H
    αD = _αD(GaussianKernel{D, T}, h)
    params = KernelParameters(k, h, h⁻¹, H, H², αD)
    return GaussianKernel{D, T}(params)
end

@inline function Wᵢⱼ(kernel::GaussianKernel{D, T}, d::T) where {D, T}
    q = d * kernel.params.h⁻¹
    return kernel.params.αD * exp(-q^2)
end

@inline function ∇Wᵢⱼ(kernel::GaussianKernel{D, T}, d::T, xᵢⱼ) where {D, T}
    q = d * kernel.params.h⁻¹
    factor = -2 * kernel.params.αD * q * kernel.params.h⁻¹ * exp(-q^2)
    return factor * (-xᵢⱼ / (d + (d == 0.0)))
end

end # module SPHKernels
