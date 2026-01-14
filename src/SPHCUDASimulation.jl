module SPHCUDASimulation

export RunSimulationCUDA, AllocateCUDASimParticles, SyncParticlesFromCUDA!

using CUDA
using StaticArrays
using Parameters
using LinearAlgebra

using ..SimulationConstantsConfiguration
using ..SimulationGeometry
using ..SimulationLoggerConfiguration
using ..SimulationMetaDataConfiguration
using ..ProduceHDFVTK
using ..SPHCUDANeighborList
using ..SPHDensityDiffusionModels
using ..SPHKernels
using ..SPHViscosityModels
using ..TimeStepping: next_output_time
using ..SimulationEquations: EquationOfStateGamma7
using ..OpenExternalPrograms

struct CUDASimParticles{D, T}
    Position::CuArray{SVector{D, T}}
    Velocity::CuArray{SVector{D, T}}
    Acceleration::CuArray{SVector{D, T}}
    Density::CuArray{T}
    Pressure::CuArray{T}
    MotionLimiter::CuArray{T}
    GravityFactor::CuArray{T}
end

@inline function AllocateCUDASimParticles(SimParticles)
    return CUDASimParticles(
        CuArray(SimParticles.Position),
        CuArray(SimParticles.Velocity),
        CuArray(SimParticles.Acceleration),
        CuArray(SimParticles.Density),
        CuArray(SimParticles.Pressure),
        CuArray(SimParticles.MotionLimiter),
        CuArray(SimParticles.GravityFactor),
    )
end

@inline function SyncParticlesFromCUDA!(SimParticles, CUDAParticles::CUDASimParticles)
    copyto!(SimParticles.Position, Array(CUDAParticles.Position))
    copyto!(SimParticles.Velocity, Array(CUDAParticles.Velocity))
    copyto!(SimParticles.Acceleration, Array(CUDAParticles.Acceleration))
    copyto!(SimParticles.Density, Array(CUDAParticles.Density))
    copyto!(SimParticles.Pressure, Array(CUDAParticles.Pressure))
    return nothing
end

@inline function ComputeBounds(Position)
    MinCorner = reduce((A, B) -> min.(A, B), Position)
    MaxCorner = reduce((A, B) -> max.(A, B), Position)
    return MinCorner, MaxCorner
end

@inline function BuildGrid(MinCorner, MaxCorner, CellSize)
    Extents = MaxCorner - MinCorner
    Dims = SVector{length(MinCorner), Int}(ceil.(Int, Extents ./ CellSize) .+ 1)
    return CUDACellGrid(MinCorner, CellSize, Dims)
end

@generated function GravityVector(::Val{D}, Value::T) where {D, T}
    Entries = [:(Index == $D ? Value : zero(T)) for Index in 1:D]
    return quote
        return SVector{$D, T}($(Entries...))
    end
end

function PressureKernel!(Pressure, Density, SimConstants)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(Pressure)
        ρ = Density[Index]
        Pressure[Index] = EquationOfStateGamma7(ρ, SimConstants.c₀, SimConstants.ρ₀)
    end
    return nothing
end

function LimitDensityKernel!(Density, ρ₀, MotionLimiter)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(Density)
        if Density[Index] < ρ₀ && MotionLimiter[Index] == zero(eltype(MotionLimiter))
            Density[Index] = ρ₀
        end
    end
    return nothing
end

function DensityEpsiKernel!(Density, DensityRate, DensityHalf, Δt)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(Density)
        Epsi = -(DensityRate[Index] / DensityHalf[Index]) * Δt
        Density[Index] *= (2 - Epsi) / (2 + Epsi)
    end
    return nothing
end

function HalfTimeStepKernel!(Position, Velocity, Acceleration, Density,
                             GravityFactor, MotionLimiter,
                             PositionHalf, VelocityHalf, DensityHalf, DensityRate,
                             DtHalf, GravityValue, Dimensions)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(Position)
        Acc = Acceleration[Index] + GravityVector(Dimensions, GravityValue * GravityFactor[Index])
        Acceleration[Index] = Acc
        Factor = MotionLimiter[Index]
        PositionHalf[Index] = Position[Index] + Velocity[Index] * DtHalf * Factor
        VelocityHalf[Index] = Velocity[Index] + Acc * DtHalf * Factor
        DensityHalf[Index] = Density[Index] + DensityRate[Index] * DtHalf
    end
    return nothing
end

function FullTimeStepKernel!(Position, Velocity, Acceleration, GravityFactor, MotionLimiter, Dt, GravityValue, Dimensions)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(Position)
        Acc = Acceleration[Index] + GravityVector(Dimensions, GravityValue * GravityFactor[Index])
        Acceleration[Index] = Acc
        Factor = MotionLimiter[Index]
        Velocity[Index] += Acc * Dt * Factor
        Position[Index] += (((Velocity[Index] + (Velocity[Index] - Acc * Dt * Factor)) / 2) * Dt) * Factor
    end
    return nothing
end

function TimeStepBuffersKernel!(MaxVisc, MinDtForce, Position, Velocity, Acceleration, SimKernel)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(Position)
        R = Position[Index]
        V = Velocity[Index]
        A = Acceleration[Index]
        RSq = sqrt(dot(R, R))^2
        CurrVisc = abs(SimKernel.h * dot(V, R) / (RSq + SimKernel.η²))
        AMag = norm(A)
        CurrDtForce = AMag > 0 ? sqrt(SimKernel.h / AMag) : typemax(eltype(MinDtForce))
        MaxVisc[Index] = CurrVisc
        MinDtForce[Index] = CurrDtForce
    end
    return nothing
end

@inline function InitAccumulatorKernel(Index, Density, Position)
    return (zero(eltype(Density)), zero(eltype(Position)))
end

@inline function InteractionKernel(Index, NeighborIndex, Accumulator,
                                   Position, Density, Pressure, Velocity,
                                   MotionLimiter, SimKernel, SimConstants)
    DensityRate, AccelerationValue = Accumulator
    Δx = Position[Index] - Position[NeighborIndex]
    Δx² = dot(Δx, Δx)
    if Δx² <= SimKernel.H²
        Distance = sqrt(abs(Δx²))
        Q = clamp(Distance * SimKernel.h⁻¹, zero(Distance), 2)
        ∇W = ∇Wᵢⱼ(SimKernel, Q, Δx)

        ρᵢ = Density[Index]
        ρⱼ = Density[NeighborIndex]

        Vᵢ = Velocity[Index]
        Vⱼ = Velocity[NeighborIndex]
        Vᵢⱼ = Vᵢ - Vⱼ
        DensityTerm = dot(-Vᵢⱼ, ∇W)
        DensityRateContribution = -ρᵢ * (SimConstants.m₀ / ρⱼ) * DensityTerm

        LinearρFactor = inv(SimConstants.Cb * SimConstants.γ) * SimConstants.ρ₀
        Pᵢⱼᴴ = SimConstants.ρ₀ * (-SimConstants.g) * -Δx[end]
        ρᵢⱼᴴ = Pᵢⱼᴴ * LinearρFactor
        InvD = one(eltype(ρᵢ)) / (Δx² + SimKernel.η²)
        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ) * (-Δx) * InvD
        MotionFactor = MotionLimiter[Index] * MotionLimiter[NeighborIndex]
        DiffusionContribution = SimConstants.δᵩ * SimKernel.h * SimConstants.c₀ * (SimConstants.m₀ / ρⱼ) * dot(ψᵢⱼ, ∇W) * MotionFactor

        DensityRate += DensityRateContribution + DiffusionContribution

        Pᵢ = Pressure[Index]
        Pⱼ = Pressure[NeighborIndex]
        Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
        Fab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, Q, SimConstants.dx)
        AccelerationContribution = -SimConstants.m₀ * (Pfac + Fab) * ∇W

        ViscTerm = zero(AccelerationContribution)
        VDotX = dot(Vᵢⱼ, Δx)
        if VDotX < 0
            ρ̄ = 0.5 * (ρᵢ + ρⱼ)
            μᵢⱼ = SimKernel.h * VDotX / (Δx² + SimKernel.η²)
            ViscTerm = -SimConstants.m₀ * (-SimConstants.α * SimConstants.c₀ * μᵢⱼ) / ρ̄ * ∇W
        end

        AccelerationValue += AccelerationContribution + ViscTerm
    end

    return DensityRate, AccelerationValue
end

@inline function FinalKernel(Index, Accumulator, DensityRate, Acceleration)
    DensityRateValue, AccelerationValue = Accumulator
    DensityRate[Index] = DensityRateValue
    Acceleration[Index] = AccelerationValue
    return nothing
end

function ComputeNextTimeStep(MaxVisc, MinDtForce, SimConstants, SimKernel)
    MaxViscHost = maximum(Array(MaxVisc))
    MinDtHost = minimum(Array(MinDtForce))
    DtLimit = SimKernel.h / (SimConstants.c₀ + MaxViscHost)
    return SimConstants.CFL * min(MinDtHost, DtLimit)
end

function RunSimulationCUDA(;SimGeometry::Vector{Geometry{Dimensions, FloatType}},
        SimMetaData::SimulationMetaData{Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, LMode},
        SimConstants::SimulationConstants,
        SimKernel::SPHKernelInstance,
        SimLogger::SimulationLogger,
        SimParticles,
        SimViscosity::ArtificialViscosity,
        SimDensityDiffusion::LinearDensityDiffusion,
        ) where {Dimensions, FloatType, LMode<:LogMode}

    NumberOfPoints = length(SimParticles)
    CUDAParticles = AllocateCUDASimParticles(SimParticles)

    DensityRate = CUDA.zeros(FloatType, NumberOfPoints)
    VelocityHalf = CUDA.zeros(SVector{Dimensions, FloatType}, NumberOfPoints)
    PositionHalf = CUDA.zeros(SVector{Dimensions, FloatType}, NumberOfPoints)
    DensityHalf = CUDA.zeros(FloatType, NumberOfPoints)
    MaxVisc = CUDA.zeros(FloatType, NumberOfPoints)
    MinDtForce = CUDA.zeros(FloatType, NumberOfPoints)

    MinCorner, MaxCorner = ComputeBounds(SimParticles.Position)
    Grid = BuildGrid(MinCorner, MaxCorner, SimKernel.H)
    NeighborList = AllocateCUDANeighborList(Grid, NumberOfPoints)

    output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    SimMetaData.OutputIterationCounter = 1
    output.enqueue_particles(SimMetaData.OutputIterationCounter)

    InitializeLog!(SimMetaData, SimLogger, SimConstants, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)

    Threads = 256
    Blocks = cld(NumberOfPoints, Threads)
    DimensionsVal = Val(Dimensions)

    CUDA.@cuda threads=Threads blocks=Blocks PressureKernel!(CUDAParticles.Pressure, CUDAParticles.Density, SimConstants)
    CUDA.@cuda threads=Threads blocks=Blocks TimeStepBuffersKernel!(
        MaxVisc, MinDtForce, CUDAParticles.Position, CUDAParticles.Velocity, CUDAParticles.Acceleration, SimKernel,
    )
    Dt = ComputeNextTimeStep(MaxVisc, MinDtForce, SimConstants, SimKernel)

    while SimMetaData.TotalTime <= next_output_time(SimMetaData)
        UpdateNeighborsCUDA!(NeighborList, CUDAParticles.Position)

        CUDA.@cuda threads=Threads blocks=Blocks PressureKernel!(CUDAParticles.Pressure, CUDAParticles.Density, SimConstants)

        NeighborLoopPerParticleCUDA!(
            InitAccumulatorKernel,
            InteractionKernel,
            FinalKernel,
            NeighborList,
            CUDAParticles.Position,
            CUDAParticles.Density,
            CUDAParticles.Pressure,
            CUDAParticles.Velocity,
            CUDAParticles.MotionLimiter,
            SimKernel,
            SimConstants,
            DensityRate,
            CUDAParticles.Acceleration,
        )

        DtHalf = Dt * 0.5
        CUDA.@cuda threads=Threads blocks=Blocks HalfTimeStepKernel!(
            CUDAParticles.Position,
            CUDAParticles.Velocity,
            CUDAParticles.Acceleration,
            CUDAParticles.Density,
            CUDAParticles.GravityFactor,
            CUDAParticles.MotionLimiter,
            PositionHalf,
            VelocityHalf,
            DensityHalf,
            DensityRate,
            DtHalf,
            SimConstants.g,
            DimensionsVal,
        )

        CUDA.@cuda threads=Threads blocks=Blocks LimitDensityKernel!(DensityHalf, SimConstants.ρ₀, CUDAParticles.MotionLimiter)

        CUDA.@cuda threads=Threads blocks=Blocks PressureKernel!(CUDAParticles.Pressure, DensityHalf, SimConstants)

        NeighborLoopPerParticleCUDA!(
            InitAccumulatorKernel,
            InteractionKernel,
            FinalKernel,
            NeighborList,
            PositionHalf,
            DensityHalf,
            CUDAParticles.Pressure,
            VelocityHalf,
            CUDAParticles.MotionLimiter,
            SimKernel,
            SimConstants,
            DensityRate,
            CUDAParticles.Acceleration,
        )

        CUDA.@cuda threads=Threads blocks=Blocks LimitDensityKernel!(CUDAParticles.Density, SimConstants.ρ₀, CUDAParticles.MotionLimiter)
        CUDA.@cuda threads=Threads blocks=Blocks DensityEpsiKernel!(CUDAParticles.Density, DensityRate, DensityHalf, Dt)

        CUDA.@cuda threads=Threads blocks=Blocks FullTimeStepKernel!(
            CUDAParticles.Position,
            CUDAParticles.Velocity,
            CUDAParticles.Acceleration,
            CUDAParticles.GravityFactor,
            CUDAParticles.MotionLimiter,
            Dt,
            SimConstants.g,
            DimensionsVal,
        )

        CUDA.@cuda threads=Threads blocks=Blocks TimeStepBuffersKernel!(
            MaxVisc, MinDtForce, CUDAParticles.Position, CUDAParticles.Velocity, CUDAParticles.Acceleration, SimKernel,
        )
        Dt = ComputeNextTimeStep(MaxVisc, MinDtForce, SimConstants, SimKernel)

        SimMetaData.Iteration += 1
        SimMetaData.CurrentTimeStep = Dt
        SimMetaData.TotalTime += Dt

        SyncParticlesFromCUDA!(SimParticles, CUDAParticles)
        LogStep!(SimMetaData, SimLogger)
        SimMetaData.OutputIterationCounter += 1
        output.enqueue_particles(SimMetaData.OutputIterationCounter)

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            output.close_files()
            FinalizeLog!(SimMetaData, SimLogger)
            AutoOpenLogFile(SimLogger, SimMetaData)
            break
        end
    end

    return nothing
end

end
