module SPHCUDA

export RunSimulationCUDA, CUDAAvailable

using CUDA
using LinearAlgebra
using Parameters
using StaticArrays
using TimerOutputs

using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..SimulationGeometry
using ..SimulationEquations
using ..OpenExternalPrograms
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..SPHNeighborList
using ..SPHCellList

using StructArrays: StructArray

CUDAAvailable() = CUDA.functional()

struct LaunchConfig
    Threads::Int
    Blocks::Int
end

@inline function KernelLaunchConfig(length)
    if length == 0
        return LaunchConfig(1, 1)
    end
    threads = min(256, length)
    return LaunchConfig(threads, cld(length, threads))
end

struct CUDAParticleBuffers{D, T}
    Position::CuArray{SVector{D, T}, 1}
    Density::CuArray{T, 1}
    Pressure::CuArray{T, 1}
    Velocity::CuArray{SVector{D, T}, 1}
    Acceleration::CuArray{SVector{D, T}, 1}
    MotionLimiter::CuArray{T, 1}
    GravityFactor::CuArray{T, 1}
end

struct CUDASupportBuffers{D, T}
    DρdtI::CuArray{T, 1}
    Velocityₙ⁺::CuArray{SVector{D, T}, 1}
    Positionₙ⁺::CuArray{SVector{D, T}, 1}
    ρₙ⁺::CuArray{T, 1}
    ∇Cᵢ::CuArray{SVector{D, T}, 1}
    ∇◌rᵢ::CuArray{T, 1}
    ΔtViscous::CuArray{T, 1}
    ΔtForce::CuArray{T, 1}
    ΔxScratch::CuArray{T, 1}
end

mutable struct CUDAGridBuffers{D}
    CellCoords::CuArray{SVector{D, Int}, 1}
    BucketCounts::CuArray{Int, 1}
    BucketEnds::CuArray{Int, 1}
    BucketStarts::CuArray{Int, 1}
    BucketWrite::CuArray{Int, 1}
    BucketParticles::CuArray{Int, 1}
    NeighborOffsets::CuArray{SVector{D, Int}, 1}
    BucketCount::Int
end

struct CUDAMotionBuffers{D, T}
    Velocity::CuArray{T, 1}
    StartTime::CuArray{T, 1}
    Duration::CuArray{T, 1}
    Direction::CuArray{SVector{D, T}, 1}
    Active::CuArray{Bool, 1}
end

function BuildCUDAParticleBuffers(SimParticles)
    return CUDAParticleBuffers(
        CuArray(SimParticles.Position),
        CuArray(SimParticles.Density),
        CuArray(SimParticles.Pressure),
        CuArray(SimParticles.Velocity),
        CuArray(SimParticles.Acceleration),
        CuArray(SimParticles.MotionLimiter),
        CuArray(SimParticles.GravityFactor),
    )
end

function BuildCUDASupportBuffers(dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ)
    return CUDASupportBuffers(
        CuArray(dρdtI),
        CuArray(Velocityₙ⁺),
        CuArray(Positionₙ⁺),
        CuArray(ρₙ⁺),
        CuArray(∇Cᵢ),
        CuArray(∇◌rᵢ),
        CuArray(similar(dρdtI)),
        CuArray(similar(dρdtI)),
        CuArray(similar(dρdtI)),
    )
end

function BuildCUDAMotionBuffers(SimParticles, MotionDefinition, ::Val{Dimensions}, ::Type{FloatType}) where {Dimensions, FloatType}
    NumberOfPoints = length(SimParticles.Position)
    ZeroVector = zero(eltype(SimParticles.Position))
    MotionVelocity = zeros(FloatType, NumberOfPoints)
    MotionStartTime = zeros(FloatType, NumberOfPoints)
    MotionDuration = zeros(FloatType, NumberOfPoints)
    MotionDirection = Vector{SVector{Dimensions, FloatType}}(undef, NumberOfPoints)
    MotionActive = falses(NumberOfPoints)

    if MotionDefinition !== nothing
        @inbounds for i in 1:NumberOfPoints
            MotionDirection[i] = ZeroVector
            if SimParticles.Type[i] == Moving
                motion = MotionDefinition[SimParticles.GroupMarker[i]]
                if motion !== nothing
                    MotionVelocity[i] = motion.Velocity
                    MotionStartTime[i] = motion.StartTime
                    MotionDuration[i] = motion.Duration
                    MotionDirection[i] = motion.Direction
                    MotionActive[i] = true
                end
            end
        end
    else
        fill!(MotionDirection, ZeroVector)
    end

    return CUDAMotionBuffers(
        CuArray(MotionVelocity),
        CuArray(MotionStartTime),
        CuArray(MotionDuration),
        CuArray(MotionDirection),
        CuArray(MotionActive),
    )
end

function BuildCUDAGridBuffers(SimParticles, FullStencil, ::Val{D}) where {D}
    NumberOfPoints = length(SimParticles.Position)
    BucketCount = max(1, 2 * NumberOfPoints)
    NeighborOffsets = [SVector{D, Int}(Tuple(offset)) for offset in FullStencil]
    return CUDAGridBuffers(
        CuArray(similar(SimParticles.Position, SVector{D, Int})),
        CuArray(zeros(Int, BucketCount)),
        CuArray(zeros(Int, BucketCount)),
        CuArray(zeros(Int, BucketCount)),
        CuArray(zeros(Int, BucketCount)),
        CuArray(zeros(Int, NumberOfPoints)),
        CuArray(NeighborOffsets),
        BucketCount,
    )
end

function SyncParticlesToHost!(SimParticles, CUDABuffers::CUDAParticleBuffers)
    CUDA.copyto!(SimParticles.Position, CUDABuffers.Position)
    CUDA.copyto!(SimParticles.Density, CUDABuffers.Density)
    CUDA.copyto!(SimParticles.Pressure, CUDABuffers.Pressure)
    CUDA.copyto!(SimParticles.Velocity, CUDABuffers.Velocity)
    CUDA.copyto!(SimParticles.Acceleration, CUDABuffers.Acceleration)
    return nothing
end

function SyncPositionsToHost!(SimParticles, CUDABuffers::CUDAParticleBuffers)
    CUDA.copyto!(SimParticles.Position, CUDABuffers.Position)
    return nothing
end

@inline function BuildNeighborCellRanges(NeighborCellLists, cell_count)
    starts = zeros(Int, cell_count + 1)
    total = 0
    @inbounds for index in 1:cell_count
        starts[index] = total + 1
        total += length(NeighborCellLists[index])
    end
    starts[cell_count + 1] = total + 1

    entries = Vector{Int}(undef, total)
    cursor = 1
    @inbounds for index in 1:cell_count
        for neighbor_index in NeighborCellLists[index]
            entries[cursor] = neighbor_index
            cursor += 1
        end
    end
    return starts, entries
end

function PressureCUDAKernel!(Press, Density, SimConstants)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Press)
        ρ = Density[i]
        Press[i] = ((SimConstants.c₀^2 * SimConstants.ρ₀) / 7) * ((ρ / SimConstants.ρ₀)^7 - 1)
    end
    return nothing
end

function PressureCUDA!(Press, Density, SimConstants, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks PressureCUDAKernel!(Press, Density, SimConstants)
    return nothing
end

function DensityEpsiCUDAKernel!(Density, dρdtIₙ⁺, ρₙ⁺, Δt)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Density)
        epsi = -(dρdtIₙ⁺[i] / ρₙ⁺[i]) * Δt
        Density[i] = Density[i] * (2 - epsi) / (2 + epsi)
    end
    return nothing
end

function DensityEpsiCUDA!(Density, dρdtIₙ⁺, ρₙ⁺, Δt, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks DensityEpsiCUDAKernel!(Density, dρdtIₙ⁺, ρₙ⁺, Δt)
    return nothing
end

function LimitDensityAtBoundaryCUDAKernel!(Density, ρ₀, MotionLimiter)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Density)
        if (Density[i] < ρ₀) * !Bool(MotionLimiter[i])
            Density[i] = ρ₀
        end
    end
    return nothing
end

function LimitDensityAtBoundaryCUDA!(Density, ρ₀, MotionLimiter, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks LimitDensityAtBoundaryCUDAKernel!(Density, ρ₀, MotionLimiter)
    return nothing
end

function HalfTimeStepCUDAKernel!(Position, Density, Velocity, Acceleration, GravityFactor, MotionLimiter,
                                 Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂, SimConstants)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        acc = Acceleration[i] + ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Acceleration[i] = acc
        limiter = MotionLimiter[i]
        Positionₙ⁺[i] = Position[i] + Velocity[i] * dt₂ * limiter
        Velocityₙ⁺[i] = Velocity[i] + acc * dt₂ * limiter
        ρₙ⁺[i] = Density[i] + dρdtI[i] * dt₂
    end
    return nothing
end

function HalfTimeStepCUDA!(CUDAParticles::CUDAParticleBuffers, CUDASupport::CUDASupportBuffers, dt₂, SimConstants, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks HalfTimeStepCUDAKernel!(
        CUDAParticles.Position,
        CUDAParticles.Density,
        CUDAParticles.Velocity,
        CUDAParticles.Acceleration,
        CUDAParticles.GravityFactor,
        CUDAParticles.MotionLimiter,
        CUDASupport.Positionₙ⁺,
        CUDASupport.Velocityₙ⁺,
        CUDASupport.ρₙ⁺,
        CUDASupport.DρdtI,
        dt₂,
        SimConstants,
    )
    return nothing
end

function FullTimeStepCUDAKernel!(Position, Velocity, Acceleration, GravityFactor, MotionLimiter, dt, SimConstants)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        acc = Acceleration[i] + ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Acceleration[i] = acc
        limiter = MotionLimiter[i]
        Velocity[i] = Velocity[i] + acc * dt * limiter
        Position[i] = Position[i] + (((Velocity[i] + (Velocity[i] - acc * dt * limiter)) / 2) * dt) * limiter
    end
    return nothing
end

function FullTimeStepCUDA!(CUDAParticles::CUDAParticleBuffers, dt, SimConstants, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks FullTimeStepCUDAKernel!(
        CUDAParticles.Position,
        CUDAParticles.Velocity,
        CUDAParticles.Acceleration,
        CUDAParticles.GravityFactor,
        CUDAParticles.MotionLimiter,
        dt,
        SimConstants,
    )
    return nothing
end

function ProgressMotionCUDAKernel!(Position, Velocity, MotionVelocity, MotionStartTime, MotionDuration,
                                   MotionDirection, MotionActive, TotalTime, dt₂)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        if MotionActive[i]
            should_move = (MotionStartTime[i] <= TotalTime) && (TotalTime <= (MotionStartTime[i] + MotionDuration[i]))
            if should_move
                Velocity[i] = MotionVelocity[i] * MotionDirection[i]
            else
                Velocity[i] = zero(Position[i])
            end
            Position[i] = Position[i] + Velocity[i] * dt₂
        end
    end
    return nothing
end

function ProgressMotionCUDA!(CUDAParticles::CUDAParticleBuffers, MotionBuffers::CUDAMotionBuffers, TotalTime, dt₂, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks ProgressMotionCUDAKernel!(
        CUDAParticles.Position,
        CUDAParticles.Velocity,
        MotionBuffers.Velocity,
        MotionBuffers.StartTime,
        MotionBuffers.Duration,
        MotionBuffers.Direction,
        MotionBuffers.Active,
        TotalTime,
        dt₂,
    )
    return nothing
end

function ΔtCUDAKernel!(ViscousBuffer, ForceBuffer, Position, Velocity, Acceleration, SimConstants, SimKernel)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        r = Position[i]
        v = Velocity[i]
        a = Acceleration[i]

        r_sq = dot(r, r)
        ViscousBuffer[i] = abs(SimKernel.h * dot(v, r) / (r_sq + SimKernel.η²))

        a_sq = dot(a, a)
        if a_sq > 0
            a_mag = sqrt(a_sq)
            ForceBuffer[i] = sqrt(SimKernel.h / a_mag)
        else
            ForceBuffer[i] = Inf
        end
    end
    return nothing
end

function ΔtCUDA(CUDASupport::CUDASupportBuffers, CUDAParticles::CUDAParticleBuffers, SimConstants, SimKernel, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks ΔtCUDAKernel!(
        CUDASupport.ΔtViscous,
        CUDASupport.ΔtForce,
        CUDAParticles.Position,
        CUDAParticles.Velocity,
        CUDAParticles.Acceleration,
        SimConstants,
        SimKernel,
    )
    max_visc = CUDA.reduce(max, CUDASupport.ΔtViscous)
    min_force = CUDA.reduce(min, CUDASupport.ΔtForce)
    return SimConstants.CFL * min(min_force, SimKernel.h / (SimConstants.c₀ + max_visc))
end

function MaxDisplacementCUDAKernel!(Scratch, Positionₙ⁺, Position)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        diff = Positionₙ⁺[i] - Position[i]
        Scratch[i] = sqrt(dot(diff, diff))
    end
    return nothing
end

function UpdateΔxCUDA!(Δx, CUDASupport::CUDASupportBuffers, CUDAParticles::CUDAParticleBuffers, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks MaxDisplacementCUDAKernel!(
        CUDASupport.ΔxScratch,
        CUDASupport.Positionₙ⁺,
        CUDAParticles.Position,
    )
    maxd = CUDA.reduce(max, CUDASupport.ΔxScratch)
    return Δx + 4 * maxd
end

@inline function MapFloorGPU(X, InverseCutOff)
    return Int(sign(X)) * unsafe_trunc(Int, muladd(abs(X), InverseCutOff, 0.5))
end

@inline function HashCell(Cell)
    if length(Cell) == 2
        return Cell[1] * 73856093 ⊻ Cell[2] * 19349663
    end
    return Cell[1] * 73856093 ⊻ Cell[2] * 19349663 ⊻ Cell[3] * 83492791
end

function ResetBucketsKernel!(BucketCounts, BucketEnds, BucketStarts, BucketWrite)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(BucketCounts)
        BucketCounts[i] = 0
        BucketEnds[i] = 0
        BucketStarts[i] = 0
        BucketWrite[i] = 0
    end
    return nothing
end

function BuildCellCoordsKernel!(CellCoords, BucketCounts, Position, InverseCutOff, BucketCount)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        pos = Position[i]
        CellCoords[i] = SVector{length(pos), Int}(ntuple(d -> MapFloorGPU(pos[d], InverseCutOff), length(pos)))
        hash_val = HashCell(CellCoords[i])
        bucket = mod(hash_val, BucketCount) + 1
        CUDA.atomic_add!(BucketCounts, bucket, 1)
    end
    return nothing
end

function InitializeBucketStartsKernel!(BucketCounts, BucketEnds, BucketStarts, BucketWrite)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(BucketCounts)
        count = BucketCounts[i]
        BucketStarts[i] = BucketEnds[i] - count + 1
        BucketWrite[i] = BucketStarts[i]
    end
    return nothing
end

function FillBucketsKernel!(BucketWrite, BucketParticles, CellCoords, BucketCount)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(CellCoords)
        hash_val = HashCell(CellCoords[i])
        bucket = mod(hash_val, BucketCount) + 1
        index = CUDA.atomic_add!(BucketWrite, bucket, 1)
        BucketParticles[index] = i
    end
    return nothing
end

function RebuildNeighborGridCUDA!(CUDAGrid::CUDAGridBuffers, CUDAParticles::CUDAParticleBuffers, SimKernel, Launch)
    @cuda threads=Launch.Threads blocks=Launch.Blocks ResetBucketsKernel!(
        CUDAGrid.BucketCounts,
        CUDAGrid.BucketEnds,
        CUDAGrid.BucketStarts,
        CUDAGrid.BucketWrite,
    )
    @cuda threads=Launch.Threads blocks=Launch.Blocks BuildCellCoordsKernel!(
        CUDAGrid.CellCoords,
        CUDAGrid.BucketCounts,
        CUDAParticles.Position,
        SimKernel.H⁻¹,
        CUDAGrid.BucketCount,
    )
    CUDAGrid.BucketEnds .= CUDA.cumsum(CUDAGrid.BucketCounts)
    @cuda threads=Launch.Threads blocks=Launch.Blocks InitializeBucketStartsKernel!(
        CUDAGrid.BucketCounts,
        CUDAGrid.BucketEnds,
        CUDAGrid.BucketStarts,
        CUDAGrid.BucketWrite,
    )
    @cuda threads=Launch.Threads blocks=Launch.Blocks FillBucketsKernel!(
        CUDAGrid.BucketWrite,
        CUDAGrid.BucketParticles,
        CUDAGrid.CellCoords,
        CUDAGrid.BucketCount,
    )
    return nothing
end

@inline function ComputeDensityDiffusionGPU(::ZeroDensityDiffusion, _SimKernel, _SimConstants,
                                            _Density, _MotionLimiter, _xᵢⱼ, _∇ᵢWᵢⱼ, d², _i, _j)
    return zero(d²)
end

@inline function ComputeDensityDiffusionGPU(::LinearDensityDiffusion, SimKernel, SimConstants,
                                            Density, MotionLimiter, xᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    @unpack ρ₀, m₀, c₀, δᵩ, Cb, γ, g = SimConstants
    @unpack h, η² = SimKernel

    Linear_ρ_factor = (1 / (Cb * γ)) * ρ₀
    ρᵢ = Density[i]
    ρⱼ = Density[j]

    Pᵢⱼᴴ = ρ₀ * (-g) * -xᵢⱼ[end]
    ρᵢⱼᴴ = Pᵢⱼᴴ * Linear_ρ_factor

    invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)
    ρⱼᵢ = ρⱼ - ρᵢ
    ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ) * (-xᵢⱼ) * invdᵢⱼ²η²

    MLcond = MotionLimiter[i] * MotionLimiter[j]
    Dᵢ = δᵩ * h * c₀ * (m₀ / ρⱼ) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) * MLcond

    return Dᵢ
end

@inline function ComputeViscosityGPU(::ZeroViscosity, _SimKernel, _SimConstants,
                                     _Density, _xᵢⱼ, _vᵢⱼ, _∇ᵢWᵢⱼ, _d², _i, _j)
    return zero(_xᵢⱼ)
end

@inline function ComputeViscosityGPU(::ArtificialViscosity, SimKernel, SimConstants,
                                     Density, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    @unpack m₀, α, c₀ = SimConstants
    @unpack h, η² = SimKernel

    ρᵢ = Density[i]
    ρⱼ = Density[j]

    v_dot_x = dot(vᵢⱼ, xᵢⱼ)
    if v_dot_x < 0
        ρ̄ = 0.5 * (ρᵢ + ρⱼ)
        μᵢⱼ = h * v_dot_x / (d² + η²)
        Π = -m₀ * (-α * c₀ * μᵢⱼ) / ρ̄ * ∇ᵢWᵢⱼ
        return Π
    end

    return zero(xᵢⱼ)
end

function NeighborLoopCUDAKernel!(dρdtI, Acceleration, Position, Density, Pressure, Velocity,
                                 MotionLimiter, CellCoords, BucketStarts, BucketEnds,
                                 BucketParticles, NeighborOffsets, BucketCount, SimKernel, SimConstants,
                                 SimDensityDiffusion, SimViscosity)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        dρdt_acc = zero(eltype(dρdtI))
        acc_acc = zero(Position[i])
        cell = CellCoords[i]
        @inbounds for offset_index in eachindex(NeighborOffsets)
            neighbor_cell = cell + NeighborOffsets[offset_index]
            hash_val = HashCell(neighbor_cell)
            bucket = mod(hash_val, BucketCount) + 1
            start_index = BucketStarts[bucket]
            end_index = BucketEnds[bucket]
            if start_index <= end_index
                @inbounds for j in start_index:end_index
                    j_index = BucketParticles[j]
                    if CellCoords[j_index] == neighbor_cell && j_index != i
                        xᵢⱼ = Position[i] - Position[j_index]
                        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
                        if xᵢⱼ² <= SimKernel.H²
                            dᵢⱼ = sqrt(abs(xᵢⱼ²))
                            q = clamp(dᵢⱼ * SimKernel.h⁻¹, zero(eltype(dᵢⱼ)), one(eltype(dᵢⱼ)) * 2)
                            ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                            ρᵢ = Density[i]
                            ρⱼ = Density[j_index]
                            vᵢⱼ = Velocity[i] - Velocity[j_index]

                            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                            dρdt⁺ = -ρᵢ * (SimConstants.m₀ / ρⱼ) * density_symmetric_term

                            Dᵢ = ComputeDensityDiffusionGPU(SimDensityDiffusion, SimKernel, SimConstants,
                                                            Density, MotionLimiter, xᵢⱼ, ∇ᵢWᵢⱼ,
                                                            dᵢⱼ^2, i, j_index)

                            dρdt_acc += dρdt⁺ + Dᵢ

                            Pᵢ = Pressure[i]
                            Pⱼ = Pressure[j_index]
                            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
                            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, SimConstants.dx)
                            dvdt⁺ = -SimConstants.m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

                            visc_term = ComputeViscosityGPU(SimViscosity, SimKernel, SimConstants,
                                                            Density, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ^2, i, j_index)

                            acc_acc += dvdt⁺ + visc_term
                        end
                    end
                end
            end
        end

        dρdtI[i] = dρdt_acc
        Acceleration[i] = acc_acc
    end
    return nothing
end

function NeighborLoopPerParticleCUDA!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimConstants, CUDAParticles::CUDAParticleBuffers,
                                      CUDAGrid::CUDAGridBuffers, dρdtI, Acceleration, Launch;
                                      Position = CUDAParticles.Position,
                                      Density = CUDAParticles.Density,
                                      Pressure = CUDAParticles.Pressure,
                                      Velocity = CUDAParticles.Velocity) where {
                                      SDD<:SPHDensityDiffusion,
                                      SV<:SPHViscosity}
    if !CUDAAvailable()
        error("CUDA is not available; cannot run GPU neighbor loop.")
    end

    if !(SimDensityDiffusion isa LinearDensityDiffusion || SimDensityDiffusion isa ZeroDensityDiffusion)
        error("CUDA neighbor loop supports LinearDensityDiffusion or ZeroDensityDiffusion.")
    end
    if !(SimViscosity isa ArtificialViscosity || SimViscosity isa ZeroViscosity)
        error("CUDA neighbor loop supports ArtificialViscosity or ZeroViscosity.")
    end

    @cuda threads=Launch.Threads blocks=Launch.Blocks NeighborLoopCUDAKernel!(
        dρdtI, Acceleration, Position, Density, Pressure,
        Velocity, CUDAParticles.MotionLimiter, CUDAGrid.CellCoords,
        CUDAGrid.BucketStarts, CUDAGrid.BucketEnds, CUDAGrid.BucketParticles,
        CUDAGrid.NeighborOffsets, CUDAGrid.BucketCount,
        SimKernel, SimConstants, SimDensityDiffusion, SimViscosity,
    )

    return nothing
end

@inbounds function SimulationLoopCUDA(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                      SimConstants, SimParticles, FullStencil,
                                      ParticleRanges, UniqueCells, CellLookup,
                                      ParticleOrder, CellOffsets,
                                      NeighborCellLists, CUDAParticles::CUDAParticleBuffers,
                                      CUDASupport::CUDASupportBuffers, MotionBuffers,
                                      CUDAGrid::CUDAGridBuffers) where {
                                      Dimensions, FloatType, SMode, KMode,
                                      BMode, LMode,
                                      SDD<:SPHDensityDiffusion,
                                      SV<:SPHViscosity}
    Launch = KernelLaunchConfig(length(CUDAParticles.Position))
    RebuildNeighborGridCUDA!(CUDAGrid, CUDAParticles, SimKernel, Launch)
    dt = ΔtCUDA(CUDASupport, CUDAParticles, SimConstants, SimKernel, Launch)
    dt₂ = dt * 0.5

    while SimMetaData.TotalTime <= next_output_time(SimMetaData)
        @timeit SimMetaData.HourGlass "01 Calculate IndexCounter" SimMetaData.Δx = UpdateΔxCUDA!(SimMetaData.Δx, CUDASupport, CUDAParticles, Launch)
        if SimMetaData.Δx >= SimKernel.h
            @timeit SimMetaData.HourGlass "01a Rebuild Neighbor Grid" RebuildNeighborGridCUDA!(CUDAGrid, CUDAParticles, SimKernel, Launch)
            SimMetaData.Δx = zero(eltype(CUDASupport.DρdtI))
        end

        if MotionBuffers !== nothing
            @timeit SimMetaData.HourGlass "Motion" ProgressMotionCUDA!(CUDAParticles, MotionBuffers, SimMetaData.TotalTime, dt₂, Launch)
        end

        @timeit SimMetaData.HourGlass "02 Pressure" PressureCUDA!(CUDAParticles.Pressure, CUDAParticles.Density, SimConstants, Launch)

        @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopPerParticleCUDA!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
            CUDAParticles, CUDAGrid, CUDASupport.DρdtI, CUDAParticles.Acceleration, Launch,
        )

        @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" HalfTimeStepCUDA!(CUDAParticles, CUDASupport, dt₂, SimConstants, Launch)

        @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary" LimitDensityAtBoundaryCUDA!(CUDASupport.ρₙ⁺, SimConstants.ρ₀, CUDAParticles.MotionLimiter, Launch)

        if MotionBuffers !== nothing
            @timeit SimMetaData.HourGlass "Motion" ProgressMotionCUDA!(CUDAParticles, MotionBuffers, SimMetaData.TotalTime, dt₂, Launch)
        end

        @timeit SimMetaData.HourGlass "07 Pressure" PressureCUDA!(CUDAParticles.Pressure, CUDASupport.ρₙ⁺, SimConstants, Launch)
        @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPerParticleCUDA!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
            CUDAParticles, CUDAGrid, CUDASupport.DρdtI, CUDAParticles.Acceleration, Launch,
            Position = CUDASupport.Positionₙ⁺,
            Density = CUDASupport.ρₙ⁺,
            Velocity = CUDASupport.Velocityₙ⁺,
        )

        @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary" LimitDensityAtBoundaryCUDA!(CUDAParticles.Density, SimConstants.ρ₀, CUDAParticles.MotionLimiter, Launch)

        @timeit SimMetaData.HourGlass "10 Final Density" DensityEpsiCUDA!(CUDAParticles.Density, CUDASupport.DρdtI, CUDASupport.ρₙ⁺, dt, Launch)

        @timeit SimMetaData.HourGlass "11 Update To Final TimeStep" FullTimeStepCUDA!(CUDAParticles, dt, SimConstants, Launch)

        @timeit SimMetaData.HourGlass "12 Update MetaData" UpdateMetaData!(SimMetaData, dt)

        @timeit SimMetaData.HourGlass "13 Update TimeStep" dt = ΔtCUDA(CUDASupport, CUDAParticles, SimConstants, SimKernel, Launch)
        dt₂ = dt * 0.5
    end

    SyncParticlesToHost!(SimParticles, CUDAParticles)

    return nothing
end

function RunSimulationCUDA(;SimGeometry::Vector{Geometry{Dimensions, FloatType}},
        SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
        SimConstants::SimulationConstants,
        SimKernel::SPHKernelInstance,
        SimLogger::SimulationLogger,
        SimParticles::StructArray,
        SimViscosity::SV,
        SimDensityDiffusion::SDD,
        ParticleNormalsPath::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType,SMode,KMode,BMode,LMode,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}
    if !CUDAAvailable()
        error("CUDA is not available; cannot run RunSimulationCUDA.")
    end
    if !(SimMetaData isa SimulationMetaData{Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, LMode} where {LMode})
        error("RunSimulationCUDA currently supports NoShifting, NoKernelOutput, and NoMDBC configurations.")
    end

    NumberOfPoints = length(SimParticles)

    dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimMetaData, SimParticles.Position)

    LoadMDBCNormals!(SimMetaData, SimParticles, ParticleNormalsPath)

    InitializeLog!(SimMetaData, SimLogger, SimConstants, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)

    Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)

    ParticleRanges = zeros(Int, NumberOfPoints + 1 + 1)
    UniqueCells = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
    CellLookup = InitializeCellIndexLookup(Val(Dimensions))
    FullStencil = ConstructStencil(Val(Dimensions))
    NeighborCellLists = [Int[] for _ in 1:length(UniqueCells)]
    ParticleOrder = zeros(Int, NumberOfPoints)
    CellOffsets = zeros(Int, length(ParticleRanges))

    output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    SimMetaData.OutputIterationCounter = 1
    output.enqueue_particles(SimMetaData.OutputIterationCounter)
    if SimMetaData.IndexCounter > 0
        unique_cells_view = view(UniqueCells, 1:SimMetaData.IndexCounter)
        cell_particle_counts = nothing
        cell_neighbor_counts = nothing
        if SimMetaData.ExportGridCellParticleCounts
            cell_particle_counts = ComputeCellParticleCounts(ParticleRanges, SimMetaData.IndexCounter)
            cell_neighbor_counts = ComputeCellNeighborCounts(ParticleRanges, NeighborCellLists, SimMetaData.IndexCounter)
        end
        output.enqueue_grid(
            SimMetaData.OutputIterationCounter,
            unique_cells_view,
            cell_particle_counts=cell_particle_counts,
            cell_neighbor_counts=cell_neighbor_counts,
        )
    end

    MotionDefinition = SPHCellList.GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)
    CUDAParticles = BuildCUDAParticleBuffers(SimParticles)
    CUDASupport = BuildCUDASupportBuffers(dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ)
    MotionBuffers = MotionDefinition === nothing ? nothing : BuildCUDAMotionBuffers(SimParticles, MotionDefinition, Val(Dimensions), FloatType)
    CUDAGrid = BuildCUDAGridBuffers(SimParticles, FullStencil, Val(Dimensions))

    @inbounds while true
        @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoopCUDA(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, FullStencil, ParticleRanges,
            UniqueCells, CellLookup, ParticleOrder, CellOffsets,
            NeighborCellLists, CUDAParticles, CUDASupport,
            MotionBuffers, CUDAGrid,
        )
        push!(SimMetaData.TimeSteps, SimMetaData.CurrentTimeStep)

        LogStep!(SimMetaData, SimLogger)

        SimMetaData.OutputIterationCounter += 1

        if SimMetaData.ExportGridCellParticleCounts
            SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, ParticleRanges, UniqueCells, CellLookup, ParticleOrder, CellOffsets)
            if SimMetaData.IndexCounter > 0
                unique_cells_view = view(UniqueCells, 1:SimMetaData.IndexCounter)
                BuildNeighborCellLists!(NeighborCellLists, FullStencil, unique_cells_view, ParticleRanges, CellLookup)
            end
        end
        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)

        @timeit SimMetaData.HourGlass "13 Determine Output" SPHCellList.DetermineOutput(
            Val(SimMetaData.ExportGridCellParticleCounts),
            SimMetaData,
            output,
            ParticleRanges,
            NeighborCellLists,
            UniqueCellsView,
        )

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

            show(SimMetaData.HourGlass, sortby=:name)
            show(SimMetaData.HourGlass)

            FinalizeLog!(SimMetaData, SimLogger)

            AutoOpenLogFile(SimLogger, SimMetaData)
            AutoOpenParaview(SimMetaData, output.variable_names)

            break
        end
    end
end

end
