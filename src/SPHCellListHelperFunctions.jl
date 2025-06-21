module SPHCellListHelperFunctions

export reduce_sum!, ResetStep!, ReductionStep!, ProgressMotion

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
using ..SimulationEquations
using ..SimulationGeometry
using ..AuxiliaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..OpenExternalPrograms
using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels

import StructArrays: StructArray, foreachfield
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using UnicodePlots
using LinearAlgebra
using Bumper

function reduce_sum!(target_array, arrays)
    n = length(target_array)
    num_threads = nthreads()
    chunk_size = ceil(Int, n / num_threads)
    @inbounds for t in 1:num_threads
        local start_idx = 1 + (t-1) * chunk_size
        local end_idx = min(t * chunk_size, n)
        for j in eachindex(arrays)
            local array = arrays[j]  # Access array only once per thread
            @simd ivdep for i in start_idx:end_idx
                @inbounds target_array[i] += array[i]
            end
        end
    end
end

function ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
    
    ResetArrays!(dρdtI, Acceleration)

    if SimMetaData.FlagOutputKernelValues
        ResetArrays!(Kernel, KernelGradient)
    end

    if SimMetaData.FlagShifting
        ResetArrays!(∇Cᵢ, ∇◌rᵢ)
    end

    foreachfield(f -> map!(v -> fill!(v, zero(eltype(v))), f, f), SimThreadedArrays)

    return nothing
end

function ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
    reduce_sum!(dρdtI, SimThreadedArrays.dρdtIThreaded)
    reduce_sum!(Acceleration, SimThreadedArrays.AccelerationThreaded)
  
    if SimMetaData.FlagOutputKernelValues
        reduce_sum!(Kernel, SimThreadedArrays.KernelThreaded)
        reduce_sum!(KernelGradient, SimThreadedArrays.KernelGradientThreaded)
    end

    if SimMetaData.FlagShifting
        reduce_sum!(∇Cᵢ, SimThreadedArrays.∇CᵢThreaded)
        reduce_sum!(∇◌rᵢ, SimThreadedArrays.∇◌rᵢThreaded)
    end

    return nothing
end

### Some functions to simplify code inside of this function
function ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionsDefinition, SimMetaData)
    @inbounds @simd ivdep for i in eachindex(Position)
        if ParticleType[i] == Moving
            motion = MotionsDefinition[ParticleMarker[i]]

            if motion !== nothing
                ShouldMove = (motion.StartTime <= SimMetaData.TotalTime) &&
                             (SimMetaData.TotalTime <= (motion.StartTime + motion.Duration))

                # Retrieve motion parameters
                MotionVel = motion.Velocity
                MotionDir = motion.Direction

                # Update Velocity and Position
                Velocity[i] = MotionVel * MotionDir * ShouldMove
                Position[i] += Velocity[i] * dt₂
            end
        end
    end

    return nothing
end
end
