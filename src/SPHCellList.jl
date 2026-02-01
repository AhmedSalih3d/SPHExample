module SPHCellList

export NeighborLoop!, ComputeInteractions!, RunSimulation

using Parameters, FastPow, StaticArrays, Base.Threads
import LinearAlgebra: dot

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
using ..SPHNeighborList: BuildNeighborCellLists!, ComputeCellNeighborCounts, ComputeCellParticleCounts, ConstructStencil, ExtractCells!, FindCellIndex, MapFloor, UpdateNeighbors!, UpdateΔx!

using StaticArrays
using StructArrays: StructArray, foreachfield
using LinearAlgebra: dot, norm, diagm, diag, cond, det
using Parameters: @unpack
using FastPow: @fastpow
using Format
using TimerOutputs
using HDF5
using Base.Threads
using LinearAlgebra
    using Bumper

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationMax;
                                      Position = SimParticles.Position,
                                      Density = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity) where {D,T,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            AccelerationMax[i] = norm(acc_acc)
        end

        return nothing
    end

    function NeighborLoopPairwise!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                   SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L},
                                   SimConstants, SimParticles, ParticleRanges,
                                   CellListIndices, NeighborCellLists, dρdtI,
                                   Acceleration, ∇Cᵢ,
                                   ∇◌rᵢ, AccelerationMax;
                                   Position = SimParticles.Position,
                                   Density = SimParticles.Density,
                                   Pressure = SimParticles.Pressure,
                                   Velocity = SimParticles.Velocity) where {D,T,
                                               B<:MDBCMode,L<:LogMode,
                                               SDD<:SPHDensityDiffusion,
                                               SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))

        @inbounds for CellListIndex in eachindex(NeighborCellLists)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

            for i in SameCellStart:SameCellEnd
                for j in (i + 1):SameCellEnd
                    dρdt_i, dρdt_j, acc_i, acc_j = ComputeInteractionsPairwiseNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, i, j,
                    )
                    dρdtI[i] += dρdt_i
                    dρdtI[j] += dρdt_j
                    Acceleration[i] += acc_i
                    Acceleration[j] += acc_j
                end
            end

            for NeighborIdx in NeighborCellLists[CellListIndex]
                if NeighborIdx > CellListIndex
                    StartIndex_ = ParticleRanges[NeighborIdx]
                    EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                    for i in SameCellStart:SameCellEnd
                        for j in StartIndex_:EndIndex_
                            dρdt_i, dρdt_j, acc_i, acc_j = ComputeInteractionsPairwiseNoKernel!(
                                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                SimConstants, SimParticles, Position, Density, Pressure,
                                Velocity, ParticleType, i, j,
                            )
                            dρdtI[i] += dρdt_i
                            dρdtI[j] += dρdt_j
                            Acceleration[i] += acc_i
                            Acceleration[j] += acc_j
                        end
                    end
                end
            end
        end

        @inbounds for i in eachindex(AccelerationMax)
            AccelerationMax[i] = norm(Acceleration[i])
        end

        return nothing
    end

    # Mod-3 coloring keeps same-color cells at least 3 apart in each dimension,
    # so their neighbor stencils do not overlap and can be updated in parallel.
    @inline function CellColorCount(::Val{D}) where D
        return 3^D
    end

    @inline function CellColorIndex(Cell::CartesianIndex{D}) where D
        color = 1
        factor = 1
        @inbounds for dim in 1:D
            color += mod(Cell[dim], 3) * factor
            factor *= 3
        end
        return color
    end

    @inline function BuildCellColorOrdering!(cell_colors, color_counts, color_offsets, color_positions,
                                             cell_order, UniqueCellsView)
        fill!(color_counts, 0)
        @inbounds for idx in eachindex(UniqueCellsView)
            color = CellColorIndex(UniqueCellsView[idx])
            cell_colors[idx] = color
            color_counts[color] += 1
        end
        color_offsets[1] = 1
        @inbounds for color in eachindex(color_counts)
            color_offsets[color + 1] = color_offsets[color] + color_counts[color]
            color_positions[color] = color_offsets[color]
        end
        @inbounds for idx in eachindex(UniqueCellsView)
            color = cell_colors[idx]
            pos = color_positions[color]
            cell_order[pos] = idx
            color_positions[color] = pos + 1
        end
        return nothing
    end

    function NeighborLoopPairwiseThreaded!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                           SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L},
                                           SimConstants, SimParticles, UniqueCellsView,
                                           ParticleRanges, CellListIndices, NeighborCellLists, dρdtI,
                                           Acceleration, ∇Cᵢ,
                                           ∇◌rᵢ, AccelerationMax;
                                           Position = SimParticles.Position,
                                           Density = SimParticles.Density,
                                           Pressure = SimParticles.Pressure,
                                           Velocity = SimParticles.Velocity) where {D,T,
                                                       B<:MDBCMode,L<:LogMode,
                                                       SDD<:SPHDensityDiffusion,
                                                       SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        thread_count = Threads.nthreads()
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))

        @no_escape begin
            cell_count = length(NeighborCellLists)
            color_count = CellColorCount(Val(D))
            cell_colors = @alloc(Int, cell_count)
            color_counts = @alloc(Int, color_count)
            color_offsets = @alloc(Int, color_count + 1)
            color_positions = @alloc(Int, color_count)
            cell_order = @alloc(Int, cell_count)
            BuildCellColorOrdering!(cell_colors, color_counts, color_offsets, color_positions, cell_order, UniqueCellsView)

            @inbounds for color in 1:color_count
                range_start = color_offsets[color]
                range_end = color_offsets[color + 1] - 1
                if range_start <= range_end
                    range_len = range_end - range_start + 1
                    chunk_count = min(thread_count, range_len)
                    chunk_size = cld(range_len, chunk_count)
                    @sync for chunk_id in 1:chunk_count
                        chunk_start = range_start + (chunk_id - 1) * chunk_size
                        chunk_end = min(chunk_start + chunk_size - 1, range_end)
                        Threads.@spawn begin
                            @inbounds for idx in chunk_start:chunk_end
                                CellListIndex = cell_order[idx]
                                SameCellStart = ParticleRanges[CellListIndex]
                                SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

                                for i in SameCellStart:SameCellEnd
                                    for j in (i + 1):SameCellEnd
                                        dρdt_i, dρdt_j, acc_i, acc_j = ComputeInteractionsPairwiseNoKernel!(
                                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                            SimConstants, SimParticles, Position, Density, Pressure,
                                            Velocity, ParticleType, i, j,
                                        )
                                        dρdtI[i] += dρdt_i
                                        dρdtI[j] += dρdt_j
                                        Acceleration[i] += acc_i
                                        Acceleration[j] += acc_j
                                    end
                                end

                                for NeighborIdx in NeighborCellLists[CellListIndex]
                                    if NeighborIdx > CellListIndex
                                        StartIndex_ = ParticleRanges[NeighborIdx]
                                        EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                                        for i in SameCellStart:SameCellEnd
                                            for j in StartIndex_:EndIndex_
                                                dρdt_i, dρdt_j, acc_i, acc_j = ComputeInteractionsPairwiseNoKernel!(
                                                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                    SimConstants, SimParticles, Position, Density, Pressure,
                                                    Velocity, ParticleType, i, j,
                                                )
                                                dρdtI[i] += dρdt_i
                                                dρdtI[j] += dρdt_j
                                                Acceleration[i] += acc_i
                                                Acceleration[j] += acc_j
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            @inbounds for i in eachindex(AccelerationMax)
                AccelerationMax[i] = norm(Acceleration[i])
            end
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationMax;
                                      Position = SimParticles.Position,
                                      Density = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity) where {D,T,
                                                  K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, i, j,
                    )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                        ComputeInteractionsPerParticle!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, Pressure,
                            Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                            kernel_grad_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
            AccelerationMax[i] = norm(acc_acc)
        end

        return nothing
    end

    function NeighborLoopPairwise!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                   SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L},
                                   SimConstants, SimParticles, ParticleRanges,
                                   CellListIndices, NeighborCellLists, dρdtI,
                                   Acceleration, ∇Cᵢ,
                                   ∇◌rᵢ, AccelerationMax;
                                   Position = SimParticles.Position,
                                   Density = SimParticles.Density,
                                   Pressure = SimParticles.Pressure,
                                   Velocity = SimParticles.Velocity) where {D,T,
                                               K<:KernelOutputMode,
                                               B<:MDBCMode,L<:LogMode,
                                               SDD<:SPHDensityDiffusion,
                                               SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))
        fill!(Kernel, zero(eltype(Kernel)))
        fill!(KernelGradient, zero(eltype(KernelGradient)))

        @inbounds for CellListIndex in eachindex(NeighborCellLists)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

            for i in SameCellStart:SameCellEnd
                for j in (i + 1):SameCellEnd
                    dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j =
                        ComputeInteractionsPairwise!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, Pressure,
                            Velocity, ParticleType, i, j,
                        )
                    dρdtI[i] += dρdt_i
                    dρdtI[j] += dρdt_j
                    Acceleration[i] += acc_i
                    Acceleration[j] += acc_j
                    Kernel[i] += kernel_i
                    Kernel[j] += kernel_j
                    KernelGradient[i] += kernel_grad_i
                    KernelGradient[j] += kernel_grad_j
                end
            end

            for NeighborIdx in NeighborCellLists[CellListIndex]
                if NeighborIdx > CellListIndex
                    StartIndex_ = ParticleRanges[NeighborIdx]
                    EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                    for i in SameCellStart:SameCellEnd
                        for j in StartIndex_:EndIndex_
                            dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j =
                                ComputeInteractionsPairwise!(
                                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                    SimConstants, SimParticles, Position, Density, Pressure,
                                    Velocity, ParticleType, i, j,
                                )
                            dρdtI[i] += dρdt_i
                            dρdtI[j] += dρdt_j
                            Acceleration[i] += acc_i
                            Acceleration[j] += acc_j
                            Kernel[i] += kernel_i
                            Kernel[j] += kernel_j
                            KernelGradient[i] += kernel_grad_i
                            KernelGradient[j] += kernel_grad_j
                        end
                    end
                end
            end
        end

        @inbounds for i in eachindex(AccelerationMax)
            AccelerationMax[i] = norm(Acceleration[i])
        end

        return nothing
    end

    function NeighborLoopPairwiseThreaded!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                           SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L},
                                           SimConstants, SimParticles, UniqueCellsView,
                                           ParticleRanges, CellListIndices, NeighborCellLists, dρdtI,
                                           Acceleration, ∇Cᵢ,
                                           ∇◌rᵢ, AccelerationMax;
                                           Position = SimParticles.Position,
                                           Density = SimParticles.Density,
                                           Pressure = SimParticles.Pressure,
                                           Velocity = SimParticles.Velocity) where {D,T,
                                                       K<:KernelOutputMode,
                                                       B<:MDBCMode,L<:LogMode,
                                                       SDD<:SPHDensityDiffusion,
                                                       SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        thread_count = Threads.nthreads()
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))
        fill!(Kernel, zero(eltype(Kernel)))
        fill!(KernelGradient, zero(eltype(KernelGradient)))

        @no_escape begin
            cell_count = length(NeighborCellLists)
            color_count = CellColorCount(Val(D))
            cell_colors = @alloc(Int, cell_count)
            color_counts = @alloc(Int, color_count)
            color_offsets = @alloc(Int, color_count + 1)
            color_positions = @alloc(Int, color_count)
            cell_order = @alloc(Int, cell_count)
            BuildCellColorOrdering!(cell_colors, color_counts, color_offsets, color_positions, cell_order, UniqueCellsView)

            @inbounds for color in 1:color_count
                range_start = color_offsets[color]
                range_end = color_offsets[color + 1] - 1
                if range_start <= range_end
                    range_len = range_end - range_start + 1
                    chunk_count = min(thread_count, range_len)
                    chunk_size = cld(range_len, chunk_count)
                    @sync for chunk_id in 1:chunk_count
                        chunk_start = range_start + (chunk_id - 1) * chunk_size
                        chunk_end = min(chunk_start + chunk_size - 1, range_end)
                        Threads.@spawn begin
                            @inbounds for idx in chunk_start:chunk_end
                                CellListIndex = cell_order[idx]
                                SameCellStart = ParticleRanges[CellListIndex]
                                SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

                                for i in SameCellStart:SameCellEnd
                                    for j in (i + 1):SameCellEnd
                                        dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j =
                                            ComputeInteractionsPairwise!(
                                                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                SimConstants, SimParticles, Position, Density, Pressure,
                                                Velocity, ParticleType, i, j,
                                            )
                                        dρdtI[i] += dρdt_i
                                        dρdtI[j] += dρdt_j
                                        Acceleration[i] += acc_i
                                        Acceleration[j] += acc_j
                                        Kernel[i] += kernel_i
                                        Kernel[j] += kernel_j
                                        KernelGradient[i] += kernel_grad_i
                                        KernelGradient[j] += kernel_grad_j
                                    end
                                end

                                for NeighborIdx in NeighborCellLists[CellListIndex]
                                    if NeighborIdx > CellListIndex
                                        StartIndex_ = ParticleRanges[NeighborIdx]
                                        EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                                        for i in SameCellStart:SameCellEnd
                                            for j in StartIndex_:EndIndex_
                                                dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j =
                                                    ComputeInteractionsPairwise!(
                                                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                        SimConstants, SimParticles, Position, Density, Pressure,
                                                        Velocity, ParticleType, i, j,
                                                    )
                                                dρdtI[i] += dρdt_i
                                                dρdtI[j] += dρdt_j
                                                Acceleration[i] += acc_i
                                                Acceleration[j] += acc_j
                                                Kernel[i] += kernel_i
                                                Kernel[j] += kernel_j
                                                KernelGradient[i] += kernel_grad_i
                                                KernelGradient[j] += kernel_grad_j
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            @inbounds for i in eachindex(AccelerationMax)
                AccelerationMax[i] = norm(Acceleration[i])
            end
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationMax;
                                      Position = SimParticles.Position,
                                      Density = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity) where {D,T,
                                                  S<:ShiftingMode,B<:MDBCMode,
                                                  L<:LogMode,SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, shift_c_acc,
                        shift_r_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, shift_c_acc,
                        shift_r_acc, i, j,
                    )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                        ComputeInteractionsPerParticleNoKernel!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, Pressure,
                            Velocity, ParticleType, dρdt_acc, acc_acc, shift_c_acc,
                            shift_r_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            ∇Cᵢ[i] = shift_c_acc
            ∇◌rᵢ[i] = shift_r_acc
            AccelerationMax[i] = norm(acc_acc)
        end

        return nothing
    end

    function NeighborLoopPairwise!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                   SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                   SimConstants, SimParticles, ParticleRanges,
                                   CellListIndices, NeighborCellLists, dρdtI,
                                   Acceleration, ∇Cᵢ,
                                   ∇◌rᵢ, AccelerationMax;
                                   Position = SimParticles.Position,
                                   Density = SimParticles.Density,
                                   Pressure = SimParticles.Pressure,
                                   Velocity = SimParticles.Velocity) where {D,T,
                                               S<:ShiftingMode,B<:MDBCMode,
                                               L<:LogMode,SDD<:SPHDensityDiffusion,
                                               SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))
        fill!(∇Cᵢ, zero(eltype(∇Cᵢ)))
        fill!(∇◌rᵢ, zero(eltype(∇◌rᵢ)))

        @inbounds for CellListIndex in eachindex(NeighborCellLists)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

            for i in SameCellStart:SameCellEnd
                for j in (i + 1):SameCellEnd
                    dρdt_i, dρdt_j, acc_i, acc_j, shift_c_i, shift_c_j, shift_r_i, shift_r_j =
                        ComputeInteractionsPairwiseNoKernel!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, Pressure,
                            Velocity, ParticleType, i, j,
                        )
                    dρdtI[i] += dρdt_i
                    dρdtI[j] += dρdt_j
                    Acceleration[i] += acc_i
                    Acceleration[j] += acc_j
                    ∇Cᵢ[i] += shift_c_i
                    ∇Cᵢ[j] += shift_c_j
                    ∇◌rᵢ[i] += shift_r_i
                    ∇◌rᵢ[j] += shift_r_j
                end
            end

            for NeighborIdx in NeighborCellLists[CellListIndex]
                if NeighborIdx > CellListIndex
                    StartIndex_ = ParticleRanges[NeighborIdx]
                    EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                    for i in SameCellStart:SameCellEnd
                        for j in StartIndex_:EndIndex_
                            dρdt_i, dρdt_j, acc_i, acc_j, shift_c_i, shift_c_j, shift_r_i, shift_r_j =
                                ComputeInteractionsPairwiseNoKernel!(
                                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                    SimConstants, SimParticles, Position, Density, Pressure,
                                    Velocity, ParticleType, i, j,
                                )
                            dρdtI[i] += dρdt_i
                            dρdtI[j] += dρdt_j
                            Acceleration[i] += acc_i
                            Acceleration[j] += acc_j
                            ∇Cᵢ[i] += shift_c_i
                            ∇Cᵢ[j] += shift_c_j
                            ∇◌rᵢ[i] += shift_r_i
                            ∇◌rᵢ[j] += shift_r_j
                        end
                    end
                end
            end
        end

        @inbounds for i in eachindex(AccelerationMax)
            AccelerationMax[i] = norm(Acceleration[i])
        end

        return nothing
    end

    function NeighborLoopPairwiseThreaded!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                           SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                           SimConstants, SimParticles, UniqueCellsView,
                                           ParticleRanges, CellListIndices, NeighborCellLists, dρdtI,
                                           Acceleration, ∇Cᵢ,
                                           ∇◌rᵢ, AccelerationMax;
                                           Position = SimParticles.Position,
                                           Density = SimParticles.Density,
                                           Pressure = SimParticles.Pressure,
                                           Velocity = SimParticles.Velocity) where {D,T,
                                                       S<:ShiftingMode,B<:MDBCMode,
                                                       L<:LogMode,SDD<:SPHDensityDiffusion,
                                                       SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        thread_count = Threads.nthreads()
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))
        fill!(∇Cᵢ, zero(eltype(∇Cᵢ)))
        fill!(∇◌rᵢ, zero(eltype(∇◌rᵢ)))

        @no_escape begin
            cell_count = length(NeighborCellLists)
            color_count = CellColorCount(Val(D))
            cell_colors = @alloc(Int, cell_count)
            color_counts = @alloc(Int, color_count)
            color_offsets = @alloc(Int, color_count + 1)
            color_positions = @alloc(Int, color_count)
            cell_order = @alloc(Int, cell_count)
            BuildCellColorOrdering!(cell_colors, color_counts, color_offsets, color_positions, cell_order, UniqueCellsView)

            @inbounds for color in 1:color_count
                range_start = color_offsets[color]
                range_end = color_offsets[color + 1] - 1
                if range_start <= range_end
                    range_len = range_end - range_start + 1
                    chunk_count = min(thread_count, range_len)
                    chunk_size = cld(range_len, chunk_count)
                    @sync for chunk_id in 1:chunk_count
                        chunk_start = range_start + (chunk_id - 1) * chunk_size
                        chunk_end = min(chunk_start + chunk_size - 1, range_end)
                        Threads.@spawn begin
                            @inbounds for idx in chunk_start:chunk_end
                                CellListIndex = cell_order[idx]
                                SameCellStart = ParticleRanges[CellListIndex]
                                SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

                                for i in SameCellStart:SameCellEnd
                                    for j in (i + 1):SameCellEnd
                                        dρdt_i, dρdt_j, acc_i, acc_j, shift_c_i, shift_c_j, shift_r_i, shift_r_j =
                                            ComputeInteractionsPairwiseNoKernel!(
                                                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                SimConstants, SimParticles, Position, Density, Pressure,
                                                Velocity, ParticleType, i, j,
                                            )
                                        dρdtI[i] += dρdt_i
                                        dρdtI[j] += dρdt_j
                                        Acceleration[i] += acc_i
                                        Acceleration[j] += acc_j
                                        ∇Cᵢ[i] += shift_c_i
                                        ∇Cᵢ[j] += shift_c_j
                                        ∇◌rᵢ[i] += shift_r_i
                                        ∇◌rᵢ[j] += shift_r_j
                                    end
                                end

                                for NeighborIdx in NeighborCellLists[CellListIndex]
                                    if NeighborIdx > CellListIndex
                                        StartIndex_ = ParticleRanges[NeighborIdx]
                                        EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                                        for i in SameCellStart:SameCellEnd
                                            for j in StartIndex_:EndIndex_
                                                dρdt_i, dρdt_j, acc_i, acc_j, shift_c_i, shift_c_j, shift_r_i, shift_r_j =
                                                    ComputeInteractionsPairwiseNoKernel!(
                                                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                        SimConstants, SimParticles, Position, Density, Pressure,
                                                        Velocity, ParticleType, i, j,
                                                    )
                                                dρdtI[i] += dρdt_i
                                                dρdtI[j] += dρdt_j
                                                Acceleration[i] += acc_i
                                                Acceleration[j] += acc_j
                                                ∇Cᵢ[i] += shift_c_i
                                                ∇Cᵢ[j] += shift_c_j
                                                ∇◌rᵢ[i] += shift_r_i
                                                ∇◌rᵢ[j] += shift_r_j
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            @inbounds for i in eachindex(AccelerationMax)
                AccelerationMax[i] = norm(Acceleration[i])
            end
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationMax;
                                      Position = SimParticles.Position,
                                      Density  = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity) where {D,T,
                                                  S<:ShiftingMode,
                                                  K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                    shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                    shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                        shift_r_acc = ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
            ∇Cᵢ[i] = shift_c_acc
            ∇◌rᵢ[i] = shift_r_acc
            AccelerationMax[i] = norm(acc_acc)
        end

        return nothing
    end

    function NeighborLoopPairwise!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                   SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                   SimConstants, SimParticles, ParticleRanges,
                                   CellListIndices, NeighborCellLists, dρdtI,
                                   Acceleration, ∇Cᵢ,
                                   ∇◌rᵢ, AccelerationMax;
                                   Position = SimParticles.Position,
                                   Density = SimParticles.Density,
                                   Pressure = SimParticles.Pressure,
                                   Velocity = SimParticles.Velocity) where {D,T,
                                               S<:ShiftingMode,
                                               K<:KernelOutputMode,
                                               B<:MDBCMode,L<:LogMode,
                                               SDD<:SPHDensityDiffusion,
                                               SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))
        fill!(Kernel, zero(eltype(Kernel)))
        fill!(KernelGradient, zero(eltype(KernelGradient)))
        fill!(∇Cᵢ, zero(eltype(∇Cᵢ)))
        fill!(∇◌rᵢ, zero(eltype(∇◌rᵢ)))

        @inbounds for CellListIndex in eachindex(NeighborCellLists)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

            for i in SameCellStart:SameCellEnd
                for j in (i + 1):SameCellEnd
                    dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j,
                        shift_c_i, shift_c_j, shift_r_i, shift_r_j = ComputeInteractionsPairwise!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, i, j,
                    )
                    dρdtI[i] += dρdt_i
                    dρdtI[j] += dρdt_j
                    Acceleration[i] += acc_i
                    Acceleration[j] += acc_j
                    Kernel[i] += kernel_i
                    Kernel[j] += kernel_j
                    KernelGradient[i] += kernel_grad_i
                    KernelGradient[j] += kernel_grad_j
                    ∇Cᵢ[i] += shift_c_i
                    ∇Cᵢ[j] += shift_c_j
                    ∇◌rᵢ[i] += shift_r_i
                    ∇◌rᵢ[j] += shift_r_j
                end
            end

            for NeighborIdx in NeighborCellLists[CellListIndex]
                if NeighborIdx > CellListIndex
                    StartIndex_ = ParticleRanges[NeighborIdx]
                    EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                    for i in SameCellStart:SameCellEnd
                        for j in StartIndex_:EndIndex_
                            dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j,
                                shift_c_i, shift_c_j, shift_r_i, shift_r_j = ComputeInteractionsPairwise!(
                                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                SimConstants, SimParticles, Position, Density, Pressure,
                                Velocity, ParticleType, i, j,
                            )
                            dρdtI[i] += dρdt_i
                            dρdtI[j] += dρdt_j
                            Acceleration[i] += acc_i
                            Acceleration[j] += acc_j
                            Kernel[i] += kernel_i
                            Kernel[j] += kernel_j
                            KernelGradient[i] += kernel_grad_i
                            KernelGradient[j] += kernel_grad_j
                            ∇Cᵢ[i] += shift_c_i
                            ∇Cᵢ[j] += shift_c_j
                            ∇◌rᵢ[i] += shift_r_i
                            ∇◌rᵢ[j] += shift_r_j
                        end
                    end
                end
            end
        end

        @inbounds for i in eachindex(AccelerationMax)
            AccelerationMax[i] = norm(Acceleration[i])
        end

        return nothing
    end

    function NeighborLoopPairwiseThreaded!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                           SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                           SimConstants, SimParticles, UniqueCellsView,
                                           ParticleRanges, CellListIndices, NeighborCellLists, dρdtI,
                                           Acceleration, ∇Cᵢ,
                                           ∇◌rᵢ, AccelerationMax;
                                           Position = SimParticles.Position,
                                           Density = SimParticles.Density,
                                           Pressure = SimParticles.Pressure,
                                           Velocity = SimParticles.Velocity) where {D,T,
                                                       S<:ShiftingMode,
                                                       K<:KernelOutputMode,
                                                       B<:MDBCMode,L<:LogMode,
                                                       SDD<:SPHDensityDiffusion,
                                                       SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        thread_count = Threads.nthreads()
        fill!(dρdtI, zero(eltype(dρdtI)))
        fill!(Acceleration, zero(eltype(Acceleration)))
        fill!(Kernel, zero(eltype(Kernel)))
        fill!(KernelGradient, zero(eltype(KernelGradient)))
        fill!(∇Cᵢ, zero(eltype(∇Cᵢ)))
        fill!(∇◌rᵢ, zero(eltype(∇◌rᵢ)))

        @no_escape begin
            cell_count = length(NeighborCellLists)
            color_count = CellColorCount(Val(D))
            cell_colors = @alloc(Int, cell_count)
            color_counts = @alloc(Int, color_count)
            color_offsets = @alloc(Int, color_count + 1)
            color_positions = @alloc(Int, color_count)
            cell_order = @alloc(Int, cell_count)
            BuildCellColorOrdering!(cell_colors, color_counts, color_offsets, color_positions, cell_order, UniqueCellsView)

            @inbounds for color in 1:color_count
                range_start = color_offsets[color]
                range_end = color_offsets[color + 1] - 1
                if range_start <= range_end
                    range_len = range_end - range_start + 1
                    chunk_count = min(thread_count, range_len)
                    chunk_size = cld(range_len, chunk_count)
                    @sync for chunk_id in 1:chunk_count
                        chunk_start = range_start + (chunk_id - 1) * chunk_size
                        chunk_end = min(chunk_start + chunk_size - 1, range_end)
                        Threads.@spawn begin
                            @inbounds for idx in chunk_start:chunk_end
                                CellListIndex = cell_order[idx]
                                SameCellStart = ParticleRanges[CellListIndex]
                                SameCellEnd = ParticleRanges[CellListIndex + 1] - 1

                                for i in SameCellStart:SameCellEnd
                                    for j in (i + 1):SameCellEnd
                                        dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j,
                                            shift_c_i, shift_c_j, shift_r_i, shift_r_j = ComputeInteractionsPairwise!(
                                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                            SimConstants, SimParticles, Position, Density, Pressure,
                                            Velocity, ParticleType, i, j,
                                        )
                                        dρdtI[i] += dρdt_i
                                        dρdtI[j] += dρdt_j
                                        Acceleration[i] += acc_i
                                        Acceleration[j] += acc_j
                                        Kernel[i] += kernel_i
                                        Kernel[j] += kernel_j
                                        KernelGradient[i] += kernel_grad_i
                                        KernelGradient[j] += kernel_grad_j
                                        ∇Cᵢ[i] += shift_c_i
                                        ∇Cᵢ[j] += shift_c_j
                                        ∇◌rᵢ[i] += shift_r_i
                                        ∇◌rᵢ[j] += shift_r_j
                                    end
                                end

                                for NeighborIdx in NeighborCellLists[CellListIndex]
                                    if NeighborIdx > CellListIndex
                                        StartIndex_ = ParticleRanges[NeighborIdx]
                                        EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                                        for i in SameCellStart:SameCellEnd
                                            for j in StartIndex_:EndIndex_
                                                dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j,
                                                    shift_c_i, shift_c_j, shift_r_i, shift_r_j = ComputeInteractionsPairwise!(
                                                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                                    SimConstants, SimParticles, Position, Density, Pressure,
                                                    Velocity, ParticleType, i, j,
                                                )
                                                dρdtI[i] += dρdt_i
                                                dρdtI[j] += dρdt_j
                                                Acceleration[i] += acc_i
                                                Acceleration[j] += acc_j
                                                Kernel[i] += kernel_i
                                                Kernel[j] += kernel_j
                                                KernelGradient[i] += kernel_grad_i
                                                KernelGradient[j] += kernel_grad_j
                                                ∇Cᵢ[i] += shift_c_i
                                                ∇Cᵢ[j] += shift_c_j
                                                ∇◌rᵢ[i] += shift_r_i
                                                ∇◌rᵢ[j] += shift_r_j
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            @inbounds for i in eachindex(AccelerationMax)
                AccelerationMax[i] = norm(Acceleration[i])
            end
        end

        return nothing
    end

    f(SimKernel, GhostPoint) = CartesianIndex(map(x -> MapFloor(x, SimKernel.H⁻¹), Tuple(GhostPoint)))
    function NeighborLoopMDBC!(SimKernel,
                               SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                               SimConstants, ParticleRanges, UniqueCellsView,
                               SimParticles, bᵧ, Aᵧ) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}

        @unpack Position, Density, GhostPoints, GhostNormals = SimParticles
        ParticleType = SimParticles.Type                       

        FullStencil = ConstructStencil(Val(Dimensions))

        @inbounds @threads for iter in eachindex(GhostPoints)
            GhostPoint = GhostPoints[iter]

            if !iszero(GhostPoint)
                # zero‐initialize per‐ghost accumulators
                b_acc = zero(bᵧ[iter])            # an SVector{D+1,FloatType}
                A_acc = zero(Aᵧ[iter])            # an SMatrix{D+1,D+1,FloatType}
            
                # compute and accumulate into the locals
                GhostCellIndex = f(SimKernel, GhostPoints[iter])
                @inbounds for offset ∈ FullStencil
                    SCellIndex = GhostCellIndex + offset

                    # Returns a range, x>:x for exact match and x=:x for no match
                    # utilizes that it is a sorted array and requires no isequal constructor,
                    # so I prefer this for now
                    NeighborIdx = FindCellIndex(UniqueCellsView, SCellIndex)

                    StartIndex_       = ParticleRanges[NeighborIdx] 
                    EndIndex_         = ParticleRanges[NeighborIdx + 1] - 1

                    for j in StartIndex_:EndIndex_
                        # change ComputeInteractions to take & return contributions, e.g.:
                        bΔ, AΔ = ComputeInteractionsMDBC!(SimKernel, SimMetaData, SimConstants,
                                                        Position, Density, ParticleType,
                                                        GhostPoints, iter, j)
                        b_acc += bΔ
                        A_acc += AΔ
                    end
                end
            
                # write out once
                bᵧ[iter] = b_acc
                Aᵧ[iter] = A_acc
            end    
        end

        return nothing
    end

    # The previous generic `ComputeInteractions!` implementation was unused
    # in favour of the per-particle variants (ComputeInteractionsPerParticle! etc.).
    # It has been removed to reduce code size and avoid dead code.

    @inline function compute_kernel_output_local(::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                                 kernel_acc, kernel_grad_acc, SimKernel,
                                                 q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                                 B<:MDBCMode,
                                                                 L<:LogMode}
        return kernel_acc, kernel_grad_acc
    end

    @inline function compute_kernel_output_local(::SimulationMetaData{D,T,S,StoreKernelOutput,B,L},
                                                 kernel_acc, kernel_grad_acc, SimKernel,
                                                 q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                                 B<:MDBCMode,
                                                                 L<:LogMode}
        Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
        return kernel_acc + Wᵢⱼ, kernel_grad_acc + ∇ᵢWᵢⱼ
    end

    Base.@propagate_inbounds function ComputeInteractionsPairwise!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        i, j) where {D,T,
                     K<:KernelOutputMode,
                     B<:MDBCMode,
                     L<:LogMode,
                     SDD<:SPHDensityDiffusion,
                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        dρdt_i = zero(eltype(Density))
        dρdt_j = zero(eltype(Density))
        acc_i = zero(eltype(Position))
        acc_j = zero(eltype(Position))
        kernel_i = zero(eltype(Density))
        kernel_j = zero(eltype(Density))
        kernel_grad_i = zero(eltype(Position))
        kernel_grad_j = zero(eltype(Position))

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt_i = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term
            dρdt_j = -ρⱼ * (m₀ / ρᵢ) * density_symmetric_term

            Dᵢ, Dⱼ = compute_density_diffusion(
                SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType,
            )
            dρdt_i += Dᵢ
            dρdt_j += Dⱼ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdtᵢ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ
            dvdtⱼ = -dvdtᵢ

            visc_i, visc_j = compute_viscosity(
                SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j,
            )

            acc_i = dvdtᵢ + visc_i
            acc_j = dvdtⱼ + visc_j

            kernel_i, kernel_grad_i = compute_kernel_output_local(SimMetaData, kernel_i, kernel_grad_i, SimKernel, q, ∇ᵢWᵢⱼ)
            kernel_j, kernel_grad_j = compute_kernel_output_local(SimMetaData, kernel_j, kernel_grad_j, SimKernel, q, -∇ᵢWᵢⱼ)
        end

        return dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j
    end

    Base.@propagate_inbounds function ComputeInteractionsPairwiseNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        i, j) where {D,T,B<:MDBCMode,L<:LogMode,
                     SDD<:SPHDensityDiffusion,
                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        dρdt_i = zero(eltype(Density))
        dρdt_j = zero(eltype(Density))
        acc_i = zero(eltype(Position))
        acc_j = zero(eltype(Position))

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt_i = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term
            dρdt_j = -ρⱼ * (m₀ / ρᵢ) * density_symmetric_term

            Dᵢ, Dⱼ = compute_density_diffusion(
                SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType,
            )
            dρdt_i += Dᵢ
            dρdt_j += Dⱼ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdtᵢ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ
            dvdtⱼ = -dvdtᵢ

            visc_i, visc_j = compute_viscosity(
                SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j,
            )

            acc_i = dvdtᵢ + visc_i
            acc_j = dvdtⱼ + visc_j
        end

        return dρdt_i, dρdt_j, acc_i, acc_j
    end

    Base.@propagate_inbounds function ComputeInteractionsPairwise!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        i, j) where {D,T,S<:ShiftingMode,
                     K<:KernelOutputMode,
                     B<:MDBCMode,
                     L<:LogMode,
                     SDD<:SPHDensityDiffusion,
                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        dρdt_i = zero(eltype(Density))
        dρdt_j = zero(eltype(Density))
        acc_i = zero(eltype(Position))
        acc_j = zero(eltype(Position))
        kernel_i = zero(eltype(Density))
        kernel_j = zero(eltype(Density))
        kernel_grad_i = zero(eltype(Position))
        kernel_grad_j = zero(eltype(Position))
        shift_c_i = zero(eltype(Position))
        shift_c_j = zero(eltype(Position))
        shift_r_i = zero(eltype(Density))
        shift_r_j = zero(eltype(Density))

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt_i = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term
            dρdt_j = -ρⱼ * (m₀ / ρᵢ) * density_symmetric_term

            Dᵢ, Dⱼ = compute_density_diffusion(
                SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType,
            )
            dρdt_i += Dᵢ
            dρdt_j += Dⱼ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdtᵢ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ
            dvdtⱼ = -dvdtᵢ

            visc_i, visc_j = compute_viscosity(
                SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j,
            )

            acc_i = dvdtᵢ + visc_i
            acc_j = dvdtⱼ + visc_j

            kernel_i, kernel_grad_i = compute_kernel_output_local(SimMetaData, kernel_i, kernel_grad_i, SimKernel, q, ∇ᵢWᵢⱼ)
            kernel_j, kernel_grad_j = compute_kernel_output_local(SimMetaData, kernel_j, kernel_grad_j, SimKernel, q, -∇ᵢWᵢⱼ)

            MotionLimiterCondition = MotionLimiterValue(eltype(ρᵢ), ParticleType[i]) * MotionLimiterValue(eltype(ρᵢ), ParticleType[j])
            shift_c_i = (m₀ / ρᵢ) * ∇ᵢWᵢⱼ
            shift_c_j = (m₀ / ρⱼ) * -∇ᵢWᵢⱼ
            shift_r_i = (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
            shift_r_j = (m₀ / ρᵢ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
        end

        return dρdt_i, dρdt_j, acc_i, acc_j, kernel_i, kernel_j, kernel_grad_i, kernel_grad_j, shift_c_i, shift_c_j, shift_r_i, shift_r_j
    end

    Base.@propagate_inbounds function ComputeInteractionsPairwiseNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        i, j) where {D,T,
                     S<:ShiftingMode,
                     B<:MDBCMode,
                     L<:LogMode,
                     SDD<:SPHDensityDiffusion,
                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        dρdt_i = zero(eltype(Density))
        dρdt_j = zero(eltype(Density))
        acc_i = zero(eltype(Position))
        acc_j = zero(eltype(Position))
        shift_c_i = zero(eltype(Position))
        shift_c_j = zero(eltype(Position))
        shift_r_i = zero(eltype(Density))
        shift_r_j = zero(eltype(Density))

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt_i = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term
            dρdt_j = -ρⱼ * (m₀ / ρᵢ) * density_symmetric_term

            Dᵢ, Dⱼ = compute_density_diffusion(
                SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType,
            )
            dρdt_i += Dᵢ
            dρdt_j += Dⱼ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdtᵢ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ
            dvdtⱼ = -dvdtᵢ

            visc_i, visc_j = compute_viscosity(
                SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j,
            )

            acc_i = dvdtᵢ + visc_i
            acc_j = dvdtⱼ + visc_j

            MotionLimiterCondition = MotionLimiterValue(eltype(ρᵢ), ParticleType[i]) * MotionLimiterValue(eltype(ρᵢ), ParticleType[j])
            shift_c_i = (m₀ / ρᵢ) * ∇ᵢWᵢⱼ
            shift_c_j = (m₀ / ρⱼ) * -∇ᵢWᵢⱼ
            shift_r_i = (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
            shift_r_j = (m₀ / ρᵢ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
        end

        return dρdt_i, dρdt_j, acc_i, acc_j, shift_c_i, shift_c_j, shift_r_i, shift_r_j
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
            dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, i, j) where {D,T,
                                                                      K<:KernelOutputMode,
                                                                      B<:MDBCMode,
                                                                      L<:LogMode,
                                                                      SDD<:SPHDensityDiffusion,
                                                                      SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc = compute_kernel_output_local(SimMetaData, kernel_acc, kernel_grad_acc, SimKernel, q, ∇ᵢWᵢⱼ)
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticleNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, i, j) where {D,T,B<:MDBCMode,L<:LogMode,
                                        SDD<:SPHDensityDiffusion,
                                        SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term
        end

        return dρdt_acc, acc_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
        shift_r_acc, i, j) where {D,T,S<:ShiftingMode,
                                  K<:KernelOutputMode,
                                  B<:MDBCMode,
                                  L<:LogMode,
                                  SDD<:SPHDensityDiffusion,
                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc = compute_kernel_output_local(SimMetaData, kernel_acc, kernel_grad_acc, SimKernel, q, ∇ᵢWᵢⱼ)

            MotionLimiterCondition = ParticleType[i]==Fluid && ParticleType[j]==Fluid #MotionLimiterValue(eltype(ρᵢ), ParticleType[i]) * MotionLimiterValue(eltype(ρᵢ), ParticleType[j])
            shift_c_acc += (m₀ / ρⱼ) * Wᵢⱼ * (m₀ / ρᵢ) * ∇ᵢWᵢⱼ * MotionLimiterCondition
            shift_r_acc += (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticleNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, shift_c_acc, shift_r_acc, i, j) where {D,T,
                                                                  S<:ShiftingMode,
                                                                  B<:MDBCMode,
                                                                  L<:LogMode,
                                                                  SDD<:SPHDensityDiffusion,
                                                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            dᵢⱼ² = dᵢⱼ^2
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            MotionLimiterCondition = ParticleType[i]==Fluid && ParticleType[j]==Fluid #MotionLimiterValue(eltype(ρᵢ), ParticleType[i]) * MotionLimiterValue(eltype(ρᵢ), ParticleType[j])
            shift_c_acc += (m₀ / ρⱼ) * Wᵢⱼ * (m₀ / ρⱼ) * ∇ᵢWᵢⱼ * MotionLimiterCondition
            shift_r_acc += (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
        end

        return dρdt_acc, acc_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsMDBC!(SimKernel, SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}, SimConstants, Position, Density, ParticleType, GhostPoints, i, j) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
        @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
        
        @unpack h⁻¹, h, η², H², αD = SimKernel 

        DimensionsPlus = Dimensions + 1
        # always zero‐initialize
        bΔ = zero(SVector{DimensionsPlus,FloatType})
        AΔ = zero(SMatrix{DimensionsPlus, DimensionsPlus,FloatType})

        # ᵢ is ghost node! ⱼ is fluid node

        if ParticleType[j] == Fluid

            xᵢⱼ  = GhostPoints[i] - Position[j]

            xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
            if xᵢⱼ² <= H²
                dᵢⱼ = sqrt(abs(xᵢⱼ²))
                q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
        
                ρⱼ = Density[j]

        
                Wᵢⱼ = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)

                ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                Vⱼ = m₀ / ρⱼ
        
                VⱼWᵢⱼ = Vⱼ * Wᵢⱼ
        
                bΔ  = SVector{DimensionsPlus, FloatType}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)

                # Filling the Aᵧ matrix is done in column-major order
                xⱼᵢ = -xᵢⱼ
                first_column = [VⱼWᵢⱼ; Vⱼ * ∇ᵢWᵢⱼ]
                AΔ = SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus*DimensionsPlus}(
                    first_column...,
                    ((xⱼᵢ * first_column')')...
                )
            end
        end
        
    
        return bΔ, AΔ
    end

    function ApplyMDBCBeforeHalf!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, _args...) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
        return nothing
    end

    function ApplyMDBCBeforeHalf!(SimMetaData::SimulationMetaData{D,T,S,K,SimpleMDBC,L},
                                  SimKernel, SimConstants, SimParticles,
                                  ParticleRanges, UniqueCells
                                 ) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        @no_escape begin
            DimensionsPlus = D + 1
            bᵧ = @alloc(SVector{DimensionsPlus, T}, length(SimParticles.Position))
            Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, T, DimensionsPlus*DimensionsPlus}, length(SimParticles.Position))
            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            NeighborLoopMDBC!(SimKernel, SimMetaData, SimConstants, ParticleRanges, UniqueCellsView, SimParticles, bᵧ, Aᵧ)
            ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)
        end

        return nothing
    end

    function ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)

        Position    = SimParticles.Position
        Density     = SimParticles.Density
        GhostPoints = SimParticles.GhostPoints

        ρ₀ = SimConstants.ρ₀
        #https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
        @inbounds @simd ivdep for i in eachindex(Position)
            A = Aᵧ[i]

            # Since Aᵧ is not reset anymore, we need to check if it is zero
            if !iszero(GhostPoints[i])
                if abs(det(A)) >= 1e-3
                        GhostPointDensity = A \ bᵧ[i]
                        diff = Position[i] - GhostPoints[i]
                        v1   = first(GhostPointDensity) + sum(GhostPointDensity[j+1] * diff[j] for j in eachindex(diff))
                        Density[i] = isnan(v1) ? ρ₀ : v1
                elseif first(A) > 0.0
                        v = first(bᵧ[i]) / first(A)
                        Density[i] = isnan(v) ? ρ₀ : v
                end
            end
        end
    end
    
    function GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)
        # Assuming group markers are sequential
        MotionDefinition = Vector{Union{Nothing, MotionDetails{Dimensions, FloatType}}}(undef, maximum(SimParticles.GroupMarker))

        for geom in SimGeometry
            group_marker = geom.GroupMarker
            if geom.Motion !== nothing
                MotionDefinition[group_marker] = geom.Motion
            else
                MotionDefinition[group_marker] = nothing
            end
        end
        if !any(!isnothing, MotionDefinition)
            MotionDefinition = nothing
        end
        return MotionDefinition
    end


    # Per-particle local Δx removed: use single scalar `SimMetaData.Δx`.

    @inbounds function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                      SimConstants, SimParticles, FullStencil,
                                      ParticleRanges, UniqueCells, CellListIndices,
                                      SortingScratchSpace,
                                      NeighborCellLists, dρdtI, Velocityₙ⁺,
                                      Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ,
                                      MotionDefinition::Union{
                                          Nothing,
                                          AbstractVector{
                                              Union{
                                                  Nothing,
                                                  MotionDetails{Dimensions, FloatType},
                                              },
                                          },
                                      }) where {
                                                Dimensions, FloatType, SMode, KMode,
                                                BMode, LMode,
                                                SDD<:SPHDensityDiffusion,
                                                SV<:SPHViscosity}
        ParticleType   = SimParticles.Type
        ParticleMarker = SimParticles.GroupMarker
        GhostPoints    = hasproperty(SimParticles, :GhostPoints) ? SimParticles.GhostPoints : nothing
        GhostNormals   = hasproperty(SimParticles, :GhostNormals) ? SimParticles.GhostNormals : nothing

        ###
        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
        # This code here is to initialize the first time step for each simulation loop
        dt = SimConstants.CFL * (SimKernel.h / SimConstants.c₀)
        TimeSteppingMode = SimMetaData.TimeSteppingMode

        @no_escape begin
            AccelerationMax = @alloc(FloatType, length(SimParticles.Position))
            dt₂ = dt * 0.5

            SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace, ParticleRanges, UniqueCells, CellListIndices)
            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges)

            if TimeSteppingMode isa SingleNeighborTimeStepping
                @timeit SimMetaData.HourGlass "00 Init Pressure"                          Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
                @timeit SimMetaData.HourGlass "00a Init MDBC"                             ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells)
                if Threads.nthreads() == 1
                    @timeit SimMetaData.HourGlass "00b Init NeighborLoop" NeighborLoopPairwise!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, ParticleRanges, CellListIndices,
                        NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                    )
                else
                    @timeit SimMetaData.HourGlass "00b Init NeighborLoop" NeighborLoopPairwiseThreaded!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, UniqueCellsView, ParticleRanges, CellListIndices,
                        NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                    )
                end
            end

            while SimMetaData.TotalTime <= next_output_time(SimMetaData)
                @timeit SimMetaData.HourGlass "01 Calculate IndexCounter"  begin

                    SimMetaData.Δx = UpdateΔx!(SimMetaData.Δx, Positionₙ⁺, SimParticles.Position)
                    ShouldRebuild = SimMetaData.Δx >= SimKernel.h

                    # println("Δx: ", Δx, "h: ", SimKernel.h," dt: ", SimMetaData.CurrentTimeStep, " Iteration: ", SimMetaData.Iteration, " TotalTime: ", SimMetaData.TotalTime, " OutputIterationCounter: ", SimMetaData.OutputIterationCounter)

                    # Note: If particles are not inside of the neighbor list visualiation, try setting this if statement to always true, since UniqueCells will be updated always then
                    # In theory, the maximal speed is the speed of sound, this should give a safe guard
                    # and ensure it is always updated in a reasonable manner. This only works well, assuming that
                    # c₀ >= maximum(norm.(Velocity))
                    # Remove if statement logic if you want to update each iteration
                    # if mod(SimMetaData.Iteration, ceil(Int, SimKernel.H / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
                    if ShouldRebuild
                        @timeit SimMetaData.HourGlass "01a Actual Calculate IndexCounter" SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace,  ParticleRanges, UniqueCells, CellListIndices)
                        SimMetaData.Δx    = zero(eltype(dρdtI))
                        UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                        BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges)
                    end
                end

                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

                if TimeSteppingMode isa SymplecticTimeStepping
                    @timeit SimMetaData.HourGlass "02 Pressure"                              Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
                    @timeit SimMetaData.HourGlass "03 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells)

                    if Threads.nthreads() == 1
                        @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopPairwise!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                        )
                    else
                        @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopPairwiseThreaded!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, UniqueCellsView, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                        )
                    end

                    @timeit SimMetaData.HourGlass "05 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)

                    @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, ParticleType)

                    @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

                    @timeit SimMetaData.HourGlass "07 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
                    if Threads.nthreads() == 1
                        @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPairwise!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                            Position = Positionₙ⁺,
                            Density = ρₙ⁺,
                            Velocity = Velocityₙ⁺,
                        )
                    else
                        @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPairwiseThreaded!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, UniqueCellsView, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                            Position = Positionₙ⁺,
                            Density = ρₙ⁺,
                            Velocity = Velocityₙ⁺,
                        )
                    end
                else
                    @timeit SimMetaData.HourGlass "02 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells)

                    @timeit SimMetaData.HourGlass "03 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)

                    @timeit SimMetaData.HourGlass "04 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, ParticleType)

                    @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

                    @timeit SimMetaData.HourGlass "05 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
                    if Threads.nthreads() == 1
                        @timeit SimMetaData.HourGlass "06 NeighborLoop" NeighborLoopPairwise!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                            Position = Positionₙ⁺,
                            Density = ρₙ⁺,
                            Velocity = Velocityₙ⁺,
                        )
                    else
                        @timeit SimMetaData.HourGlass "06 NeighborLoop" NeighborLoopPairwiseThreaded!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, UniqueCellsView, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
                            Position = Positionₙ⁺,
                            Density = ρₙ⁺,
                            Velocity = Velocityₙ⁺,
                        )
                    end
                end

                @timeit SimMetaData.HourGlass "07 Final Density"                         DensityEpsi!(SimParticles.Density, dρdtI, ρₙ⁺, dt)

                @timeit SimMetaData.HourGlass "08 Final LimitDensityAtBoundary"          LimitDensityAtBoundary!(SimParticles.Density, SimConstants.ρ₀, ParticleType)

                @timeit SimMetaData.HourGlass "09 Update To Final TimeStep"              FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, Velocityₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt)

                @timeit SimMetaData.HourGlass "10 Update MetaData"                       UpdateMetaData!(SimMetaData, dt)

                @timeit SimMetaData.HourGlass "11 Update TimeStep"                       dt = UpdateTimeStep(AccelerationMax, SimConstants, SimKernel)
            end
        end
        
        return nothing
    end
    
    ###===
    function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}}, #Don't further specify type for now
        SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
        SimConstants::SimulationConstants,
        SimKernel::SPHKernelInstance,
        SimLogger::SimulationLogger,
        SimParticles::StructArray,
        SimViscosity::SV,
        SimDensityDiffusion::SDD,
        SimTimeStepping::TimeSteppingMode,
        ParticleNormalsPath::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType,SMode,KMode,BMode,LMode,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}

        NumberOfPoints = length(SimParticles)

        SimMetaData.TimeSteppingMode = SimTimeStepping

        dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimMetaData, SimParticles.Position)

        LoadMDBCNormals!(SimMetaData, SimParticles, ParticleNormalsPath)

        InitializeLog!(SimMetaData, SimLogger, SimConstants, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        
        Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    
        # Produce sorting related variables
        ParticleRanges         = zeros(Int, NumberOfPoints + 1 + 1) # +1 for the last particle, +1 for dummy entry
        UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
        CellListIndices        = zeros(Int, NumberOfPoints)
        FullStencil            = ConstructStencil(Val(Dimensions))
        NeighborCellLists      = [Int[] for _ in 1:length(UniqueCells)]
        _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)

        output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

        # Save initial state, use 1 else this cannot be used to index fid vector
        SimMetaData.OutputIterationCounter = 1
        output.enqueue_particles(SimMetaData.OutputIterationCounter)
        if SimMetaData.IndexCounter > 0
            unique_cells_view = view(UniqueCells, 1:SimMetaData.IndexCounter)
            cell_particle_counts = nothing
            cell_neighbor_counts = nothing
            if SimMetaData.ExportGridCellParticleCounts
                cell_particle_counts = ComputeCellParticleCounts(
                    ParticleRanges,
                    SimMetaData.IndexCounter,
                )
                cell_neighbor_counts = ComputeCellNeighborCounts(
                    ParticleRanges,
                    NeighborCellLists,
                    SimMetaData.IndexCounter,
                )
            end
            output.enqueue_grid(
                SimMetaData.OutputIterationCounter,
                unique_cells_view,
                cell_particle_counts=cell_particle_counts,
                cell_neighbor_counts=cell_neighbor_counts,
            )
        end


        MotionDefinition = GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)

        @inbounds while true

            @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoop(
                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                SimConstants, SimParticles, FullStencil, ParticleRanges,
                UniqueCells, CellListIndices, SortingScratchSpace,
                NeighborCellLists, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺,
                ∇Cᵢ, ∇◌rᵢ, MotionDefinition,
            )
            push!(SimMetaData.TimeSteps, SimMetaData.CurrentTimeStep)

            LogStep!(SimMetaData, SimLogger)

            SimMetaData.OutputIterationCounter += 1

            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            cell_particle_counts = nothing
            cell_neighbor_counts = nothing
            if SimMetaData.ExportGridCellParticleCounts
                cell_particle_counts = ComputeCellParticleCounts(
                    ParticleRanges,
                    length(UniqueCellsView),
                )
                cell_neighbor_counts = ComputeCellNeighborCounts(
                    ParticleRanges,
                    NeighborCellLists,
                    length(UniqueCellsView),
                )
            end
            
            @timeit SimMetaData.HourGlass "13 Save Particle Data"  begin
                output.enqueue_particles(SimMetaData.OutputIterationCounter)
                output.enqueue_grid(SimMetaData.OutputIterationCounter, UniqueCellsView, cell_particle_counts=cell_particle_counts, cell_neighbor_counts=cell_neighbor_counts)
            end

            if SimMetaData.TotalTime > SimMetaData.SimulationTime

                # At end of simulation
                @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

                show(SimMetaData.HourGlass,sortby=:name)
                show(SimMetaData.HourGlass)

                AutoOpenParaview(SimMetaData, output.variable_names)

                FinalizeLog!(SimMetaData, SimLogger)
                AutoOpenLogFile(SimLogger, SimMetaData)

                break
            end
        end
    end
    

end
