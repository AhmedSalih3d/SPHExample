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
using ..SPHNeighborList: MinCell
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
                dρdt_acc, acc_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc = ComputeInteractionsPerParticle!(
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
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, shift_c_acc,
                        shift_r_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticle!(
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
                        ComputeInteractionsPerParticle!(
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

    f(SimKernel, GhostPoint) = CartesianIndex(map(x -> MapFloor(x, SimKernel.H⁻¹), Tuple(GhostPoint)))
    function NeighborLoopMDBC!(SimKernel,
                               SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                               SimConstants, ParticleRanges,
                               SimParticles, bᵧ, Aᵧ) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}

        @unpack Position, Density, GhostPoints = SimParticles
        ParticleType = SimParticles.Type
        GhostIndices = SimMetaData.GhostIndices
        GhostNeighborCellLists = SimMetaData.GhostNeighborCellLists

        if isempty(GhostIndices)
            return nothing
        end

        @inbounds @threads for gpos in eachindex(GhostIndices)
            iter = GhostIndices[gpos]
            GhostPoint = GhostPoints[iter]

            if !iszero(GhostPoint)
                # zero‐initialize per‐ghost accumulators
                b_acc = zero(bᵧ[iter])            # an SVector{D+1,FloatType}
                A_acc = zero(Aᵧ[iter])            # an SMatrix{D+1,D+1,FloatType}

                NeighborCellIndices = GhostNeighborCellLists[gpos]
                @inbounds for NeighborIdx in NeighborCellIndices
                    StartIndex_ = ParticleRanges[NeighborIdx]
                    EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1

                    for j in StartIndex_:EndIndex_
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

    @inline function UpdateGhostNeighborCellLists!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, _args...) where {D,T,S<:ShiftingMode,
                                                                                                          K<:KernelOutputMode,
                                                                                                          L<:LogMode}
        return nothing
    end

    @inline function UpdateGhostIndices!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, _args...) where {D,T,S<:ShiftingMode,
                                                                                                 K<:KernelOutputMode,
                                                                                                 L<:LogMode}
        return nothing
    end

    function UpdateGhostIndices!(SimMetaData::SimulationMetaData{D,T,S,K,SimpleMDBC,L}, SimParticles) where {D,T,S<:ShiftingMode,
                                                                                                            K<:KernelOutputMode,
                                                                                                            L<:LogMode}
        GhostIndices = SimMetaData.GhostIndices
        empty!(GhostIndices)
        @inbounds for i in eachindex(SimParticles.GhostPoints)
            if !iszero(SimParticles.GhostPoints[i])
                push!(GhostIndices, i)
            end
        end

        GhostNeighborCellLists = SimMetaData.GhostNeighborCellLists
        if length(GhostNeighborCellLists) != length(GhostIndices)
            resize!(GhostNeighborCellLists, length(GhostIndices))
            @inbounds for idx in eachindex(GhostNeighborCellLists)
                GhostNeighborCellLists[idx] = Int[]
            end
        end

        return nothing
    end

    function UpdateGhostNeighborCellLists!(SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, SimpleMDBC, LMode},
                                           SimKernel,
                                           SimParticles,
                                           ParticleRanges,
                                           UniqueCellsView,
                                           FullStencil) where {Dimensions, FloatType, SMode, KMode, LMode}
        GhostIndices = SimMetaData.GhostIndices
        if isempty(GhostIndices)
            return nothing
        end

        GhostNeighborCellLists = SimMetaData.GhostNeighborCellLists
        if length(GhostNeighborCellLists) != length(GhostIndices)
            resize!(GhostNeighborCellLists, length(GhostIndices))
            @inbounds for idx in eachindex(GhostNeighborCellLists)
                GhostNeighborCellLists[idx] = Int[]
            end
        end

        @inbounds for (gpos, iter) in enumerate(GhostIndices)
            NeighborCellIndices = GhostNeighborCellLists[gpos]
            empty!(NeighborCellIndices)
            GhostPoint = SimParticles.GhostPoints[iter]
            if iszero(GhostPoint)
                continue
            end

            GhostCellIndex = f(SimKernel, GhostPoint)
            @inbounds for offset ∈ FullStencil
                SCellIndex = GhostCellIndex + offset
                NeighborIdx = FindCellIndex(UniqueCellsView, SCellIndex)
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                if StartIndex_ <= EndIndex_
                    push!(NeighborCellIndices, NeighborIdx)
                end
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


    @inline function ShiftCScale(::SimulationMetaData{D,T,S,K,B,L},
                                 m₀, ρᵢ, ρⱼ) where {D,T,S<:ShiftingMode,
                                                        K<:KernelOutputMode,
                                                        B<:MDBCMode,
                                                        L<:LogMode}
        return (m₀ / ρⱼ) * (m₀ / ρᵢ)
    end

    @inline function ComputeInteractionsPerParticleNoShiftingCore!(
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
        return ComputeInteractionsPerParticleNoShiftingCore!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
            SimParticles, Position, Density, Pressure, Velocity, ParticleType,
            dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, i, j,
        )
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, i, j) where {D,T,B<:MDBCMode,L<:LogMode,
                                        SDD<:SPHDensityDiffusion,
                                        SV<:SPHViscosity}
        dρdt_acc, acc_acc, _, _ = ComputeInteractionsPerParticleNoShiftingCore!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
            SimParticles, Position, Density, Pressure, Velocity, ParticleType,
            dρdt_acc, acc_acc, nothing, nothing, i, j,
        )

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
            shift_c_acc += ShiftCScale(SimMetaData, m₀, ρᵢ, ρⱼ) * Wᵢⱼ * ∇ᵢWᵢⱼ * MotionLimiterCondition
            shift_r_acc += (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, shift_c_acc, shift_r_acc, i, j) where {D,T,
                                                                  S<:ShiftingMode,
                                                                  B<:MDBCMode,
                                                                  L<:LogMode,
                                                                  SDD<:SPHDensityDiffusion,
                                                                  SV<:SPHViscosity}
        dρdt_acc, acc_acc, _, _, shift_c_acc, shift_r_acc = ComputeInteractionsPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, Position, Density, Pressure,
            Velocity, ParticleType, dρdt_acc, acc_acc, nothing, nothing,
            shift_c_acc, shift_r_acc, i, j,
        )

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
                                  ParticleRanges
                                 ) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        if isempty(SimMetaData.GhostIndices)
            return nothing
        end

        @no_escape begin
            DimensionsPlus = D + 1
            bᵧ = @alloc(SVector{DimensionsPlus, T}, length(SimParticles.Position))
            Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, T, DimensionsPlus*DimensionsPlus}, length(SimParticles.Position))
            NeighborLoopMDBC!(SimKernel, SimMetaData, SimConstants, ParticleRanges, SimParticles, bᵧ, Aᵧ)
            ApplyMDBCCorrection(SimConstants, SimParticles, SimMetaData.GhostIndices, bᵧ, Aᵧ)
        end

        return nothing
    end

    function ApplyMDBCCorrection(SimConstants, SimParticles, GhostIndices, bᵧ, Aᵧ)

        Position    = SimParticles.Position
        Density     = SimParticles.Density
        GhostPoints = SimParticles.GhostPoints

        ρ₀ = SimConstants.ρ₀
        #https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
        @inbounds @simd ivdep for gpos in eachindex(GhostIndices)
            i = GhostIndices[gpos]
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

    @inline function CollectGridCounts(::Val{true}, ParticleRanges, NeighborCellLists, CellCount)
        return ComputeCellParticleCounts(ParticleRanges, CellCount),
               ComputeCellNeighborCounts(ParticleRanges, NeighborCellLists, CellCount)
    end

    @inline function CollectGridCounts(::Val{false}, _ParticleRanges, _NeighborCellLists, _CellCount)
        return nothing, nothing
    end

    @inline function ViewUniqueCells(UniqueCells, IndexCounter)
        if IndexCounter <= 0
            return view(UniqueCells, 1:0)
        end
        return view(UniqueCells, 1:IndexCounter)
    end

    @inline function EnqueueGridSnapshot!(output, iteration, SimMetaData, UniqueCellsView, ParticleRanges, NeighborCellLists)
        if isempty(UniqueCellsView)
            return nothing
        end
        cell_particle_counts, cell_neighbor_counts = CollectGridCounts(
            Val(SimMetaData.ExportGridCellParticleCounts), ParticleRanges, NeighborCellLists, length(UniqueCellsView),
        )
        cells_for_output = UniqueCellsView
        if UniqueCellsView[1] == MinCell(eltype(UniqueCellsView))
            if length(UniqueCellsView) == 1
                return nothing
            end
            cells_for_output = view(UniqueCellsView, 2:length(UniqueCellsView))
            if cell_particle_counts !== nothing
                cell_particle_counts = view(cell_particle_counts, 2:length(cell_particle_counts))
                cell_neighbor_counts = view(cell_neighbor_counts, 2:length(cell_neighbor_counts))
            end
        end
        output.enqueue_grid(
            iteration,
            cells_for_output,
            cell_particle_counts=cell_particle_counts,
            cell_neighbor_counts=cell_neighbor_counts,
        )
        return nothing
    end

    @inline function InitializeTimeStepping!(::SymplecticTimeStepping, _args...)
        return nothing
    end

    function InitializeTimeStepping!(::SingleNeighborTimeStepping,
                                     SimDensityDiffusion::SDD,
                                     SimViscosity::SV,
                                     SimKernel,
                                     SimMetaData,
                                     SimConstants,
                                     SimParticles,
                                     ParticleRanges,
                                     CellListIndices,
                                     NeighborCellLists,
                                     dρdtI,
                                     ∇Cᵢ,
                                     ∇◌rᵢ,
                                     AccelerationMax,
                                     UniqueCells) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @timeit SimMetaData.HourGlass "00 Init Pressure"                          Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
        @timeit SimMetaData.HourGlass "00a Init MDBC"                             ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges)
        @timeit SimMetaData.HourGlass "00b Init NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
        )
        return nothing
    end

    function AdvanceTimeStep!(::SymplecticTimeStepping,
                              SimDensityDiffusion::SDD,
                              SimViscosity::SV,
                              SimKernel,
                              SimMetaData,
                              SimConstants,
                              SimParticles,
                              ParticleRanges,
                              CellListIndices,
                              NeighborCellLists,
                              dρdtI,
                              AccelerationMax,
                              Positionₙ⁺,
                              Velocityₙ⁺,
                              ρₙ⁺,
                              ∇Cᵢ,
                              ∇◌rᵢ,
                              dt₂,
                              ParticleType,
                              MotionDefinition,
                              UniqueCells) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @timeit SimMetaData.HourGlass "02 Pressure"                              Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
        @timeit SimMetaData.HourGlass "03 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges)

        @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
        )

        @timeit SimMetaData.HourGlass "05 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)

        @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, ParticleType)

        @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

        @timeit SimMetaData.HourGlass "07 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
        @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
            Position = Positionₙ⁺,
            Density = ρₙ⁺,
            Velocity = Velocityₙ⁺,
        )
        return nothing
    end

    function AdvanceTimeStep!(::SingleNeighborTimeStepping,
                              SimDensityDiffusion::SDD,
                              SimViscosity::SV,
                              SimKernel,
                              SimMetaData,
                              SimConstants,
                              SimParticles,
                              ParticleRanges,
                              CellListIndices,
                              NeighborCellLists,
                              dρdtI,
                              AccelerationMax,
                              Positionₙ⁺,
                              Velocityₙ⁺,
                              ρₙ⁺,
                              ∇Cᵢ,
                              ∇◌rᵢ,
                              dt₂,
                              ParticleType,
                              MotionDefinition,
                              UniqueCells) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @timeit SimMetaData.HourGlass "02 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges)

        @timeit SimMetaData.HourGlass "03 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)

        @timeit SimMetaData.HourGlass "04 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, ParticleType)

        @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

        @timeit SimMetaData.HourGlass "05 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
        @timeit SimMetaData.HourGlass "06 NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
            Position = Positionₙ⁺,
            Density = ρₙ⁺,
            Velocity = Velocityₙ⁺,
        )
        return nothing
    end

    function FinalizeSimulation!(SimMetaData, SimLogger, output)
        # At end of simulation
        @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

        show(SimMetaData.HourGlass, sortby=:name)
        show(SimMetaData.HourGlass)

        AutoOpenParaview(SimMetaData, output.variable_names)

        FinalizeLog!(SimMetaData, SimLogger)
        AutoOpenLogFile(SimLogger, SimMetaData)

        return nothing
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

        ###
        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
        # This code here is to initialize the first time step for each simulation loop
        dt = SimConstants.CFL * (SimKernel.h / SimConstants.c₀)

        @no_escape begin
            AccelerationMax = @alloc(FloatType, length(SimParticles.Position))
            dt₂ = dt * 0.5

            SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace, ParticleRanges, UniqueCells, CellListIndices)
            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges)
            UpdateGhostIndices!(SimMetaData, SimParticles)
            UpdateGhostNeighborCellLists!(SimMetaData, SimKernel, SimParticles, ParticleRanges, UniqueCellsView, FullStencil)

            InitializeTimeStepping!(
                SimMetaData.TimeSteppingMode,
                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                SimConstants, SimParticles, ParticleRanges, CellListIndices,
                NeighborCellLists, dρdtI, ∇Cᵢ, ∇◌rᵢ, AccelerationMax, UniqueCells,
            )

            NextOutputTime = next_output_time(SimMetaData)
            while SimMetaData.TotalTime <= NextOutputTime
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
                        UpdateGhostIndices!(SimMetaData, SimParticles)
                        UpdateGhostNeighborCellLists!(SimMetaData, SimKernel, SimParticles, ParticleRanges, UniqueCellsView, FullStencil)
                    end
                end

                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

                AdvanceTimeStep!(
                    SimMetaData.TimeSteppingMode,
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, ParticleRanges, CellListIndices,
                    NeighborCellLists, dρdtI, AccelerationMax, Positionₙ⁺,
                    Velocityₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt₂, ParticleType,
                    MotionDefinition, UniqueCells,
                )

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
        UniqueCellsView = ViewUniqueCells(UniqueCells, SimMetaData.IndexCounter)
        EnqueueGridSnapshot!(
            output,
            SimMetaData.OutputIterationCounter,
            SimMetaData,
            UniqueCellsView,
            ParticleRanges,
            NeighborCellLists,
        )


        MotionDefinition = GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)

        @inbounds while SimMetaData.TotalTime <= SimMetaData.SimulationTime

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

            UniqueCellsView = ViewUniqueCells(UniqueCells, SimMetaData.IndexCounter)
            
            @timeit SimMetaData.HourGlass "13 Save Particle Data"  begin
                output.enqueue_particles(SimMetaData.OutputIterationCounter)
                EnqueueGridSnapshot!(
                    output,
                    SimMetaData.OutputIterationCounter,
                    SimMetaData,
                    UniqueCellsView,
                    ParticleRanges,
                    NeighborCellLists,
                )
            end
        end

        FinalizeSimulation!(SimMetaData, SimLogger, output)
    end
    

end
