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
using ..SPHNeighborList: BuildNeighborCellLists!, ComputeCellNeighborCounts, ComputeCellParticleCounts, ConstructStencil, ExtractCells!, UpdateNeighbors!, UpdateΔx!
using ..SPHMDBC: InitializeGhostDataRuntime, UpdateGhostIndices!, UpdateGhostNeighborCellLists!, ApplyMDBCBeforeHalf!

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
                                      Acceleration, Cᵢ, ∇Cᵢ,
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
                                      Acceleration, Cᵢ, ∇Cᵢ,
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
                                      Acceleration, Cᵢ, ∇Cᵢ,
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
            concentration_acc = zero(Cᵢ[i])
            shift_grad_raw_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, concentration_acc,
                        shift_grad_raw_acc,
                        shift_r_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, concentration_acc,
                        shift_grad_raw_acc,
                        shift_r_acc, i, j,
                    )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc =
                        ComputeInteractionsPerParticle!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, Pressure,
                            Velocity, ParticleType, dρdt_acc, acc_acc, concentration_acc,
                            shift_grad_raw_acc,
                            shift_r_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            # Thesis Eq. (5.14): Cᵢ = Σⱼ (mⱼ/ρⱼ) Wᵢⱼ
            Cᵢ[i] = concentration_acc
            # Thesis Eq. (5.15): ∇Cᵢ = Cᵢ Σⱼ (mⱼ/ρⱼ) ∇Wᵢⱼ
            ∇Cᵢ[i] = concentration_acc * shift_grad_raw_acc
            ∇◌rᵢ[i] = shift_r_acc
            AccelerationMax[i] = norm(acc_acc)
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, Cᵢ, ∇Cᵢ,
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
            concentration_acc = zero(Cᵢ[i])
            shift_grad_raw_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, concentration_acc, shift_grad_raw_acc,
                    shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, concentration_acc, shift_grad_raw_acc,
                    shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc, i, j,
                )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, concentration_acc, shift_grad_raw_acc,
                        shift_r_acc = ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
            # Thesis Eq. (5.14): Cᵢ = Σⱼ (mⱼ/ρⱼ) Wᵢⱼ
            Cᵢ[i] = concentration_acc
            # Thesis Eq. (5.15): ∇Cᵢ = Cᵢ Σⱼ (mⱼ/ρⱼ) ∇Wᵢⱼ
            ∇Cᵢ[i] = concentration_acc * shift_grad_raw_acc
            ∇◌rᵢ[i] = shift_r_acc
            AccelerationMax[i] = norm(acc_acc)
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

    @inline function ResolveBoundaryMassFactor(::SimulationMetaData{D,T,S,K,B,L}, SimParticles, ParticleType, j) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode,L<:LogMode}
        return one(eltype(SimParticles.Density))
    end

    @inline function ResolveBoundaryMassFactor(::SimulationMetaData{D,T,S,K,UpdatedMDBC,L}, SimParticles, ParticleType, j) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        return ParticleType[j] == Fluid ? one(eltype(SimParticles.Density)) : SimParticles.MDBCBoundaryFactor[j]
    end

    @inline function ResolveDensityVelocityAtJ(::SimulationMetaData{D,T,S,K,B,L}, SimParticles, Velocity, ParticleType, j) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode,L<:LogMode}
        return Velocity[j]
    end

    @inline function ResolveDensityVelocityAtJ(::SimulationMetaData{D,T,S,K,UpdatedMDBC,L}, SimParticles, Velocity, ParticleType, j) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        return ParticleType[j] == Fluid ? Velocity[j] : SimParticles.MDBCMotionVelocity[j]
    end

    @inline function ResolveViscosityVelocityAtJ(::SimulationMetaData{D,T,S,K,B,L}, SimParticles, Velocity, ParticleType, j) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode,L<:LogMode}
        return Velocity[j]
    end

    @inline function ResolveViscosityVelocityAtJ(::SimulationMetaData{D,T,S,K,UpdatedMDBC,L}, SimParticles, Velocity, ParticleType, j) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        return ParticleType[j] == Fluid ? Velocity[j] : SimParticles.MDBCTangentVelocity[j]
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
            dᵢⱼ² = xᵢⱼ²
            dᵢⱼ = sqrt(dᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            InteractionFactor = ResolveBoundaryMassFactor(SimMetaData, SimParticles, ParticleType, j)
            if InteractionFactor > zero(InteractionFactor)
                mⱼ = m₀ * InteractionFactor

                ρᵢ = Density[i]
                ρⱼ = Density[j]

                vᵢ = Velocity[i]
                vⱼ_density = ResolveDensityVelocityAtJ(SimMetaData, SimParticles, Velocity, ParticleType, j)
                vᵢⱼ_density = vᵢ - vⱼ_density
                density_symmetric_term = dot(-vᵢⱼ_density, ∇ᵢWᵢⱼ)
                dρdt⁺ = -ρᵢ * (mⱼ / ρⱼ) * density_symmetric_term

                Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType)

                dρdt_acc += dρdt⁺ + Dᵢ * InteractionFactor

                Pᵢ = Pressure[i]
                Pⱼ = Pressure[j]
                Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
                f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
                dvdt⁺ = -mⱼ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

                vⱼ_visc = ResolveViscosityVelocityAtJ(SimMetaData, SimParticles, Velocity, ParticleType, j)
                vᵢⱼ_visc = vᵢ - vⱼ_visc
                visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ_visc, ∇ᵢWᵢⱼ, dᵢⱼ², i, j)

                acc_acc += dvdt⁺ + visc_term * InteractionFactor

                kernel_acc, kernel_grad_acc = compute_kernel_output_local(SimMetaData, kernel_acc, kernel_grad_acc, SimKernel, q, ∇ᵢWᵢⱼ)
            end
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
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, concentration_acc,
        shift_grad_raw_acc, shift_r_acc, i, j) where {D,T,S<:ShiftingMode,
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
            dᵢⱼ² = xᵢⱼ²
            dᵢⱼ = sqrt(dᵢⱼ²)
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            InteractionFactor = ResolveBoundaryMassFactor(SimMetaData, SimParticles, ParticleType, j)
            if InteractionFactor > zero(InteractionFactor)
                mⱼ = m₀ * InteractionFactor

                ρᵢ = Density[i]
                ρⱼ = Density[j]

                vᵢ = Velocity[i]
                vⱼ_density = ResolveDensityVelocityAtJ(SimMetaData, SimParticles, Velocity, ParticleType, j)
                vᵢⱼ_density = vᵢ - vⱼ_density
                density_symmetric_term = dot(-vᵢⱼ_density, ∇ᵢWᵢⱼ)
                dρdt⁺ = -ρᵢ * (mⱼ / ρⱼ) * density_symmetric_term

                Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², i, j, ParticleType)

                dρdt_acc += dρdt⁺ + Dᵢ * InteractionFactor

                Pᵢ = Pressure[i]
                Pⱼ = Pressure[j]
                Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
                f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
                dvdt⁺ = -mⱼ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

                vⱼ_visc = ResolveViscosityVelocityAtJ(SimMetaData, SimParticles, Velocity, ParticleType, j)
                vᵢⱼ_visc = vᵢ - vⱼ_visc
                visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ_visc, ∇ᵢWᵢⱼ, dᵢⱼ², i, j)

                acc_acc += dvdt⁺ + visc_term * InteractionFactor

                kernel_acc, kernel_grad_acc = compute_kernel_output_local(SimMetaData, kernel_acc, kernel_grad_acc, SimKernel, q, ∇ᵢWᵢⱼ)

                if ParticleType[i] == Fluid
                    Vⱼ = mⱼ / ρⱼ
                    # Concentration and concentration-gradient core sums (Eq. 5.14, 5.15).
                    concentration_acc += Vⱼ * Wᵢⱼ
                    shift_grad_raw_acc += Vⱼ * ∇ᵢWᵢⱼ
                    shift_r_acc += Vⱼ * dot(-xᵢⱼ, ∇ᵢWᵢⱼ)
                end
            end
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc, i, j) where {D,T,
                                                                  S<:ShiftingMode,
                                                                  B<:MDBCMode,
                                                                  L<:LogMode,
                                                                  SDD<:SPHDensityDiffusion,
                                                                  SV<:SPHViscosity}
        dρdt_acc, acc_acc, _, _, concentration_acc, shift_grad_raw_acc, shift_r_acc = ComputeInteractionsPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, Position, Density, Pressure,
            Velocity, ParticleType, dρdt_acc, acc_acc, nothing, nothing,
            concentration_acc, shift_grad_raw_acc, shift_r_acc, i, j,
        )

        return dρdt_acc, acc_acc, concentration_acc, shift_grad_raw_acc, shift_r_acc
    end

    @inline function LimitDensityAtBoundaryForMode!(::SimulationMetaData{D,T,S,K,UpdatedMDBC,L}, Density, ρ₀, ParticleType) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        return nothing
    end

    @inline function LimitDensityAtBoundaryForMode!(::SimulationMetaData{D,T,S,K,B,L}, Density, ρ₀, ParticleType) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode,L<:LogMode}
        LimitDensityAtBoundary!(Density, ρ₀, ParticleType)
        return nothing
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
                                     GhostData,
                                     SimConstants,
                                     SimParticles,
                                     ParticleRanges,
                                     CellListIndices,
                                     NeighborCellLists,
                                     dρdtI,
                                     Cᵢ,
                                     ∇Cᵢ,
                                     ∇◌rᵢ,
                                     AccelerationMax,
                                     UniqueCells) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @timeit SimMetaData.HourGlass "00 Init Pressure"                          Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
        @timeit SimMetaData.HourGlass "00a Init MDBC"                             ApplyMDBCBeforeHalf!(SimMetaData, GhostData, SimKernel, SimConstants, SimParticles, ParticleRanges)
        @timeit SimMetaData.HourGlass "00b Init NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, Cᵢ, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
        )
        return nothing
    end

    @inline function ApplyMDBCBeforeCorrector!(SimMetaData::SimulationMetaData{D,T,S,K,UpdatedMDBC,L},
                                                GhostData,
                                                SimKernel,
                                                SimConstants,
                                                SimParticles,
                                                ParticleRanges) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        ApplyMDBCBeforeHalf!(SimMetaData, GhostData, SimKernel, SimConstants, SimParticles, ParticleRanges)
        return nothing
    end

    @inline function ApplyMDBCBeforeCorrector!(::SimulationMetaData{D,T,S,K,B,L},
                                                _GhostData,
                                                _SimKernel,
                                                _SimConstants,
                                                _SimParticles,
                                                _ParticleRanges) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode,L<:LogMode}
        return nothing
    end

    function AdvanceTimeStep!(::SymplecticTimeStepping,
                              SimDensityDiffusion::SDD,
                              SimViscosity::SV,
                              SimKernel,
                              SimMetaData,
                              GhostData,
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
                              Cᵢ,
                              ∇Cᵢ,
                              ∇◌rᵢ,
                              dt₂,
                              ParticleType,
                              MotionDefinition,
                              UniqueCells,
                              FluidAcceleration,
                              RigidMotionModel) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @timeit SimMetaData.HourGlass "02 Pressure"                              Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
        @timeit SimMetaData.HourGlass "03 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, GhostData, SimKernel, SimConstants, SimParticles, ParticleRanges)

        @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, Cᵢ, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
        )

        @timeit SimMetaData.HourGlass "05 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂, FluidAcceleration)

        @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"           LimitDensityAtBoundaryForMode!(SimMetaData, ρₙ⁺, SimConstants.ρ₀, ParticleType)

        @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

        @timeit SimMetaData.HourGlass "07 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
        @timeit SimMetaData.HourGlass "07a Apply MDBC before Corrector"          ApplyMDBCBeforeCorrector!(SimMetaData, GhostData, SimKernel, SimConstants, SimParticles, ParticleRanges)
        @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, Cᵢ, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
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
                              GhostData,
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
                              Cᵢ,
                              ∇Cᵢ,
                              ∇◌rᵢ,
                              dt₂,
                              ParticleType,
                              MotionDefinition,
                              UniqueCells,
                              FluidAcceleration,
                              RigidMotionModel) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @timeit SimMetaData.HourGlass "02 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, GhostData, SimKernel, SimConstants, SimParticles, ParticleRanges)

        @timeit SimMetaData.HourGlass "03 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂, FluidAcceleration)

        @timeit SimMetaData.HourGlass "04 Half LimitDensityAtBoundary"           LimitDensityAtBoundaryForMode!(SimMetaData, ρₙ⁺, SimConstants.ρ₀, ParticleType)

        @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

        @timeit SimMetaData.HourGlass "05 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
        @timeit SimMetaData.HourGlass "06 NeighborLoop" NeighborLoopPerParticle!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellListIndices,
            NeighborCellLists, dρdtI, SimParticles.Acceleration, Cᵢ, ∇Cᵢ, ∇◌rᵢ, AccelerationMax,
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
                                      GhostData,
                                      SimConstants, SimParticles, FullStencil,
                                      ParticleRanges, UniqueCells, CellListIndices,
                                      SortingScratchSpace,
                                      NeighborCellLists, dρdtI, Velocityₙ⁺,
                                      Positionₙ⁺, ρₙ⁺, Cᵢ, ∇Cᵢ, ∇◌rᵢ,
                                      MotionDefinition::Union{
                                          Nothing,
                                          AbstractVector{
                                              Union{
                                                  Nothing,
                                                  MotionDetails{Dimensions, FloatType},
                                              },
                                          },
                                      },
                                      FluidAccelerationModel::Union{
                                          Nothing,
                                          FluidAccelerationSeries{Dimensions, FloatType},
                                          FluidAccelerationByGroup{Dimensions, FloatType},
                                          FluidAccelerationInputSeries{Dimensions, FloatType},
                                          FluidAccelerationInputByGroup{Dimensions, FloatType},
                                      },
                                      RigidMotionModel::Union{
                                          Nothing,
                                          RigidRotationMotionSeries{Dimensions, FloatType},
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
            TimeStepping.RefreshRigidRotationParticleIndices!(RigidMotionModel, SimParticles)
            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges)
            UpdateGhostIndices!(SimMetaData, GhostData, SimParticles)
            UpdateGhostNeighborCellLists!(SimMetaData, GhostData, SimKernel, SimParticles, ParticleRanges, UniqueCellsView, FullStencil)

            InitializeTimeStepping!(
                SimMetaData.TimeSteppingMode,
                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, GhostData,
                SimConstants, SimParticles, ParticleRanges, CellListIndices,
                NeighborCellLists, dρdtI, Cᵢ, ∇Cᵢ, ∇◌rᵢ, AccelerationMax, UniqueCells,
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
                        TimeStepping.RefreshRigidRotationParticleIndices!(RigidMotionModel, SimParticles)
                        SimMetaData.Δx    = zero(eltype(dρdtI))
                        UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                        BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges)
                        UpdateGhostIndices!(SimMetaData, GhostData, SimParticles)
                        UpdateGhostNeighborCellLists!(SimMetaData, GhostData, SimKernel, SimParticles, ParticleRanges, UniqueCellsView, FullStencil)
                    end
                end

                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)
                @timeit SimMetaData.HourGlass "Rigid Motion"                             ApplyRigidRotationMotion!(SimParticles, RigidMotionModel, FloatType, SimMetaData.TotalTime)

                CurrentFluidAcceleration = EvaluateFluidAcceleration(
                    FluidAccelerationModel, FloatType, Val(Dimensions), SimMetaData.TotalTime,
                )

                AdvanceTimeStep!(
                    SimMetaData.TimeSteppingMode,
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, GhostData,
                    SimConstants, SimParticles, ParticleRanges, CellListIndices,
                    NeighborCellLists, dρdtI, AccelerationMax, Positionₙ⁺,
                    Velocityₙ⁺, ρₙ⁺, Cᵢ, ∇Cᵢ, ∇◌rᵢ, dt₂, ParticleType,
                    MotionDefinition, UniqueCells, CurrentFluidAcceleration, RigidMotionModel,
                )

                @timeit SimMetaData.HourGlass "07 Final Density"                         DensityEpsi!(SimParticles.Density, dρdtI, ρₙ⁺, dt)

                @timeit SimMetaData.HourGlass "08 Final LimitDensityAtBoundary"          LimitDensityAtBoundaryForMode!(SimMetaData, SimParticles.Density, SimConstants.ρ₀, ParticleType)

                @timeit SimMetaData.HourGlass "09 Update To Final TimeStep"              FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, Velocityₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt, CurrentFluidAcceleration)

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
        ParticleNormalsPath::Union{Nothing,String} = nothing,
        FluidAccelerationModel::Union{
            Nothing,
            FluidAccelerationSeries{Dimensions, FloatType},
            FluidAccelerationByGroup{Dimensions, FloatType},
            FluidAccelerationInputSeries{Dimensions, FloatType},
            FluidAccelerationInputByGroup{Dimensions, FloatType},
        } = nothing,
        RigidMotionModel::Union{Nothing, RigidRotationMotionSeries{Dimensions, FloatType}} = nothing
        ) where {Dimensions,FloatType,SMode,KMode,BMode,LMode,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}

        NumberOfPoints = length(SimParticles)

        SimMetaData.TimeSteppingMode = SimTimeStepping

        dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, Cᵢ, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimMetaData, SimParticles.Position)

        GhostData = InitializeGhostDataRuntime(SimMetaData)
        LoadMDBCNormals!(SimMetaData, SimParticles, ParticleNormalsPath, GhostData)

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
                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, GhostData,
                SimConstants, SimParticles, FullStencil, ParticleRanges,
                UniqueCells, CellListIndices, SortingScratchSpace,
                NeighborCellLists, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺,
                Cᵢ, ∇Cᵢ, ∇◌rᵢ, MotionDefinition, FluidAccelerationModel, RigidMotionModel,
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
