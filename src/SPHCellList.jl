module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!, RunSimulation

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

using StaticArrays
import StructArrays: StructArray, foreachfield
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using Base.Threads
using UnicodePlots
using LinearAlgebra
    using Bumper

    function ConstructFullStencil(v::Val{d}) where d
        return CartesianIndices(ntuple(_->-1:1, v))
    end

    function BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView,
                                     ParticleRanges, CellDict)
        target_len = length(UniqueCellsView)
        original_len = length(NeighborCellLists)
        resize!(NeighborCellLists, target_len)
        if target_len > original_len
            @inbounds for idx in (original_len + 1):target_len
                NeighborCellLists[idx] = Int[]
            end
        end
        @inbounds for cell_idx in eachindex(UniqueCellsView)
            neighbors = NeighborCellLists[cell_idx]
            empty!(neighbors)
            cell = UniqueCellsView[cell_idx]
            for offset in FullStencil
                neighbor_cell = cell + offset
                neighbor_idx = get(CellDict, neighbor_cell, 1)
                start_idx = ParticleRanges[neighbor_idx]
                end_idx = ParticleRanges[neighbor_idx + 1] - 1
                if start_idx <= end_idx && neighbor_idx != cell_idx
                    push!(neighbors, neighbor_idx)
                end
            end
        end

        return nothing
    end

    """
    Extracts the cells for each particle based on their positions and the inverse cutoff value.

    # Arguments
    - `Particles`: The particles whose cells are to be extracted.
    - `::Val{InverseCutOff}`: The inverse cutoff value used for cell extraction.

    # Returns
    - `nothing`: This function modifies the `Particles` in place.
    """
    # Replace unsafe_trunc with trunc if this ever errors
    @inline function map_floor(x, InverseCutOff)
        # This is different than just doing muladd(x,InverseCutOff,0.5) because it rounds towards zero.
        # Consider -1.7 + 0.5, this would give -1.2 and then trunced 1, but we want -2, therefore absolute addition before hand
        # We add 0.5 instead of 1, to ensure proper rounding behavior when restoring the sign for negative numbers.
        Int(sign(x)) * unsafe_trunc(Int, muladd(abs(x),InverseCutOff,0.5))
    end

    # Add contributions related to particle shifting. Dispatch on `SimulationMetaData`
    # so that no runtime checks are required.
    @inline function add_shifting_terms!(::SimulationMetaData{D,T,NoShifting,K,B,L}, SimThreadedArrays,
                                MotionLimiter, xᵢⱼ, ∇ᵢWᵢⱼ, m₀, ρᵢ, ρⱼ, i, j, ichunk) where {D,T,
                                                                                                K<:KernelOutputMode,
                                                                                                B<:MDBCMode,
                                                                                                L<:LogMode}
        return nothing
    end

    @inline function add_shifting_terms!(::SimulationMetaData{D,T,PlanarShifting,K,B,L}, SimThreadedArrays,
                                MotionLimiter, xᵢⱼ, ∇ᵢWᵢⱼ, m₀, ρᵢ, ρⱼ, i, j, ichunk) where {D,T,
                                                                                                   K<:KernelOutputMode,
                                                                                                   B<:MDBCMode,
                                                                                                   L<:LogMode}
        MLcond = MotionLimiter[i] * MotionLimiter[j]

        SimThreadedArrays.∇CᵢThreaded[ichunk][i]   += (m₀/ρᵢ) *  ∇ᵢWᵢⱼ
        SimThreadedArrays.∇CᵢThreaded[ichunk][j]   += (m₀/ρⱼ) * -∇ᵢWᵢⱼ

        # Switch signs compared to DSPH, else free surface detection does not make sense
        # Agrees, https://arxiv.org/abs/2110.10076, it should have been r_ji
        mark_pair!(SimThreadedArrays.∇CᵢTouched[ichunk], SimThreadedArrays.∇CᵢMask[ichunk], i, j)
        SimThreadedArrays.∇◌rᵢThreaded[ichunk][i]  += (m₀/ρⱼ) * dot(-xᵢⱼ , ∇ᵢWᵢⱼ) * MLcond
        SimThreadedArrays.∇◌rᵢThreaded[ichunk][j]  += (m₀/ρᵢ) * dot( xᵢⱼ ,-∇ᵢWᵢⱼ) * MLcond
        mark_pair!(SimThreadedArrays.∇◌rᵢTouched[ichunk], SimThreadedArrays.∇◌rᵢMask[ichunk], i, j)
        return nothing
    end

    # Optionally record kernel values and gradients based on `SimulationMetaData`
    # This function is designed to be a no-op when kernel output is not requested.
    # The `@inline` annotation encourages the compiler to substitute the function call
    # with its body, which in this case is `nothing`. When the `SimMetaData` type
    # is concrete at the call site, the compiler can completely eliminate this call,
    # resulting in zero runtime overhead.
    @inline function KernelOutput!(::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimKernel,
                            SimThreadedArrays, q, ∇ᵢWᵢⱼ, i, j, ichunk) where {D,T,S<:ShiftingMode,
                                                                               B<:MDBCMode,
                                                                               L<:LogMode}
        return nothing
    end

    # This version is called when kernel output is requested.
    # The `@inline` annotation helps reduce function call overhead, especially
    # since this is called inside a tight loop (`ComputeInteractions!`).
    @inline function KernelOutput!(::SimulationMetaData{D,T,S,StoreKernelOutput,B,L}, SimKernel,
                            SimThreadedArrays, q, ∇ᵢWᵢⱼ, i, j, ichunk) where {D,T,S<:ShiftingMode,
                                                                            B<:MDBCMode,
                                                                            L<:LogMode}
        Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
        mark_pair!(SimThreadedArrays.KernelTouched[ichunk], SimThreadedArrays.KernelMask[ichunk], i, j)
        SimThreadedArrays.KernelThreaded[ichunk][i]         += Wᵢⱼ
        SimThreadedArrays.KernelThreaded[ichunk][j]         += Wᵢⱼ
        mark_pair!(SimThreadedArrays.KernelGradientTouched[ichunk], SimThreadedArrays.KernelGradientMask[ichunk], i, j)
        SimThreadedArrays.KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
        SimThreadedArrays.KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
        return nothing
    end

    @inline function KernelOutputLocal!(::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                        kernel_acc, kernel_grad_acc, SimKernel,
                                        q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                        B<:MDBCMode,
                                                        L<:LogMode}
        return kernel_acc, kernel_grad_acc
    end

    @inline function KernelOutputLocal!(::SimulationMetaData{D,T,S,StoreKernelOutput,B,L},
                                        kernel_acc, kernel_grad_acc, SimKernel,
                                        q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                        B<:MDBCMode,
                                                        L<:LogMode}
        Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
        return kernel_acc + Wᵢⱼ, kernel_grad_acc + ∇ᵢWᵢⱼ
    end
   
    @inline function ExtractCells!(Particles, InverseCutOff)
        @inbounds @simd ivdep for i ∈ eachindex(Particles.Cells)
            Particles.Cells[i] = CartesianIndex(map(x -> map_floor(x, InverseCutOff), Tuple(Particles.Position[i])))
        end
        return nothing
    end

    """
    Updates the neighbor list and sorts particles by their cell indices.

    # Arguments
    - `Particles`: The particles whose neighbors are to be updated.
    - `CutOff`: The cutoff value used for cell extraction.
    - `SortingScratchSpace`: Scratch space for sorting.
    - `ParticleRanges`: Array to store the ranges of particles in each cell.
    - `UniqueCells`: Array to store the unique cells.

    # Returns
    - `IndexCounter`: The number of unique cells identified.
    """
    function UpdateNeighbors!(Particles, InverseCutOff, SortingScratchSpace,
                              ParticleRanges, UniqueCells, CellDict)
        ExtractCells!(Particles, InverseCutOff)

        sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)
        Cells = @views Particles.Cells
        @. ParticleRanges             = zero(eltype(ParticleRanges))
        ParticleRanges[1] = 1
        IndexCounter                  = 2
        ParticleRanges[IndexCounter]  = 1
        UniqueCells[IndexCounter]     = Cells[1]
        empty!(CellDict)
        CellDict[Cells[1]] = IndexCounter

        @inbounds @simd ivdep for i in eachindex(Cells)[2:end]
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                IndexCounter                 += 1
                ParticleRanges[IndexCounter]  = i
                UniqueCells[IndexCounter]     = Cells[i]
                CellDict[Cells[i]]           = IndexCounter
            end
        end
        ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

        return IndexCounter 
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellDict, NeighborCellLists, Position, Density,
                                      Pressure, Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ) where {D,T,B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        Cells = SimParticles.Cells
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            CellIndex = Cells[i]
            CellListIndex = get(CellDict, CellIndex, 1)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, i, j,
                )
            end
            for NeighborIdx in NeighborCellIndices
                StartIndex_ = ParticleRanges[NeighborIdx]
                EndIndex_ = ParticleRanges[NeighborIdx + 1] - 1
                @inbounds for j in StartIndex_:EndIndex_
                    dρdt_acc, acc_acc = ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellDict, NeighborCellLists, Position, Density,
                                      Pressure, Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ) where {D,T,K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        Cells = SimParticles.Cells
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            CellIndex = Cells[i]
            CellListIndex = get(CellDict, CellIndex, 1)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
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
                            Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                            kernel_grad_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellDict, NeighborCellLists, Position, Density,
                                      Pressure, Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ) where {D,T,S<:ShiftingMode,B<:MDBCMode,
                                                  L<:LogMode,SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        Cells = SimParticles.Cells
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellIndex = Cells[i]
            CellListIndex = get(CellDict, CellIndex, 1)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, shift_c_acc,
                        shift_r_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticleNoKernel!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, Pressure,
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, shift_c_acc,
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
                            Velocity, MotionLimiter, dρdt_acc, acc_acc, shift_c_acc,
                            shift_r_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            ∇Cᵢ[i] = shift_c_acc
            ∇◌rᵢ[i] = shift_r_acc
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellDict, NeighborCellLists, Position, Density,
                                      Pressure, Velocity, MotionLimiter, dρdtI,
                                      Acceleration, Kernel, KernelGradient, ∇Cᵢ,
                                      ∇◌rᵢ) where {D,T,S<:ShiftingMode,
                                                  K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        Cells = SimParticles.Cells
        @inbounds Threads.@threads for i in eachindex(Position)
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellIndex = Cells[i]
            CellListIndex = get(CellDict, CellIndex, 1)
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, Pressure,
                    Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
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
                        Velocity, MotionLimiter, dρdt_acc, acc_acc, kernel_acc,
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
        end

        return nothing
    end

    f(SimKernel, GhostPoint) = CartesianIndex(map(x->map_floor(x,SimKernel.H⁻¹), Tuple(GhostPoint)))
    function NeighborLoopMDBC!(SimKernel,
                               SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                               SimConstants, ParticleRanges, CellDict, Position,
                               Density, GhostPoints, GhostNormals, ParticleType,
                               bᵧ, Aᵧ) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
        
        FullStencil = ConstructFullStencil(Val(Dimensions))

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
                    NeighborIdx = get(CellDict, SCellIndex, 1)

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

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, i, j) where {D,T,
                                                                     K<:KernelOutputMode,
                                                                     B<:MDBCMode,
                                                                     L<:LogMode,
                                                                     SDD<:SPHDensityDiffusion,
                                                                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc =
                KernelOutputLocal!(SimMetaData, kernel_acc, kernel_grad_acc,
                                   SimKernel, q, ∇ᵢWᵢⱼ)
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticleNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, i, j) where {D,T,B<:MDBCMode,L<:LogMode,
                                        SDD<:SPHDensityDiffusion,
                                        SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term
        end

        return dρdt_acc, acc_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,K,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
        shift_r_acc, i, j) where {D,T,S<:ShiftingMode,
                                  K<:KernelOutputMode,
                                  B<:MDBCMode,
                                  L<:LogMode,
                                  SDD<:SPHDensityDiffusion,
                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc =
                KernelOutputLocal!(SimMetaData, kernel_acc, kernel_grad_acc,
                                   SimKernel, q, ∇ᵢWᵢⱼ)

            MLcond = MotionLimiter[i] * MotionLimiter[j]
            shift_c_acc += (m₀ / ρᵢ) * ∇ᵢWᵢⱼ
            shift_r_acc += (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticleNoKernel!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, Pressure, Velocity, MotionLimiter,
        dρdt_acc, acc_acc, shift_c_acc, shift_r_acc, i, j) where {D,T,
                                                                  S<:ShiftingMode,
                                                                  B<:MDBCMode,
                                                                  L<:LogMode,
                                                                  SDD<:SPHDensityDiffusion,
                                                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ = sqrt(abs(xᵢⱼ²))
            q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺ = -ρᵢ * (m₀ / ρⱼ) * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel,
                                              SimConstants, SimParticles, xᵢⱼ,
                                              ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants,
                                             SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ,
                                             xᵢⱼ², i, j)

            acc_acc += dvdt⁺ + visc_term

            MLcond = MotionLimiter[i] * MotionLimiter[j]
            shift_c_acc += (m₀ / ρᵢ) * ∇ᵢWᵢⱼ
            shift_r_acc += (m₀ / ρⱼ) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
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

    function prepare_shifting_arrays!(::SimulationMetaData{D,T,NoShifting,K,B,L}, ∇Cᵢ, ∇◌rᵢ) where {D,T,
                                                                                                K<:KernelOutputMode,
                                                                                                B<:MDBCMode,
                                                                                                L<:LogMode}
        resize!(∇Cᵢ, 0)
        resize!(∇◌rᵢ, 0)
        return nothing
    end
    prepare_shifting_arrays!(::SimulationMetaData{D,T,S,K,B,L}, ∇Cᵢ, ∇◌rᵢ) where {D,T,S<:ShiftingMode,
                                                                               K<:KernelOutputMode,
                                                                               B<:MDBCMode,
                                                                               L<:LogMode} = nothing


    function ApplyMDBCBeforeHalf!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, _args...) where {D,T,S<:ShiftingMode,
                                                                                           K<:KernelOutputMode,
                                                                                           L<:LogMode}
        return nothing
    end
    function ApplyMDBCBeforeHalf!(SimMetaData::SimulationMetaData{D,T,S,K,SimpleMDBC,L},
                                  SimKernel, SimConstants, SimParticles,
                                  ParticleRanges, CellDict, Position, Density,
                                  GhostPoints, GhostNormals, ParticleType
                                 ) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        @no_escape begin
            DimensionsPlus = D + 1
            bᵧ = @alloc(SVector{DimensionsPlus, T}, length(Position))
            Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, T, DimensionsPlus*DimensionsPlus}, length(Position))
            NeighborLoopMDBC!(SimKernel, SimMetaData, SimConstants, ParticleRanges, CellDict, Position, Density, GhostPoints,GhostNormals, ParticleType, bᵧ, Aᵧ)
            ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)
        end

        return nothing
    end

    function LoadMDBCNormals!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, SimParticles, path) where {D,T,S<:ShiftingMode,
                                                                                                K<:KernelOutputMode,
                                                                                                L<:LogMode}
        return nothing
    end
    function LoadMDBCNormals!(::SimulationMetaData{D,T,S,K,SimpleMDBC,L}, SimParticles, path) where {D,T,S<:ShiftingMode,
                                                                                                   K<:KernelOutputMode,
                                                                                                   L<:LogMode}
        if isnothing(path)
            return nothing
        end
        _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(D), T, path)
        for gi ∈ eachindex(GhostPoints)
            SimParticles.GhostPoints[gi]  = GhostPoints[gi]
            SimParticles.GhostNormals[gi] = GhostNormals[gi]
        end
        return nothing
    end

    function initialize_log!(::SimulationMetaData{D,T,S,K,B,NoLog}, _args...) where {D,T,S<:ShiftingMode,
                                                                                      K<:KernelOutputMode,
                                                                                      B<:MDBCMode}
        return nothing
    end
    function initialize_log!(SimMetaData::SimulationMetaData{D,T,S,K,B,StoreLog}, SimLogger,
                             SimConstants, SimKernel, SimViscosity, SimDensityDiffusion,
                             SimGeometry, SimParticles) where {D,T,S<:ShiftingMode,
                                                              K<:KernelOutputMode,
                                                              B<:MDBCMode}
        InitializeLogger(SimLogger, SimConstants, SimMetaData, SimKernel,
                         SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        LogStep(SimLogger, SimMetaData, SimMetaData.HourGlass)
        SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        return nothing
    end

    function log_step!(::SimulationMetaData{D,T,S,K,B,NoLog}, _...) where {D,T,S<:ShiftingMode,
                                                                           K<:KernelOutputMode,
                                                                           B<:MDBCMode}
        return nothing
    end
    function log_step!(SimMetaData::SimulationMetaData{D,T,S,K,B,StoreLog}, SimLogger) where {D,T,S<:ShiftingMode,
                                                                                             K<:KernelOutputMode,
                                                                                             B<:MDBCMode}
        LogStep(SimLogger, SimMetaData, SimMetaData.HourGlass)
        SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        return nothing
    end

    function finalize_log!(::SimulationMetaData{D,T,S,K,B,NoLog}, _...) where {D,T,S<:ShiftingMode,
                                                                              K<:KernelOutputMode,
                                                                              B<:MDBCMode}
        return nothing
    end
    function finalize_log!(SimMetaData::SimulationMetaData{D,T,S,K,B,StoreLog}, SimLogger,
                           HourGlass, UnicodeTimeStepsGraph) where {D,T,S<:ShiftingMode,
                                                                      K<:KernelOutputMode,
                                                                      B<:MDBCMode}
        LogFinal(SimLogger, HourGlass)
        with_logger(SimLogger.Logger) do
            @info ""
            show(SimLogger.LoggerIo, UnicodeTimeStepsGraph)
        end
        close(SimLogger.LoggerIo)
        AutoOpenLogFile(SimLogger, SimMetaData)
        return nothing
    end

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
    
    function HalfTimeStep(::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                          SimConstants, SimParticles, Positionₙ⁺,
                          Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
        @unpack Position, Density, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles

        @inbounds @simd ivdep for i in eachindex(Position)
            Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
            Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
            ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
        end


        return nothing
    end

    function FullTimeStep(::SimulationMetaData{D,T,NoShifting,K,B,L}, SimKernel,
                          SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,
                                                                             K<:KernelOutputMode,
                                                                             B<:MDBCMode,
                                                                             L<:LogMode}
        @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles
        @inbounds @simd ivdep for i in eachindex(Position)
            Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
            Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
        end
        return nothing
    end

    function FullTimeStep(::SimulationMetaData{D,T,S,K,B,L}, SimKernel, SimConstants,
                          SimParticles, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,S<:ShiftingMode,
                                                             K<:KernelOutputMode,
                                                             B<:MDBCMode,
                                                             L<:LogMode}
        @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles
        A     = 2# Value between 1 to 6 advised
        A_FST = 0; # zero for internal flows
        A_FSM = length(first(Position)); #2d, 3d val different
        @inbounds @simd ivdep for i in eachindex(Position)
            Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]

            A_FSC                  = (∇◌rᵢ[i] - A_FST)/(A_FSM - A_FST)
            if A_FSC < 0
                δxᵢ = zero(eltype(Position))
            else
                δxᵢ = -A_FSC * A * SimKernel.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
            end

            Position[i]           += (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt + δxᵢ) * MotionLimiter[i]
        end
        return nothing
    end

    function UpdateMetaData!(SimMetaData, dt)
        SimMetaData.Iteration      += 1
        SimMetaData.CurrentTimeStep = dt
        SimMetaData.TotalTime      += dt

        return nothing
    end

    @inline next_output_time(SimMetaData) = next_output_time(SimMetaData.OutputTimes, SimMetaData)

    @inline next_output_time(interval::Real, SimMetaData) = interval * SimMetaData.OutputIterationCounter

    @inline function next_output_time(times::AbstractVector, SimMetaData)
        idx = SimMetaData.OutputIterationCounter
        if idx < length(times)
            return times[idx]
        else
            return SimMetaData.SimulationTime
        end
    end

    """
        update_delta_x!(Δx, posₙ⁺, pos)

    Increment Δx by twice the maximum ‖posₙ⁺[i] – pos[i]‖, without ever allocating.
    Returns the new Δx.
    """
    @inline function update_delta_x!(Δx::T,
                                    posₙ⁺::AbstractVector{SVector{D, T}},
                                    pos   ::AbstractVector{SVector{D, T}}) where {D, T<:Real}
        maxd = zero(T)
        @inbounds for i in eachindex(posₙ⁺, pos)
            # compute squared norm manually
            sumsq = zero(T)
            @inbounds for j in 1:D
                d = posₙ⁺[i][j] - pos[i][j]
                sumsq += d*d
            end
            # sqrt/T is allocation-free on scalars
            nrm = sqrt(sumsq)
            if nrm > maxd
                maxd = nrm
            end
        end
        return Δx + 4*maxd
    end

    # Per-particle local Δx removed: use single scalar `SimMetaData.Δx`.

    
    @inbounds function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                      SimConstants, SimParticles, FullStencil,
                                      ParticleRanges, UniqueCells, CellDict,
                                      SortingScratchSpace, SimThreadedArrays,
                                      NeighborCellLists, dρdtI, Velocityₙ⁺,
                                      Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ,
                                      MotionDefinition) where {Dimensions, FloatType,
                                                               SMode, KMode, BMode,
                                                               LMode, SDD<:SPHDensityDiffusion,
                                                               SV<:SPHViscosity}
        @unpack Position, Density, Pressure, Velocity, Acceleration, MotionLimiter, GroupMarker, Kernel, KernelGradient, GhostPoints, GhostNormals = SimParticles
        ParticleType   = SimParticles.Type
        ParticleMarker = GroupMarker

        ###
        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)

            while SimMetaData.TotalTime <= next_output_time(SimMetaData)

                SimMetaData.Δx = update_delta_x!(SimMetaData.Δx, Positionₙ⁺, SimParticles.Position)
                ShouldRebuild = SimMetaData.Δx >= SimKernel.h

                # println("Δx: ", Δx, "h: ", SimKernel.h," dt: ", SimMetaData.CurrentTimeStep, " Iteration: ", SimMetaData.Iteration, " TotalTime: ", SimMetaData.TotalTime, " OutputIterationCounter: ", SimMetaData.OutputIterationCounter)

                @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants, SimKernel)
                dt₂ = dt * 0.5

                @timeit SimMetaData.HourGlass "02 Calculate IndexCounter"  begin
                    # Note: If particles are not inside of the neighbor list visualiation, try setting this if statement to always true, since UniqueCells will be updated always then
                    # In theory, the maximal speed is the speed of sound, this should give a safe guard
                    # and ensure it is always updated in a reasonable manner. This only works well, assuming that
                    # c₀ >= maximum(norm.(Velocity))
                    # Remove if statement logic if you want to update each iteration
                    # if mod(SimMetaData.Iteration, ceil(Int, SimKernel.H / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
                    if ShouldRebuild
                        @timeit SimMetaData.HourGlass "02a Actual Calculate IndexCounter" SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace,  ParticleRanges, UniqueCells, CellDict)
                        SimMetaData.Δx    = zero(eltype(dρdtI))
                        UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                        BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges, CellDict)
                    end
                end

                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
            
                @timeit SimMetaData.HourGlass "03 Pressure"                              Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
                @timeit SimMetaData.HourGlass "04 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, CellDict, Position, Density, GhostPoints, GhostNormals, ParticleType)

                @timeit SimMetaData.HourGlass "05 First NeighborLoop" NeighborLoopPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, ParticleRanges, CellDict,
                    NeighborCellLists, Position, Density, Pressure, Velocity,
                    MotionLimiter, dρdtI, Acceleration, Kernel,
                    KernelGradient, ∇Cᵢ, ∇◌rᵢ,
                )


                @timeit SimMetaData.HourGlass "06 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)


                @timeit SimMetaData.HourGlass "07 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)
            
                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
            
                @timeit SimMetaData.HourGlass "03 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
                @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, ParticleRanges, CellDict,
                    NeighborCellLists, Positionₙ⁺, ρₙ⁺, Pressure, Velocityₙ⁺,
                    MotionLimiter, dρdtI, Acceleration, Kernel,
                    KernelGradient, ∇Cᵢ, ∇◌rᵢ,
                )

            
                @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary"          LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)
            
                @timeit SimMetaData.HourGlass "10 Final Density"                         DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)
            
                @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"              FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
            
                @timeit SimMetaData.HourGlass "12 Update MetaData"                       UpdateMetaData!(SimMetaData, dt)

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
        ParticleNormalsPath::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType,SMode,KMode,BMode,LMode,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}

        # Unpack the relevant simulation meta data
        @unpack HourGlass = SimMetaData;
        
        dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimMetaData, SimParticles.Position)

        LoadMDBCNormals!(SimMetaData, SimParticles, ParticleNormalsPath)

        prepare_shifting_arrays!(SimMetaData, ∇Cᵢ, ∇◌rᵢ)

        initialize_log!(SimMetaData, SimLogger, SimConstants, SimKernel,
                        SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        
        NumberOfPoints = length(SimParticles)::Int
        Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    
        SimThreadedArrays = AllocateThreadedArrays(SimMetaData, SimParticles, dρdtI, ∇Cᵢ, ∇◌rᵢ)
        SimMetaData.Δx = zero(eltype(dρdtI))
    
        # Produce sorting related variables
        ParticleRanges         = zeros(Int, NumberOfPoints + 1 + 1) # +1 for the last particle, +1 for dummy entry
        UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
        CellDict               = Dict{CartesianIndex{Dimensions}, Int}()
        FullStencil            = ConstructFullStencil(Val(Dimensions))
        NeighborCellLists      = [Int[] for _ in 1:length(UniqueCells)]
        _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)

        output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

        # Save initial state, use 1 else this cannot be used to index fid vector
        SimMetaData.OutputIterationCounter = 1
        output.save_particles(SimMetaData.OutputIterationCounter)
        output.save_grid(SimMetaData.OutputIterationCounter, UniqueCells, SimParticles)


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

        # Normal run and save data
        generate_showvalues(Iteration, TotalTime, TimeLeftInSeconds) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime)), (:(TimeLeftInSeconds),format(FormatExpr("{1:3.1f} [s]"), TimeLeftInSeconds))]
        

        if !SimLogger.ToConsole
            @timeit HourGlass "14 Next TimeStep" next!(
                SimMetaData.ProgressSpecification;
                showvalues = generate_showvalues(
                    SimMetaData.Iteration,
                    SimMetaData.TotalTime,
                    1e6,
                ),
            )
        end

        @inbounds while true

            @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoop(
                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                SimConstants, SimParticles, FullStencil, ParticleRanges,
                UniqueCells, CellDict, SortingScratchSpace, SimThreadedArrays,
                NeighborCellLists, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺,
                ∇Cᵢ, ∇◌rᵢ, MotionDefinition,
            )
            push!(SimMetaData.TimeSteps, SimMetaData.CurrentTimeStep)

            log_step!(SimMetaData, SimLogger)
    
            SimMetaData.OutputIterationCounter += 1

            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            @timeit SimMetaData.HourGlass "13 Save Particle Data"  begin
                output.save_particles(SimMetaData.OutputIterationCounter)
                output.save_grid(SimMetaData.OutputIterationCounter, UniqueCellsView, SimParticles)
            end
    
            if !SimLogger.ToConsole
                TimeLeftInSeconds = (SimMetaData.SimulationTime - SimMetaData.TotalTime) *
                                    (TimerOutputs.tottime(HourGlass) / 1e9 / SimMetaData.TotalTime)
                @timeit HourGlass "14 Next TimeStep" next!(
                    SimMetaData.ProgressSpecification;
                    showvalues = generate_showvalues(
                        SimMetaData.Iteration,
                        SimMetaData.TotalTime,
                        TimeLeftInSeconds,
                    ),
                )
            end
    
            if SimMetaData.TotalTime > SimMetaData.SimulationTime
                
                # At end of simulation
                @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

                if !SimLogger.ToConsole
                    finish!(SimMetaData.ProgressSpecification)
                end
                show(HourGlass,sortby=:name)
                show(HourGlass)

                AutoOpenParaview(SimMetaData, output.variable_names)

                # Time steps line plot
                UnicodeTimeStepsGraph = lineplot(1:length(SimMetaData.TimeSteps), SimMetaData.TimeSteps, title="Time Steps [s] as a function of iteration", name="Time Steps", xlabel="Iterations [-]", ylabel="Time Step Size [s]")

                finalize_log!(SimMetaData, SimLogger, HourGlass, UnicodeTimeStepsGraph)

                break
            end
        end
    end
    

end
