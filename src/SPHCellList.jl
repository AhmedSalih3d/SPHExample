module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!, RunSimulation

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
import LinearAlgebra: dot

using ..SimulationEquations
using ..SimulationGeometry
using ..AuxillaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..OpenExternalPrograms

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

    function ConstructStencil(v::Val{d}) where d
        n_ = CartesianIndices(ntuple(_->-1:1,v))
        half_length = length(n_) ÷ 2
        n  = n_[1:half_length]

        return n
    end

    """
    Extracts the cells for each particle based on their positions and the inverse cutoff value.

    # Arguments
    - `Particles`: The particles whose cells are to be extracted.
    - `::Val{InverseCutOff}`: The inverse cutoff value used for cell extraction.

    # Returns
    - `nothing`: This function modifies the `Particles` in place.
    """
    @inline function ExtractCells!(Particles, ::Val{InverseCutOff}) where InverseCutOff
        # Replace unsafe_trunc with trunc if this ever errors
        function map_floor(x)
            # This is different than just doing muladd(x,InverseCutOff,0.5) because it rounds towards zero.
            # Consider -1.7 + 0.5, this would give -1.2 and then trunced 1, but we want -2, therefore absolute addition before hand
            # We add 0.5 instead of 1, to ensure proper rounding behavior when restoring the sign for negative numbers.
            Int(sign(x)) * unsafe_trunc(Int, muladd(abs(x),InverseCutOff,0.5))
        end

        for i ∈ eachindex(Particles.Cells)
            t = map(map_floor, Tuple(Particles.Position[i]))
            Particles.Cells[i] = CartesianIndex(t)
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
    function UpdateNeighbors!(Particles, CutOff, SortingScratchSpace, ParticleRanges, UniqueCells)
        ExtractCells!(Particles, CutOff)

        sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)

        Cells = @views Particles.Cells
        @. ParticleRanges             = zero(eltype(ParticleRanges))
        IndexCounter                  = 1
        ParticleRanges[IndexCounter]  = 1
        UniqueCells[IndexCounter]     = Cells[1]

        for i in eachindex(Cells)[2:end]
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                IndexCounter                 += 1
                ParticleRanges[IndexCounter]  = i
                UniqueCells[IndexCounter]     = Cells[i]
            end
        end
        ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

        return IndexCounter 
    end

    
# Neither Polyester.@batch per core or thread is faster
###=== Function to process each cell and its neighbors
    function NeighborLoop!(SimMetaData, SimConstants, SimThreadedArrays, ParticleRanges, Stencil, Position, Density, Pressure, Velocity, MotionLimiter, UniqueCells, EnumeratedIndices)
        @sync tasks = map(EnumeratedIndices) do (ichunk, inds)
            @spawn for iter ∈ inds

                CellIndex = UniqueCells[iter]

                StartIndex = ParticleRanges[iter]
                EndIndex   = ParticleRanges[iter+1] - 1

                @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
                    @inline ComputeInteractions!(SimMetaData, SimConstants, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
                end

                @inbounds for S ∈ Stencil
                    SCellIndex = CellIndex + S

                    # Returns a range, x:x for exact match and x:(x-1) for no match
                    # utilizes that it is a sorted array and requires no isequal constructor,
                    # so I prefer this for now
                    NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

                    if length(NeighborCellIndex) != 0
                        StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                        EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                        @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
                            @inline ComputeInteractions!(SimMetaData, SimConstants, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
                        end
                    end
                end
            end
        end
        
        return nothing
    end

    function NeighborLoopMDBC!(SimMetaData, SimConstants, ParticleRanges, Stencil, Position, Density, UniqueCells, GhostPoints, GhostNormals,  ParticleType, bᵧ, Aᵧ, ::Val{InverseCutOff}) where InverseCutOff
        
        FullStencil = CartesianIndices(ntuple(_->-1:1, 2))

        function map_floor(x)
            # This is different than just doing muladd(x,InverseCutOff,0.5) because it rounds towards zero.
            # Consider -1.7 + 0.5, this would give -1.2 and then trunced 1, but we want -2, therefore absolute addition before hand
            # We add 0.5 instead of 1, to ensure proper rounding behavior when restoring the sign for negative numbers.
            Int(sign(x)) * unsafe_trunc(Int, muladd(abs(x),InverseCutOff,0.5))
        end

        # No @threads initially, just to check that algorithm used is correct
        # @threads for iter ∈ eachindex(GhostPoints)
        @threads for iter ∈ eachindex(GhostPoints)

            GhostPoint = GhostPoints[iter]
            
            if !iszero(GhostPoint)
                
                GhostCellIndex = CartesianIndex(map(map_floor, Tuple(GhostPoint)))

                # FullStencil includes Cell I - Some neighbours can potentially be in the neighbouring cells
                @inbounds for S ∈ FullStencil
                    SCellIndex = GhostCellIndex + S
                    # Returns a range, x:x for exact match and x:(x-1) for no match
                    # utilizes that it is a sorted array and requires no isequal constructor,
                    # so I prefer this for now
                    NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

                    if length(NeighborCellIndex) != 0
                        StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                        EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1
                        @inbounds for j = StartIndex_:EndIndex_
                            @inline ComputeInteractionsMDBC!(SimMetaData, SimConstants, Position, Density, ParticleType,  GhostPoints, bᵧ, Aᵧ, iter, j)
                        end
                        
                    end
                end

            end
        end

        return nothing
    end

    function ComputeInteractions!(SimMetaData, SimConstants, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
        @unpack FlagViscosityTreatment, FlagDensityDiffusion, FlagOutputKernelValues, FlagLinearizedDDT = SimMetaData
        @unpack ρ₀, h, h⁻¹, m₀, αD, α, γ, g, c₀, δᵩ, η², H², Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants

        Linear_ρ_factor = (1/(Cb*γ))*ρ₀

        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
        if  xᵢⱼ² <= H²
            #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
            dᵢⱼ  = sqrt(abs(xᵢⱼ²))

            # clamp seems faster than min, no util
            q         = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0) #min(dᵢⱼ * h⁻¹, 2.0) - 8% util no DDT
            invd²η²   =  1.0 / (dᵢⱼ*dᵢⱼ+η²)
            ∇ᵢWᵢⱼ     = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
            dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

            # Density diffusion
            if FlagDensityDiffusion
                if SimConstants.g == 0
                    ρᵢⱼᴴ  = 0.0
                    # ρⱼᵢᴴ  = 0.0
                else
                    Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
                    # Pⱼᵢᴴ  = -Pᵢⱼᴴ
                    
                    if FlagLinearizedDDT
                        ρᵢⱼᴴ  = Pᵢⱼᴴ * Linear_ρ_factor
                        # ρⱼᵢᴴ  = -ρᵢⱼᴴ
                    else
                        ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
                        # ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
                    end
                end

                ρⱼᵢ   = ρⱼ - ρᵢ

                Ψᵢⱼ   = 2( ρⱼᵢ - ρᵢⱼᴴ) * (-xᵢⱼ) * invd²η²
                #Ψⱼᵢ   = -Ψᵢⱼ #2(-ρⱼᵢ - ρⱼᵢᴴ) * ( xᵢⱼ) * invd²η²

                MLcond = MotionLimiter[i] * MotionLimiter[j]
                Dᵢ    =  δᵩ * h * c₀ * (m₀/ρⱼ) * dot(Ψᵢⱼ ,  ∇ᵢWᵢⱼ) * MLcond
                Dⱼ    =  -Dᵢ #δᵩ * h * c₀ * (m₀/ρᵢ) * dot(Ψⱼᵢ , -∇ᵢWᵢⱼ) * MLcond
            else
                Dᵢ  = 0.0
                Dⱼ  = 0.0
            end
            SimThreadedArrays.dρdtIThreaded[ichunk][i] += dρdt⁺ + Dᵢ
            SimThreadedArrays.dρdtIThreaded[ichunk][j] += dρdt⁻ + Dⱼ


            Pᵢ      =  Pressure[i]
            Pⱼ      =  Pressure[j]
            Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
            #dvdt⁻   = - dvdt⁺

            if FlagViscosityTreatment == :ArtificialViscosity
                ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
                cond      = dot(vᵢⱼ, xᵢⱼ)
                cond_bool = eltype(cond)(cond < 0.0)
                μᵢⱼ       = h*cond * invd²η²
                Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
                Πⱼ        = - Πᵢ
            else
                Πᵢ        = zero(xᵢⱼ)
                Πⱼ        = Πᵢ
            end
        
            if FlagViscosityTreatment == :Laminar || FlagViscosityTreatment == :LaminarSPS
                # 4 comes from 2 divided by 0.5 from average density
                # should divide by ρᵢ eq 6 DPC
                # ν₀∇²uᵢ = (1/ρᵢ) * ( (4 * m₀ * (ρᵢ * ν₀) * dot( xᵢⱼ, ∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) *  vᵢⱼ
                # ν₀∇²uⱼ = (1/ρⱼ) * ( (4 * m₀ * (ρⱼ * ν₀) * dot(-xᵢⱼ,-∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) * -vᵢⱼ
                visc_symmetric_term = (4 * m₀ * ν₀ * dot( xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²))
                # ν₀∇²uᵢ = (1/ρᵢ) * visc_symmetric_term *  vᵢⱼ * ρᵢ
                # ν₀∇²uⱼ = (1/ρⱼ) * visc_symmetric_term * -vᵢⱼ * ρⱼ
                ν₀∇²uᵢ =  visc_symmetric_term *  vᵢⱼ
                ν₀∇²uⱼ = -ν₀∇²uᵢ #visc_symmetric_term * -vᵢⱼ
            else
                ν₀∇²uᵢ = zero(xᵢⱼ)
                ν₀∇²uⱼ = ν₀∇²uᵢ
            end
        
            if FlagViscosityTreatment == :LaminarSPS 
                Iᴹ       = diagm(one.(xᵢⱼ))
                #julia> a .- a'
                # 3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
                # 0.0  0.0  0.0
                # 0.0  0.0  0.0
                # 0.0  0.0  0.0
                # Strain *rate* tensor is the gradient of velocity
                Sᵢ = ∇vᵢ =  (m₀/ρⱼ) * (vⱼ - vᵢ) * ∇ᵢWᵢⱼ'
                norm_Sᵢ  = sqrt(2 * sum(Sᵢ .^ 2))
                νtᵢ      = (SmagorinskyConstant * dx)^2 * norm_Sᵢ
                trace_Sᵢ = sum(diag(Sᵢ))
                τᶿᵢ      = 2*νtᵢ*ρᵢ * (Sᵢ - (1/3) * trace_Sᵢ * Iᴹ) - (2/3) * ρᵢ * BlinConstant * dx^2 * norm_Sᵢ^2 * Iᴹ
                Sⱼ = ∇vⱼ =  (m₀/ρᵢ) * (vᵢ - vⱼ) * -∇ᵢWᵢⱼ'
                norm_Sⱼ  = sqrt(2 * sum(Sⱼ .^ 2))
                νtⱼ      = (SmagorinskyConstant * dx)^2 * norm_Sⱼ
                trace_Sⱼ = sum(diag(Sⱼ))
                τᶿⱼ      = 2*νtⱼ*ρⱼ * (Sⱼ - (1/3) * trace_Sⱼ * Iᴹ) - (2/3) * ρⱼ * BlinConstant * dx^2 * norm_Sⱼ^2 * Iᴹ
        
                # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA TO REPRESENT IT
                dτdtᵢ = (m₀/(ρⱼ * ρᵢ)) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ 
                dτdtⱼ = dτdtᵢ #(m₀/(ρᵢ * ρⱼ)) * (τᶿᵢ + τᶿⱼ) * -∇ᵢWᵢⱼ 
            else
                dτdtᵢ  = zero(xᵢⱼ)
                dτdtⱼ  = dτdtᵢ
            end
        
            uₘ = dvdt⁺ + Πᵢ + ν₀∇²uᵢ + dτdtᵢ
            SimThreadedArrays.AccelerationThreaded[ichunk][i] += uₘ
            SimThreadedArrays.AccelerationThreaded[ichunk][j] -= uₘ #dvdt⁻ + Πⱼ + ν₀∇²uⱼ + dτdtⱼ

            
            if FlagOutputKernelValues
                Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
                SimThreadedArrays.KernelThreaded[ichunk][i]         += Wᵢⱼ
                SimThreadedArrays.KernelThreaded[ichunk][j]         += Wᵢⱼ
                SimThreadedArrays.KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
                SimThreadedArrays.KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
            end


            if SimMetaData.FlagShifting
                Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
        
                MLcond = MotionLimiter[i] * MotionLimiter[j]

                SimThreadedArrays.∇CᵢThreaded[ichunk][i]   += (m₀/ρᵢ) *  ∇ᵢWᵢⱼ
                SimThreadedArrays.∇CᵢThreaded[ichunk][j]   += (m₀/ρⱼ) * -∇ᵢWᵢⱼ
        
                # Switch signs compared to DSPH, else free surface detection does not make sense
                # Agrees, https://arxiv.org/abs/2110.10076, it should have been r_ji
                SimThreadedArrays.∇◌rᵢThreaded[ichunk][i]  += (m₀/ρⱼ) * dot(-xᵢⱼ , ∇ᵢWᵢⱼ)  * MLcond
                SimThreadedArrays.∇◌rᵢThreaded[ichunk][j]  += (m₀/ρᵢ) * dot( xᵢⱼ ,-∇ᵢWᵢⱼ)  * MLcond
            end
        end

        return nothing
    end


    @generated function UpdateAMatrix!(::Dimensions, ::FloatType) where {Dimensions, FloatType}
        DimensionsPlus = Dimensions + 1
        quote
            Aᵧ[i] += SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus*DimensionsPlus}(
                #     # First row
                    VⱼWᵢⱼ, (Vⱼ * ∇ᵢWᵢⱼ)...,
                #     # # Second row
                    # xⱼᵢ[1] * VⱼWᵢⱼ, (xⱼᵢ[1] * Vⱼ * ∇ᵢWᵢⱼ)...,
                #     # # # Third row
                    # xⱼᵢ[2] * VⱼWᵢⱼ, (xⱼᵢ[2] * Vⱼ * ∇ᵢWᵢⱼ)...,
                    
                    Base.Cartesian.@nexprs $Dimensions d -> (xⱼᵢ[d] * VⱼWᵢⱼ, (xⱼᵢ[d] * Vⱼ * ∇ᵢWᵢⱼ))...
                )
        end
    end

    function ComputeInteractionsMDBC!(SimMetaData::SimulationMetaData{Dimensions, FloatType}, SimConstants, Position, Density, ParticleType, GhostPoints, bᵧ, Aᵧ, i, j) where {Dimensions, FloatType}
        @unpack FlagViscosityTreatment, FlagDensityDiffusion, FlagOutputKernelValues, FlagLinearizedDDT = SimMetaData
        @unpack ρ₀, h, h⁻¹, m₀, αD, α, γ, g, c₀, δᵩ, η², H², Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    
        # ᵢ is ghost node! j is fluid node

        
        if ParticleType[j] == Fluid

            DimensionsPlus = Dimensions + 1
    
            xᵢⱼ  = GhostPoints[i] - Position[j]

            xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
            if xᵢⱼ² <= H²
                dᵢⱼ = sqrt(abs(xᵢⱼ²))
                q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
        
                ρⱼ = Density[j]

        
                Wᵢⱼ = @fastpow αD * (1 - q / 2)^4 * (2 * q + 1)


                ∇ᵢWᵢⱼ = @fastpow (αD * 5 * (q - 2)^3 * q / (8h * (q * h + η²))) * -xᵢⱼ

                Vⱼ = m₀ / ρⱼ
        
                VⱼWᵢⱼ = Vⱼ * Wᵢⱼ
        
                bᵧ_ = SVector{DimensionsPlus, FloatType}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)
                bᵧ[i] += bᵧ_

                # Filling the Aᵧ matrix is done in column-major order
                xⱼᵢ = -xᵢⱼ

                if Dimensions == 2
                    Aᵧ[i] += SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus*DimensionsPlus}(
                        # First row
                        VⱼWᵢⱼ, (Vⱼ * ∇ᵢWᵢⱼ)...,
                        # Second row
                        xⱼᵢ[1] * VⱼWᵢⱼ, (xⱼᵢ[1] * Vⱼ * ∇ᵢWᵢⱼ)...,
                        # Third row
                        xⱼᵢ[2] * VⱼWᵢⱼ, (xⱼᵢ[2] * Vⱼ * ∇ᵢWᵢⱼ)...,
                    )
                elseif Dimensions == 3
                    Aᵧ[i] += SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus*DimensionsPlus}(
                        # First row
                        VⱼWᵢⱼ, (Vⱼ * ∇ᵢWᵢⱼ)...,
                        # Second row
                        xⱼᵢ[1] * VⱼWᵢⱼ, (xⱼᵢ[1] * Vⱼ * ∇ᵢWᵢⱼ)...,
                        # Third row
                        xⱼᵢ[2] * VⱼWᵢⱼ, (xⱼᵢ[2] * Vⱼ * ∇ᵢWᵢⱼ)...,
                        # Fourth row
                        xⱼᵢ[3] * VⱼWᵢⱼ, (xⱼᵢ[3] * Vⱼ * ∇ᵢWᵢⱼ)...,
                    )
                end
            end
        end
        
    
        return nothing
    end

    function reduce_sum!(target_array, arrays)
        n = length(target_array)
        num_threads = nthreads()
        chunk_size = ceil(Int, n / num_threads)
        @inbounds for t in 1:num_threads
            local start_idx = 1 + (t-1) * chunk_size
            local end_idx = min(t * chunk_size, n)
            for j in eachindex(arrays)
                local array = arrays[j]  # Access array only once per thread
                @simd for i in start_idx:end_idx
                    @inbounds target_array[i] += array[i]
                end
            end
        end
    end

    function ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ, bᵧ, Aᵧ)
        
        ResetArrays!(dρdtI, Acceleration)

        if SimMetaData.FlagOutputKernelValues
            ResetArrays!(Kernel, KernelGradient)
        end

        if SimMetaData.FlagShifting
            ResetArrays!(∇Cᵢ, ∇◌rᵢ)
        end

        if SimMetaData.FlagMDBCSimple 
            ResetArrays!(bᵧ, Aᵧ)
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
        @inbounds for i in eachindex(Position)
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
    
    function HalfTimeStep(::SimulationMetaData{Dimensions, FloatType}, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, GhostPoints, GhostNormals, bᵧ, Aᵧ, dt₂) where {Dimensions, FloatType}
        Position       = SimParticles.Position
        Density        = SimParticles.Density
        Velocity       = SimParticles.Velocity
        Acceleration   = SimParticles.Acceleration
        GravityFactor  = SimParticles.GravityFactor
        MotionLimiter  = SimParticles.MotionLimiter

        ρ₀ = SimConstants.ρ₀

        #https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
        @inbounds @threads for i in eachindex(Position)
            A = Aᵧ[i]

            if abs(det(A)) >= 1e-3
                    GhostPointDensity = A \ bᵧ[i]
                    v1 = first(GhostPointDensity) + dot(SimParticles.Position[i] - GhostPoints[i], GhostPointDensity[2:end])
                    Density[i] = isnan(v1) ? ρ₀ : v1
            elseif first(A) > 0.0
                    v = first(bᵧ[i]) / first(A)
                    Density[i] = isnan(v) ? ρ₀ : v
            end
        end

        @inbounds for i in eachindex(Position)
            Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
            Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
            ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
        end


        return nothing
    end

    function FullTimeStep(SimMetaData, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
        Position       = SimParticles.Position
        Velocity       = SimParticles.Velocity
        Acceleration   = SimParticles.Acceleration
        GravityFactor  = SimParticles.GravityFactor
        MotionLimiter  = SimParticles.MotionLimiter
  
        if !SimMetaData.FlagShifting
            @inbounds for i in eachindex(Position)
                Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
                Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
                Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
            end
        else
            A     = 2# Value between 1 to 6 advised
            A_FST = 0; # zero for internal flows
            A_FSM = length(first(Position)); #2d, 3d val different
            @inbounds for i in eachindex(Position)
                Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
                Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        
                A_FSC                  = (∇◌rᵢ[i] - A_FST)/(A_FSM - A_FST)
                if A_FSC < 0
                    δxᵢ = zero(eltype(Position))
                else
                    δxᵢ = -A_FSC * A * SimConstants.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
                end
        
                Position[i]           += (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt + δxᵢ) * MotionLimiter[i]
            end
        end

        return nothing
    end

    function UpdateMetaData!(SimMetaData, dt)
        SimMetaData.Iteration      += 1
        SimMetaData.CurrentTimeStep = dt
        SimMetaData.TotalTime      += dt

        return nothing
    end
    
    @inbounds function SimulationLoop(SimMetaData, SimConstants, SimParticles, Stencil,  ParticleRanges, UniqueCells, SortingScratchSpace, SimThreadedArrays, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ,  bᵧ, Aᵧ, MotionDefinition, InverseCutOff)
        Position       = SimParticles.Position
        Density        = SimParticles.Density
        Pressure       = SimParticles.Pressure
        Velocity       = SimParticles.Velocity
        Acceleration   = SimParticles.Acceleration
        GravityFactor  = SimParticles.GravityFactor
        MotionLimiter  = SimParticles.MotionLimiter
        ParticleType   = SimParticles.Type
        ParticleMarker = SimParticles.GroupMarker
        Kernel         = SimParticles.Kernel
        KernelGradient = SimParticles.KernelGradient
        GhostPoints    = SimParticles.GhostPoints
        GhostNormals   = SimParticles.GhostNormals

        ###

        
        while SimMetaData.TotalTime <= SimMetaData.OutputEach * SimMetaData.OutputIterationCounter

            @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants)
            dt₂ = dt * 0.5

            @timeit SimMetaData.HourGlass "02 Calculate IndexCounter"  begin
                # Note: If particles are not inside of the neighbor list visualiation, try setting this if statement to always true, since UniqueCells will be updated always then
                # In theory, the maximal speed is the speed of sound, this should give a safe guard
                # any ensure it is always updated in a reasonable manner. This only works well, assuming that
                # c₀ >= maximum(norm.(Velocity))
                # Remove if statement logic if you want to update each iteration
                if mod(SimMetaData.Iteration, ceil(Int, SimConstants.H / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
                    SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, InverseCutOff, SortingScratchSpace,  ParticleRanges, UniqueCells)
                end

                UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                EnumeratedIndices = enumerate(index_chunks(UniqueCellsView; n=nthreads()))
            end


            @timeit SimMetaData.HourGlass "Motion"                               ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
        
            if !SimMetaData.FlagSingleStepTimeStepping
                ###=== First step of resetting arrays
                @timeit SimMetaData.HourGlass "ResetArrays"                          ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ, bᵧ, Aᵧ)
                ###===
            
                @timeit SimMetaData.HourGlass "03 Pressure"                          Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
                if SimMetaData.FlagMDBCSimple
                    @timeit SimMetaData.HourGlass "04 First NeighborLoopMDBC"        NeighborLoopMDBC!(SimMetaData, SimConstants, ParticleRanges, Stencil, Position, Density, UniqueCellsView, GhostPoints, GhostNormals, ParticleType, bᵧ, Aᵧ, InverseCutOff)
                end
                @timeit SimMetaData.HourGlass "04 First NeighborLoop"                NeighborLoop!(SimMetaData, SimConstants, SimThreadedArrays, ParticleRanges, Stencil, Position, Density, Pressure, Velocity, MotionLimiter, UniqueCellsView, EnumeratedIndices)
                @timeit SimMetaData.HourGlass "Reduction"                            ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
            end

            @timeit SimMetaData.HourGlass "05 Update To Half TimeStep"           HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, GhostPoints, GhostNormals, bᵧ, Aᵧ, dt₂)

            @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"       LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)
        
            ###=== Second step of resetting arrays
            @timeit SimMetaData.HourGlass "ResetArrays"                          ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ, bᵧ, Aᵧ)
            ###===

            @timeit SimMetaData.HourGlass "Motion"                               ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
        
            @timeit SimMetaData.HourGlass "03 Pressure"                          Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
            @timeit SimMetaData.HourGlass "08 Second NeighborLoop"               NeighborLoop!(SimMetaData, SimConstants, SimThreadedArrays, ParticleRanges, Stencil, Positionₙ⁺, ρₙ⁺, Pressure, Velocityₙ⁺, MotionLimiter, UniqueCellsView, EnumeratedIndices)
            @timeit SimMetaData.HourGlass "Reduction"                            ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)

        
            @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary"      LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)
        
            @timeit SimMetaData.HourGlass "10 Final Density"                     DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)
        
            @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"          FullTimeStep(SimMetaData, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
        
            @timeit SimMetaData.HourGlass "12 Update MetaData"                   UpdateMetaData!(SimMetaData, dt)

        end
        
        return nothing
    end
    
    ###===
    function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}}, #Don't further specify type for now
        SimMetaData::SimulationMetaData{Dimensions, FloatType},
        SimConstants::SimulationConstants,
        SimLogger::SimulationLogger,
        SimParticles::StructArray,
        path_mdbc::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType}

        # Unpack the relevant simulation meta data
        @unpack HourGlass = SimMetaData;

        # Vector of time steps
        TimeSteps = Vector{FloatType}()
        
        dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimParticles.Position)

        bᵧ = zeros(SVector{Dimensions + 1,FloatType}, length(SimParticles.Position))
        DimensionsPlus = Dimensions + 1
        Aᵧ = zeros(SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus * DimensionsPlus}, length(SimParticles.Position))

        if !isnothing(path_mdbc)
            _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(Dimensions), FloatType, path_mdbc)

            #TODO: In the future decide on one of the two in shaa Allah
            for gi ∈ eachindex(GhostPoints)
                SimParticles.GhostPoints[gi]  = GhostPoints[gi]
                SimParticles.GhostNormals[gi] = GhostNormals[gi]
            end
        end

        if !SimMetaData.FlagShifting
            resize!(∇Cᵢ , 0)
            resize!(∇◌rᵢ, 0)
        end

        if SimMetaData.FlagLog
            InitializeLogger(SimLogger,SimConstants,SimMetaData, SimGeometry, SimParticles)
        end
        
        NumberOfPoints = length(SimParticles)::Int
        Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    
        SimThreadedArrays = AllocateThreadedArrays(SimMetaData, SimParticles, dρdtI, ∇Cᵢ, ∇◌rᵢ)
    
        # Produce sorting related variables
        ParticleRanges         = zeros(Int, NumberOfPoints + 1)
        UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
        Stencil                = ConstructStencil(Val(Dimensions))
        _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)

        output = SetupVTKOutput(SimMetaData, SimParticles, SimConstants, Dimensions)

        # Save initial state
        SimMetaData.OutputIterationCounter = 1
        output.save_particles(SimMetaData.OutputIterationCounter)
        output.save_grid(SimMetaData.OutputIterationCounter, UniqueCells)

        InverseCutOff = Val(1/(SimConstants.H))

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
    
        @inbounds while true
    
            @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoop(SimMetaData, SimConstants, SimParticles, Stencil, ParticleRanges, UniqueCells, SortingScratchSpace, SimThreadedArrays, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ, bᵧ, Aᵧ, MotionDefinition, InverseCutOff)
            push!(TimeSteps, SimMetaData.CurrentTimeStep)
    
            SimMetaData.OutputIterationCounter += 1

            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            @sync Threads.@spawn begin
                @timeit SimMetaData.HourGlass "13A Save Particle Data" output.save_particles(SimMetaData.OutputIterationCounter)
                @timeit SimMetaData.HourGlass "13A Save CellGrid Data" output.save_grid(SimMetaData.OutputIterationCounter, UniqueCellsView)
            end
    
            if SimMetaData.FlagLog
                LogStep(SimLogger, SimMetaData, HourGlass)
                SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
            end
    
            TimeLeftInSeconds = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime)
            @timeit HourGlass "14 Next TimeStep" next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime, TimeLeftInSeconds))
    
            if SimMetaData.TotalTime > SimMetaData.SimulationTime
                
                # At end of simulation
                @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

                finish!(SimMetaData.ProgressSpecification)
                show(HourGlass,sortby=:name)
                show(HourGlass)

                AutoOpenParaview(SimMetaData, output.variable_names)

                # Time steps line plot
                UnicodeTimeStepsGraph = lineplot(1:length(TimeSteps), TimeSteps, title="Time Steps [s] as a function of iteration", name="Time Steps", xlabel="Iterations [-]", ylabel="Time Step Size [s]")

                if SimMetaData.FlagLog
                    LogFinal(SimLogger, HourGlass)
             
                    with_logger(SimLogger.Logger) do
                        @info ""
                        show(SimLogger.LoggerIo, UnicodeTimeStepsGraph)
                    end

                    close(SimLogger.LoggerIo)
                    AutoOpenLogFile(SimLogger, SimMetaData)
                end

                break
            end
        end
    end
    

end
