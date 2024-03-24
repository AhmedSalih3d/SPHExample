module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!

using Parameters, FastPow, Distances
import LinearAlgebra: dot

include("SimulationEquations.jl"); using .SimulationEquations

include("AuxillaryFunctions.jl"); using .AuxillaryFunctions

    function ConstructStencil(v::Val{d}) where d
        n_ = CartesianIndices(ntuple(_->-1:1,v))
        half_length = length(n_) ÷ 2
        n  = n_[1:half_length]

        return n
    end

    function ExtractCells!(Cells, Points, CutOff)
        for i ∈ eachindex(Cells)
            Cells[i]  =  CartesianIndex(@. Int(fld(Points[i], CutOff)) ...)
            Cells[i] +=  2 * one(Cells[i])  # + CartesianIndex(1,1) + CartesianIndex(1,1) #+ ZeroOffset + HalfPad
        end
        return nothing
    end

    ###=== Function to update ordering
    function UpdateNeighbors!(Cells, CutOff, SortedIndices, SortingScratchSpace, Position, Density, Acceleration, Velocity, GravityFactor, MotionLimiter, ParticleRanges, UniqueCells)
        ExtractCells!(Cells,Position,CutOff)

        # First call allocates, which is why TimerOutputs shows allocs - it should be alloc free otherwise
        sortperm!(SortedIndices,Cells; scratch=SortingScratchSpace)

        RearrangeVector!(Cells         , SortedIndices)
        RearrangeVector!(Position      , SortedIndices)
        RearrangeVector!(Density       , SortedIndices)
        RearrangeVector!(Acceleration  , SortedIndices)
        RearrangeVector!(Velocity      , SortedIndices)    
        RearrangeVector!(GravityFactor , SortedIndices)    
        RearrangeVector!(MotionLimiter , SortedIndices)    

        IndexCounter = 1
        for i in 2:length(Cells)
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                ParticleRanges[IndexCounter] = i
                UniqueCells[IndexCounter]    = Cells[i]
                IndexCounter                += 1
            end
        end

        return IndexCounter
    end

    function ComputeInteractions!(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, BoolDDT)
        xᵢⱼ² = evaluate(SqEuclidean(), Position[i], Position[j])
        if  xᵢⱼ² <= H²
            xᵢⱼ  = Position[i] - Position[j]
            
            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            # Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
            invd²η² = inv(dᵢⱼ*dᵢⱼ+η²)
            ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ
            symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ) # = dot(vᵢⱼ , -∇ᵢWᵢⱼ)
            dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  symmetric_term
            dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  symmetric_term
            # Density diffusion
            if BoolDDT
                Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
                ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
                Pⱼᵢᴴ  = -Pᵢⱼᴴ
                ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
            
                ρⱼᵢ   = ρⱼ - ρᵢ
                MLcond = MotionLimiter[i] * MotionLimiter[j]
                ddt_symmetric_term =  δᵩ * h * c₀ * 2 * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond #  dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) =  dot( xᵢⱼ, -∇ᵢWᵢⱼ)
                Dᵢ  = ddt_symmetric_term * (m₀/ρⱼ) * ( ρⱼᵢ - ρᵢⱼᴴ)
                Dⱼ  = ddt_symmetric_term * (m₀/ρᵢ) * (-ρⱼᵢ - ρⱼᵢᴴ)
                dρdtI[i] += dρdt⁺ + Dᵢ
                dρdtI[j] += dρdt⁻ + Dⱼ
            else
                dρdtI[i] += dρdt⁺
                dρdtI[j] += dρdt⁻
            end
            Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
            Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
            Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ
            dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
            dvdt⁻ = - dvdt⁺ + Πⱼ
            dvdtI[i] +=  dvdt⁺
            dvdtI[j] +=  dvdt⁻
            # Kernel[i] += Wᵢⱼ
            # Kernel[j] += Wᵢⱼ
            # KernelGradient[i] +=  ∇ᵢWᵢⱼ
            # KernelGradient[j] += -∇ᵢWᵢⱼ
        end
    
        return nothing
    end

    
    function SimStepLocalCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, MotionLimiter, ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, BoolDDT)

        @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
            @inline ComputeInteractions!(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, BoolDDT)
        end

        return nothing
    end


    function SimStepNeighborCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, StartIndex_, EndIndex_, MotionLimiter, ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, BoolDDT)
        @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
            @inline ComputeInteractions!(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, BoolDDT)
        end
        return nothing
    end


###=== Function to process each cell and its neighbors
    function NeighborLoop!(SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI,  MotionLimiter, UniqueCells, IndexCounter, BoolDDT)
        @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H², Cb⁻¹ = SimConstants

        UniqueCells = view(UniqueCells, 1:IndexCounter)
        @inbounds for iter ∈ eachindex(UniqueCells)
            CellIndex = UniqueCells[iter]

            StartIndex = ParticleRanges[iter] 
            EndIndex   = ParticleRanges[iter+1] - 1

            @inline SimStepLocalCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, MotionLimiter, ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, BoolDDT)

            @inbounds for S ∈ Stencil
                SCellIndex = CellIndex + S

                # Returns a range, x:x for exact match and x:(x-1) for no match
                # utilizes that it is a sorted array and requires no isequal constructor,
                # so I prefer this for now
                NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

                if length(NeighborCellIndex) != 0
                    StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                    EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                    @inline SimStepNeighborCell(Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, StartIndex_, EndIndex_, MotionLimiter, ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, BoolDDT)
                end
            end
        end

        return nothing
    end

end
