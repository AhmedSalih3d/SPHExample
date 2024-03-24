module SPHCellList

export ConstructStencil, ExtractCells!

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

end
