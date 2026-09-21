using Test
using SPHExample
using StaticArrays
using StructArrays

@testset "known neighbor-reuse coverage limitation" begin
    Kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h=0.5)
    Offset = Kernel.H / 1024
    Initial = [SVector(0.5 - Offset, 0.0), SVector(1.5 + Offset, 0.0)]
    Final = [SVector(0.5 + Offset, 0.0), SVector(1.5 - Offset, 0.0)]
    Half = (Initial .+ Final) ./ 2
    Particles = StructArray((Position=copy(Initial), Cells=fill(CartesianIndex(0, 0), 2)))
    Ranges = zeros(Int, 4)
    Cells = zeros(CartesianIndex{2}, 3)
    Indices = zeros(Int, 2)
    Count = UpdateNeighbors!(Particles, Kernel.H⁻¹, NeighborSortScratch(2), Ranges, Cells, Indices)
    Lists = PackedNeighborCellLists(3)
    BuildNeighborCellLists!(Lists, ConstructStencil(Val(2)), view(Cells, 1:Count), Ranges)

    @test sum(abs2, Initial[2] - Initial[1]) > Kernel.H²
    @test sum(abs2, Final[2] - Final[1]) < Kernel.H²
    # With CFL=0.2, dt=CFL*h/c₀, the speed is still below c₀/10.
    @test (2 * Offset) / (0.2 * Kernel.h) < 0.1
    MotionBudget = SPHExample.SPHNeighborList.UpdateΔx!(0.0, Half, Final)
    # Mirrors SimulationLoop's present decision. A skin-based replacement must
    # cover this pair or rebuild before evaluating it. Keep the known failure
    # visible until that policy is replaced and tested at every evaluation stage.
    @test_broken MotionBudget >= Kernel.h || Indices[2] in Lists[Indices[1]]
end
