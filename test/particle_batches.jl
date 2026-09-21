using Test
using SPHExample

@testset "particle batches finish every index exactly once" begin
    # Cover empty input, both sides of a batch boundary, partial final batches,
    # and enough batches to exercise several workers.
    for Count in (0, 1, 63, 64, 65, 127, 128, 129, 513)
        Visits = [Threads.Atomic{Int}(0) for _ in 1:Count]
        Values = zeros(Int, Count)
        Result = SPHExample.SPHCellList.ForEachParticle!(1:Count) do Index
            Threads.atomic_add!(Visits[Index], 1)
            Values[Index] = Index^2
        end
        @test Result === nothing
        @test all(Visit -> Visit[] == 1, Visits)
        @test Values == (1:Count).^2
    end

    Failure = try
        SPHExample.SPHCellList.ForEachParticle!(1:129) do Index
            Index == 65 && error("intentional particle worker failure")
        end
        nothing
    catch Exception
        Exception
    end
    @test Failure isa Exception
    @test occursin("intentional particle worker failure", sprint(showerror, Failure))
end
