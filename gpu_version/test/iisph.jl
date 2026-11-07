# Focused IISPH suite, runnable independently of the full WCSPH/CPU comparisons.
include(joinpath(@__DIR__,"iisph_operator.jl"))
include(joinpath(@__DIR__,"iisph_hydrostatic.jl"))
include(joinpath(@__DIR__,"iisph_dam_break.jl"))
