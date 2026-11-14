# Full investigation first, then warm, alternating whole-solver timing pairs.
include(joinpath(@__DIR__,"dual_compare.jl"))
Test06DualComparison.study()
Test06DualComparison.timing_study()
