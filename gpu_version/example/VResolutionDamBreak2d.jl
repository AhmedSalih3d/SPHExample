# Native CUDA variable-resolution 2D dam break, based on DualSPHysics 5.4
# examples/vresolution/01_DamBreak2D. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/VResolutionDamBreak2d.jl
#
# For customized runs, include VResolutionDamBreak2d/Setup.jl instead.
include(joinpath(@__DIR__, "VResolutionDamBreak2d", "Setup.jl"))

isempty(ARGS) || throw(ArgumentError("this example takes no arguments; include VResolutionDamBreak2d/Setup.jl for customized runs"))
run_vresolution_dambreak()
