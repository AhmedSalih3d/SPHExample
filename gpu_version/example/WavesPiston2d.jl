# DualSPHysics 5.4 examples/main/09_WavesPiston: second-order piston waves
# in a 0.66 m deep 2D flume, with a 1:10 beach or an absorbing end zone.
# Running or including this file starts the simulation. Requires an NVIDIA GPU.
#
# julia --project=gpu_version -t 2,0 gpu_version/example/WavesPiston2d.jl beach regular
# julia --project=gpu_version -t 2,0 gpu_version/example/WavesPiston2d.jl damping irregular
# For customized runs, include WavesPiston/WavesPistonSetup.jl instead.
include(joinpath(@__DIR__, "WavesPiston", "WavesPistonSetup.jl"))

length(ARGS) <= 2 || throw(ArgumentError("usage: WavesPiston2d.jl [beach|damping] [regular|irregular]"))
run_waves_piston(termination = isempty(ARGS) ? :damping : Symbol(ARGS[1]),
    waves = length(ARGS) < 2 ? :regular : Symbol(ARGS[2]))
