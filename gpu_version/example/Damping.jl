# DualSPHysics 5.4 examples/main/20_Damping.
# Select one of the planar, angled, box, or cylindrical damping cases.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/Damping.jl plane_mdbc
#     julia --project=gpu_version -t 2,0 gpu_version/example/Damping.jl plane
#     julia --project=gpu_version -t 2,0 gpu_version/example/Damping.jl angle
#     julia --project=gpu_version -t 2,0 gpu_version/example/Damping.jl box
#     julia --project=gpu_version -t 2,0 gpu_version/example/Damping.jl cylinder
include(joinpath(@__DIR__, "Damping", "DampingSetup.jl"))

length(ARGS) <= 1 ||
    throw(ArgumentError("usage: Damping.jl [plane_mdbc|plane|angle|box|cylinder]"))
variant = isempty(ARGS) ? :plane_mdbc : Symbol(ARGS[1])
run_damping_case20(; variant)
