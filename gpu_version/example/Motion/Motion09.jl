include(joinpath(@__DIR__, "MotionExamples.jl"))
using .MotionExamples

# Run from the repository root:
# julia --project=gpu_version -t 2,0 gpu_version/example/Motion/Motion09.jl [output_dir] [duration] [dimensions]
if abspath(PROGRAM_FILE) == @__FILE__
    run_motion_example(9;
        save_location = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "output", "Motion", "Motion09"),
        simulation_time = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 4.0,
        dimensions = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2)
end
