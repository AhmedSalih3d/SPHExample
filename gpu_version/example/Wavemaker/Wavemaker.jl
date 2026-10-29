# DualSPHysics main/06_Wavemaker: sinusoidal piston and a sloping beach.
# Running or including this file starts the case. Requires an NVIDIA GPU.
# From the repository root:
#   julia --project=gpu_version -t 2,0 gpu_version/example/Wavemaker/Wavemaker.jl [output_dir] [dx] [duration] [images|queries]
# From gpu_version/:
#   julia --project=. -t 2,0 example/Wavemaker/Wavemaker.jl
# For a customized run, include WavemakerSetup.jl and call run_wavemaker(...).
include(joinpath(@__DIR__, "WavemakerSetup.jl"))

# Julia/VS Code includes use the case defaults, independent of the arguments
# used to start the surrounding Julia process.
let args = abspath(PROGRAM_FILE) == abspath(@__FILE__) ? ARGS : String[]
    length(args) <= 4 || throw(ArgumentError("usage: Wavemaker.jl [output_dir] [dx] [duration] [images|queries]"))
    backend = length(args) >= 4 ? args[4] : "images"
    backend in ("images", "queries") || throw(ArgumentError("backend must be images or queries"))
    run_wavemaker(;
        save_location = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "..", "..", "output", "wavemaker_periodic_copies"),
        dx = length(args) >= 2 ? parse(Float64, args[2]) : 0.02,
        simulation_time = length(args) >= 3 ? parse(Float64, args[3]) : 10.0,
        periodic_images = backend == "images",
    )
end
