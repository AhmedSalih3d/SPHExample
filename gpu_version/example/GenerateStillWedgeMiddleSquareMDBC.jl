# Export the existing middle-square still-wedge particles for ParaView.
#
# Run from the repository root with:
#     julia --project=gpu_version \
#         gpu_version/example/GenerateStillWedgeMiddleSquareMDBC.jl [output_dir] [dx]
using SPHExampleGPU
using CSV
using StaticArrays

const MIDDLE_SQUARE_INPUT_DIR = normpath(joinpath(
    @__DIR__, "..", "input", "still_wedge_middle_square_mdbc"))

function read_middle_square_region(path)
    positions = SVector{2, Float64}[]
    density = Float64[]
    pressure = Float64[]

    for row in CSV.File(path)
        push!(positions, SVector(
            Float64(row[Symbol("Points:0")]),
            Float64(row[Symbol("Points:2")]),
        ))
        push!(density, Float64(row.Rhop))
        push!(pressure, Float64(row.Press))
    end

    isempty(positions) && throw(ArgumentError("No particles found in $path"))
    return (; positions, density, pressure)
end

"""
    generate_still_wedge_middle_square_geometry(output_dir;
        input_dir = MIDDLE_SQUARE_INPUT_DIR, dx = 0.02)

Export the case's existing `Bound` and `Fluid` CSV particles to a VTKHDF
point-cloud file for ParaView. The input CSVs remain unchanged. Returns the
generated file path.
"""
function generate_still_wedge_middle_square_geometry(output_dir;
        input_dir = MIDDLE_SQUARE_INPUT_DIR, dx = 0.02)
    prefix = "StillWedge_MiddleSquare_Dp$(dx)"
    bound = read_middle_square_region(
        joinpath(input_dir, "$(prefix)_Bound.csv"))
    fluid = read_middle_square_region(
        joinpath(input_dir, "$(prefix)_Fluid.csv"))

    positions = to_3d(vcat(bound.positions, fluid.positions))
    density = vcat(bound.density, fluid.density)
    pressure = vcat(bound.pressure, fluid.pressure)
    types = vcat(fill(Int8(Fixed), length(bound.positions)),
                 fill(Int8(Fluid), length(fluid.positions)))
    markers = vcat(fill(1, length(bound.positions)),
                   fill(2, length(fluid.positions)))

    mkpath(output_dir)
    output_path = joinpath(output_dir, "$(prefix)_Particles.vtkhdf")
    SaveVTKHDF(output_path, positions,
               ["Density", "Pressure", "Type", "GroupMarker"],
               density, pressure, types, markers)
    return output_path
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    output_dir = isempty(ARGS) ? MIDDLE_SQUARE_INPUT_DIR : ARGS[1]
    dx = length(ARGS) < 2 ? 0.02 : parse(Float64, ARGS[2])
    output_path = generate_still_wedge_middle_square_geometry(output_dir; dx)
    @info "Saved middle-square particle geometry" output_path
end
