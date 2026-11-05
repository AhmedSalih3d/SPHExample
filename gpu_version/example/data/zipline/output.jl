# CAD is used only for display. Contact search uses the simulated particles.
using SPHExampleGPU, StaticArrays, CSV

function zipline_stl_surface(path, displacement)
    points = SVector{3,Float64}[]
    faces = Vector{Int}[]
    open(path, "r") do io
        skip(io, 80)
        count = ltoh(read(io, UInt32))
        filesize(path) == 84 + 50count || error("expected the bundled binary STL: $path")
        for _ in 1:count
            skip(io, 12) # stored face normal
            first = length(points) + 1
            for _ in 1:3
                p = SVector(ntuple(_ -> Float64(reinterpret(Float32, ltoh(read(io, UInt32)))), 3))
                push!(points, p + displacement)
            end
            push!(faces, collect(first:first+2))
            skip(io, 2)
        end
    end
    return PolygonSurface(points, faces)
end

function save_zipline_surfaces(directory, geometry)
    names = (:Head, :Body, :ArmLeft, :ArmRight, :ForearmLeft, :ForearmRight, :LegLeft, :LegRight, :CalfLeft, :CalfRight)
    files = ("head", "body", "arm_left", "arm_right", "forearm_left", "forearm_right", "leg_left", "leg_right", "calf_left", "calf_right")
    moves = ((0,0,0.03), (0,0,0), (0,-0.03,0), (0,0.03,0), (0,-0.03,0), (0,0.03,0), (0,0,-0.03), (0,0,-0.03), (0,0,-0.06), (0,0,-0.06))
    surfaces = NamedTuple{names}(Tuple(zipline_stl_surface(joinpath(@__DIR__, "ragdoll", name * ".stl"), SVector(move)) for (name,move) in zip(files,moves)))
    rows = collect(CSV.File(joinpath(directory, "ZipLine3D_Floating.csv")))
    times = sort!(unique(row.Time for row in rows))
    length(times) >= 2 || return nothing
    bodies = filter(g -> g.Type == Floating, geometry)
    reference = [sum(SVector{3,Float64}.(g.Particles.Position)) / length(g.Particles) for g in bodies]
    states = Dict((row.Body, row.Time) => row for row in rows)
    motions = NamedTuple{names}(Tuple(let body = b
        (point, time) -> begin
            row = states[(body, time)]
            center = SVector(ntuple(d -> row[Symbol("Center:", d - 1)], 3))
            q = SVector(ntuple(d -> row[Symbol("Orientation:", d - 1)], 4))
            center + SPHExampleGPU.GPUFloating.rotate_vector(q, point - reference[body])
        end
    end for b in eachindex(names)))
    filepath = SavePolygonMotionSequence(joinpath(directory, "ZipLine3D.vtkhdf"), surfaces;
        motions, times, block_name = "RagdollSurfaces")
    state_path = joinpath(directory, "ZipLine3D_SingleVTKHDFStateFile.py")
    open(state_path, "a") do io
        write(io, """

# Display CAD surfaces following the simulated native body poses.
Ragdoll = ExtractBlock(registrationName='Ragdoll surfaces', Input=Simulation_vtkhdf)
Ragdoll.Selectors = ['/Root/RagdollSurfaces']
RagdollDisplay = Show(Ragdoll, renderView1, 'GeometryRepresentation')
RagdollDisplay.SetRepresentationType('Surface')
ColorBy(RagdollDisplay, ('CELLS', 'Region'))
RagdollDisplay.RescaleTransferFunctionToDataRange(True, False)
Simulation_vtkhdfDisplay.ColorArrayName = ['POINTS', '']
Simulation_vtkhdfDisplay.DiffuseColor = [0.55, 0.55, 0.6]
Simulation_vtkhdfDisplay.Opacity = 0.35
SetActiveSource(Ragdoll)
RenderAllViews()
""")
    end
    return filepath
end

function open_zipline_output(metadata)
    metadata.VisualizeInParaview || return nothing
    executable = Sys.which("paraview")
    if executable === nothing
        @warn "ParaView was not found on PATH; open ZipLine3D_SingleVTKHDFStateFile.py manually."
        return nothing
    end
    state = joinpath(metadata.SaveLocation, "ZipLine3D_SingleVTKHDFStateFile.py")
    run(`$executable --state=$state`; wait = false)
    return nothing
end
