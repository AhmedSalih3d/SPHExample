# CAD is used only for display. Contact search uses the simulated particles.
using SPHExampleGPU, StaticArrays, CSV

function zipline_reference_surface(group, points, cells)
    faces = [[row[Symbol("v", i)] for i in 1:row.n] for row in cells if row.group == group && row.n >= 3]
    indices = sort!(unique(vcat(faces...)))
    mapping = Dict(index => local_index for (local_index,index) in enumerate(indices))
    return PolygonSurface(points[indices], [[mapping[index] for index in face] for face in faces])
end

function save_zipline_surfaces(directory, geometry; write_geometry = true)
    names = (:Head, :Body, :ArmLeft, :ArmRight, :ForearmLeft, :ForearmRight, :LegLeft, :LegRight, :CalfLeft, :CalfRight)
    # GenCase's actual surface points and indexed cells, including its draw
    # translations. Shared vertices preserve the reference mesh's shading.
    points = [SVector(row.x, row.y, row.z) for row in CSV.File(joinpath(@__DIR__, "surface_points.csv"))]
    cells = collect(CSV.File(joinpath(@__DIR__, "surface_cells.csv")))
    surfaces = NamedTuple{names}(Tuple(zipline_reference_surface(group, points, cells) for group in 3:12))
    fixed_path = joinpath(directory, "ZipLine3D_FixedGeometry.vtkhdf")
    write_geometry && SavePolygonVTKHDF(fixed_path, (; Domain = zipline_reference_surface(2, points, cells)))
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
    filepath = joinpath(directory, "ZipLine3D.vtkhdf")
    write_geometry && SavePolygonMotionSequence(filepath, surfaces;
        motions, times, block_name = "RagdollSurfaces")
    state_path = joinpath(directory, "ZipLine3D_SingleVTKHDFStateFile.py")
    fixed_file = repr(replace(abspath(fixed_path), '\\' => '/'))
    line = only(filter(row -> row.group == 1, cells))
    first_point, last_point = Tuple(points[line.v1]), Tuple(points[line.v2])
    open(state_path, "a") do io
        write(io, """

# Display CAD surfaces following the simulated native body poses.
LoadPalette(paletteName='WhiteBackground')
Ragdoll = ExtractBlock(registrationName='Ragdoll surfaces', Input=Simulation_vtkhdf)
Ragdoll.Selectors = ['/Root/RagdollSurfaces']
RagdollDisplay = Show(Ragdoll, renderView1, 'GeometryRepresentation')
RagdollDisplay.SetRepresentationType('Surface')
RagdollDisplay.Interpolation = 'Gouraud'
RagdollDisplay.ColorArrayName = ['CELLS', '']
RagdollDisplay.DiffuseColor = [0.78, 0.72, 0.55]
FixedGeometry = VTKHDFReader(registrationName='DualSPHysics container surface', FileName=[$fixed_file])
FixedGeometryDisplay = Show(FixedGeometry, renderView1, 'GeometryRepresentation')
FixedGeometryDisplay.SetRepresentationType('Wireframe')
FixedGeometryDisplay.DiffuseColor = [0.4, 0.4, 0.45]
FixedGeometryDisplay.Opacity = 0.35
CableAxis = Line(registrationName='Analytical sliding axis (visual guide)', Point1=$first_point, Point2=$last_point, Resolution=1)
CableDisplay = Show(CableAxis, renderView1, 'GeometryRepresentation')
CableDisplay.LineWidth = 2.0
CableDisplay.DiffuseColor = [0.2, 0.2, 0.25]
Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1, False)
Simulation_vtkhdfDisplay.ColorArrayName = ['POINTS', '']
Simulation_vtkhdfDisplay.DiffuseColor = [0.55, 0.55, 0.6]
Simulation_vtkhdfDisplay.Opacity = 0.35
SetActiveSource(Ragdoll)
# Gravity is along -Z. Use an oblique Z-up overview rather than a plan view.
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraPosition = [-4.0, -20.0, -2.0]
renderView1.CameraFocalPoint = [5.0, 0.0, -5.0]
renderView1.CameraParallelProjection = 1
renderView1.ResetCamera()
if has_native_forces:
    # Keep the complete geometry visible and provide a close view in which
    # force arrows can be read while the body travels along the line.
    layout1.SplitHorizontal(0, 0.6)
    detailView = CreateView('RenderView')
    layout1.AssignView(2, detailView)
    detailView.InteractionMode = '3D'
    detailView.CameraViewUp = [0.0, 0.0, 1.0]
    detailView.CameraPosition = [-2.0, -6.0, 0.2]
    detailView.CameraFocalPoint = [0.0, 0.0, -0.5]
    detailView.CameraParallelProjection = 1
    detailBodyDisplay = Show(Ragdoll, detailView, 'GeometryRepresentation')
    detailBodyDisplay.SetRepresentationType('Surface')
    detailBodyDisplay.DiffuseColor = [0.78, 0.72, 0.55]
    detailBodyDisplay.ColorArrayName = ['CELLS', '']
    followed_sources = [Ragdoll]
    for vector, arrows, colour in native_force_arrows:
        if vector == 'TotalForce':
            continue
        display = Show(arrows, detailView, 'GeometryRepresentation')
        display.SetRepresentationType('Surface')
        display.DiffuseColor = colour
        display.ColorArrayName = ['POINTS', '']
        followed_sources.append(arrows)
    ForceKeyDisplay.Visibility = 0
    detailKey = Show(ForceKey, detailView, 'TextSourceRepresentation')
    detailKey.WindowLocation = 'Upper Left Corner'
    detailKey.FontSize = 12
    detailKey.Color = [0.12, 0.12, 0.12]
    detailView.ResetCamera()
    source_ids = [source.GetGlobalIDAsString() for source in followed_sources]
    view_id = detailView.GetGlobalIDAsString()
    FollowRagdoll = PythonAnimationCue(registrationName='Follow ragdoll in detail view')
    FollowRagdoll.TimeMode = 'Relative'
    FollowRagdoll.Script = f'''def tick(self):
    import math
    from paraview.simple import GetViews, GetSources, GetAnimationScene
    time = self.GetAnimationTime()
    bounds = None
    for source in GetSources().values():
        if source.GetGlobalIDAsString() in {source_ids!r}:
            source.UpdatePipeline(time)
            if source.GetGlobalIDAsString() == {source_ids[0]!r}:
                bounds = source.GetDataInformation().GetBounds()
    for view in GetViews():
        if view.GetGlobalIDAsString() == {view_id!r}:
            view.ViewTime = time
            if bounds is not None:
                center = [(bounds[2*d] + bounds[2*d+1]) / 2 for d in range(3)]
                offset = [view.CameraPosition[d] - view.CameraFocalPoint[d] for d in range(3)]
                view.CameraFocalPoint = center
                view.CameraPosition = [center[d] + offset[d] for d in range(3)]
                radius = math.sqrt(sum((bounds[2*d+1] - bounds[2*d])**2 for d in range(3))) / 2
                aspect = max(1.0, view.ViewSize[1] / max(1, view.ViewSize[0]))
                view.CameraParallelScale = (radius + {force_arrow_length!r}) * aspect
def start_cue(self):
    pass
def end_cue(self):
    pass
'''
    animation = GetAnimationScene()
    animation.UpdateAnimationUsingDataTimeSteps()
    FollowRagdoll.StartTime = animation.StartTime
    FollowRagdoll.EndTime = animation.EndTime
    animation.Cues.append(FollowRagdoll)
SetActiveView(renderView1)
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
