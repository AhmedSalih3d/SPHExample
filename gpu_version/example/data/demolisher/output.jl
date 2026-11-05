# CAD is used for display; native rigid contacts use the particle samples.
using SPHExampleGPU, StaticArrays, CSV

function demolisher_reference_surface(group, points, cells)
    faces = [[row[Symbol("v", i)] for i in 1:row.n] for row in cells if row.group == group && row.n >= 3]
    indices = sort!(unique(vcat(faces...)))
    mapping = Dict(index => local_index for (local_index,index) in enumerate(indices))
    return PolygonSurface(points[indices], [[mapping[index] for index in face] for face in faces])
end

function save_demolisher_surfaces(directory, geometry; write_geometry = true)
    names = (:Arm1, :Arm2, :Arm3, :Arm4, :Ball, (Symbol("Block",i) for i in 1:16)...)
    groups = (1,2,3,4,5,(52:67)...)
    # Keep GenCase's shared vertices and original polygon connectivity.
    points = [SVector(row.x, row.y, row.z) for row in CSV.File(joinpath(@__DIR__, "surface_points.csv"))]
    cells = collect(CSV.File(joinpath(@__DIR__, "surface_cells.csv")))
    surfaces = NamedTuple{names}(Tuple(demolisher_reference_surface(group, points, cells) for group in groups))
    fixed_path = joinpath(directory, "Demolisher3D_FixedGeometry.vtkhdf")
    write_geometry && SavePolygonVTKHDF(fixed_path, (; Domain = demolisher_reference_surface(11, points, cells)))
    rows = collect(CSV.File(joinpath(directory, "Demolisher3D_Floating.csv")))
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
    filepath = joinpath(directory, "Demolisher3D.vtkhdf")
    write_geometry && SavePolygonMotionSequence(filepath, surfaces;
        motions, times, block_name = "DemolisherSurfaces")
    state_path = joinpath(directory, "Demolisher3D_SingleVTKHDFStateFile.py")
    fixed_file = repr(replace(abspath(fixed_path), '\\' => '/'))
    state_text = read(state_path, String)
    appended = findfirst("\n# Display CAD surfaces following the simulated native body poses.", state_text)
    appended === nothing || write(state_path, state_text[begin:prevind(state_text, first(appended))])
    open(state_path, "a") do io
        write(io, """

# Display CAD surfaces following the simulated native body poses.
LoadPalette(paletteName='WhiteBackground')
Demolisher = ExtractBlock(registrationName='Demolisher surfaces', Input=Simulation_vtkhdf)
Demolisher.Selectors = ['/Root/DemolisherSurfaces']
DemolisherDisplay = Show(Demolisher, renderView1, 'GeometryRepresentation')
DemolisherDisplay.SetRepresentationType('Surface')
DemolisherDisplay.Interpolation = 'Gouraud'
DemolisherDisplay.ColorArrayName = ['CELLS', '']
DemolisherDisplay.DiffuseColor = [0.78, 0.72, 0.55]
BodyColors = GetColorTransferFunction('Region')
BodyColors.InterpretValuesAsCategories = 1
body_names = ['Arm1','Arm2','Arm3','Arm4','Ball'] + ['Block' + str(i) for i in range(1,17)]
BodyColors.Annotations = [item for i,name in enumerate(body_names,1) for item in (str(i),name)]
block_palette = [[0.05,0.55,0.22],[0.10,0.40,0.85],[0.95,0.55,0.08],[0.82,0.12,0.15],[0.55,0.18,0.75]]
body_colours = [[0.55,0.52,0.43]] * 4 + [[0.42,0.44,0.47]] + [block_palette[i % 5] for i in range(16)]
BodyColors.IndexedColors = [value for colour in body_colours for value in colour]
ColorBy(DemolisherDisplay, ('CELLS','Region'))
DemolisherDisplay.LookupTable = BodyColors
DemolisherDisplay.SetScalarBarVisibility(renderView1, False)
FixedGeometry = VTKHDFReader(registrationName='DualSPHysics container surface', FileName=[$fixed_file])
FixedGeometryDisplay = Show(FixedGeometry, renderView1, 'GeometryRepresentation')
FixedGeometryDisplay.SetRepresentationType('Wireframe')
FixedGeometryDisplay.DiffuseColor = [0.4, 0.4, 0.45]
FixedGeometryDisplay.Opacity = 0.35
Pivot = Sphere(registrationName='Analytical spherical pivot (visual guide)', Center=[5,5,5.2], Radius=0.075)
PivotDisplay = Show(Pivot, renderView1, 'GeometryRepresentation')
PivotDisplay.DiffuseColor = [0.2, 0.2, 0.25]
Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1, False)
Simulation_vtkhdfDisplay.ColorArrayName = ['POINTS', '']
Simulation_vtkhdfDisplay.DiffuseColor = [0.55, 0.55, 0.6]
Simulation_vtkhdfDisplay.Opacity = 0.35
Simulation_vtkhdfDisplay.Visibility = 0
SetActiveSource(Demolisher)
# Gravity is along -Z. Use an oblique Z-up overview rather than a plan view.
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraPosition = [15.0, -12.0, 10.0]
renderView1.CameraFocalPoint = [5.5, 5.0, 2.5]
renderView1.CameraParallelProjection = 1
renderView1.ResetCamera()
if has_native_forces:
    # Keep the complete geometry visible and provide a close view in which
    # force arrows can be read while the articulated demolisher swings.
    layout1.SplitHorizontal(0, 0.6)
    detailView = CreateView('RenderView')
    layout1.AssignView(2, detailView)
    layout1.SetSize(1600, 900)
    detailView.InteractionMode = '3D'
    detailView.CameraViewUp = [0.0, 0.0, 1.0]
    detailView.CameraPosition = [13.0, -9.0, 7.0]
    detailView.CameraFocalPoint = [5.5, 5.0, 2.5]
    detailView.CameraParallelProjection = 1
    detailBodyDisplay = Show(Demolisher, detailView, 'GeometryRepresentation')
    detailBodyDisplay.SetRepresentationType('Surface')
    detailBodyDisplay.DiffuseColor = [0.78, 0.72, 0.55]
    detailBodyDisplay.ColorArrayName = ['CELLS', '']
    ColorBy(detailBodyDisplay, ('CELLS','Region'))
    detailBodyDisplay.LookupTable = BodyColors
    detailBodyDisplay.SetScalarBarVisibility(detailView, False)
    followed_sources = [Demolisher]
    force_arrow_length = 0.65
    for vector, arrows, colour in native_force_arrows:
        # Display the wrecking ball's forces clearly; all 21 bodies retain
        # their complete force data in the Body forces [N] source and CSV.
        arrows.Input.Function = 'if(abs(Body-5) < 0.5, ' + arrows.Input.Function + ', 0)'
        arrows.ScaleFactor = force_arrow_length
        Hide(arrows, renderView1)
        if vector == 'TotalForce':
            continue
        display = Show(arrows, detailView, 'GeometryRepresentation')
        display.SetRepresentationType('Surface')
        display.DiffuseColor = colour
        display.ColorArrayName = ['POINTS', '']
        followed_sources.append(arrows)
    ForceKeyDisplay.Visibility = 0
    ForceKey.Text = ('Wrecking ball forces [N]\\nBlue: gravity. Red: joint/contact reaction.\\n'
        'Equal lengths show direction\\nInspect body arrays for magnitudes')
    detailKey = Show(ForceKey, detailView, 'TextSourceRepresentation')
    detailKey.WindowLocation = 'Upper Left Corner'
    detailKey.FontSize = 12
    detailKey.Justification = 'Left'
    detailKey.Color = [0.12, 0.12, 0.12]
    detailView.ResetCamera()
    source_ids = [source.GetGlobalIDAsString() for source in followed_sources]
    view_id = detailView.GetGlobalIDAsString()
    FollowDemolisher = PythonAnimationCue(registrationName='Follow demolisher in detail view')
    FollowDemolisher.TimeMode = 'Relative'
    FollowDemolisher.Script = f'''def tick(self):
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
    FollowDemolisher.StartTime = animation.StartTime
    FollowDemolisher.EndTime = animation.EndTime
    animation.Cues.append(FollowDemolisher)
SetActiveView(renderView1)
layout1.SetSize(1600, 900)
TimeLabel = AnnotateTimeFilter(registrationName='Simulation time', Input=Simulation_vtkhdf)
TimeLabel.Format = 'Time: {time:.2f} s'
TimeDisplay = Show(TimeLabel, renderView1, 'TextSourceRepresentation')
TimeDisplay.WindowLocation = 'Lower Left Corner'
TimeDisplay.FontSize = 16
TimeDisplay.Color = [0.12,0.12,0.12]
RenderAllViews()
""")
    end
    return filepath
end

function open_demolisher_output(metadata)
    metadata.VisualizeInParaview || return nothing
    executable = Sys.which("paraview")
    if executable === nothing
        @warn "ParaView was not found on PATH; open Demolisher3D_SingleVTKHDFStateFile.py manually."
        return nothing
    end
    state = joinpath(metadata.SaveLocation, "Demolisher3D_SingleVTKHDFStateFile.py")
    run(`$executable --state=$state`; wait = false)
    return nothing
end
