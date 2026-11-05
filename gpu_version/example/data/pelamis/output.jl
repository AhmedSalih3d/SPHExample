using SPHExampleGPU,StaticArrays,CSV

function pelamis_reference_surface(group,points,cells)
    faces = [[r[Symbol("v",i)] for i in 1:r.n] for r in cells if r.group == group]
    ids = sort!(unique(vcat(faces...)))
    mapping = Dict(id => i for (i,id) in enumerate(ids))
    return PolygonSurface(points[ids],[[mapping[id] for id in face] for face in faces])
end

function save_pelamis_surfaces(directory,geometry;piston_state,write_geometry = true)
    points = [SVector(r.x,r.y,r.z) for r in CSV.File(joinpath(@__DIR__,"surface_points.csv"))]
    cells = collect(CSV.File(joinpath(@__DIR__,"surface_cells.csv")))
    template = pelamis_reference_surface(10,points,cells)
    names = (:Segment1,:Segment2,:Segment3,:Segment4)
    surfaces = NamedTuple{names}(Tuple(PolygonSurface([p+SVector(1.3*(b-1),0,0) for p in template.points],template.faces) for b in 1:4))
    bodies = filter(g -> g.Type == Floating,geometry)
    reference = [sum(SVector{2,Float64}.(g.Particles.Position))/length(g.Particles) for g in bodies]
    rows = collect(CSV.File(joinpath(directory,"Pelamis2D_Floating.csv")))
    times = sort!(unique(r.Time for r in rows))
    length(times) >= 2 || return nothing
    states = Dict((r.Body,r.Time) => r for r in rows)
    motions = NamedTuple{names}(Tuple(let body = b
        (point,time) -> begin
            row = states[(body,time)]
            r = point-SVector(reference[body]...,0.)
            s,c = sincos(row.Angle)
            SVector(row[Symbol("Center:0")]+c*r[1]-s*r[2],row[Symbol("Center:1")]+s*r[1]+c*r[2],r[3])
        end
    end for b in 1:4))
    filepath = joinpath(directory,"Pelamis2D.vtkhdf")
    fixed_path = joinpath(directory,"Pelamis2D_Beach.vtkhdf")
    if write_geometry
        SavePolygonVTKHDF(fixed_path,(;Beach = pelamis_reference_surface(21,points,cells)))
        SavePolygonMotionSequence(filepath,surfaces;motions,times,block_name = "PelamisSurfaces")
        SavePolygonMotionSequence(filepath,(;Piston = pelamis_reference_surface(20,points,cells));
            motions = (;Piston = (p,t) -> p+SVector(piston_state(t)[1],0,0)),times,block_name = "PistonSurface")
    end
    path = joinpath(directory,"Pelamis2D_SingleVTKHDFStateFile.py")
    text = read(path,String)
    appended = findfirst("\n# Pelamis coupled display",text)
    appended === nothing || write(path,text[begin:prevind(text,first(appended))])
    fixed_file = repr(replace(abspath(fixed_path),'\\' => '/'))
    open(path,"a") do io
        write(io,"""

# Pelamis coupled display: copied CAD, fluid speed and analytical constraints.
LoadPalette(paletteName='WhiteBackground')
Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1, False)
Simulation_vtkhdfDisplay.Visibility = 0
Particles = ExtractBlock(registrationName='SPH particles', Input=Simulation_vtkhdf)
Particles.Selectors = ['/Root/Particles']
ParticleVertices = ConvertToPointCloud(registrationName='Particle vertices', Input=Particles)
ParticleVertices.CellGenerationMode = 'Vertex cells'
Fluid = Threshold(registrationName='Fluid particles', Input=ParticleVertices)
Fluid.Scalars = ['POINTS','Type']
Fluid.LowerThreshold = 1
Fluid.UpperThreshold = 1
Fluid.ThresholdMethod = 'Between'
WaterSurface = Delaunay2D(registrationName='Fluid speed surface', Input=Fluid)
WaterSurface.Alpha = 0.045
WaterSurface.UpdatePipeline()
WaterDisplay = Show(WaterSurface, renderView1, 'GeometryRepresentation')
WaterDisplay.SetRepresentationType('Surface')
ColorBy(WaterDisplay, ('POINTS','Velocity','Magnitude'))
SpeedColors = GetColorTransferFunction('Velocity')
SpeedColors.RGBPoints = [0,0,0,1, 0.25,0,1,1, 0.5,0,1,0, 0.75,1,1,0, 1,1,0,0]
SpeedColors.ColorSpace = 'RGB'
SpeedColors.RescaleTransferFunction(0,1)
SpeedColors.AutomaticRescaleRangeMode = 'Never'
WaterDisplay.SetScalarBarVisibility(renderView1, True)
SpeedBar = GetScalarBar(SpeedColors, renderView1)
SpeedBar.Title = 'Speed [m/s]'
SpeedBar.ComponentTitle = ''
SpeedBar.Orientation = 'Horizontal'
SpeedBar.WindowLocation = 'Lower Right Corner'
SpeedBar.ScalarBarLength = 0.25
Pelamis = ExtractBlock(registrationName='Pelamis segment surfaces', Input=Simulation_vtkhdf)
Pelamis.Selectors = ['/Root/PelamisSurfaces']
Pelamis.UpdatePipeline()
PelamisDisplay = Show(Pelamis, renderView1, 'GeometryRepresentation')
PelamisDisplay.SetRepresentationType('Surface')
PelamisDisplay.Interpolation = 'Gouraud'
BodyColors = GetColorTransferFunction('Region')
BodyColors.InterpretValuesAsCategories = 1
BodyColors.Annotations = ['1','Segment 1','2','Segment 2','3','Segment 3','4','Segment 4']
BodyColors.IndexedColors = [0.1,0.25,0.85, 0.08,0.65,0.25, 0.95,0.65,0.12, 0.65,0.05,0.25]
ColorBy(PelamisDisplay, ('CELLS','Region'))
PelamisDisplay.LookupTable = BodyColors
PelamisDisplay.SetScalarBarVisibility(renderView1, False)
Beach = VTKHDFReader(registrationName='Reference wet beach faces', FileName=[$fixed_file])
BeachDisplay = Show(Beach, renderView1, 'GeometryRepresentation')
BeachDisplay.SetRepresentationType('Wireframe')
BeachDisplay.ColorArrayName = ['CELLS','']
BeachDisplay.DiffuseColor = [0.12,0.12,0.12]
BeachDisplay.LineWidth = 2
PistonSurface = ExtractBlock(registrationName='Moving piston surface', Input=Simulation_vtkhdf)
PistonSurface.Selectors = ['/Root/PistonSurface']
PistonDisplay = Show(PistonSurface, renderView1, 'GeometryRepresentation')
PistonDisplay.SetRepresentationType('Wireframe')
PistonDisplay.ColorArrayName = ['CELLS','']
PistonDisplay.DiffuseColor = [0.12,0.12,0.12]
PistonDisplay.LineWidth = 2
renderView1.InteractionMode = '2D'
renderView1.CameraParallelProjection = 1
renderView1.CameraViewUp = [0,1,0]
renderView1.CameraPosition = [13.8,1.25,100]
renderView1.CameraFocalPoint = [13.8,1.25,0]
renderView1.CameraParallelScale = 3.4
layout1.SplitVertical(0,0.5)
detailView = CreateView('RenderView')
layout1.AssignView(2, detailView)
layout1.SetSize(1600,900)
detailView.InteractionMode = '2D'
detailView.CameraParallelProjection = 1
detailView.CameraViewUp = [0,1,0]
detailView.CameraPosition = [8.45,1.5,100]
detailView.CameraFocalPoint = [8.45,1.5,0]
detailView.CameraParallelScale = 1.2
DetailWaterDisplay = Show(WaterSurface, detailView, 'GeometryRepresentation')
ColorBy(DetailWaterDisplay, ('POINTS','Velocity','Magnitude'))
DetailWaterDisplay.LookupTable = SpeedColors
DetailWaterDisplay.SetScalarBarVisibility(detailView, False)
DetailBodyDisplay = Show(Pelamis, detailView, 'GeometryRepresentation')
DetailBodyDisplay.SetRepresentationType('Surface')
DetailBodyDisplay.Interpolation = 'Gouraud'
ColorBy(DetailBodyDisplay, ('CELLS','Region'))
DetailBodyDisplay.LookupTable = BodyColors
DetailBodyDisplay.SetScalarBarVisibility(detailView, False)
SlidingAxis = Line(registrationName='Analytical vertical sliding line', Point1=[5.8,0.9,0.25], Point2=[5.8,2.1,0.25])
AxisDisplay = Show(SlidingAxis, detailView, 'GeometryRepresentation')
AxisDisplay.DiffuseColor = [0.35,0.35,0.35]
AxisDisplay.Opacity = 0.4
if has_native_forces:
    ForceLocations = Transform(registrationName='Force display positions above bodies', Input=FloatingBodies)
    ForceLocations.Transform.Translate = [0,0.55,0]
    mask = Calculator(registrationName='Fluid force mask', Input=FloatingBodies)
    mask.ResultArrayName = 'VisibleForce'
    mask.Function = 'if(ForceMagnitude > 0,1,0)'
    arrows = Glyph(registrationName='Fluid load (orange)', Input=mask, GlyphType='Arrow')
    arrows.OrientationArray = ['POINTS','Force']
    arrows.ScaleArray = ['POINTS','VisibleForce']
    arrows.GlyphMode = 'All Points'
    native_force_arrows.append(('Force',arrows,[0.95,0.45,0.05]))
    for vector,arrows,colour in native_force_arrows:
        arrows.Input.Input = ForceLocations
        Show(arrows,renderView1,'GeometryRepresentation')
        Hide(arrows,renderView1)
        arrows.ScaleFactor = 0.32
        if vector == 'TotalForce':
            continue
        display = Show(arrows,detailView,'GeometryRepresentation')
        display.ColorArrayName = ['POINTS','']
        display.DiffuseColor = colour
    ForceKeyDisplay.Visibility = 0
    ForceKey.Text = ('Forces displayed 0.55 m above each body.\\n'
        'Blue: gravity. Orange: fluid. Red: joints. Equal lengths show direction.\\n'
        'Magnitude arrays: N for 1 m width.')
    DetailKey = Show(ForceKey,detailView,'TextSourceRepresentation')
    DetailKey.WindowLocation = 'Upper Left Corner'
    DetailKey.FontSize = 13
    DetailKey.Justification = 'Left'
    DetailKey.Color = [0.12,0.12,0.12]
Pelamis.UpdatePipeline()
source_id = Pelamis.GetGlobalIDAsString()
view_id = detailView.GetGlobalIDAsString()
FollowPelamis = PythonAnimationCue(registrationName='Follow Pelamis')
FollowPelamis.TimeMode = 'Relative'
FollowPelamis.Script = f'''def tick(self):
    from paraview.simple import GetSources, GetViews
    time = self.GetAnimationTime()
    for source in GetSources().values():
        if source.GetGlobalIDAsString() == {source_id!r}:
            source.UpdatePipeline(time)
            bounds = source.GetDataInformation().GetBounds()
            for view in GetViews():
                if view.GetGlobalIDAsString() == {view_id!r}:
                    view.ViewTime = time
                    center = [(bounds[2*d]+bounds[2*d+1])/2 for d in range(3)]
                    view.CameraFocalPoint = center
                    view.CameraPosition = [center[0],center[1],center[2]+100]
                    aspect = view.ViewSize[1]/max(1,view.ViewSize[0])
                    view.CameraParallelScale = max(1.2,((bounds[1]-bounds[0])/2+0.25)*aspect,(bounds[3]-bounds[2])/2+0.35)
def start_cue(self):
    pass
def end_cue(self):
    pass
'''
animation = GetAnimationScene()
animation.UpdateAnimationUsingDataTimeSteps()
FollowPelamis.StartTime = animation.StartTime
FollowPelamis.EndTime = animation.EndTime
animation.Cues.append(FollowPelamis)
TimeLabel = AnnotateTimeFilter(registrationName='Simulation time', Input=Simulation_vtkhdf)
TimeLabel.Format = 'Time: {time:.2f} s'
TimeDisplay = Show(TimeLabel,renderView1,'TextSourceRepresentation')
TimeDisplay.WindowLocation = 'Upper Right Corner'
TimeDisplay.FontSize = 16
TimeDisplay.Color = [0.12,0.12,0.12]
SetActiveSource(WaterSurface)
SetActiveView(renderView1)
SpeedColors.RescaleTransferFunction(0,1)
RenderAllViews()
""")
    end
    return filepath
end

function open_pelamis_output(metadata)
    metadata.VisualizeInParaview || return nothing
    executable = Sys.which("paraview")
    executable === nothing && return nothing
    state = joinpath(metadata.SaveLocation,"Pelamis2D_SingleVTKHDFStateFile.py")
    run(`$executable --state=$state`;wait = false)
    return nothing
end
