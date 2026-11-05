# Portable reference fixtures: ordinary runs require neither GenCase nor Chrono.
using SPHExampleGPU,StaticArrays,CSV,LinearAlgebra

function reference_geometry(directory,::Val{D},::Type{T};dx,origin,weights,details = Dict()) where {D,T}
    rows = collect(CSV.File(joinpath(directory,"lattice.csv")))
    groups = sort!(unique(r.group for r in rows);by = g -> (first(r.type for r in rows if r.group == g),g))
    kinds = (Fixed,Moving,Floating,Fluid)
    return [let data = filter(r -> r.group == group,rows),kind = kinds[first(data).type+1]
        points = [D == 2 ? SVector(origin[1]+dx*r.i,origin[3]+dx*r.k) :
            SVector(origin[1]+dx*r.i,origin[2]+dx*r.j,origin[3]+dx*r.k) for r in data]
        floating = kind == Floating ? get(details,group,FloatingDetails{T}(RelativeWeight = weights[group])) : nothing
        SPHGeometry{D,T}(points;Density = T[r.density for r in data],GroupMarker = group,Type = kind,Floating = floating)
    end for group in groups]
end

function reference_surface(group,points,cells)
    faces = [[r[Symbol("v",i)] for i in 1:r.n] for r in cells if r.group == group]
    isempty(faces) && throw(ArgumentError("missing reference surface for group $group"))
    ids = sort!(unique(vcat(faces...)))
    mapping = Dict(id => i for (i,id) in enumerate(ids))
    return PolygonSurface(points[ids],[[mapping[id] for id in face] for face in faces])
end

function reference_rotate(q,r)
    v = SVector(q[2],q[3],q[4])
    return r+2cross(v,cross(v,r)+q[1]*r)
end

function save_reference_surfaces(directory,fixture,geometry,name;moving = Dict(),force_offset = 0.2,
        force_offset_vector = nothing,fluid_alpha = 0.03,write_geometry = true)
    D = length(first(first(geometry).Particles.Position))
    points = [SVector(r.x,r.y,r.z) for r in CSV.File(joinpath(fixture,"surface_points.csv"))]
    cells = collect(CSV.File(joinpath(fixture,"surface_cells.csv")))
    rows = collect(CSV.File(joinpath(directory,name*"_Floating.csv")))
    times = sort!(unique(r.Time for r in rows))
    length(times) >= 2 || return nothing
    states = Dict((r.Body,r.Time) => r for r in rows)
    bodies = filter(g -> g.Type == Floating,geometry)
    names = Tuple(Symbol("Body",b) for b in eachindex(bodies))
    reference = [sum(SVector{D,Float64}.(g.Particles.Position))/length(g.Particles) for g in bodies]
    surfaces = NamedTuple{names}(Tuple(reference_surface(g.GroupMarker,points,cells) for g in bodies))
    motions = NamedTuple{names}(Tuple(let body = b
        (point,time) -> begin
            row = states[(body,time)]
            if D == 2
                r = point-SVector(reference[body]...,0.)
                s,c = sincos(row.Angle)
                SVector(row[Symbol("Center:0")]+c*r[1]-s*r[2],row[Symbol("Center:1")]+s*r[1]+c*r[2],r[3])
            else
                q = SVector(ntuple(d -> row[Symbol("Orientation:",d-1)],4))
                center = SVector(ntuple(d -> row[Symbol("Center:",d-1)],3))
                center+reference_rotate(q,point-reference[body])
            end
        end
    end for b in eachindex(bodies)))
    file = joinpath(directory,name*".vtkhdf")
    fixed = filter(g -> g.Type == Fixed,geometry)
    fixed_file = joinpath(directory,name*"_Fixed.vtkhdf")
    if write_geometry
        SavePolygonMotionSequence(file,surfaces;motions,times,block_name = "RigidSurfaces")
        if !isempty(fixed)
            labels = Tuple(Symbol("Fixed",g.GroupMarker) for g in fixed)
            SavePolygonVTKHDF(fixed_file,NamedTuple{labels}(Tuple(reference_surface(g.GroupMarker,points,cells) for g in fixed)))
        end
        if !isempty(moving)
            markers = sort!(collect(keys(moving)))
            labels = Tuple(Symbol("Moving",g) for g in markers)
            SavePolygonMotionSequence(file,NamedTuple{labels}(Tuple(reference_surface(g,points,cells) for g in markers));
                motions = NamedTuple{labels}(Tuple(moving[g] for g in markers)),times,block_name = "MovingSurfaces")
        end
    end
    path = joinpath(directory,name*"_SingleVTKHDFStateFile.py")
    text = read(path,String)
    appended = findfirst("\n# Coupled reference display",text)
    appended === nothing || write(path,text[begin:prevind(text,first(appended))])
    fixed_path = repr(replace(abspath(fixed_file),'\\' => '/'))
    offset = something(force_offset_vector,D == 2 ? (0,force_offset,0) : (0,0,force_offset))
    open(path,"a") do io
        write(io,"""

# Coupled reference display: original GenCase CAD driven by simulated poses.
LoadPalette(paletteName='WhiteBackground')
Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1,False)
Hide(Simulation_vtkhdf,renderView1)
Particles = ExtractBlock(registrationName='SPH particles',Input=Simulation_vtkhdf)
Particles.Selectors = ['/Root/Particles']
Vertices = ConvertToPointCloud(Input=Particles)
Vertices.CellGenerationMode = 'Vertex cells'
Fluid = Threshold(Input=Vertices)
Fluid.Scalars = ['POINTS','Type']
Fluid.LowerThreshold = 1
Fluid.UpperThreshold = 1
Fluid.ThresholdMethod = 'Between'
Water = Delaunay2D(Input=Fluid) if $D == 2 else Fluid
if $D == 2:
    Water.Alpha = $fluid_alpha
WaterDisplay = Show(Water,renderView1,'GeometryRepresentation')
WaterDisplay.Representation = 'Surface' if $D == 2 else 'Points'
WaterDisplay.PointSize = 2
ColorBy(WaterDisplay,('POINTS','Velocity','Magnitude'))
WaterDisplay.SetScalarBarVisibility(renderView1,True)
Colors = GetColorTransferFunction('Velocity')
Colors.ApplyPreset('Blue to Red Rainbow',True)
Bodies = ExtractBlock(registrationName='Reference rigid surfaces',Input=Simulation_vtkhdf)
Bodies.Selectors = ['/Root/RigidSurfaces']
BodyDisplay = Show(Bodies,renderView1,'GeometryRepresentation')
BodyDisplay.Interpolation = 'Gouraud'
ColorBy(BodyDisplay,('CELLS','Region'))
BodyDisplay.SetScalarBarVisibility(renderView1,False)
if $(isempty(fixed) ? "False" : "True"):
    FixedSurfaces = VTKHDFReader(FileName=[$fixed_path])
    FixedDisplay = Show(FixedSurfaces,renderView1,'GeometryRepresentation')
    FixedDisplay.Representation = 'Wireframe'
    FixedDisplay.ColorArrayName = ['CELLS','']
    FixedDisplay.DiffuseColor = [0.2,0.2,0.2]
if $(isempty(moving) ? "False" : "True"):
    MovingSurfaces = ExtractBlock(Input=Simulation_vtkhdf)
    MovingSurfaces.Selectors = ['/Root/MovingSurfaces']
    MovingDisplay = Show(MovingSurfaces,renderView1,'GeometryRepresentation')
    MovingDisplay.Representation = 'Wireframe'
    MovingDisplay.ColorArrayName = ['CELLS','']
    MovingDisplay.DiffuseColor = [0.2,0.2,0.2]
if has_native_forces:
    ForceLocations = Transform(Input=FloatingBodies)
    ForceLocations.Transform.Translate = [$(offset[1]),$(offset[2]),$(offset[3])]
    FluidForceMask = Calculator(Input=ForceLocations)
    FluidForceMask.ResultArrayName = 'VisibleForce'
    FluidForceMask.Function = 'if(ForceMagnitude > 0,1,0)'
    FluidForceArrows = Glyph(Input=FluidForceMask,GlyphType='Arrow')
    FluidForceArrows.OrientationArray = ['POINTS','Force']
    FluidForceArrows.ScaleArray = ['POINTS','VisibleForce']
    FluidForceArrows.GlyphMode = 'All Points'
    native_force_arrows.append(('Force',FluidForceArrows,[0.95,0.45,0.05]))
    for vector,arrows,colour in native_force_arrows:
        arrows.Input.Input = ForceLocations
        arrows.ScaleFactor = $(force_offset/2)
        if vector != 'TotalForce':
            display = Show(arrows,renderView1,'GeometryRepresentation')
            display.ColorArrayName = ['POINTS','']
            display.DiffuseColor = colour
    ForceKey.Text = 'Blue: gravity. Red: joint/contact reaction. Orange: SPH/spring force.\\nEqual arrow lengths show direction. Display positions offset from body centres; magnitude arrays in N.'
TimeLabel = AnnotateTimeFilter(Input=Simulation_vtkhdf)
TimeLabel.Format = 'Time: {time:.2f} s'
TimeDisplay = Show(TimeLabel,renderView1,'TextSourceRepresentation')
TimeDisplay.WindowLocation = 'Upper Right Corner'
TimeDisplay.Color = [0.1,0.1,0.1]
renderView1.CameraViewUp = [0,1,0] if $D == 2 else [0,0,1]
renderView1.CameraParallelProjection = 1
if $D == 2:
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0,0,100]
    renderView1.CameraFocalPoint = [0,0,0]
ResetCamera(renderView1)
renderView1.CameraParallelScale *= 0.75
GetLayout(renderView1).SetSize(1600,900)
Colors.RescaleTransferFunction(0,2)
Colors.AutomaticRescaleRangeMode = 'Never'
RenderAllViews()
""")
    end
    return file
end
