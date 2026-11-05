using SPHExampleGPU,CSV,StaticArrays,LinearAlgebra

function reference_velocity_file(file)
    rows = collect(CSV.File(file;delim = ';',types = String))
    times = parse.(Float64,[r[1] for r in rows])
    values = [all(v -> lowercase(strip(v)) == "none",Tuple(r)[2:4]) ? nothing :
        Tuple(parse.(Float64,Tuple(r)[2:4])) for r in rows]
    return VelocitySeries(times,values)
end

function save_external_velocity_surfaces(directory,fixture,geometry,name,variant;write_geometry = true)
    file = save_reference_surfaces(directory,fixture,geometry,name;force_offset = variant == :ship ? 0.12 : variant == :hauling ? 0.6 : 0.8,
        force_offset_vector = variant == :ship ? (0,0.2,0) : variant == :hauling ? (0,1.2,0) : (0,0,1.),
        fluid_alpha = 0.015,speed_limit = 2.,write_geometry)
    if variant == :hauling
        rows = collect(CSV.File(joinpath(directory,name*"_Floating.csv")))
        states = Dict((r.GroupMarker,r.Time) => r for r in rows)
        times = sort(unique(r.Time for r in rows))
        bodies = Dict(g.GroupMarker => g for g in geometry if g.Type == Floating)
        initial = Dict(g => sum(SVector{2,Float64}.(b.Particles.Position))/length(b.Particles) for (g,b) in bodies)
        function endpoint(group,point,time)
            row = states[(group,time)]
            r = SVector{2,Float64}(point)-initial[group]
            s,c = sincos(row.Angle)
            arm = SVector(c*r[1]-s*r[2],s*r[1]+c*r[2])
            position = SVector(row[Symbol("Center:0")],row[Symbol("Center:1")])+arm
            velocity = SVector(row[Symbol("Velocity:0")],row[Symbol("Velocity:1")])+row.Omega*SVector(-arm[2],arm[1])
            return position,velocity
        end
        records = Dict()
        for time in times
            a,va = endpoint(61,(0.6,0.4),time)
            b,vb = endpoint(60,(1.2,0.4),time)
            load = axial_spring_force(a-b,va-vb,4000.,0.,1.)
            records[time] = (;a,b,load)
        end
        CSV.write(joinpath(directory,name*"_BodySpring.csv"),[(;Time = t,Length = norm(r.a-r.b),ForceMagnitude = norm(r.load),
            AttachmentX = r.a[1],AttachmentZ = r.a[2],AnchorX = r.b[1],AnchorZ = r.b[2]) for (t,r) in sort(collect(records);by = first)])
        spring = LinearSpring{2,Float64}(Anchor = (1.2,0.4),Attachment = (0.6,0.4),Stiffness = 4000.,Damping = 0.,RestLength = 1.)
        coil = reference_spring_surface(spring;radius = 0.025,wire_radius = 0.003)
        function motion(point,time)
            state = records[time]
            axis = SVector((state.a-state.b)...,0.)
            direction = norm(axis) > 0 ? normalize(axis) : SVector(-1.,0.,0.)
            radial = cross(direction,SVector(0.,0.,1.))
            delta = point-SVector(1.2,0.4,0.)
            return SVector(state.b...,0.14)+(-delta[1]/0.6)*axis+delta[2]*radial+SVector(0.,0.,delta[3])
        end
        write_geometry && SavePolygonMotionSequence(file,(;Coil = coil);motions = (;Coil = motion),times,block_name = "BodySpring",
            cell_fields = (;ForceMagnitude = (region,t) -> norm(records[t].load)))
        open(joinpath(directory,name*"_SingleVTKHDFStateFile.py"),"a") do io
            write(io,"""

# Coil follows both moving attachment points; depth offset is display only.
Coil = ExtractBlock(registrationName='Moving body spring',Input=Simulation_vtkhdf)
Coil.Selectors = ['/Root/BodySpring']
CoilDisplay = Show(Coil,renderView1,'GeometryRepresentation')
if 'SpringBlock' in globals():
    Hide(SpringBlock,renderView1)
    SpringDisplay.SetScalarBarVisibility(renderView1,False)
ColorBy(CoilDisplay,('CELLS','ForceMagnitude'))
BodySpringLUT = GetColorTransferFunction('ForceMagnitude')
BodySpringLUT.ApplyPreset('Blue to Red Rainbow',True)
CoilDisplay.LookupTable = BodySpringLUT
CoilDisplay.SetScalarBarVisibility(renderView1,True)
BodySpringBar = GetScalarBar(BodySpringLUT,renderView1)
BodySpringBar.Title = 'Moving-body spring force (N)'
BodySpringBar.ComponentTitle = ''
BodySpringBar.WindowLocation = 'Any Location'
BodySpringBar.Position = [0.25,0.06]
BodySpringBar.Orientation = 'Horizontal'
BodySpringBar.ScalarBarLength = 0.3
BodySpringForceLimit = $(max(1e-12,maximum(norm(r.load) for r in values(records))))
BodySpringLUT.RescaleTransferFunction(0,BodySpringForceLimit)
BodySpringLUT.AutomaticRescaleRangeMode = 'Never'
RenderAllViews()
""")
        end
    end
    open(joinpath(directory,name*"_SingleVTKHDFStateFile.py"),"a") do io
        write(io,"\nForceKey.Text = 'Blue: gravity. Red: motor/joint/contact reaction. Orange: SPH/spring force.\\nEqual lengths show direction; magnitude arrays are in N.\\nTorque: ConstraintTorque (N m); arrows show forces.'\n")
        variant != :ship && write(io,"Hide(Water,renderView1)\nWaterDisplay.SetScalarBarVisibility(renderView1,False)\n")
        if variant in (:boxes,:breaking_wall,:many_blocks)
            write(io,"FixedDisplay.Representation = 'Surface'\nFixedDisplay.DiffuseColor = [0.72,0.70,0.62]\nFixedDisplay.Opacity = 0.18\n")
        end
        write(io,"RenderAllViews()\n")
    end
    return file
end
