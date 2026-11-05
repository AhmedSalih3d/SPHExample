using SPHExampleGPU,StaticArrays,CSV

function save_current_wheel_surfaces(directory,fixture,geometry,name,variant;write_geometry = true,dx = 0.0075)
    file = save_reference_surfaces(directory,fixture,geometry,name;force_offset = 0.2,
        force_offset_vector = (0.5,0.1,0),fluid_alpha = 2dx,speed_limit = 1.5,write_geometry)
    variant == :pulley || return file
    body = only(filter(g -> g.GroupMarker == 61 && g.Type == Floating,geometry))
    reference = sum(SVector{2,Float64}.(body.Particles.Position))/length(body.Particles)
    rows = filter(r -> r.GroupMarker == 61,collect(CSV.File(joinpath(directory,name*"_Floating.csv"))))
    times = [r.Time for r in rows]
    states = Dict(r.Time => r for r in rows)
    # A display-only shaft pointer makes rotation of the otherwise uniform
    # sphere visible, without changing its original CAD or sampled mechanics.
    pointer = PolygonSurface([SVector(reference[1],reference[2]-0.001,0.2),
        SVector(reference[1]+0.045,reference[2],0.2),SVector(reference[1],reference[2]+0.001,0.2)],[[1,2,3]])
    function motion(point,time)
        row = states[time]
        r = point-SVector(reference...,0.)
        s,c = sincos(row.Angle)
        return SVector(row[Symbol("Center:0")]+c*r[1]-s*r[2],row[Symbol("Center:1")]+s*r[1]+c*r[2],r[3])
    end
    write_geometry && SavePolygonMotionSequence(file,(;Pointer = pointer);motions = (;Pointer = motion),times,block_name = "ShaftPointer")
    open(joinpath(directory,name*"_SingleVTKHDFStateFile.py"),"a") do io
        write(io,"""

# Analytical shaft indicator; original reference surfaces remain unchanged.
Pointer = ExtractBlock(registrationName='Gear rotation pointer (display only)',Input=Simulation_vtkhdf)
Pointer.Selectors = ['/Root/ShaftPointer']
PointerDisplay = Show(Pointer,renderView1,'GeometryRepresentation')
PointerDisplay.ColorArrayName = ['CELLS','']
PointerDisplay.DiffuseColor = [0.05,0.05,0.05]
RenderAllViews()
""")
    end
    return file
end
