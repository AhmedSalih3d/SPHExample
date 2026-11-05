# Coils follow the accepted spring endpoints; force fields are constitutive
# loads at those accepted poses, distinct from the body's last half-state load.
function reference_spring_surface(spring;radius = 0.02,turns = spring isa CoulombDamper ? 0 : 8,wire_radius = 0.0015,wire_sides = 8)
    anchor,attachment = SVector(spring.Anchor...,0.),SVector(spring.Attachment...,0.)
    axis = attachment-anchor
    direction = normalize(axis)
    radial = normalize(cross(direction,SVector(0.,0.,1.)))
    transverse = cross(direction,radial)
    path = if turns == 0
        [anchor,attachment]
    else
        points = [anchor,anchor+0.08axis]
        for i in 1:16turns-1
            s = i/(16turns)
            envelope = min(1.,2turns*s,2turns*(1-s))
            push!(points,anchor+(0.08+0.84s)*axis+radius*envelope*(cos(2pi*turns*s)*radial+sin(2pi*turns*s)*transverse))
        end
        append!(points,(anchor+0.92axis,attachment))
    end
    points = SVector{3,Float64}[]
    for i in eachindex(path)
        tangent = normalize(path[min(i+1,end)]-path[max(i-1,1)])
        normal = normalize(radial-dot(radial,tangent)*tangent)
        binormal = cross(tangent,normal)
        for side in 0:wire_sides-1
            angle = 2pi*side/wire_sides
            push!(points,path[i]+wire_radius*(cos(angle)*normal+sin(angle)*binormal))
        end
    end
    faces = [collect(wire_sides:-1:1)]
    for i in 1:length(path)-1,side in 1:wire_sides
        a,b = (i-1)*wire_sides+side,(i-1)*wire_sides+mod1(side+1,wire_sides)
        push!(faces,[a,b,b+wire_sides],[a,b+wire_sides,a+wire_sides])
    end
    push!(faces,collect((length(path)-1)*wire_sides+1:length(path)*wire_sides))
    return PolygonSurface(points,faces)
end

function save_reference_springs(file,bodies,states,reference,times;write_geometry = true)
    links = [(;body = b,spring) for (b,g) in enumerate(bodies) for spring in g.Floating.Springs]
    isempty(links) && return nothing
    all(link -> length(link.spring.Anchor) == 2,links) || throw(ArgumentError("the reference coil display currently requires planar springs"))
    records = Dict()
    for (id,link) in enumerate(links),time in times
        b,spring = link.body,link.spring
        row = states[(b,time)]
        s,c = sincos(row.Angle)
        r = spring.Attachment-reference[b]
        arm = SVector(c*r[1]-s*r[2],s*r[1]+c*r[2])
        center = SVector(row[Symbol("Center:0")],row[Symbol("Center:1")])
        attachment = center+arm
        velocity = SVector(row[Symbol("Velocity:0")],row[Symbol("Velocity:1")])+row.Omega*SVector(-arm[2],arm[1])
        delta = attachment-spring.Anchor
        force = spring isa CoulombDamper ? axial_coulomb_force(delta,velocity,spring.Force) :
            axial_spring_force(delta,velocity,spring.Stiffness,spring.Damping,spring.RestLength)
        length = norm(delta)
        axial_force = length > 0 ? -dot(force,delta)/length : 0.
        elastic = spring isa CoulombDamper ? 0. : spring.Stiffness*(length-spring.RestLength)
        records[(id,time)] = (;body = b,anchor = spring.Anchor,attachment,force,length,
            extension = length-spring.RestLength,axial_force,elastic,damping = axial_force-elastic)
    end
    path = splitext(file)[1]*"_Springs.csv"
    open(path,"w") do io
        println(io,"Time,Body,Spring,Anchor:0,Anchor:1,Attachment:0,Attachment:1,Length,Extension,Force:0,Force:1,ForceMagnitude,ElasticAxialForce,DampingAxialForce")
        for time in times,id in eachindex(links)
            r = records[(id,time)]
            println(io,join((time,r.body,id,r.anchor...,r.attachment...,r.length,r.extension,r.force...,norm(r.force),r.elastic,r.damping),','))
        end
    end
    force_range = (0.,max(1e-12,maximum(norm(r.force) for r in values(records))))
    write_geometry || return force_range
    names = Tuple(Symbol("Spring",id) for id in eachindex(links))
    surfaces = NamedTuple{names}(Tuple(reference_spring_surface(link.spring) for link in links))
    motions = NamedTuple{names}(Tuple(let id = i,link = link
        (point,time) -> begin
            r = records[(id,time)]
            anchor = SVector(r.anchor...,0.)
            initial_axis = SVector(link.spring.Attachment...,0.)-anchor
            initial_direction = normalize(initial_axis)
            axis = SVector((r.attachment-r.anchor)...,0.)
            direction = norm(axis) > 0 ? normalize(axis) : initial_direction
            radial0,radial = cross(initial_direction,SVector(0.,0.,1.)),cross(direction,SVector(0.,0.,1.))
            delta = point-anchor
            anchor+(dot(delta,initial_direction)/norm(initial_axis))*axis+dot(delta,radial0)*radial+SVector(0.,0.,delta[3])
        end
    end for (i,link) in enumerate(links)))
    field(name) = (region,time) -> getproperty(records[(parse(Int,replace(string(region),"Spring" => "")),time)],name)
    id(region) = parse(Int,replace(string(region),"Spring" => ""))
    SavePolygonMotionSequence(file,surfaces;motions,times,block_name = "Springs",cell_fields = (
        Force = (region,t) -> SVector(records[(id(region),t)].force...,0.),
        ForceMagnitude = (region,t) -> norm(records[(id(region),t)].force),
        AxialForce = field(:axial_force),ElasticAxialForce = field(:elastic),DampingAxialForce = field(:damping),
        Extension = field(:extension),Anchor = (region,t) -> SVector(records[(id(region),t)].anchor...,0.),
        Attachment = (region,t) -> SVector(records[(id(region),t)].attachment...,0.)))
    return force_range
end
