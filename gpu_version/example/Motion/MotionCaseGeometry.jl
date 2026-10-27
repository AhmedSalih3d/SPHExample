# Reference-case shapes are built as Julia polygon surfaces, independently of
# GenCase. Each part retains the upstream mkbound value for motion ownership.
const Point3 = SVector{3,Float64}

function box_surface(point, size)
    a, b = min.(Point3(point), Point3(point)+Point3(size)), max.(Point3(point), Point3(point)+Point3(size))
    points = [Point3(a[1],a[2],a[3]), Point3(b[1],a[2],a[3]), Point3(b[1],b[2],a[3]), Point3(a[1],b[2],a[3]),
        Point3(a[1],a[2],b[3]), Point3(b[1],a[2],b[3]), Point3(b[1],b[2],b[3]), Point3(a[1],b[2],b[3])]
    return PolygonSurface(points, [[4,3,2,1],[5,6,7,8],[1,2,6,5],[2,3,7,6],[3,4,8,7],[4,1,5,8]])
end

function tapered_surface(bottom, top)
    n = length(bottom)
    length(top) == n || throw(DimensionMismatch("prism rings must match"))
    points = Point3[bottom..., top...]
    faces = [[1,i+1,i] for i in 2:(n-1)]
    append!(faces,[[n+1,n+i,n+i+1] for i in 2:(n-1)])
    append!(faces, [[i,mod1(i+1,n),n+mod1(i+1,n),n+i] for i in 1:n])
    return PolygonSurface(points, faces)
end

function sphere_surface(center, radius; sides = 32)
    c = Point3(center)
    rings = sides
    points = [c + Point3(0,0,radius)]
    for j in 1:(rings-1), i in 0:(sides-1)
        phi, theta = pi*j/rings, 2pi*i/sides
        push!(points, c + radius*Point3(sin(phi)*cos(theta),sin(phi)*sin(theta),cos(phi)))
    end
    push!(points, c-Point3(0,0,radius))
    index(j,i) = 2+(j-1)*sides+mod(i,sides)
    faces = [[1,index(1,i),index(1,i+1)] for i in 0:(sides-1)]
    for j in 1:(rings-2), i in 0:(sides-1)
        push!(faces, [index(j,i),index(j+1,i),index(j+1,i+1),index(j,i+1)])
    end
    append!(faces, [[length(points),index(rings-1,i+1),index(rings-1,i)] for i in 0:(sides-1)])
    return PolygonSurface(points, faces)
end

function cylinder_surface(a, b, radius; sides = 32)
    a, b = Point3(a), Point3(b)
    axis = normalize(b-a)
    seed = abs(axis[1]) < 0.9 ? Point3(1,0,0) : Point3(0,1,0)
    u = normalize(seed-dot(seed,axis)*axis)
    v = cross(axis,u)
    ring(c) = [c+radius*(cos(2pi*i/sides)*u+sin(2pi*i/sides)*v) for i in 0:(sides-1)]
    points = [ring(a)...,ring(b)...,b]
    # The supplied cases use mask=1: lateral quads and the cap at the second
    # endpoint. Its opposite end is open against the corresponding backdrop.
    faces = [[i,mod1(i+1,sides),sides+mod1(i+1,sides),sides+i] for i in 1:sides]
    append!(faces,[[2sides+1,sides+i,sides+mod1(i+1,sides)] for i in 1:sides])
    return PolygonSurface(points,faces)
end

function case_parts(number)
    part(name, mk, shape) = (;name,mk,shape)
    box(name,mk,point,size) = part(name,mk,box_surface(point,size))
    if number == 1
        return [box(:base,0,(0,0,0),(1.85,0.1,0.02)), box(:slider,1,(0,0.03,0.02),(0.05,0.04,0.02))]
    elseif number == 2
        return [box(:base,0,(0,0,0),(2.05,1.35,0.02)), box(:slider,1,(0,0,0.02),(0.05,0.05,0.05))]
    elseif number == 3
        return [box(:base,0,(0,0,0),(2,1.6,0.02)), box(:parent,1,(0,0,0.02),(0.5,0.5,0.02)),
            box(:child,2,(0,0,0.04),(0.05,0.05,0.02))]
    elseif number == 4
        return [box(:base,0,(0,0,0),(2.1,1.6,0.02)), box(:slider,1,(0,0,0.02),(0.1,0.1,0.02))]
    elseif number == 5
        tick = box_surface((0.495,0.96,0.02),(0.01,0.03,0.005))
        ticks = map(0:11) do i
            pose = motion_state(PrescribedMotion(1;center=(0.5,0.5,0),
                rotation=SampledTrajectory([0,1],[deg2rad(-30i),deg2rad(-30i)])),0)
            PolygonSurface([motion_position(pose,p) for p in tick.points],tick.faces)
        end
        hour = PolygonSurface([(0.5,0.5,0.06),(0.47,0.47,0.035),(0.53,0.47,0.035),(0.5,0.75,0.035)],
            [[2,4,3],[1,2,3],[1,3,4],[1,4,2]])
        minute = tapered_surface([(0.47,0.47,0.02),(0.53,0.47,0.02),(0.5,0.94,0.02)],
            [(0.47,0.47,0.03),(0.53,0.47,0.03),(0.5,0.94,0.03)])
        return [box(:base,0,(0,0,0),(1,1,0.02)), part(:hour_marks,1,ticks),
            box(:spindle,2,(0.4975,0.4975,0),(0.005,0.005,0.062)),part(:hour_hand,3,hour),part(:minute_hand,4,minute)]
    elseif number == 6
        tower = tapered_surface([(-0.3,-0.3,0),(0.3,-0.3,0),(0.3,0.3,0),(-0.3,0.3,0)],
            [(-0.2,-0.2,2),(0.2,-0.2,2),(0.2,0.2,2),(-0.2,0.2,2)])
        return [part(:tower,0,tower),box(:axis,1,(-0.05,0,1.8),(0.1,-0.7,0.1)),
            box(:blade,3,(-1.4,-0.65,1.75),(2.8,0.1,0.2)),
            box(:rotating_block,4,(-1.7,-0.7,1.45),(0.8,0.05,0.8)),box(:orbiting_block,5,(0.9,-0.7,1.45),(0.8,0.05,0.8))]
    elseif number == 7
        tower = tapered_surface([(-0.2,-0.2,0),(0.2,-0.2,0),(0.2,0.2,0),(-0.2,0.2,0)],
            [(-0.1,-0.1,3),(0.1,-0.1,3),(0.1,0.1,3),(-0.1,0.1,3)])
        return [box(:base,0,(-1.5,-1.2,-0.01),(3,2,0.01)),part(:tower,1,tower),
            box(:axis,2,(-0.05,0,2.8),(0.1,-0.7,0.1)),box(:blade,4,(-0.1,-0.65,0.2),(0.2,0.1,2.75)),
            box(:block,5,(-0.35,-0.7,0.1),(0.7,0.05,0.4)),box(:shadow,6,(-0.35,-0.75,0),(0.7,0.15,0.02))]
    elseif number == 8
        return [box(:base,0,(0,0,0),(6.4,3.4,0.02)),part(:sphere,200,sphere_surface((0.2,3.2,2.22),0.2)),
            part(:shadow_xy,150,cylinder_surface((0.2,3.2,0.02),(0.2,3.2,0.03),0.12)),
            part(:shadow_xz,151,cylinder_surface((0.2,3.41,2.22),(0.2,3.4,2.22),0.2)),
            part(:shadow_yz,152,cylinder_surface((-0.01,3.2,2.22),(0,3.2,2.22),0.2))]
    else
        return [box(:base,0,(0,0,0),(2,0.1,0.02)),box(:paddle,1,(0,0.03,0.02),(0.99,0.04,0.02)),
            box(:hinge,2,(0.99,0.02,0.02),(0.02,0.06,0.02))]
    end
end
