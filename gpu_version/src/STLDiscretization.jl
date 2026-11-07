module STLDiscretization

using StaticArrays, LinearAlgebra
export TriangleSurface, load_stl, signed_distance, sample_solid, packing_quality

const Point = SVector{3,Float64}
const Face = NTuple{3,Point}

struct MeshNode
    lower::Point
    upper::Point
    left::Int
    right::Int
    faces::Vector{Int}
end

"""Closed triangle surface with a bounding-volume hierarchy for distance and inside queries."""
struct TriangleSurface
    faces::Vector{Face}
    nodes::Vector{MeshNode}
end

function TriangleSurface(faces::AbstractVector; check_closed=true)
    triangles = Face[Tuple(Point.(f)) for f in faces]
    isempty(triangles) && throw(ArgumentError("surface contains no triangles"))
    edges = Dict{Tuple{Point,Point},Tuple{Int,Int}}()
    for (a,b,c) in triangles
        all(isfinite,a) && all(isfinite,b) && all(isfinite,c) || throw(ArgumentError("nonfinite mesh vertex"))
        norm(cross(b-a,c-a)) > 0 || throw(ArgumentError("degenerate mesh triangle"))
        for (u,v) in ((a,b),(b,c),(c,a))
            forward = isless(Tuple(u),Tuple(v))
            key = forward ? (u,v) : (v,u)
            count,balance = get(edges,key,(0,0))
            edges[key] = (count+1,balance+(forward ? 1 : -1))
        end
    end
    if check_closed
        bad = count(e -> e[1] != 2 || e[2] != 0,values(edges))
        bad == 0 || throw(ArgumentError("surface has $bad open, nonmanifold or inconsistently oriented edges; repair/weld the STL before volume sampling"))
    end
    nodes = MeshNode[]
    function build(indices)
        lo = reduce((a,b) -> min.(a,b),(v for i in indices for v in triangles[i]))
        hi = reduce((a,b) -> max.(a,b),(v for i in indices for v in triangles[i]))
        slot = length(nodes)+1
        push!(nodes,MeshNode(lo,hi,0,0,Int[]))
        if length(indices) <= 8
            nodes[slot] = MeshNode(lo,hi,0,0,indices)
        else
            axis = argmax(hi-lo)
            sort!(indices;by=i -> sum(v[axis] for v in triangles[i]))
            middle = length(indices)÷2
            left,right = build(indices[1:middle]),build(indices[middle+1:end])
            nodes[slot] = MeshNode(lo,hi,left,right,Int[])
        end
        return slot
    end
    build(collect(eachindex(triangles)))
    return TriangleSurface(triangles,nodes)
end

"""
    load_stl(path; scale=1, translation=(0,0,0), check_closed=true)

Read binary or ASCII STL in Float64. Scale and translate explicitly (STL has
no units). Closed, consistently oriented, nondegenerate triangles are required
by default; open or defective surfaces must be repaired before solid filling.
Normals in the STL are ignored and recomputed from the vertices.
"""
function load_stl(path;scale=1,translation=(0,0,0),check_closed=true)
    isfinite(scale) && scale > 0 || throw(ArgumentError("scale must be finite and positive"))
    offset = Point(translation)
    all(isfinite,offset) || throw(ArgumentError("translation must be finite"))
    faces = Face[]
    open(path,"r") do io
        binary = false
        if filesize(path) >= 84
            seek(io,80)
            count = ltoh(read(io,UInt32))
            binary = 84+50Int64(count) == filesize(path)
            if binary
                for _ in 1:count
                    read!(io,Vector{UInt8}(undef,12))
                    vertices = ntuple(3) do _
                        Point(ntuple(_ -> Float64(reinterpret(Float32,ltoh(read(io,UInt32)))),3))*scale+offset
                    end
                    read(io,UInt16)
                    push!(faces,vertices)
                end
            end
        end
        if !binary
            seekstart(io)
            vertices = Point[]
            for line in eachline(io)
                fields = split(strip(line))
                if !isempty(fields) && fields[1] == "vertex"
                    length(fields) == 4 || throw(ArgumentError("invalid STL vertex"))
                    push!(vertices,Point(parse.(Float64,fields[2:4]))*scale+offset)
                end
            end
            length(vertices)%3 == 0 || throw(ArgumentError("incomplete STL triangle"))
            append!(faces,(Tuple(vertices[i:i+2]) for i in 1:3:length(vertices)))
        end
    end
    return TriangleSurface(faces;check_closed)
end

# Closest point on a triangle (Voronoi regions, including edges and corners).
function closest_point(p,a,b,c)
    ab,ac,ap = b-a,c-a,p-a
    d1,d2 = dot(ab,ap),dot(ac,ap)
    d1 <= 0 && d2 <= 0 && return a
    bp = p-b
    d3,d4 = dot(ab,bp),dot(ac,bp)
    d3 >= 0 && d4 <= d3 && return b
    vc = d1*d4-d3*d2
    vc <= 0 && d1 >= 0 && d3 <= 0 && return a+(d1/(d1-d3))*ab
    cp = p-c
    d5,d6 = dot(ab,cp),dot(ac,cp)
    d6 >= 0 && d5 <= d6 && return c
    vb = d5*d2-d1*d6
    vb <= 0 && d2 >= 0 && d6 <= 0 && return a+(d2/(d2-d6))*ac
    va = d3*d6-d5*d4
    va <= 0 && d4-d3 >= 0 && d5-d6 >= 0 && return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
    denom = 1/(va+vb+vc)
    return a+(vb*denom)*ab+(vc*denom)*ac
end

box_distance2(p,node) = sum(abs2,max.(node.lower-p,0)+min.(node.upper-p,0))
function nearest_surface(mesh,p)
    best2,point = Inf,zero(Point)
    stack = [1]
    while !isempty(stack)
        node = mesh.nodes[pop!(stack)]
        box_distance2(p,node) > best2 && continue
        if node.left == 0
            for i in node.faces
                q = closest_point(p,mesh.faces[i]...)
                d2 = sum(abs2,p-q)
                if d2 < best2
                    best2,point = d2,q
                end
            end
        else
            left,right = node.left,node.right
            if box_distance2(p,mesh.nodes[left]) < box_distance2(p,mesh.nodes[right])
                push!(stack,right,left)
            else
                push!(stack,left,right)
            end
        end
    end
    return sqrt(best2),point
end

function ray_box(p,d,node)
    near,far = -Inf,Inf
    for k in 1:3
        a,b = (node.lower[k]-p[k])/d[k],(node.upper[k]-p[k])/d[k]
        near,far = max(near,min(a,b)),min(far,max(a,b))
    end
    return far >= max(near,0)
end

# Non-axis-aligned rays avoid systematic edge hits on lattice-aligned meshes.
# Shared-edge hits are merged before parity counting.
function inside_surface(mesh,p)
    d = normalize(Point(1,sqrt(2),sqrt(3)))
    hits = Float64[]
    stack = [1]
    while !isempty(stack)
        node = mesh.nodes[pop!(stack)]
        ray_box(p,d,node) || continue
        if node.left != 0
            push!(stack,node.left,node.right)
            continue
        end
        for i in node.faces
            a,b,c = mesh.faces[i]
            e1,e2 = b-a,c-a
            h = cross(d,e2)
            det = dot(e1,h)
            abs(det) <= 32eps(Float64)*norm(e1)*norm(e2) && continue
            s = p-a
            u = dot(s,h)/det
            -1e-12 <= u <= 1+1e-12 || continue
            q = cross(s,e1)
            v = dot(d,q)/det
            v >= -1e-12 && u+v <= 1+1e-12 || continue
            t = dot(e2,q)/det
            t > 0 && push!(hits,t)
        end
    end
    sort!(hits)
    unique_hits = 0
    last_hit = -Inf
    for t in hits
        if t-last_hit > 1e-10*max(1,abs(t))
            unique_hits += 1
            last_hit = t
        end
    end
    return isodd(unique_hits)
end

"""Signed distance to the triangle surface, negative inside and positive outside."""
function signed_distance(mesh::TriangleSurface,x)
    p = Point(x)
    distance,_ = nearest_surface(mesh,p)
    return inside_surface(mesh,p) ? -distance : distance
end

"""
    sample_solid(mesh, spacing; layers=nothing, iterations=30)

Seed a Cartesian volume and relax spacing with local repulsion constrained by
exact triangle distance. The outer particle centers stay at least `spacing/2`
inside the surface, placing the nominal solid interface half a spacing away.
`layers` retains an interior shell of that many spacings; `nothing` fills the
whole solid. The BVH accelerates surface queries and a spatial hash accelerates
relaxation. Returns positions and spacing/surface diagnostics. This is a
geometric relaxation, not an implementation of TrixiParticles' SPH packing ODE.
"""
function sample_solid(mesh::TriangleSurface,spacing;layers=nothing,iterations=30)
    dx = Float64(spacing)
    isfinite(dx) && dx > 0 || throw(ArgumentError("spacing must be finite and positive"))
    iterations isa Integer && iterations >= 0 || throw(ArgumentError("iterations must be a nonnegative integer"))
    layers === nothing || (layers isa Integer && layers >= 1) || throw(ArgumentError("layers must be a positive integer or nothing"))
    lo,hi = mesh.nodes[1].lower,mesh.nodes[1].upper
    inner = layers === nothing ? Inf : layers*dx
    positions = Point[]
    for z in lo[3]+dx/2:dx:hi[3], y in lo[2]+dx/2:dx:hi[2], x in lo[1]+dx/2:dx:hi[1]
        p = Point(x,y,z)
        depth = -signed_distance(mesh,p)
        if 0 <= depth <= inner
            if depth < dx/2
                _,surface = nearest_surface(mesh,p)
                direction = p-surface
                norm(direction) > 0 || continue
                p = surface+dx/2*normalize(direction)
                -signed_distance(mesh,p) >= dx/2-1e-8dx || continue
            end
            push!(positions,p)
        end
    end
    isempty(positions) && throw(ArgumentError("no solid particles at this spacing; refine the mesh sampling"))
    key(p) = Tuple(floor.(Int,p/dx))
    # Projection of different lattice sites can crowd a sharp edge. Remove
    # those collisions before relaxation rather than retaining near duplicates.
    accepted = Point[]
    seed_bins = Dict{NTuple{3,Int},Vector{Int}}()
    for p in positions
        cell = key(p)
        crowded = false
        for dz in -1:1,dy in -1:1,dxcell in -1:1
            for j in get(seed_bins,cell.+(dxcell,dy,dz),Int[])
                if norm(p-accepted[j]) < 0.75dx
                    crowded = true
                end
            end
        end
        if !crowded
            push!(accepted,p)
            push!(get!(seed_bins,cell,Int[]),length(accepted))
        end
    end
    positions = accepted
    for _ in 1:iterations
        bins = Dict{NTuple{3,Int},Vector{Int}}()
        for (i,p) in enumerate(positions)
            push!(get!(bins,key(p),Int[]),i)
        end
        updated = similar(positions)
        for (i,p) in enumerate(positions)
            force = zero(Point)
            cell = key(p)
            for dz in -1:1,dy in -1:1,dxcell in -1:1
                for j in get(bins,cell.+(dxcell,dy,dz),Int[])
                    i == j && continue
                    delta = p-positions[j]
                    r = norm(delta)
                    0 < r < dx && (force += (1-r/dx)*delta/r)
                end
            end
            displacement = 0.15dx*force
            norm(displacement) > 0.1dx && (displacement *= 0.1dx/norm(displacement))
            q = p+displacement
            for _ in 1:8
                distance,surface = nearest_surface(mesh,q)
                inside = inside_surface(mesh,q)
                depth = inside ? distance : -distance
                depth >= dx/2-1e-10dx && break
                direction = inside ? q-surface : surface-q
                norm(direction) > 0 || (direction = p-surface)
                norm(direction) > 0 || break
                q = surface+dx/2*normalize(direction)
            end
            new_depth = -signed_distance(mesh,q)
            updated[i] = dx/2-1e-8dx <= new_depth <= inner ? q : p
        end
        positions = updated
    end
    return (;positions,quality=packing_quality(mesh,positions,dx))
end

"""Report minimum center clearance, nearest-neighbor spacing and represented volume."""
function packing_quality(mesh,positions,spacing)
    dx = Float64(spacing)
    bins = Dict{NTuple{3,Int},Vector{Int}}()
    key(p) = Tuple(floor.(Int,p/dx))
    for (i,p) in enumerate(positions)
        push!(get!(bins,key(p),Int[]),i)
    end
    nearest = fill(Inf,length(positions))
    for (i,p) in enumerate(positions)
        cell = key(p)
        for dz in -2:2,dy in -2:2,dxcell in -2:2
            for j in get(bins,cell.+(dxcell,dy,dz),Int[])
                i == j || (nearest[i] = min(nearest[i],norm(p-positions[j])))
            end
        end
    end
    return (;particles=length(positions),represented_volume=length(positions)*dx^3,
        minimum_spacing=minimum(nearest;init=Inf),
        minimum_clearance=minimum(p -> -signed_distance(mesh,p),positions;init=Inf))
end
end
