using CSV
using LinearAlgebra: dot, norm

function falling_wedge_shapes(regions)
    tank = polyline([(-1.5,2.0),(-1.5,0.0),(1.5,0.0),(1.5,2.0)];thickness=0.015,side=:right)
    wedge = triangle((0.0,1.5),(0.25,1.64433),(-0.25,1.64433))
    outer,inner = map(regions) do r
        rectangle(Tuple(r.lower),r.upper[1]-r.lower[1],r.upper[2]-r.lower[2])
    end
    return (;tank,wedge,outer,inner)
end

function falling_wedge_ghost(point,spacing)
    vertices = (SVector(0.0,1.5),SVector(0.25,1.64433),SVector(-0.25,1.64433))
    projections,normals,distances = SVector{2,Float64}[],SVector{2,Float64}[],Float64[]
    for i in 1:3
        a,b = vertices[i],vertices[mod1(i+1,3)]
        edge = b-a
        projected = a+clamp(dot(point-a,edge)/dot(edge,edge),0.0,1.0)*edge
        distance = sum(abs2,point-projected)
        push!(projections,projected)
        push!(normals,SVector(edge[2],-edge[1])/norm(edge))
        push!(distances,distance)
    end
    best = minimum(distances)
    faces = findall(d -> abs(d-best) <= 1e-10*spacing^2,distances)
    if length(faces) == 2
        i,j = faces
        vertex = only(intersect((vertices[i],vertices[mod1(i+1,3)]),(vertices[j],vertices[mod1(j+1,3)])))
        offset = 0.5spacing*(normals[i]+normals[j])/(1+dot(normals[i],normals[j]))
        return 2(vertex+offset)-point
    end
    i = first(faces)
    return 2(projections[i]+0.5spacing*normals[i])-point
end

function falling_wedge_particles(spacing,kernel,region; finest=false)
    fixed = SVector{2,Float64}[]
    for i in floor(Int,-1.5/spacing)-3:ceil(Int,1.5/spacing)+3, j in -3:floor(Int,2.0/spacing)
        x,y = i*spacing,j*spacing
        y <= 0 || x <= -1.5+1e-10 || x >= 1.5-1e-10 || continue
        push!(fixed,SVector(x,y))
    end
    lower = region === nothing ? SVector(-1.5,0.0) : region.lower .- region.buffer_h*kernel.h
    upper = region === nothing ? SVector(1.5,1.5) : region.upper .+ region.buffer_h*kernel.h
    if region !== nothing
        filter!(x -> all(lower .- kernel.H .<= x) && all(x .<= upper .+ kernel.H),fixed)
    end
    ghosts = [SVector(x[1] <= -1.5+1e-10 ? -3+spacing-x[1] :
        x[1] >= 1.5-1e-10 ? 3-spacing-x[1] : x[1],x[2] <= 0 ? spacing-x[2] : x[2]) for x in fixed]
    water = [SVector(i*spacing,j*spacing)
        for i in max(ceil(Int,lower[1]/spacing),floor(Int,-1.5/spacing)+1):min(floor(Int,upper[1]/spacing),ceil(Int,1.5/spacing)-1)
        for j in max(1,ceil(Int,lower[2]/spacing)):min(floor(Int,upper[2]/spacing),ceil(Int,1.5/spacing)-1)]
    body = finest ? only(sample_particles([ParticleRegion("Wedge",
        triangle((0.0,1.5),(0.25,1.64433),(-0.25,1.64433)),Floating;
        sampling=:conforming,layers=4)],spacing)).positions : SVector{2,Float64}[]
    return (;fixed,ghosts,water,body)
end

function save_falling_wedge_geometry(case,result)
    rows = collect(CSV.File(joinpath(case.save_location,"FallingWedge2D_Floating.csv")))
    times = Float64[r.Time for r in rows]
    center = sum(case.geometry[end][end].Particles.Position)/length(case.geometry[end][end].Particles)
    transform = function(point,time)
        row = rows[searchsortedfirst(times,time)]
        s,c = sincos(row.Angle)
        x,y = point[1]-center[1],point[2]-center[2]
        SVector(c*x-s*y+getproperty(row,Symbol("Center:0")),s*x+c*y+getproperty(row,Symbol("Center:1")),point[3])
    end
    SavePolygonMotionSequence(joinpath(case.save_location,"FallingWedge2D.vtkhdf"),falling_wedge_shapes(case.regions);
        times,motions=(;wedge=transform,outer=transform,inner=transform),block_name="Geometry")
    return result
end
