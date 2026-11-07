# Case-specific conversion of the principal hull from DualSPHysics 5.4.
# Run from gpu_version/: julia --project=. example/data/current_hull/convert_reference.jl path/to/Hull.vtk
using StaticArrays, SHA

function convert_reference(source, destination=joinpath(@__DIR__,"principal_hull.stl"))
    Point = SVector{3,Float64}
    faces = NTuple{3,Point}[]
    open(source,"r") do io
        header = [strip(readline(io)) for _ in 1:4]
        header[3:4] == ["BINARY","DATASET POLYDATA"] || error("expected binary legacy VTK POLYDATA")
        label,count,kind = split(readline(io))
        label == "POINTS" && kind == "float" || error("expected Float32 points")
        vertices = [Point(ntuple(_ -> Float64(reinterpret(Float32,ntoh(read(io,UInt32)))),3)) for _ in 1:parse(Int,count)]
        line = strip(readline(io))
        while isempty(line) && !eof(io)
            line = strip(readline(io))
        end
        label,count,_ = split(line)
        label == "POLYGONS" || error("expected polygons")
        for _ in 1:parse(Int,count)
            size = Int(ntoh(read(io,UInt32)))
            ids = [Int(ntoh(read(io,UInt32)))+1 for _ in 1:size]
            for j in 2:size-1
                push!(faces,(vertices[ids[1]],vertices[ids[j]],vertices[ids[j+1]]))
            end
        end
    end
    edge_key(a,b) = isless(Tuple(a),Tuple(b)) ? (a,b) : (b,a)
    edge_faces = Dict{Tuple{Point,Point},Vector{Int}}()
    for (i,(a,b,c)) in enumerate(faces), (u,v) in ((a,b),(b,c),(c,a))
        push!(get!(edge_faces,edge_key(u,v),Int[]),i)
    end
    adjacency = [Int[] for _ in faces]
    for indices in values(edge_faces), i in indices
        append!(adjacency[i],indices)
    end
    seen = falses(length(faces))
    components = Vector{Int}[]
    for i in eachindex(faces)
        seen[i] && continue
        stack,component = [i],Int[]
        seen[i] = true
        while !isempty(stack)
            j = pop!(stack)
            push!(component,j)
            for k in adjacency[j]
                if !seen[k]
                    seen[k] = true
                    push!(stack,k)
                end
            end
        end
        push!(components,component)
    end
    principal = components[argmax(length.(components))]
    length(principal) == 2440 || error("unexpected principal hull topology; review source before conversion")
    welded = Dict{NTuple{3,Int},Point}()
    # Choose a deterministic representative independently of component traversal.
    for i in principal, p in faces[i]
        key = Tuple(round.(Int,p/1e-6))
        previous = get(welded,key,p)
        welded[key] = isless(Tuple(previous),Tuple(p)) ? p : previous
    end
    weld(p) = welded[Tuple(round.(Int,p/1e-6))]
    hull = [Tuple(weld(p) for p in faces[i]) for i in sort(principal)]
    edge_owners = Dict{Tuple{Point,Point},Vector{Tuple{Point,Point}}}()
    for (a,b,c) in hull, (u,v) in ((a,b),(b,c),(c,a))
        push!(get!(edge_owners,edge_key(u,v),Tuple{Point,Point}[]),(u,v))
    end
    any(v -> length(v) > 2,values(edge_owners)) && error("principal hull has nonmanifold edges")
    boundary = sort!([only(owners) for owners in values(edge_owners) if length(owners) == 1];by=edge -> Tuple(edge[1]))
    all(edge -> min(edge[1][3],edge[2][3])+0.13 > 0.2,boundary) || error("refusing to cap an opening at or below the waterline")
    successors = Dict(boundary)
    length(successors) == length(boundary) && Set(keys(successors)) == Set(values(successors)) || error("boundary is not a simple directed loop")
    start = first(sort!(collect(keys(successors));by=Tuple))
    cursor,loop = start,Point[]
    while cursor ∉ loop
        push!(loop,cursor)
        cursor = successors[cursor]
    end
    cursor == start && length(loop) == length(boundary) || error("expected exactly one boundary loop")
    center = sum(loop)/length(loop)
    append!(hull,((b,a,center) for (a,b) in boundary))
    paired = Dict{Tuple{Point,Point},Vector{Tuple{Point,Point}}}()
    for (a,b,c) in hull, (u,v) in ((a,b),(b,c),(c,a))
        push!(get!(paired,edge_key(u,v),Tuple{Point,Point}[]),(u,v))
    end
    all(v -> length(v) == 2 && v[1] == reverse(v[2]),values(paired)) || error("repaired surface is not closed and consistently oriented")
    mkpath(dirname(destination))
    open(destination,"w") do io
        write(io,rpad("DualSPHysics 5.4 principal CurrentHull; dry opening capped",80))
        write(io,htol(UInt32(length(hull))))
        for face in hull
            write(io,zeros(UInt8,12))
            for p in face, x in p
                write(io,htol(reinterpret(UInt32,Float32(x))))
            end
            write(io,UInt16(0))
        end
    end
    lowest = minimum(p[3] for edge in boundary for p in edge)
    provenance = """
    {
      "source": $(repr(abspath(source))),
      "source_sha256": "$(bytes2hex(sha256(read(source))))",
      "source_triangles": $(length(faces)),
      "retained_triangles": $(length(principal)),
      "cap_triangles": $(length(boundary)),
      "weld_tolerance_m": 1e-6,
      "lowest_cap_z_m": $lowest,
      "omitted_triangles": $(length(faces)-length(principal))
    }
    """
    write(splitext(destination)[1]*".json",provenance)
    println(provenance)
    return destination
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("Usage: julia --project=. example/data/current_hull/convert_reference.jl path/to/Hull.vtk [output.stl]")
    convert_reference(ARGS[1],length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__,"principal_hull.stl"))
end

