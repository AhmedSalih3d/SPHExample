# CPU preprocessing diagnostics for the retained DualSPHysics principal hull.
# Run: julia --project=. -t 2,0 benchmark/boat_discretization.jl
using SPHExampleGPU, LinearAlgebra, Printf
using SPHExampleGPU.STLDiscretization: Point
mesh = load_stl(joinpath(@__DIR__,"..","example","data","current_hull","principal_hull.stl"))
# Translate the volume integral to the mesh origin to limit cancellation.
origin = mesh.nodes[1].lower
volume = abs(sum(dot(a-origin,cross(b-origin,c-origin)) for (a,b,c) in mesh.faces)/6)
println("Closed mesh volume (m^3): ",volume)
println("dx,particles,represented_volume,relative_volume_error,min_spacing_over_dx,min_clearance_over_dx,max_sampled_surface_gap_over_dx")
for dx in (0.025,0.02,0.01)
    packed = sample_solid(mesh,dx;iterations=30)
    bins = Dict{NTuple{3,Int},Vector{Point}}()
    key(p) = Tuple(floor.(Int,p/dx))
    for p in packed.positions
        push!(get!(bins,key(p),Point[]),p)
    end
    gap = 0.
    # Vertices and centroids catch coarse-sampling omissions. This sampled
    # maximum is not a rigorous Hausdorff-distance bound over every triangle.
    for (a,b,c) in mesh.faces, p in (a,b,c,(a+b+c)/3)
        cell = key(p)
        nearest = Inf
        for z in -3:3,y in -3:3,x in -3:3
            for q in get(bins,cell.+(x,y,z),Point[])
                nearest = min(nearest,norm(p-q))
            end
        end
        gap = max(gap,nearest)
    end
    q = packed.quality
    @printf("%.4f,%d,%.8f,%.6f,%.4f,%.4f,%.4f\n",dx,q.particles,q.represented_volume,
        (q.represented_volume-volume)/volume,q.minimum_spacing/dx,q.minimum_clearance/dx,gap/dx)
end
