module ParticleGenerator

"""
Fill Meshes.jl polygons with SPH particles.

One lattice of spacing `dx` is laid over all regions with `RegularSampling`
and every lattice point is handed to the *first* region that contains it.
Listing the walls before the fluid therefore guarantees that boundary and
fluid particles never overlap, without any pairwise distance checks.

Selected 2D regions can instead be sampled along their own shape
(`sampling = :conforming`): their particles lie on layers parallel to the
outline, so arcs and slanted walls are followed closely at the right spacing,
while every other region keeps the common lattice.
"""

export ParticleRegion, sample_particles, hydrostatic_density, write_particle_csv

using LinearAlgebra: norm
using Meshes
using StaticArrays

using ..SimulationGeometry: ParticleType, Fluid
using ..PolygonDrawing: ExtrudedPolygon, normalized_rings, layer_points,
                        shapes_signed_distance

const PolygonGeometry = Union{PolyArea, Multi}
const GeometryUnion   = Union{Tuple, AbstractVector}

"""
    ParticleRegion(name, geometry, type; include_surface = type != Fluid, offset = 0,
                   sampling = :lattice, layers = nothing)

A geometry to fill with particles of a given `ParticleType`: a `PolyArea` or a
`Multi` of `PolyArea`s in 2D, an `ExtrudedPolygon` (see `prism`) in 3D, or a
tuple or vector of such geometries filled as their union. With
`include_surface = true` lattice points on the outline belong to the region, so
walls own their surfaces. With `false` only the open interior is filled and the
particles stop one lattice spacing short of the outline, which is what a fluid
resting against a wall or a free surface needs.

`offset` shrinks the region by that distance: only lattice points at least
`offset` inside the outline are filled, so the particles keep this gap to the
original edges (a negative `offset` grows the region instead). The surface rule
then applies to the shrunk outline. For a union the distance is measured to the
nearest part, which is exact outside but may underestimate the depth near seams
where parts overlap.

`sampling = :conforming` (2D only) samples the region along its own shape
instead of on the lattice. The particles lie on layers parallel to the
outline (every ring, holes included), at depths `0, dx, 2dx, …` inside it, and
each layer is split evenly into steps of about `dx` with a particle on every
sharp corner. The first layer lies on the outline with `include_surface`
and one spacing inside without, both moved inwards by `offset`. `layers` limits
the number of layers; by default the region is filled to its core. A particle
closer than `dx / 2` to one already placed is dropped. See `sample_particles`
for how conforming and lattice regions share the domain.
"""
struct ParticleRegion{G}
    name::String
    geometry::G
    type::ParticleType
    include_surface::Bool
    offset::Float64
    sampling::Symbol
    layers::Union{Nothing, Int}
end

function ParticleRegion(name, geometry, type::ParticleType; include_surface = type != Fluid,
                        offset::Real = 0, sampling::Symbol = :lattice,
                        layers::Union{Nothing, Integer} = nothing)
    sampling in (:lattice, :conforming) ||
        throw(ArgumentError("sampling must be :lattice or :conforming, got $(repr(sampling))"))
    if layers !== nothing
        sampling === :conforming ||
            throw(ArgumentError("layers only applies to sampling = :conforming"))
        layers >= 1 || throw(ArgumentError("layers must be at least 1, got $layers"))
    end
    return ParticleRegion(String(name), geometry, type, include_surface, Float64(offset),
                          sampling, layers === nothing ? nothing : Int(layers))
end

"""
    sample_particles(regions, dx; tolerance = 1e-6 * dx)

Sample all `regions` with particle spacing `dx` and return a vector with one
entry `(; name, type, positions)` per region, in the order of `regions`.
`tolerance` is the distance within which a point counts as lying on an outline.

Lattice regions share one lattice and each lattice point is given to the first
region that contains it. A region with `sampling = :conforming` places its own
particles along its shape (see `ParticleRegion`) and claims the band they cover,
reaching `dx / 2` beyond its first and last layers. Later regions skip the
lattice points in that band, so they keep at least half a spacing from these
particles. Its own particles are dropped where an earlier region owns the
point or within `dx / 2` of an earlier particle. List conforming boundaries
before the fluid and the walls around them.
"""
function sample_particles(regions::AbstractVector{<:ParticleRegion}, dx::Real;
                          tolerance::Real = 1e-6 * dx)
    lattice = particle_lattice(regions, dx)
    claimed = falses(length(lattice))
    placed = PointHash{length(eltype(lattice))}(dx)
    territories = Function[]  # whether an earlier region owns a point

    return map(regions) do region
        positions, territory = if region.sampling === :conforming
            conforming_positions!(claimed, lattice, region, dx, tolerance, territories, placed)
        else
            lattice_positions!(claimed, lattice, region, tolerance),
            x -> owns(region, x, tolerance)
        end
        push!(territories, territory)
        foreach(x -> push!(placed, x), positions)
        (; region.name, region.type, positions)
    end
end

"""Unclaimed lattice points owned by `region`, which are marked as claimed."""
function lattice_positions!(claimed, lattice, region, tolerance)
    # Points further than this outside the bounding box cannot be owned,
    # which spares the distance computation for most of the lattice.
    margin = max(-region.offset, 0.0) + tolerance
    positions = eltype(lattice)[]
    for i in candidate_indices(claimed, lattice, region.geometry, margin)
        if owns(region, lattice[i], tolerance)
            claimed[i] = true
            push!(positions, lattice[i])
        end
    end
    return positions
end

"""Indices of unclaimed lattice points within `margin` of the bounding box of `geometry`."""
function candidate_indices(claimed, lattice, geometry, margin)
    lo, hi = geometry_bounds(geometry)
    lo, hi = lo .- margin, hi .+ margin
    return (i for (i, x) in enumerate(lattice) if !claimed[i] && all(lo .<= x .<= hi))
end

# Turns sharper than this make a vertex a corner that always gets a particle.
const CONFORMING_CORNER_ANGLE = deg2rad(30)

"""
    conforming_positions!(claimed, lattice, region, dx, tolerance, territories, placed)

Particles of a conforming `region` and the test for its territory. Particles in
the territory of an earlier region or within `dx / 2` of a particle already
placed are dropped. The lattice points the region's band covers are marked as
claimed.
"""
function conforming_positions!(claimed, lattice, region, dx, tolerance, territories, placed)
    geometry_dim(region.geometry) == 2 ||
        throw(ArgumentError("conforming sampling supports 2D regions only"))
    clearance = dx / 2
    first_depth = region.offset + (region.include_surface ? 0.0 : Float64(dx))
    shapes = [normalized_rings(p) for p in conforming_polygons(region.geometry)]
    candidates, last_depth = conforming_layers(shapes, region.geometry, dx, first_depth,
                                               region.layers)
    own = PointHash{2}(dx)
    positions = eltype(lattice)[]
    for x in candidates
        any(owned -> owned(x), territories) && continue
        (isnear(placed, x, clearance) || isnear(own, x, clearance)) && continue
        push!(own, x)
        push!(positions, x)
    end

    shallowest, deepest = first_depth - clearance, last_depth + clearance
    function territory(x)
        depth = shapes_signed_distance(shapes, x)
        return shallowest + tolerance < depth < deepest - tolerance
    end
    for i in candidate_indices(claimed, lattice, region.geometry, max(-shallowest, 0.0))
        territory(lattice[i]) && (claimed[i] = true)
    end
    return positions, territory
end

"""
    conforming_layers(shapes, geometry, dx, first_depth, layers)

Points on the layers `first_depth + k dx` (`k = 0, 1, …`) inside the union of
`shapes` (the rings of the polygons of `geometry`), at most `layers` of them,
and the depth of the last layer (`Inf` when the geometry is filled to its
core). Each layer is sampled with `layer_points`.
"""
function conforming_layers(shapes, geometry, dx, first_depth, layers)
    lo, hi = geometry_bounds(geometry)
    max_layers = something(layers, ceil(Int, (maximum(hi - lo) + abs(first_depth)) / dx) + 2)
    points = SVector{2, Float64}[]
    for k in 0:(max_layers - 1)
        layer = layer_points(shapes, first_depth + k * dx, dx, CONFORMING_CORNER_ANGLE)
        isempty(layer) && return points, Inf
        append!(points, (round.(x; sigdigits = 12) for x in layer))
    end
    return points, layers === nothing ? Inf : first_depth + (max_layers - 1) * dx
end

conforming_polygons(geometry::PolyArea)      = [geometry]
conforming_polygons(geometry::Multi)         = reduce(vcat, map(conforming_polygons, parent(geometry)))
conforming_polygons(geometry::GeometryUnion) = reduce(vcat, map(conforming_polygons, collect(geometry)))
conforming_polygons(geometry) = unsupported_geometry(geometry)

"""Points bucketed in square cells of `cell`, to find neighbours within one cell size."""
struct PointHash{D}
    cell::Float64
    buckets::Dict{NTuple{D, Int}, Vector{SVector{D, Float64}}}
end
PointHash{D}(cell) where {D} =
    PointHash{D}(Float64(cell), Dict{NTuple{D, Int}, Vector{SVector{D, Float64}}}())

bucket(hash::PointHash{D}, x) where {D} = ntuple(d -> floor(Int, x[d] / hash.cell), D)

Base.push!(hash::PointHash, x) = push!(get!(Vector{eltype(valtype(hash.buckets))},
                                            hash.buckets, bucket(hash, x)), x)

"""Whether a point of `hash` lies closer than `radius` (at most the cell size) to `x`."""
function isnear(hash::PointHash{D}, x, radius) where {D}
    centre = bucket(hash, x)
    for shift in CartesianIndices(ntuple(_ -> -1:1, D))
        points = get(hash.buckets, centre .+ Tuple(shift), nothing)
        points === nothing && continue
        any(p -> norm(p - x) < radius, points) && return true
    end
    return false
end

"""
    particle_lattice(regions, dx)

Lattice points of spacing `dx` covering every region, sampled with
`RegularSampling` over the common bounding box. The box is widened to whole
multiples of `dx` so that the lattice passes through the origin, and the
coordinates are rounded to remove the floating point noise of the sampling.
"""
function particle_lattice(regions, dx::Real)
    D = geometry_dim(first(regions).geometry)
    all(region -> geometry_dim(region.geometry) == D, regions) ||
        throw(ArgumentError("all particle regions must have the same dimension"))
    lo = fill(Inf, D)
    hi = fill(-Inf, D)
    for region in regions
        box_lo, box_hi = geometry_bounds(region.geometry)
        growth = max(-region.offset, 0.0)  # a negative offset extends past the outline
        lo .= min.(lo, box_lo .- growth)
        hi .= max.(hi, box_hi .+ growth)
    end
    lo = floor.(lo ./ dx) .* dx
    hi = ceil.(hi ./ dx) .* dx
    sizes = round.(Int, (hi .- lo) ./ dx) .+ 1

    box = Box(Point(lo...), Point(hi...))
    return [snap_to_lattice(plain_coordinates(p), dx)
            for p in sample(box, RegularSampling(sizes...))]
end

plain_coordinates(p::Point) = SVector(Meshes.ustrip.(to(p))...)

# Round to the nearest lattice node and drop binary noise (0.8600000000000001 → 0.86).
snap_to_lattice(x, dx) = round.(round.(x ./ dx) .* dx; sigdigits = 12)

"""
Whether `region` claims the lattice point `x`; see `ParticleRegion` for the
surface and offset rules. The test uses the signed distance to the outline
(positive inside), so `offset` shrinks or grows the region without rebuilding
its polygons.
"""
function owns(region::ParticleRegion, x, tolerance)
    signed_distance = geometry_signed_distance(region.geometry, x)
    return region.include_surface ? signed_distance >= region.offset - tolerance :
                                    signed_distance >  region.offset + tolerance
end

"""
    geometry_signed_distance(geometry, x)

Distance from the point `x` (plain coordinates) to the outline of `geometry`,
positive inside and negative outside.
"""
function geometry_signed_distance(geometry::PolygonGeometry, x)
    point = Point(x...)
    distance = minimum(outline_segments(geometry)) do segment
        Meshes.ustrip(Meshes.evaluate(Meshes.Euclidean(), point, segment))
    end
    return point in geometry ? distance : -distance
end

# A prism is the intersection of its base column and the slab bottom ≤ z ≤ top.
function geometry_signed_distance(geometry::ExtrudedPolygon, x)
    across = geometry_signed_distance(geometry.base, SVector(x[1], x[2]))
    along  = min(x[3] - geometry.bottom, geometry.top - x[3])
    across >= 0 && along >= 0 && return min(across, along)
    return -hypot(min(across, 0.0), min(along, 0.0))
end

geometry_signed_distance(geometry::GeometryUnion, x) =
    maximum(part -> geometry_signed_distance(part, x), geometry)

geometry_signed_distance(geometry, x) = unsupported_geometry(geometry)

outline_segments(geometry) = (s for ring in rings(geometry) for s in segments(ring))

"""Lower and upper corners of the bounding box of `geometry`, as plain coordinates."""
function geometry_bounds(geometry::PolygonGeometry)
    box = boundingbox(geometry)
    return plain_coordinates(minimum(box)), plain_coordinates(maximum(box))
end

function geometry_bounds(geometry::ExtrudedPolygon)
    lo, hi = geometry_bounds(geometry.base)
    return SVector(lo[1], lo[2], geometry.bottom), SVector(hi[1], hi[2], geometry.top)
end

function geometry_bounds(geometry::GeometryUnion)
    isempty(geometry) && throw(ArgumentError("a union of geometries cannot be empty"))
    bounds = map(geometry_bounds, collect(geometry))
    return reduce((a, b) -> min.(a, b), first.(bounds)),
           reduce((a, b) -> max.(a, b), last.(bounds))
end

geometry_bounds(geometry) = unsupported_geometry(geometry)

geometry_dim(geometry::PolygonGeometry) = embeddim(geometry)
geometry_dim(::ExtrudedPolygon)         = 3
function geometry_dim(geometry::GeometryUnion)
    isempty(geometry) && throw(ArgumentError("a union of geometries cannot be empty"))
    dims = unique(geometry_dim(part) for part in geometry)
    length(dims) == 1 ||
        throw(ArgumentError("a union cannot mix 2D and 3D geometries"))
    return only(dims)
end
geometry_dim(geometry) = unsupported_geometry(geometry)

unsupported_geometry(geometry) =
    throw(ArgumentError("particle regions must be a PolyArea, a Multi of PolyAreas, " *
                        "an ExtrudedPolygon or a tuple or vector of those; got " *
                        "$(typeof(geometry))"))

"""
    hydrostatic_density(positions, SimConstants; water_level = highest particle)

Initial densities for which the solver's equation of state returns the
hydrostatic pressure `P = ρ₀ g (water_level - z)`, so a fluid at rest starts
with the correct pressure profile. `z` is the vertical (last) coordinate, the
`y` of 2D positions. `ρ₀`, `g` and `c₀` are taken from `SimConstants`.

By default the water level is the height of the highest particle, the
convention of DualSPHysics that reproduces `input/still_wedge`. Particles above
`water_level` get `ρ₀`.
"""
function hydrostatic_density(positions, SimConstants; water_level = maximum(last, positions))
    (; ρ₀, g, c₀) = SimConstants
    # `Pressure!` uses `EquationOfStateGamma7`, so invert it with γ = 7 as well.
    # The exact root is used: `Estimate7thRoot` (inside
    # `InverseHydrostaticEquationOfState`) is ~1e-13 off even at P = 0.
    invCb = SimConstants.γ / (c₀^2 * ρ₀)
    return map(positions) do x
        depth = max(water_level - last(x), zero(water_level))
        ρ₀ * (1 + ρ₀ * g * depth * invCb)^(1 / SimConstants.γ)
    end
end

"""
    write_particle_csv(path, positions; density, first_id = 0)

Write particles in the CSV layout read by `SPHGeometry` (`Idp`, `Vel:0..2`,
`Rhop`, `Points:0..2`). 2D positions are stored as `(x, 0, y)` because the
loader reads `Points:0` and `Points:2` in 2D. `density` is a number or one
value per particle. Coordinates and densities are written as `Float64`, the
precision used by the CSV loader, so `Float32` inputs round-trip exactly when
promoted for double-position simulations. Returns the id following the last
particle written, so consecutive files can continue the numbering.
"""
function write_particle_csv(path::AbstractString, positions; density, first_id::Integer = 0)
    densities = density isa Number ? Iterators.repeated(density) : density
    header    = ["Idp", "Vel:0", "Vel:1", "Vel:2", "Rhop", "Points:0", "Points:1", "Points:2"]
    open(path, "w") do io
        println(io, join("\"" .* header .* "\"", ","))
        for (k, (x, ρ)) in enumerate(zip(positions, densities))
            xyz = length(x) == 2 ? (Float64(x[1]), 0.0, Float64(x[2])) :
                                   Float64.(x)
            println(io, first_id + k - 1, ",0,0,0,", Float64(ρ), ",",
                    join(xyz, ","))
        end
    end
    return first_id + length(positions)
end

end # module ParticleGenerator