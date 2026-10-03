module ParticleGenerator

"""
Fill Meshes.jl polygons with SPH particles.

One lattice of spacing `dx` is laid over all regions with `RegularSampling`
and every lattice point is handed to the *first* region that contains it.
Listing the walls before the fluid therefore guarantees that boundary and
fluid particles never overlap, without any pairwise distance checks.
"""

export ParticleRegion, sample_particles, hydrostatic_density, write_particle_csv

using Meshes
using StaticArrays

using ..SimulationGeometry: ParticleType, Fluid
using ..PolygonDrawing: ExtrudedPolygon

"""
    ParticleRegion(name, geometry, type; include_surface = type != Fluid, offset = 0)

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
"""
struct ParticleRegion{G}
    name::String
    geometry::G
    type::ParticleType
    include_surface::Bool
    offset::Float64
end

ParticleRegion(name, geometry, type::ParticleType; include_surface = type != Fluid,
               offset::Real = 0) =
    ParticleRegion(String(name), geometry, type, include_surface, Float64(offset))

"""
    sample_particles(regions, dx; tolerance = 1e-6 * dx)

Sample all `regions` on one lattice of spacing `dx` and return a vector with one
entry `(; name, type, positions)` per region, in the order of `regions`. Each
lattice point is given to the first region that contains it. `tolerance` is the
distance within which a point counts as lying on an outline.
"""
function sample_particles(regions::AbstractVector{<:ParticleRegion}, dx::Real;
                          tolerance::Real = 1e-6 * dx)
    lattice = particle_lattice(regions, dx)
    claimed = falses(length(lattice))

    return map(regions) do region
        # Points further than this outside the bounding box cannot be owned,
        # which spares the distance computation for most of the lattice.
        margin = max(-region.offset, 0.0) + tolerance
        lo, hi = geometry_bounds(region.geometry)
        lo, hi = lo .- margin, hi .+ margin
        positions = eltype(lattice)[]
        for (i, x) in enumerate(lattice)
            claimed[i] && continue
            all(lo .<= x .<= hi) || continue
            if owns(region, x, tolerance)
                claimed[i] = true
                push!(positions, x)
            end
        end
        (; region.name, region.type, positions)
    end
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

const PolygonGeometry = Union{PolyArea, Multi}
const GeometryUnion   = Union{Tuple, AbstractVector}

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
value per particle. Returns the id following the last particle written, so
consecutive files can continue the numbering.
"""
function write_particle_csv(path::AbstractString, positions; density, first_id::Integer = 0)
    densities = density isa Number ? Iterators.repeated(density) : density
    header    = ["Idp", "Vel:0", "Vel:1", "Vel:2", "Rhop", "Points:0", "Points:1", "Points:2"]
    open(path, "w") do io
        println(io, join("\"" .* header .* "\"", ","))
        for (k, (x, ρ)) in enumerate(zip(positions, densities))
            xyz = length(x) == 2 ? (x[1], 0.0, x[2]) : (x[1], x[2], x[3])
            println(io, first_id + k - 1, ",0,0,0,", ρ, ",", join(xyz, ","))
        end
    end
    return first_id + length(positions)
end

end # module ParticleGenerator