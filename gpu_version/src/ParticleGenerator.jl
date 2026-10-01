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

"""
    ParticleRegion(name, geometry, type; include_surface = type != Fluid)

A `PolyArea` (or `Multi` of `PolyArea`s) to fill with particles of a given
`ParticleType`. With `include_surface = true` lattice points on the outline
belong to the region, so walls own their surfaces. With `false` only the open
interior is filled and the particles stop one lattice spacing short of the
outline, which is what a fluid resting against a wall or a free surface needs.
"""
struct ParticleRegion{G}
    name::String
    geometry::G
    type::ParticleType
    include_surface::Bool
end

ParticleRegion(name, geometry, type::ParticleType; include_surface = type != Fluid) =
    ParticleRegion(String(name), geometry, type, include_surface)

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
        positions = eltype(lattice)[]
        for (i, x) in enumerate(lattice)
            claimed[i] && continue
            if owns(region, Point(x...), tolerance)
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
    D = embeddim(first(regions).geometry)
    lo = fill(Inf, D)
    hi = fill(-Inf, D)
    for region in regions
        box = boundingbox(region.geometry)
        lo .= min.(lo, plain_coordinates(minimum(box)))
        hi .= max.(hi, plain_coordinates(maximum(box)))
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

"""Whether `region` claims `point`; see `ParticleRegion` for the surface rule."""
function owns(region::ParticleRegion, point::Point, tolerance)
    on_outline = any(outline_segments(region.geometry)) do segment
        Meshes.ustrip(Meshes.evaluate(Meshes.Euclidean(), point, segment)) <= tolerance
    end
    inside = point in region.geometry
    return region.include_surface ? (inside || on_outline) : (inside && !on_outline)
end

outline_segments(geometry) = (s for ring in rings(geometry) for s in segments(ring))

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
function hydrostatic_density(positions, SimConstants;
                             water_level = maximum(last, positions))
    (; ρ₀, g, c₀) = SimConstants
    # `Pressure!` uses `EquationOfStateGamma7`, so invert it with γ = 7 as well.
    # The exact root is used: `Estimate7thRoot` (inside
    # `InverseHydrostaticEquationOfState`) is ~1e-13 off even at P = 0.
    invCb = 7 / (c₀^2 * ρ₀)
    return map(positions) do x
        depth = max(water_level - last(x), zero(water_level))
        ρ₀ * (1 + ρ₀ * g * depth * invCb)^(1 / 7)
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