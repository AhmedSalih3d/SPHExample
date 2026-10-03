module PolygonDrawing

"""
Draw the geometry of a particle case as Meshes.jl polygons.

Every 2D helper returns a `PolyArea` (or a `Multi` of them) whose outer ring is
counter clockwise and whose holes are clockwise, so the result can be passed
straight to `ParticleRegion`, combined in a `Multi` or written with
`SavePolygonVTKHDF`. Points can be given as tuples, vectors, `SVector`s or
Meshes `Point`s; lengths are plain numbers (metres) and angles are in radians
(use `deg2rad` for degrees).

* Filled shapes: `polygon`, `triangle`, `rectangle`, `square`,
  `regular_polygon`, `circle`.
* Walls with a thickness: `line`, `polyline`, `outline`. The `side` keyword
  places the wall relative to the drawn path along its normal, `offset` moves
  it further along that normal.
* `offset_polygon` grows or shrinks a shape, `arc` returns points on a circular
  arc for `polyline` and `polygon`, and `translate`, `rotate` and `mirror`
  move shapes.
* `prism` extrudes a 2D shape along `z` for 3D particle regions.

Corners of offset paths are mitred, so straight walls keep sharp corners;
corners sharper than `miter_limit` are bevelled on their convex side. A wall that
is too thick for its path (an inner offset that crosses itself or collapses)
throws an `ArgumentError` instead of returning an invalid polygon.
"""

export polygon, triangle, rectangle, square, regular_polygon, circle, arc,
       line, polyline, outline, offset_polygon,
       translate, rotate, mirror,
       ExtrudedPolygon, prism

using LinearAlgebra: dot, norm
using Meshes
using StaticArrays

const Vec2 = SVector{2, Float64}
const Ring = Vector{Vec2}

#  Points and rings

as_vec2(p::Point) = as_vec2(Meshes.ustrip.(to(p)))
function as_vec2(p)
    length(p) == 2 || throw(ArgumentError("expected a 2D point, got $(repr(p))"))
    x = Vec2(Float64(p[1]), Float64(p[2]))
    all(isfinite, x) ||
        throw(ArgumentError("point coordinates must be finite, got $(repr(p))"))
    return x
end

as_tuple(x::Vec2) = (x[1], x[2])

cross2(a, b)   = a[1] * b[2] - a[2] * b[1]
left_normal(u) = Vec2(-u[2], u[1])

function signed_area(ring::Ring)
    n = length(ring)
    return sum(cross2(ring[i], ring[mod1(i + 1, n)]) for i in 1:n) / 2
end

# Length scale of a set of rings, for relative tolerances.
function extent(rings)
    points = reduce(vcat, rings)
    lo = reduce((a, b) -> min.(a, b), points)
    hi = reduce((a, b) -> max.(a, b), points)
    return max(maximum(hi - lo), maximum(abs, hi), maximum(abs, lo), eps())
end

"""
Plain vertices of a point list with consecutive duplicates removed (and the
closing duplicate when `closed`), so that every edge has a direction.
"""
function clean_points(points; closed::Bool)
    raw = Vec2[as_vec2(p) for p in points]
    isempty(raw) && return raw
    tolerance = 1e-12 * extent([raw])
    cleaned = Vec2[raw[1]]
    for x in Iterators.drop(raw, 1)
        norm(x - cleaned[end]) > tolerance && push!(cleaned, x)
    end
    if closed && length(cleaned) > 1 && norm(cleaned[end] - cleaned[1]) <= tolerance
        pop!(cleaned)
    end
    return cleaned
end

shape_rings(shape::PolyArea) =
    Ring[Vec2[as_vec2(p) for p in vertices(r)] for r in rings(shape)]

"""
The first ring of `ring_list` made counter clockwise and the others (holes)
clockwise, so the left normal of every ring points into the shape.
"""
function normalized_rings(ring_list::Vector{Ring})
    outer = signed_area(ring_list[1]) < 0 ? reverse(ring_list[1]) : ring_list[1]
    holes = [signed_area(h) > 0 ? reverse(h) : h for h in ring_list[2:end]]
    return vcat([outer], holes)
end
normalized_rings(shape::PolyArea) = normalized_rings(shape_rings(shape))

"""`PolyArea` with outer ring `outer` and `holes`, oriented as Meshes expects."""
function make_polyarea(outer::Ring, holes = Ring[])
    oriented = normalized_rings(vcat([outer], holes))
    return PolyArea([as_tuple.(ring) for ring in oriented])
end

#  Validation

# Sign of the orientation of `c` relative to the line `a → b`, zero within `tolerance`.
function orientation(a, b, c, tolerance)
    o = cross2(b - a, c - a)
    return abs(o) <= tolerance ? 0 : Int(sign(o))
end

within_box(a, b, p, tol) = all(min.(a, b) .- tol .<= p .<= max.(a, b) .+ tol)

"""Whether the closed segments `a–b` and `c–d` touch or cross."""
function segments_meet(a, b, c, d, area_tolerance, length_tolerance)
    o1 = orientation(a, b, c, area_tolerance)
    o2 = orientation(a, b, d, area_tolerance)
    o3 = orientation(c, d, a, area_tolerance)
    o4 = orientation(c, d, b, area_tolerance)
    o1 * o2 < 0 && o3 * o4 < 0 && return true
    o1 == 0 && within_box(a, b, c, length_tolerance) && return true
    o2 == 0 && within_box(a, b, d, length_tolerance) && return true
    o3 == 0 && within_box(c, d, a, length_tolerance) && return true
    o4 == 0 && within_box(c, d, b, length_tolerance) && return true
    return false
end

"""
Throw an `ArgumentError` mentioning `what` unless every ring has at least
three vertices and a non zero area and no two edges of any rings meet, apart
from consecutive edges of the same ring at their shared vertex.
"""
function check_simple(rings::Vector{Ring}, what)
    scale = extent(rings)
    for ring in rings
        length(ring) >= 3 ||
            throw(ArgumentError("$what needs at least three distinct vertices"))
        abs(signed_area(ring)) > 1e-12 * scale^2 ||
            throw(ArgumentError("$what has zero area"))
    end

    edges = [(r, i) for (r, ring) in enumerate(rings) for i in eachindex(ring)]
    endpoints(r, i) = (rings[r][i], rings[r][mod1(i + 1, length(rings[r]))])
    adjacent(r, i, s, j) = r == s && (j == mod1(i + 1, length(rings[r])) ||
                                      i == mod1(j + 1, length(rings[r])))
    for k in eachindex(edges), l in (k + 1):lastindex(edges)
        (r, i), (s, j) = edges[k], edges[l]
        adjacent(r, i, s, j) && continue
        a, b = endpoints(r, i)
        c, d = endpoints(s, j)
        if segments_meet(a, b, c, d, 1e-12 * scale^2, 1e-12 * scale)
            throw(ArgumentError("$what intersects itself; for a wall, reduce the " *
                                "thickness or offset, or lengthen the short edges"))
        end
    end
    return nothing
end

function point_in_ring(p, ring::Ring)
    inside = false
    n = length(ring)
    for i in 1:n
        a, b = ring[i], ring[mod1(i + 1, n)]
        if (a[2] > p[2]) != (b[2] > p[2]) &&
           p[1] < a[1] + (p[2] - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
            inside = !inside
        end
    end
    return inside
end

#  Offsetting

unit(v) = v / norm(v)

"""
Offset vertices for the joint at `p` between the incoming direction `u1` and
the outgoing direction `u2`, moved by `distance` along the left normal.
"""
function join_points(p, u1, u2, distance, miter_limit)
    n1, n2 = left_normal(u1), left_normal(u2)
    denominator = 1 + dot(n1, n2)
    denominator > 1e-12 ||
        throw(ArgumentError("the path doubles back on itself at $(as_tuple(p))"))
    miter = (n1 + n2) / denominator              # norm = 1 / cos(turn / 2)
    convex_side = distance * cross2(u1, u2) < 0
    if convex_side && norm(miter) > miter_limit
        return (p + distance * n1, p + distance * n2)
    end
    return (p + distance * miter,)
end

"""
    offset_path(points, distance; closed, miter_limit)

Vertices of the path through `points` moved by `distance` along its left
normal (`distance < 0` moves to the right), with mitred joints. The ends of an
open path are moved along the normal of their edge. Throws an `ArgumentError`
when an offset edge reverses its direction, i.e. the distance is larger than
the inner corners around that edge allow.
"""
function offset_path(points::Ring, distance; closed::Bool, miter_limit)
    iszero(distance) && return copy(points)
    n = length(points)
    edge_count = closed ? n : n - 1
    directions = [unit(points[mod1(i + 1, n)] - points[i]) for i in 1:edge_count]
    joints = map(1:n) do i
        if !closed && i == 1
            (points[1] + distance * left_normal(directions[1]),)
        elseif !closed && i == n
            (points[n] + distance * left_normal(directions[end]),)
        else
            incoming = directions[mod1(i - 1, edge_count)]
            join_points(points[i], incoming, directions[i], distance, miter_limit)
        end
    end
    for i in 1:edge_count
        shifted_edge = first(joints[mod1(i + 1, n)]) - last(joints[i])
        if !(dot(shifted_edge, directions[i]) > 0)
            a, b = as_tuple(points[i]), as_tuple(points[mod1(i + 1, n)])
            throw(ArgumentError("offsetting by $distance reverses the edge from $a " *
                                "to $b; the wall is too thick or the offset too " *
                                "large for it"))
        end
    end
    return Vec2[p for joint in joints for p in joint]
end

"""
Closed ring offset by `distance` along its left normal, checked to keep its
orientation (it has not collapsed through itself).
"""
function offset_ring(ring::Ring, distance, miter_limit, what)
    shifted = offset_path(ring, distance; closed = true, miter_limit)
    if sign(signed_area(shifted)) != sign(signed_area(ring))
        throw(ArgumentError("$what collapses: the offset is larger than the shape allows"))
    end
    return shifted
end

"""
The area between two non crossing closed loops as a `PolyArea`, the larger
loop being the outer ring.
"""
function band_between(loop_a::Ring, loop_b::Ring, what)
    check_simple([loop_a, loop_b], what)
    outer, inner = abs(signed_area(loop_a)) >= abs(signed_area(loop_b)) ?
                   (loop_a, loop_b) : (loop_b, loop_a)
    return make_polyarea(outer, [inner])
end

"""
Normal distance range `(low, high)` covered by a wall of `thickness` drawn on
`side`, measured positive towards `positive_side`, then moved by `offset`.
"""
function normal_range(side::Symbol, thickness, offset, positive_side, negative_side)
    (isfinite(thickness) && thickness > 0) ||
        throw(ArgumentError("thickness must be finite and positive, got $thickness"))
    isfinite(offset) || throw(ArgumentError("offset must be finite, got $offset"))
    t = Float64(thickness)
    low, high = side === positive_side ? (0.0, t) :
                side === negative_side ? (-t, 0.0) :
                side === :center ? (-t / 2, t / 2) :
                throw(ArgumentError("side must be :$positive_side, :$negative_side " *
                                    "or :center, got $(repr(side))"))
    return low + offset, high + offset
end

check_miter_limit(miter_limit) = miter_limit >= 1 ||
    throw(ArgumentError("miter_limit must be at least 1, got $miter_limit"))

#  Filled shapes

as_ring(shape::PolyArea) = first(shape_rings(shape))
as_ring(points) = clean_points(points; closed = true)

"""
    polygon(vertices; holes = ())

Arbitrary simple polygon through `vertices`, in either orientation; a repeated
closing vertex is ignored. `holes` are vertex lists or `PolyArea`s (their
outer ring is used) lying strictly inside. When `vertices` is a `PolyArea` its
own holes are kept and `holes` are added to them.

Throws an `ArgumentError` for fewer than three vertices, zero area, crossing
edges or holes outside the polygon.

```julia
water = polygon([(0, 0), (2, 0), (2, 1), (0, 1)]; holes = [circle((1, 0.5), 0.2)])
```
"""
function polygon(outline_points; holes = ())
    outer, existing = if outline_points isa PolyArea
        all_rings = shape_rings(outline_points)
        first(all_rings), all_rings[2:end]
    else
        as_ring(outline_points), Ring[]
    end
    inner = vcat(existing, Ring[as_ring(h) for h in holes])
    all_rings = vcat([outer], inner)
    check_simple(all_rings, "the polygon")
    for (k, hole) in enumerate(inner)
        point_in_ring(hole[1], outer) ||
            throw(ArgumentError("hole $k lies outside the polygon"))
        for (l, other) in enumerate(inner)
            l != k && point_in_ring(hole[1], other) &&
                throw(ArgumentError("hole $k lies inside hole $l"))
        end
    end
    return make_polyarea(outer, inner)
end

"""
    triangle(a, b, c)

Filled triangle with corners `a`, `b` and `c`, in any order.
"""
triangle(a, b, c) = polygon((a, b, c))

check_length(name, value) = (isfinite(value) && value > 0) ||
    throw(ArgumentError("$name must be finite and positive, got $value"))

"""
    rectangle(corner, width, height; angle = 0, centered = false)

Filled `width × height` rectangle whose lower left corner is `corner`, or whose
centre is `corner` with `centered = true`. The rectangle is turned
counter clockwise by `angle` (radians) about that point.
"""
function rectangle(corner, width, height; angle::Real = 0, centered::Bool = false)
    check_length("width", width)
    check_length("height", height)
    anchor = as_vec2(corner)
    lower_left = centered ? anchor - Vec2(width, height) / 2 : anchor
    corners = [lower_left, lower_left + Vec2(width, 0), lower_left + Vec2(width, height),
               lower_left + Vec2(0, height)]
    return polygon(rotate_points(corners, angle, anchor))
end

"""
    square(corner, side; angle = 0, centered = false)

Filled square of edge `side`; see `rectangle` for the placement keywords.
"""
square(corner, side; kwargs...) = rectangle(corner, side, side; kwargs...)

"""
    regular_polygon(center, radius, sides; angle = 0)

Filled regular polygon with `sides` corners on the circle of `radius` around
`center`, the first at `angle` (radians) from the `x` axis.
"""
function regular_polygon(center, radius, sides::Integer; angle::Real = 0)
    sides >= 3 ||
        throw(ArgumentError("a regular polygon needs at least 3 sides, got $sides"))
    return polygon(arc_points(as_vec2(center), radius, angle, 2π, sides)[1:sides])
end

"""
    circle(center, radius; segments = 128)

Filled circle drawn as a regular polygon of `segments` edges with its vertices
on the circle. The first vertex lies on the `+x` axis, so with `segments`
divisible by 4 the four extreme points are exact. The polygon lies inside the
true circle by at most `radius * (1 - cos(π / segments))` (`3e-4 radius` for the
default), so raise `segments` when lattice points on the circle must be kept.
"""
circle(center, radius; segments::Integer = 128) = regular_polygon(center, radius, segments)

"""
    arc(center, radius, start_angle, stop_angle; segments)

Points along a circular arc from `start_angle` to `stop_angle` (radians,
counter clockwise when `stop_angle > start_angle`), both ends included. The
default `segments` matches `circle`'s 128 per full turn. The points are
returned as tuples to build paths for `polyline` or outlines for `polygon`:

```julia
baffle = polyline(arc((1, 0), 0.5, 0, π); thickness = 0.04)
dome   = polygon(arc((1, 0), 0.5, 0, π))
```
"""
function arc(center, radius, start_angle::Real, stop_angle::Real;
             segments::Integer = max(1, ceil(Int, 64 * abs(stop_angle - start_angle) / π)))
    sweep = stop_angle - start_angle
    (isfinite(sweep) && 0 < abs(sweep) <= 2π + eps(2π)) ||
        throw(ArgumentError("the arc must sweep a non zero angle of at most 2π"))
    segments >= 1 || throw(ArgumentError("an arc needs at least one segment"))
    return as_tuple.(arc_points(as_vec2(center), radius, start_angle, sweep, segments))
end

function arc_points(center::Vec2, radius, start_angle, sweep, segments)
    check_length("radius", radius)
    return map(0:segments) do k
        # `sincospi` keeps multiples of π/2 exact (cos(π/2) is not 0 in Float64).
        s, c = sincospi(start_angle / π + (sweep / π) * k / segments)
        center + radius * Vec2(c, s)
    end
end

#  Walls with a thickness

"""
    polyline(points; thickness, side = :center, offset = 0, closed = false,
             miter_limit = 4)

Wall of `thickness` along the path through `points`, as a `PolyArea`.

`side` places the wall relative to the path, looking along the drawing
direction: `:left`, `:right` or `:center`. `offset` then moves the wall by that
distance along the left normal (negative: to the right), leaving a gap between
the path and the wall. Joints are mitred; joints whose miter would reach more
than `miter_limit` times the offset distance are bevelled. Open paths end
square to their first and last edges. With `closed = true` the path returns to
its first point and the wall is a ring.

```julia
# Open tank whose inner faces are the path: the walls grow outwards (right).
tank = polyline([(0, 1), (0, 0), (2, 0), (2, 1)]; thickness = 0.06, side = :right)
```
"""
function polyline(points; thickness::Real, side::Symbol = :center, offset::Real = 0,
                  closed::Bool = false, miter_limit::Real = 4)
    check_miter_limit(miter_limit)
    path = clean_points(points; closed)
    length(path) >= (closed ? 3 : 2) ||
        throw(ArgumentError("a $(closed ? "closed" : "open") path needs at least " *
                            "$(closed ? 3 : 2) distinct points"))
    low, high = normal_range(side, thickness, offset, :left, :right)
    if closed
        check_simple([path], "the closed path")
        loops = (offset_ring(path, low, miter_limit, "the wall"),
                 offset_ring(path, high, miter_limit, "the wall"))
        return band_between(loops..., "the wall")
    end
    right = offset_path(path, low; closed = false, miter_limit)
    left  = offset_path(path, high; closed = false, miter_limit)
    band  = vcat(right, reverse(left))
    check_simple([band], "the wall")
    return make_polyarea(band)
end

"""
    line(a, b; thickness, side = :center, offset = 0)

Straight wall of `thickness` from `a` to `b`; `side` (`:left`, `:right` or
`:center`, looking from `a` to `b`) and `offset` work as for `polyline`.
"""
line(a, b; kwargs...) = polyline((a, b); kwargs...)

"""
    outline(shape; thickness, side = :outward, offset = 0, miter_limit = 4)

Wall of `thickness` along the outline of a filled `shape` (a `PolyArea`, a
`Multi` of them or a vertex list).

`side = :outward` (default) builds the wall outside the shape, so the shape
itself stays free, e.g. for the fluid inside a tank; `:inward` builds it inside
the shape and `:center` straddles the outline. `offset` moves the wall away
from the shape (negative: into it). Every ring of the shape gets a wall; for a
hole, "outward" points into the hole. Returns a `PolyArea`, or a `Multi` when
the shape has several rings.

```julia
interior = rectangle((0, 0), 2, 1)
tank     = outline(interior; thickness = 0.04)                 # closed tank
pipe     = outline(circle((1, 0.5), 0.2); thickness = 0.02, side = :inward)
```
"""
function outline(shape::PolyArea; thickness::Real, side::Symbol = :outward,
                 offset::Real = 0, miter_limit::Real = 4)
    check_miter_limit(miter_limit)
    low, high = normal_range(side, thickness, offset, :outward, :inward)
    # The left normal of every normalized ring points into the shape, so
    # outward distances are negated.
    walls = map(normalized_rings(shape)) do ring
        loops = (offset_ring(ring, -low, miter_limit, "the wall"),
                 offset_ring(ring, -high, miter_limit, "the wall"))
        band_between(loops..., "the wall")
    end
    return length(walls) == 1 ? only(walls) : Multi(walls)
end
outline(shape::Multi; kwargs...) =
    Multi(reduce(vcat, collect_polygons(outline(p; kwargs...)) for p in parent(shape)))
outline(points; kwargs...) = outline(polygon(points); kwargs...)

collect_polygons(shape::PolyArea) = [shape]
collect_polygons(shape::Multi)    = collect(parent(shape))

"""
    offset_polygon(shape, distance; miter_limit = 4)

`shape` grown by `distance` along the outward normal of every edge (shrunk for
`distance < 0`), with mitred corners. Holes shrink as the shape grows. Throws an
`ArgumentError` when the shape would collapse or its rings would cross.
"""
function offset_polygon(shape::PolyArea, distance::Real; miter_limit::Real = 4)
    check_miter_limit(miter_limit)
    isfinite(distance) || throw(ArgumentError("distance must be finite, got $distance"))
    shifted = [offset_ring(ring, -distance, miter_limit, "the offset polygon")
               for ring in normalized_rings(shape)]
    check_simple(shifted, "the offset polygon")
    return make_polyarea(shifted[1], shifted[2:end])
end
offset_polygon(shape::Multi, distance::Real; kwargs...) =
    Multi([offset_polygon(p, distance; kwargs...) for p in parent(shape)])
offset_polygon(points, distance::Real; kwargs...) =
    offset_polygon(polygon(points), distance; kwargs...)

#  Transformations

function rotate_points(points, angle, origin::Vec2)
    s, c = sincospi(angle / π)                   # exact for quarter turns
    return [origin + Vec2(c * d[1] - s * d[2], s * d[1] + c * d[2])
            for d in (p - origin for p in points)]
end

function reflect_points(points, origin::Vec2, direction::Vec2)
    u = unit(direction)
    return [origin + 2 * dot(p - origin, u) * u - (p - origin) for p in points]
end

function map_shape(f, shape::PolyArea)
    mapped = map(f, shape_rings(shape))
    return make_polyarea(mapped[1], mapped[2:end])
end
map_shape(f, shape::Multi)    = Multi([map_shape(f, p) for p in parent(shape)])
map_shape(f, points)          = as_tuple.(f(Vec2[as_vec2(p) for p in points]))

"""
    translate(shape, displacement)

`shape` (a `PolyArea`, `Multi`, vertex list or `ExtrudedPolygon`) moved by
`displacement`; give three components to also move a prism along `z`.
"""
function translate(shape, displacement)
    d = as_vec2(displacement)
    return map_shape(points -> [p + d for p in points], shape)
end

"""
    rotate(shape, angle; origin = (0, 0))

`shape` turned counter clockwise by `angle` (radians) about `origin`; prisms
turn about the vertical axis through `origin`.
"""
rotate(shape, angle::Real; origin = (0.0, 0.0)) =
    map_shape(points -> rotate_points(points, angle, as_vec2(origin)), shape)

"""
    mirror(shape; origin = (0, 0), direction = (0, 1))

`shape` reflected across the line through `origin` along `direction`; the
default mirrors `x → -x`. Orientation is restored, so the result is valid.
"""
function mirror(shape; origin = (0.0, 0.0), direction = (0.0, 1.0))
    u = as_vec2(direction)
    norm(u) > 0 || throw(ArgumentError("the mirror direction must be non zero"))
    return map_shape(points -> reflect_points(points, as_vec2(origin), u), shape)
end

#  Prisms

"""
    ExtrudedPolygon(base, bottom, top)

The 2D `base` (a `PolyArea` or `Multi`) extruded along `z` from `bottom` to
`top`; build it with `prism`. `ParticleRegion` and `sample_particles` fill it
on a 3D lattice and `SavePolygonVTKHDF` writes its surface.
"""
struct ExtrudedPolygon{G}
    base::G
    bottom::Float64
    top::Float64
end

"""
    prism(base, bottom, top)

3D prism with cross section `base` (a `PolyArea`, `Multi` or vertex list)
between the heights `bottom` and `top`. A rectangle gives a box, a circle a
cylinder and an `outline` a wall. Combine prisms in a tuple or vector to fill
them as one `ParticleRegion`:

```julia
interior = rectangle((0, 0), 1.0, 0.5)
tank = (prism(outline(interior; thickness = 0.04), -0.04, 0.6),      # walls
        prism(offset_polygon(interior, 0.04), -0.04, 0.0))           # floor
```
"""
function prism(base::Union{PolyArea, Multi}, bottom::Real, top::Real)
    embeddim(base) == 2 || throw(ArgumentError("the prism base must be 2D"))
    (isfinite(bottom) && isfinite(top) && bottom < top) ||
        throw(ArgumentError("prism heights must be finite with bottom < top, " *
                            "got $bottom and $top"))
    return ExtrudedPolygon(base, Float64(bottom), Float64(top))
end
prism(points, bottom::Real, top::Real) = prism(polygon(points), bottom, top)

function translate(shape::ExtrudedPolygon, displacement)
    length(displacement) in (2, 3) ||
        throw(ArgumentError("a prism moves by a 2D or 3D displacement"))
    dz = length(displacement) == 3 ? Float64(displacement[3]) : 0.0
    base = translate(shape.base, (displacement[1], displacement[2]))
    return prism(base, shape.bottom + dz, shape.top + dz)
end
map_shape(f, shape::ExtrudedPolygon) =
    prism(map_shape(f, shape.base), shape.bottom, shape.top)

end # module PolygonDrawing
