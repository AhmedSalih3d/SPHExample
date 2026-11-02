# Geometry generation, wave theory and reference comparison helpers.
# User-facing input structs and the simulation driver are in the main example.
using CSV
using LinearAlgebra: normalize, norm, dot, cross

const POINT_ABSORBER_DATA = joinpath(@__DIR__, "..", "data", "vresolution_special")

point_absorber_bed(x) = max(0.0, x - 2.5) * 0.6 / 6.7

"""
    point_absorber_water(spacing, lower, upper; exclude_body=true)

Fill the tank with rows parallel to the bed, clipped at the still-water level.
The shear `(x, y) -> (x, y + bed(x))` preserves particle area `spacing^2`.
It keeps the first row one vertical spacing above the bed instead of cutting
horizontal rows into steps with a clearance varying between one and two
spacings. The flat part of the tank retains its Cartesian lattice.
"""
function point_absorber_water(spacing, lower, upper; exclude_body = true)
    depth = 0.4
    tolerance = 1e-10 * spacing
    water = SVector{2,Float64}[]
    for i in max(1, ceil(Int, lower[1] / spacing)):floor(Int, upper[1] / spacing)
        x = i * spacing
        bed = point_absorber_bed(x)
        for j in 1:(ceil(Int, depth / spacing) - 1)
            y = bed + j * spacing
            lower[2] - tolerance <= y <= upper[2] + tolerance && y < depth - tolerance || continue
            if exclude_body && 1.85 - 1e-10 <= x <= 2.15 + 1e-10 && 0.32 - 1e-10 <= y <= 0.48 + 1e-10
                continue
            end
            push!(water, SVector(x, y))
        end
    end
    return water
end

"Second-order regular piston motion, with the reference's smooth one-period ramp."
function point_absorber_piston_state(t; depth = 0.4, height = 0.1, period = 1.2, gravity = 9.81)
    omega = 2pi / period
    k = omega^2 / gravity
    for _ in 1:50
        k -= (gravity*k*tanh(k*depth)-omega^2)/(gravity*tanh(k*depth)+gravity*k*depth*sech(k*depth)^2)
    end
    transfer = 2sinh(k*depth)^2/(sinh(k*depth)*cosh(k*depth)+k*depth)
    a1 = height/(2transfer)
    a2 = height^2/(32depth)*(3cosh(k*depth)/sinh(k*depth)^3-2/transfer)
    s = clamp(t/period,0.0,1.0)
    ramp = s^2 * (6 - 8s + 3s^2)
    rate = 0 < t < period ? 12s * (1 - s)^2 / period : 0.0
    displacement = a1*sin(omega*t)+a2*sin(2omega*t)
    velocity = a1*omega*cos(omega*t)+2a2*omega*cos(2omega*t)
    return ramp * displacement, rate * displacement + ramp * velocity
end

function point_absorber_level(spacing, constants::SimulationConstants{T}, kernel, region, floating_details; fine) where T
    depth = 0.4
    if !fine && isapprox(spacing, 0.01; rtol = 1e-10)
        # Preserve the already generated coarse sloping skin; no GenCase call.
        fixed = [SVector(Float64(r.x), Float64(r.y))
            for r in CSV.File(joinpath(POINT_ABSORBER_DATA, "BoundarySection.csv"))]
    else
        fixed = SVector{2,Float64}[]
        for i in ceil(Int, -0.2 / spacing):floor(Int, 9.2 / spacing), layer in 0:2
            x = i * spacing
            normal = x <= 2.5 ? SVector(0.0, 1.0) : normalize(SVector(-0.6 / 6.7, 1.0))
            push!(fixed, SVector(x, point_absorber_bed(x)) - layer * spacing * normal)
        end
    end
    lo = fine ? region.lower .- region.buffer_h * kernel.h : SVector(0.0, 0.0)
    hi = fine ? region.upper .+ region.buffer_h * kernel.h : SVector(9.2, depth)
    water = point_absorber_water(spacing, lo, hi)
    boundary = SPHGeometry{2,T}(fixed;
        Density = constants.ρ₀, Type = Fixed, GroupMarker = 1)
    fluid = SPHGeometry{2,T}(water;
        Density = hydrostatic_density(water, constants; water_level = depth),
        Type = Fluid, GroupMarker = 2)
    geometry = [boundary, fluid]
    if fine
        nx,ny = round(Int,0.3/spacing),round(Int,0.16/spacing)
        isapprox(nx*spacing,0.3;atol=1e-10) && isapprox(ny*spacing,0.16;atol=1e-10) ||
            throw(ArgumentError("fine spacing must divide the body's 0.3 m width and 0.16 m height"))
        body = [SVector(1.85+i*spacing,0.32+j*spacing) for i in 0:nx for j in 0:ny]
        push!(geometry, SPHGeometry{2,T}(body;
            Density = constants.ρ₀, Type = Floating, GroupMarker = 50,
            Floating = floating_details))
    else
        nx,ny = round(Int,0.05/spacing),round(Int,0.6/spacing)
        piston = [SVector(-i*spacing,j*spacing) for i in 0:nx for j in 1:ny]
        push!(geometry, SPHGeometry{2,T}(piston;
            Density = constants.ρ₀, Type = Moving, GroupMarker = 10))
    end
    return (; geometry, constants, kernel)
end

function point_absorber_geometry(case, heave = 0.0)
    shift = SVector(0.0, heave)
    lower = case.region.lower + shift
    tank = polyline([(-0.2, 0.0), (2.5, 0.0), (9.2, 0.6)]; thickness = 0.02, side = :right)
    body = rectangle((1.85, 0.32 + heave), 0.3, 0.16)
    patch = rectangle(Tuple(lower), case.region.upper[1] - case.region.lower[1], case.region.upper[2] - case.region.lower[2])
    return (; tank, body, refinement = patch)
end

"A display-only helical wire with fixed turns and tapered ends at the two attachment points."
function point_absorber_spring_surface(spring;radius=0.02,turns=8,sections_per_turn=16,wire_radius=0.0015,wire_sides=8)
    all(x -> isfinite(x) && x > 0,(radius,wire_radius)) && wire_radius < radius ||
        throw(ArgumentError("spring display radii must be positive, with wire_radius < radius"))
    turns isa Integer && turns > 0 && sections_per_turn isa Integer && sections_per_turn >= 8 &&
        wire_sides isa Integer && wire_sides >= 3 || throw(ArgumentError("spring display needs positive integer turns, at least 8 sections per turn and 3 wire sides"))
    anchor = SVector(spring.Anchor[1],spring.Anchor[2],0.0)
    attachment = SVector(spring.Attachment[1],spring.Attachment[2],0.0)
    axis = attachment-anchor
    norm(axis) > 0 || throw(ArgumentError("spring display needs distinct endpoints"))
    direction = normalize(axis)
    radial = normalize(cross(direction,SVector(0.0,0.0,1.0)))
    transverse = cross(direction,radial)
    path = [anchor,anchor+0.08axis]
    for i in 1:turns*sections_per_turn-1
        s = i/(turns*sections_per_turn)
        envelope = min(1.0,2turns*s,2turns*(1-s))
        angle = 2pi*turns*s
        push!(path,anchor+(0.08+0.84s)*axis+radius*envelope*(cos(angle)*radial+sin(angle)*transverse))
    end
    append!(path,(anchor+0.92axis,attachment))
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
    for i in 1:length(path)-1, side in 1:wire_sides
        a,b = (i-1)*wire_sides+side,(i-1)*wire_sides+mod1(side+1,wire_sides)
        c,d = b+wire_sides,a+wire_sides
        # Triangles keep every exported face planar along the curved wire.
        push!(faces,[a,b,c],[a,c,d])
    end
    push!(faces,collect((length(path)-1)*wire_sides+1:length(path)*wire_sides))
    return PolygonSurface(points,faces)
end

function point_absorber_outputs(case,result;spring_display=(;))
    body_rows = collect(CSV.File(joinpath(case.save_location,"PointAbsorberSpring2D_Floating.csv")))
    spring_rows = collect(CSV.File(joinpath(case.save_location,"PointAbsorberSpring2D_Springs.csv")))
    times = Float64[r.Time for r in body_rows]
    Float64[r.Time for r in spring_rows] == times || throw(ArgumentError("spring and body frame times must match"))
    function interpolate(values,time)
        length(times) == 1 && return first(values)
        i = clamp(searchsortedlast(times,time),1,length(times)-1)
        weight = (time-times[i])/(times[i+1]-times[i])
        return (1-weight)*values[i]+weight*values[i+1]
    end
    native_y = Float64[getproperty(r,Symbol("Center:1")) for r in body_rows]
    native_v = Float64[getproperty(r,Symbol("Velocity:1")) for r in body_rows]
    native_f = Float64[getproperty(r,Symbol("Force:1")) for r in spring_rows]
    open(joinpath(case.save_location,"PointAbsorberSpring2D_Comparison.csv"),"w") do io
        println(io,"time,native_center_y,reference_center_y,native_velocity_y,reference_velocity_y,native_spring_force_y,reference_spring_force_y")
        for r in CSV.File(joinpath(POINT_ABSORBER_DATA,"FloatingReference.csv"))
            first(times) <= r.time <= last(times) || continue
            println(io,join((r.time,interpolate(native_y,r.time),r.center_y,interpolate(native_v,r.time),r.velocity_y,
                interpolate(native_f,r.time),r.spring_force_y),','))
        end
    end
    heave = native_y .- 0.4
    function displacement(time)
        index = searchsortedfirst(times, time)
        return heave[index]
    end
    translate_body(point, time) = point + SVector(0.0, displacement(time), 0.0)
    # Keep the anchor fixed while the upper endpoint follows the attachment.
    stretch_spring(point, time) = point + SVector(0.0,
        displacement(time) * (point[2] - case.spring.Anchor[2]) /
        (case.spring.Attachment[2] - case.spring.Anchor[2]), 0.0)
    SavePolygonMotionSequence(
        joinpath(case.save_location, "PointAbsorberSpring2D.vtkhdf"),
        point_absorber_geometry(case);
        times,
        motions = (; body = translate_body, refinement = translate_body),
        block_name = "Geometry",
    )
    spring_state(time) = spring_rows[searchsortedfirst(times,time)]
    force(time) = let row = spring_state(time)
        SVector(getproperty(row,Symbol("Force:0")),getproperty(row,Symbol("Force:1")),0.0)
    end
    endpoint(time,kind) = let row = spring_state(time)
        SVector(getproperty(row,Symbol("$kind:0")),getproperty(row,Symbol("$kind:1")),0.0)
    end
    axial_force(time) = -dot(force(time),normalize(endpoint(time,"Attachment")-endpoint(time,"Anchor")))
    SavePolygonMotionSequence(
        joinpath(case.save_location,"PointAbsorberSpring2D.vtkhdf"),
        (;spring=point_absorber_spring_surface(case.spring;spring_display...));
        times,
        motions = (;spring=stretch_spring),
        block_name = "Springs",
        cell_fields = (;
            Force = (_,time) -> force(time),
            ForceMagnitude = (_,time) -> norm(force(time)),
            AxialForce = (_,time) -> axial_force(time),
            ElasticAxialForce = (_,time) -> case.spring.Stiffness*spring_state(time).Extension,
            DampingAxialForce = (_,time) -> axial_force(time)-case.spring.Stiffness*spring_state(time).Extension,
            Extension = (_,time) -> spring_state(time).Extension,
            Anchor = (_,time) -> endpoint(time,"Anchor"),
            Attachment = (_,time) -> endpoint(time,"Attachment"),
        ),
    )
    return result
end
