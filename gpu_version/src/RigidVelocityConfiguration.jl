"""
    VelocitySeries(times, values)

Piecewise linear body velocity, sampled at the existing predictor/corrector
stage times. Equal times describe a right-continuous jump. `nothing` releases
that complete vector (or planar angular scalar) to dynamics. A released
interval stays free until the next finite key; the last key is held.
"""
struct VelocitySeries{V}
    times::Vector{Float64}
    values::Vector{Union{Nothing,V}}
end
function VelocitySeries(times,values)
    t = Float64.(collect(times))
    length(t) == length(values) && !isempty(t) && all(isfinite,t) && all(>=(0),t) && issorted(t) ||
        throw(ArgumentError("velocity keys need equally sized nonempty values and sorted finite nonnegative times"))
    finite = filter(!isnothing,collect(values))
    isempty(finite) && throw(ArgumentError("a velocity series needs at least one finite key"))
    scalar = first(finite) isa Real
    all(v -> (v isa Real) == scalar,finite) || throw(ArgumentError("velocity key shapes must match"))
    scalar || all(v -> length(v) == length(first(finite)),finite) || throw(ArgumentError("velocity vector dimensions must match"))
    V = scalar ? Float64 : SVector{length(first(finite)),Float64}
    result = Union{Nothing,V}[v === nothing ? nothing : V(v) for v in values]
    all(v -> v === nothing || all(isfinite,v),result) || throw(ArgumentError("velocity keys must be finite or nothing"))
    return VelocitySeries{V}(t,result)
end

"""
    RigidVelocity(group; Linear=nothing, Angular=nothing)

Prescribe complete world-space translation and/or angular velocity of a
native rigid body. Linear keys use native 2D/3D coordinates; angular keys
use the native planar scalar or world XYZ vector in 3D. Other degrees of
freedom retain fluid, contact and joint response. Put in
`RigidBodyDynamics(Velocities=(...,))`. No particle axis or external engine.
"""
struct RigidVelocity
    group::Int
    linear::Union{Nothing,VelocitySeries}
    angular::Union{Nothing,VelocitySeries}
end
function RigidVelocity(group::Integer;Linear=nothing,Angular=nothing)
    group > 0 && (Linear !== nothing || Angular !== nothing) || throw(ArgumentError("a rigid velocity needs a positive marker and a velocity series"))
    all(s -> s === nothing || s isa VelocitySeries,(Linear,Angular)) || throw(ArgumentError("Linear/Angular must be VelocitySeries"))
    return RigidVelocity(Int(group),Linear,Angular)
end

"""
    BodySpring(a,b; Attachment, Anchor, Stiffness, Damping=0, RestLength)

Axial spring/damper between two floating markers, with initial world-space
attachment points in native simulation coordinates. Applies equal/opposite
loads and attachment torques on CUDA through the existing floating stages.
Put in `RigidBodyDynamics(Springs=(...,))`.
"""
struct BodySpring{D}
    a::Int
    b::Int
    attachment::SVector{D,Float64}
    anchor::SVector{D,Float64}
    stiffness::Float64
    damping::Float64
    rest_length::Float64
end
function BodySpring(a::Integer,b::Integer;Attachment,Anchor,Stiffness,Damping=0,RestLength)
    a > 0 && b > 0 && a != b || throw(ArgumentError("a body spring needs two distinct positive markers"))
    D = length(Attachment)
    D in (2,3) && length(Anchor) == D || throw(ArgumentError("spring attachment dimensions must match (2 or 3)"))
    p,q = SVector{D,Float64}(Attachment),SVector{D,Float64}(Anchor)
    k,c,l = Float64.((Stiffness,Damping,RestLength))
    all(isfinite,(p...,q...,k,c,l)) && min(k,c,l) >= 0 || throw(ArgumentError("spring points must be finite and coefficients nonnegative"))
    return BodySpring(Int(a),Int(b),p,q,k,c,l)
end
