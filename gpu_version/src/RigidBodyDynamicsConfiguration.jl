# Included in Constraints. These host specifications are converted once into
# device joints, local attachment offsets and a body-disjoint graph colouring.
export BallJoint, HingeJoint, PointLineJoint, RigidContacts, RigidBodyDynamics

struct RigidJoint
    kind::UInt8
    body_a::Int
    body_b::Int
    attachment::SVector{3,Float64}
    anchor::SVector{3,Float64}
    axis::SVector{3,Float64}
end

function rigid_joint(kind, a, b, attachment, anchor, axis)
    a > 0 && b >= 0 && a != b || throw(ArgumentError("joint markers must be distinct, with body A positive and body B nonnegative; zero means the fixed world"))
    p, q, d = SVector{3,Float64}(attachment), SVector{3,Float64}(anchor), SVector{3,Float64}(axis)
    all(isfinite, p) && all(isfinite, q) && all(isfinite, d) || throw(ArgumentError("joint points and axes must be finite"))
    scale = maximum(abs, d)
    scale > 0 || throw(ArgumentError("joint axis must be nonzero"))
    d /= scale
    return RigidJoint(UInt8(kind), Int(a), Int(b), p, q, d / sqrt(dot(d, d)))
end

"""
    BallJoint(body_a, body_b = 0; Anchor)

Join two floating `GroupMarker`s at a common initial world-space `Anchor`.
Translation at that point is constrained; all relative rotation is free.
Marker zero denotes the fixed world. Solved on the GPU with `RigidBodyDynamics`.
"""
BallJoint(a::Integer, b::Integer = 0; Anchor) = rigid_joint(0, a, b, Anchor, Anchor, (0, 0, 1))

"""
    HingeJoint(body_a, body_b = 0; Anchor, Axis)

Join two floating markers at `Anchor`, permitting only relative rotation about
the initial world-space `Axis`. Unlike `FloatingConstraints.HingePoint`, both
bodies may move. Marker zero denotes the fixed world.
"""
HingeJoint(a::Integer, b::Integer = 0; Anchor, Axis) = rigid_joint(1, a, b, Anchor, Anchor, Axis)

"""
    PointLineJoint(body; Point, Direction, Attachment = Point)

Keep a body's initial world-space attachment on the infinite fixed line
`Point + s*Direction`. Translation along the line and all body rotation remain
free. An off-centre attachment transfers constraint torque to the body.
"""
PointLineJoint(a::Integer; Point, Direction, Attachment = Point) = rigid_joint(2, a, 0, Attachment, Point, Direction)

"""
    RigidContacts(; ParticleDiameter, Friction = 0.5, FixedFriction = Friction,
        Restitution = 0, FixedRestitution = Restitution, Slop = 0,
        Margin = ParticleDiameter, BounceThreshold = 0.01, FixedGroups = nothing)

Native particle-sampled rigid contacts. Distinct bodies and static `Fixed`
particles collide; same-body particles and fluid are excluded. The minimum
allowed centre distance is `ParticleDiameter - Slop`. Search candidates extend
by `Margin`, which must fit inside the SPH search support. Friction is Coulomb
friction (including sticking); pair friction/restitution use the smaller of
the two material coefficients. Fixed surfaces use the `Fixed*` coefficients.
`BounceThreshold` suppresses restitution below that incident normal speed.
`FixedGroups = nothing` enables all fixed groups; a tuple restricts contacts
to those fixed markers, allowing other fixed particles to be visual guides.

Particle contact represents the sampled geometry, not the original CAD mesh.
Use spacing/convergence checks when comparing with mesh-contact solvers.
"""
struct RigidContacts
    diameter::Float64
    friction::Float64
    fixed_friction::Float64
    restitution::Float64
    fixed_restitution::Float64
    slop::Float64
    margin::Float64
    bounce_threshold::Float64
    fixed_groups::Union{Nothing,Tuple}
end

function RigidContacts(; ParticleDiameter, Friction = 0.5, FixedFriction = Friction,
        Restitution = 0, FixedRestitution = Restitution, Slop = 0,
        Margin = ParticleDiameter, BounceThreshold = 0.01, FixedGroups = nothing)
    values = Float64.((ParticleDiameter, Friction, FixedFriction, Restitution, FixedRestitution, Slop, Margin, BounceThreshold))
    all(isfinite, values) || throw(ArgumentError("contact parameters must be finite"))
    d, f, ff, e, ef, s, m, b = values
    d > 0 && f >= 0 && ff >= 0 && 0 <= e <= 1 && 0 <= ef <= 1 && 0 <= s < d && m >= 0 && b >= 0 ||
        throw(ArgumentError("contact diameter must be positive, 0 <= slop < diameter, restitution in [0,1], and friction, margin and bounce threshold nonnegative"))
    fixed_groups = FixedGroups === nothing ? nothing : Tuple(Int.(FixedGroups))
    fixed_groups === nothing || all(>(0), fixed_groups) || throw(ArgumentError("FixedGroups must contain positive group markers"))
    return RigidContacts(values..., fixed_groups)
end

"""
    RigidBodyDynamics(; Joints = (), Contacts = nothing, Iterations = 32, MaxTimeStep = nothing)

Opt-in native GPU multibody dynamics for nonperiodic 3D simulations. Pass as
`SimRigidBodyDynamics` to `RunSimulation`. Coupled joints and contacts are
projected at both predictor and corrector stages; all iteration, contact
search and impulse application stay on CUDA, including graph replay.

Body-disjoint graph colours permit parallel joint/body-pair updates without
racing on poses or velocities. Contact manifolds retain up to 24 samples per
body pair (normal direction and tangent quadrant). No external rigid engine
is used. Individual `FloatingConstraints` and moving contact surfaces cannot
be combined with this mode. Native mode requires dense particle IDs 1:N.

For a case with no fluid, `MaxTimeStep` optionally selects a rigid-body time
controller instead of the SPH acoustic criterion. The CUDA controller caps
surface travel by `CFL*dx` (or the contact distance), limits acceleration and
centripetal travel, and retains the floating-spring stability cap. Validate
timestep and iteration convergence for the chosen ceiling. The default
`nothing` retains the usual SPH criterion, including in fluid-coupled runs.
"""
struct RigidBodyDynamics{J<:Tuple}
    joints::J
    contacts::Union{Nothing,RigidContacts}
    iterations::Int
    max_timestep::Union{Nothing,Float64}
end

function RigidBodyDynamics(; Joints = (), Contacts = nothing, Iterations::Integer = 32, MaxTimeStep = nothing)
    joints = Tuple(Joints)
    all(j -> j isa RigidJoint, joints) || throw(ArgumentError("Joints must contain native rigid joints"))
    Contacts === nothing || Contacts isa RigidContacts || throw(ArgumentError("Contacts must be RigidContacts or nothing"))
    0 < Iterations <= typemax(Int32) || throw(ArgumentError("Iterations must be positive and fit Int32"))
    ceiling = MaxTimeStep === nothing ? nothing : Float64(MaxTimeStep)
    ceiling === nothing || (isfinite(ceiling) && ceiling > 0) || throw(ArgumentError("MaxTimeStep must be finite and positive"))
    return RigidBodyDynamics(joints, Contacts, Int(Iterations), ceiling)
end
