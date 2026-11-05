# Included in Constraints. These host specifications are converted once into
# device joints, local attachment offsets and a body-disjoint graph colouring.
export BallJoint, HingeJoint, PointLineJoint, RigidMaterial, RigidContacts, RigidBodyDynamics

struct RigidJoint
    kind::UInt8
    body_a::Int
    body_b::Int
    attachment::SVector{3,Float64}
    anchor::SVector{3,Float64}
    axis::SVector{3,Float64}
    stiffness::Float64
    damping::Float64
    rest_angle::Float64
end

function rigid_joint(kind, a, b, attachment, anchor, axis)
    a > 0 && b >= 0 && a != b || throw(ArgumentError("joint markers must be distinct, with body A positive and body B nonnegative; zero means the fixed world"))
    p, q, d = SVector{3,Float64}(attachment), SVector{3,Float64}(anchor), SVector{3,Float64}(axis)
    all(isfinite, p) && all(isfinite, q) && all(isfinite, d) || throw(ArgumentError("joint points and axes must be finite"))
    scale = maximum(abs, d)
    scale > 0 || throw(ArgumentError("joint axis must be nonzero"))
    d /= scale
    return RigidJoint(UInt8(kind), Int(a), Int(b), p, q, d / sqrt(dot(d, d)),0.,0.,0.)
end

"""
    BallJoint(body_a, body_b = 0; Anchor)

Join two floating `GroupMarker`s at a common initial world-space `Anchor`.
Translation at that point is constrained; all relative rotation is free.
Marker zero denotes the fixed world. Solved on the GPU with `RigidBodyDynamics`.
"""
BallJoint(a::Integer, b::Integer = 0; Anchor) = rigid_joint(0, a, b, Anchor, Anchor, (0, 0, 1))

"""
    HingeJoint(body_a, body_b = 0; Anchor, Axis, Stiffness = 0, Damping = 0, RestAngle = 0)

Join two floating markers at `Anchor`, permitting only relative rotation about
the initial world-space `Axis`. Unlike `FloatingConstraints.HingePoint`, both
bodies may move. Marker zero denotes the fixed world. Optional torsional
stiffness (Nm/rad) and damping (Nms/rad) apply equal and opposite torques
through the existing floating integrator. `RestAngle` is the signed relative
rotation in radians about `Axis` from the initial pose. In 3D the elastic
angle uses the principal twist, so elastic motion must remain within pi radians.
The spring/damping stability bound also limits the existing SPH timestep.
"""
function HingeJoint(a::Integer, b::Integer = 0; Anchor, Axis, Stiffness = 0, Damping = 0, RestAngle = 0)
    joint = rigid_joint(1,a,b,Anchor,Anchor,Axis)
    k,c,theta = Float64.((Stiffness,Damping,RestAngle))
    isfinite(k) && k >= 0 && isfinite(c) && c >= 0 && isfinite(theta) ||
        throw(ArgumentError("hinge stiffness/damping must be finite and nonnegative, and RestAngle finite"))
    return RigidJoint(joint.kind,joint.body_a,joint.body_b,joint.attachment,joint.anchor,joint.axis,k,c,theta)
end

"""
    PointLineJoint(body; Point, Direction, Attachment = Point, LockRotation = false)

Keep a body's initial world-space attachment on the infinite fixed line
`Point + s*Direction`. Translation along the line and all body rotation remain
free. An off-centre attachment transfers constraint torque to the body.
`LockRotation = true` also fixes the initial planar angle, making a 2D
prismatic/heave guide. Rotation locking currently requires a 2D simulation.
"""
PointLineJoint(a::Integer; Point, Direction, Attachment = Point,LockRotation::Bool = false) =
    rigid_joint(LockRotation ? 3 : 2,a,0,Attachment,Point,Direction)

"""
    RigidMaterial(; Friction = 0.5, Restitution = 0)

Contact coefficients for a floating group in `RigidContacts.Materials`.
Pair coefficients use the smaller value from the two contacting materials.
"""
struct RigidMaterial
    friction::Float64
    restitution::Float64
    function RigidMaterial(; Friction = 0.5, Restitution = 0)
        f, e = Float64(Friction), Float64(Restitution)
        isfinite(f) && f >= 0 && isfinite(e) && 0 <= e <= 1 ||
            throw(ArgumentError("rigid material requires finite friction >= 0 and restitution in [0,1]"))
        new(f,e)
    end
end

"""
    RigidContacts(; ParticleDiameter, Friction = 0.5, FixedFriction = Friction,
        Restitution = 0, FixedRestitution = Restitution, Slop = 0,
        Margin = ParticleDiameter, BounceThreshold = 0.01, FixedGroups = nothing,
        FloatingGroups = nothing, Materials = ())

Native particle-sampled rigid contacts. Distinct bodies and static `Fixed`
particles collide; same-body particles and fluid are excluded. The minimum
allowed centre distance is `ParticleDiameter - Slop`. Search candidates extend
by `Margin`, which must fit inside the SPH search support. Friction is Coulomb
friction (including sticking); pair friction/restitution use the smaller of
the two material coefficients. Fixed surfaces use the `Fixed*` coefficients.
`BounceThreshold` suppresses restitution below that incident normal speed.
`FixedGroups = nothing` enables all fixed groups; a tuple restricts contacts
to those fixed markers, allowing other fixed particles to be visual guides.
`FloatingGroups = nothing` enables all floating groups; a tuple restricts
contacts to the selected bodies without disabling their joints or fluid coupling.
`Materials = (group => RigidMaterial(...), ...)` overrides floating-group
coefficients. Other bodies use `Friction`/`Restitution`; fixed surfaces retain
`FixedFriction`/`FixedRestitution`. Material mixing occurs once during setup.

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
    floating_groups::Union{Nothing,Tuple}
    materials::Tuple
end

function RigidContacts(; ParticleDiameter, Friction = 0.5, FixedFriction = Friction,
        Restitution = 0, FixedRestitution = Restitution, Slop = 0,
        Margin = ParticleDiameter, BounceThreshold = 0.01, FixedGroups = nothing,
        FloatingGroups = nothing, Materials = ())
    values = Float64.((ParticleDiameter, Friction, FixedFriction, Restitution, FixedRestitution, Slop, Margin, BounceThreshold))
    all(isfinite, values) || throw(ArgumentError("contact parameters must be finite"))
    d, f, ff, e, ef, s, m, b = values
    d > 0 && f >= 0 && ff >= 0 && 0 <= e <= 1 && 0 <= ef <= 1 && 0 <= s < d && m >= 0 && b >= 0 ||
        throw(ArgumentError("contact diameter must be positive, 0 <= slop < diameter, restitution in [0,1], and friction, margin and bounce threshold nonnegative"))
    fixed_groups = FixedGroups === nothing ? nothing : Tuple(Int.(FixedGroups))
    fixed_groups === nothing || all(>(0), fixed_groups) || throw(ArgumentError("FixedGroups must contain positive group markers"))
    floating_groups = FloatingGroups === nothing ? nothing : Tuple(Int.(FloatingGroups))
    floating_groups === nothing || all(>(0), floating_groups) || throw(ArgumentError("FloatingGroups must contain positive group markers"))
    materials = Tuple(Materials)
    all(p -> p isa Pair && first(p) isa Integer && first(p) > 0 && last(p) isa RigidMaterial, materials) ||
        throw(ArgumentError("Materials must contain positive floating group markers paired with RigidMaterial"))
    length(unique(first.(materials))) == length(materials) || throw(ArgumentError("duplicate material group"))
    return RigidContacts(values..., fixed_groups, floating_groups, materials)
end

# Preserve the original positional construction with uniform materials.
RigidContacts(d,f,ff,e,ef,s,m,b,groups) = RigidContacts(d,f,ff,e,ef,s,m,b,groups,nothing,())

"""
    RigidBodyDynamics(; Joints = (), Contacts = nothing, Iterations = 32, MaxTimeStep = nothing)

Opt-in native GPU multibody dynamics for nonperiodic 2D/3D simulations. Pass as
`SimRigidBodyDynamics` to `RunSimulation`. Coupled joints and contacts are
projected at both predictor and corrector stages; all iteration, contact
search and impulse application stay on CUDA, including graph replay.

Body-disjoint graph colours permit parallel joint/body-pair updates without
racing on poses or velocities. Contact manifolds retain up to 24 samples per
body pair (normal direction and tangent quadrant). No external rigid engine
is used. Individual `FloatingConstraints` and moving contact surfaces cannot
be combined with this mode. Native mode requires dense particle IDs 1:N.

In 2D, bilateral joints act in the world X-Z plane: specify points as
`(x,0,z)`, hinge axes parallel to world Y and sliding directions in X-Z.
The SPH coordinates remain `(x,z)`. Ball and hinge joints both allow the
single planar relative rotation. Particle-based rigid contacts require 3D.

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
