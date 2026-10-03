"""
Floating rigid bodies (the DualSPHysics "floating" objects) on the GPU.

The particles of a body (`Type == Floating`) take part in the neighbour loops
like boundary particles, but their momentum terms are always evaluated. After
each neighbour loop the accelerations of a body's particles are summed into
the force and the torque about its centre (`launch_floating_forces!`), the
body is advanced by one stage of the symplectic scheme
(`launch_floating_update!`) and its particles are placed rigidly
(`launch_floating_particles!`):

* predictor: `V½ = Vₙ + (F/M + g) dt/2`, `ω½ = ωₙ + τ/I dt/2`,
  `C½ = Cₙ + Vₙ dt/2`, particles turned by `ωₙ dt/2` about the centre;
* corrector: `Vₙ₊₁ = Vₙ + (F½/M + g) dt`, `ωₙ₊₁ = ωₙ + τ½/I dt`,
  `Cₙ₊₁ = Cₙ + V½ dt`, particles turned by `ω½ dt`.

The force on a body is `m₀ Σ aₖ` over its particles, `m₀` being the particle
mass the neighbour loops use for every neighbour, so the fluid and the body
exchange equal and opposite forces. The mass of the body is
`RelativeWeight m₀ N` and its moment of inertia `Σ RelativeWeight m₀ |rₖ|²`
about the initial centre of its particles. Bodies are 2D; the body state lives
on the device, so the step stays capturable as a CUDA graph.
"""
module GPUFloating

export FloatingArrays, launch_floating_forces!, launch_floating_update!,
       launch_floating_particles!, floating_state

using CUDA
using StaticArrays

using ..SimulationGeometry
using ..GPUCellGrid: thread_index
using ..GPUStepState
using ..GPUKernels: lanes_sum, store_pos_cell!, ELEMENTWISE_THREADS

"""
    FloatingArrays(SimGeometry, SimParticles, SimConstants; position_type) -> NamedTuple

Device state of the floating bodies, one entry per `SPHGeometry` of type
`Floating`, in the order of `SimGeometry`. `active` is false when the case
has none, in which case nothing is launched. `body` maps a group marker to
its body (0 for other groups).
"""
function FloatingArrays(SimGeometry::Vector{SPHGeometry{D, T}}, SimParticles, SimConstants;
                        position_type::Type{TP} = T) where {D, T, TP}
    floating = filter(geom -> geom.Type == Floating || geom.Floating !== nothing, SimGeometry)
    for geom in floating
        (geom.Type == Floating && geom.Floating !== nothing) ||
            error("Floating bodies need `Type = Floating` together with `Floating = FloatingDetails(...)`" *
                  " (group marker $(geom.GroupMarker)).")
    end
    nbodies = length(floating)
    nbodies > 0 && D != 2 && error("Floating bodies are only supported in 2D.")

    ngroups = max(1, Int(maximum(SimParticles.GroupMarker; init = 0)))
    body     = zeros(Int32, ngroups)
    mass     = ones(T, max(nbodies, 1))
    inertia  = ones(T, max(nbodies, 1))
    pause    = zeros(T, max(nbodies, 1))
    center   = zeros(SVector{D, TP}, max(nbodies, 1))
    count    = zeros(Int, max(nbodies, 1))
    for (b, geom) in enumerate(floating)
        g = geom.GroupMarker
        body[g] != 0 && error("Two floating bodies share the group marker $g.")
        body[g] = b
        members = findall(k -> SimParticles.GroupMarker[k] == g && SimParticles.Type[k] == Floating,
                          eachindex(SimParticles.GroupMarker))
        isempty(members) && error("The floating body with group marker $g has no particles.")
        mₖ = T(geom.Floating.RelativeWeight) * SimConstants.m₀
        positions  = SimParticles.Position[members]
        c          = sum(SVector{D, TP}.(positions)) / length(members)
        count[b]   = length(members)
        center[b]  = c
        mass[b]    = mₖ * length(members)
        inertia[b] = mₖ * sum(x -> T(sum(abs2, SVector{D, TP}(x) - c)), positions)
        pause[b]   = T(geom.Floating.PauseTime)
    end

    zv(E) = CUDA.zeros(E, max(nbodies, 1))
    return (active = nbodies > 0, nbodies = Int32(nbodies), count = count,
            body = CuArray(body), mass = CuArray(mass), inertia = CuArray(inertia),
            pause = CuArray(pause),
            center = CuArray(center), center_n = CuArray(center), center_half = CuArray(center),
            velocity = zv(SVector{D, T}), velocity_half = zv(SVector{D, T}),
            omega = zv(T), omega_half = zv(T), angle = zv(T), turn = zv(T),
            force = CUDA.zeros(T, 3 * max(nbodies, 1)))
end

"""
    floating_state(floating) -> NamedTuple of host vectors

Centre, velocity, angle and angular velocity of every body (synchronizes).
"""
floating_state(f) = (center = Array(f.center), velocity = Array(f.velocity),
                     angle = Array(f.angle), omega = Array(f.omega))

#---------------------------------------------------------------
# Force and torque on every body
#---------------------------------------------------------------

function floating_force_kernel!(force::AbstractVector{T}, Acceleration, Position, ParticleType, GroupMarker,
                                body, center, m₀::T, step, nbodies::Int32, n::Int32) where {T}
    step_active(step) || return nothing
    i = thread_index()
    a = zero(SVector{2, T})
    r = zero(SVector{2, T})
    b = Int32(0)
    @inbounds if i <= n && ParticleType[i] == Floating
        b = body[GroupMarker[i]]
        a = Acceleration[i]
        r = SVector{2, T}(Position[i] - center[b])
    end
    # Every lane of the warp takes part in the shuffles; one atomic per warp
    # and body remains.
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    for k in Int32(1):nbodies
        mine = b == k
        fx = lanes_sum(mine ? m₀ * a[1] : zero(T), Val(32))
        fy = lanes_sum(mine ? m₀ * a[2] : zero(T), Val(32))
        τ  = lanes_sum(mine ? m₀ * (r[1] * a[2] - r[2] * a[1]) : zero(T), Val(32))
        if lane == Int32(0) && (fx != zero(T) || fy != zero(T) || τ != zero(T))
            @inbounds CUDA.@atomic force[3k - 2] += fx
            @inbounds CUDA.@atomic force[3k - 1] += fy
            @inbounds CUDA.@atomic force[3k]     += τ
        end
    end
    return nothing
end

"""
    launch_floating_forces!(floating, Acceleration, Position, ParticleType, GroupMarker, center, m₀, step)

Add the force and the torque about `center` (a vector of body centres) of the
accelerations of every body's particles to `floating.force`.
"""
function launch_floating_forces!(floating, Acceleration, Position, ParticleType, GroupMarker, center, m₀, step)
    n = length(Position)
    n == 0 && return nothing
    T = eltype(floating.force)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) floating_force_kernel!(
        floating.force, Acceleration, Position, ParticleType, GroupMarker, floating.body, center,
        T(m₀), step, floating.nbodies, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Rigid body update (one thread)
#---------------------------------------------------------------

function floating_update_kernel!(f, step, g::T, ::Val{Final}) where {T, Final}
    step_active(step) || return nothing
    thread_index() == 1 || return nothing
    dt   = step_dt(step)
    time = step_time(step)
    gvec = SVector{2, T}(zero(T), -g)
    @inbounds for b in 1:Int(f.nbodies)
        F = SVector{2, T}(f.force[3b - 2], f.force[3b - 1])
        τ = f.force[3b]
        f.force[3b - 2] = zero(T)
        f.force[3b - 1] = zero(T)
        f.force[3b]     = zero(T)
        held = time < f.pause[b]
        Vₙ = f.velocity[b]
        ωₙ = f.omega[b]
        a  = F / f.mass[b] + gvec
        α  = τ / f.inertia[b]
        if !Final
            Cₙ = f.center[b]
            f.center_n[b] = Cₙ
            if held
                f.velocity_half[b] = zero(Vₙ)
                f.omega_half[b]    = zero(T)
                f.center_half[b]   = Cₙ
                f.turn[b]          = zero(T)
            else
                f.velocity_half[b] = Vₙ + a * (dt / 2)
                f.omega_half[b]    = ωₙ + α * (dt / 2)
                f.center_half[b]   = Cₙ + Vₙ * (dt / 2)
                f.turn[b]          = ωₙ * (dt / 2)
            end
        else
            Cₙ = f.center_n[b]
            if held
                f.velocity[b] = zero(Vₙ)
                f.omega[b]    = zero(T)
                f.center[b]   = Cₙ
                f.turn[b]     = zero(T)
            else
                f.velocity[b] = Vₙ + a * dt
                f.omega[b]    = ωₙ + α * dt
                f.center[b]   = Cₙ + f.velocity_half[b] * dt
                f.turn[b]     = f.omega_half[b] * dt
                f.angle[b]   += f.turn[b]
            end
        end
    end
    return nothing
end

"""
    launch_floating_update!(floating, step, g, final::Bool)

Advance every body by the predictor (`final = false`) or the corrector stage
from the force accumulated since the last update, and clear the force.
"""
function launch_floating_update!(floating, step, g, final::Bool)
    T = eltype(floating.force)
    f = floating
    device = (; f.nbodies, f.force, f.pause, f.mass, f.inertia, f.center, f.center_n, f.center_half,
                f.velocity, f.velocity_half, f.omega, f.omega_half, f.angle, f.turn)
    @cuda threads=1 blocks=1 floating_update_kernel!(device, step, T(g), Val(final))
    return nothing
end

#---------------------------------------------------------------
# Rigid placement of the body particles
#---------------------------------------------------------------

function floating_particles_kernel!(PosOut, VelOut, PosIn, ParticleType, GroupMarker, body, origin,
                                    center, velocity, omega, turn, step, PosCellsOut, CellID, gridarg, H,
                                    n::Int32)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    @inbounds if ParticleType[i] == Floating
        b = body[GroupMarker[i]]
        r = PosIn[i] - origin[b]
        s, c = sincos(turn[b])
        r = typeof(r)(c * r[1] - s * r[2], s * r[1] + c * r[2])
        x = center[b] + r
        PosOut[i] = x
        V = velocity[b]
        VelOut[i] = V + omega[b] * typeof(V)(-r[2], r[1])
        store_pos_cell!(PosCellsOut, i, x, CellID, gridarg, H)
    end
    return nothing
end

"""
    launch_floating_particles!(floating, PosOut, VelOut, PosIn, ParticleType, GroupMarker, step, final::Bool;
                               pos_cells = nothing, CellID = nothing, grid = nothing, H = 0)

Move the body particles rigidly from their start-of-step positions `PosIn`:
turned by the stage's rotation about the start-of-step centre and carried to
the stage's centre, with the rigid velocity `V + ω × r`. The predictor writes
the half step arrays (and their cell relative form with `pos_cells`), the
corrector the start-of-step arrays of the next step.
"""
function launch_floating_particles!(floating, PosOut, VelOut, PosIn, ParticleType, GroupMarker, step,
                                    final::Bool; pos_cells = nothing, CellID = nothing, grid = nothing,
                                    H = zero(eltype(floating.force)))
    n = length(PosIn)
    n == 0 && return nothing
    center, velocity, omega = final ? (floating.center, floating.velocity, floating.omega) :
                                      (floating.center_half, floating.velocity_half, floating.omega_half)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) floating_particles_kernel!(
        PosOut, VelOut, PosIn, ParticleType, GroupMarker, floating.body, floating.center_n,
        center, velocity, omega, floating.turn, step, pos_cells, CellID, grid, H, Int32(n))
    return nothing
end

end # module GPUFloating
