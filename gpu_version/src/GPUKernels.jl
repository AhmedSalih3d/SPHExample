"""
CUDA kernels for the SPH time step.

All kernels use a *gather* formulation: the thread(s) of particle `i` loop
over the particles of the neighbouring cells and sum the contributions into
registers, then write the totals once. There are no atomics, no per thread
accumulation arrays and no reduction step.

Small cases do not have enough particles to fill the GPU with one thread per
particle, so the neighbour loop of a particle can be split over `K` lanes of
a warp (`K` a power of two up to 32): lane `k` visits every `K`-th candidate
of each cell row and the partial sums are combined with warp shuffles.

The CPU code evaluates every pair once and assigns the two halves of a pair
to `i` and `j`. Reference density diffusion assigns equal-and-opposite terms,
but its neighbor reciprocal-density weight makes their magnitude depend on
which particle is chosen first. A user supplied model may also depend on that
choice. To reproduce the CPU results exactly the gather kernel reconstructs which particle of a pair
was the CPU's `i`: for pairs inside one cell the lower index, for pairs in
different cells the particle in the cell that is processed (the higher index,
because the CPU stencil only visits cells with a lower linear index). Both
particles of a pair reach the same decision, so the rule stays consistent for
any grid; it only reproduces the CPU's choice with the CPU's cells of edge
`H` (`GPUCellSubdivision = 1`). Opt-in radial linear diffusion instead
averages the two reciprocal-density weights and evaluates from the target
particle's viewpoint, removing that diffusion term's orientation dependence.
The separate Fourtakas inner model also evaluates from the target viewpoint;
it suppresses the whole diffusion sum when the support contains a wall.

Every kernel of a time step takes the device resident step state (`step`, see
`GPUStepState`) and reads the step size and the simulated time from it; the
first thing each kernel does is to return when the stop flag is set. The grid
description is read from a one element device vector (`load_grid`), so none
of the launch arguments of a step change between steps.

Positions may be stored in a higher precision than the working precision
(`GPUDoublePosition`). The pair loops then read cell relative positions
(`PosCell`, see `GPUCellGrid`) instead of the positions: the interaction and
mDBC kernels take a *pair source* argument, which is either the position
array itself or the `PosCell` array, and form every pair vector with
`pair_vector`. The element-wise kernels update the positions in their own
precision (the increments are promoted) and write the cell relative form of
the positions they produce: the half step kernel for the half step positions,
the final step kernel for the start-of-step positions of the next step. The
kernels that move particles between two final steps (prescribed motion,
floating body placement) refresh the form of the particles they move, and a
cell list rebuild refreshes it for every particle (`launch_pos_cells!`).
"""
module GPUKernels

using CUDA
using Adapt
using StaticArrays
using LinearAlgebra

using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SimulationEquations
using ..SimulationGeometry
using ..PeriodicBoundaries: PeriodicBoundary
using ..GPUCellGrid
using ..GPUReductions
using ..GPUStepState
using ..DampingZones: damped_velocity
using ..SimulationMetaDataConfiguration: NoMDBC, SimpleMDBC, UpdatedMDBC, DualSPHysicsSymplectic

export launch_interactions!, launch_mdbc!, launch_motion!, launch_half_step!, launch_final_step!,
       launch_prepare_interactions!, launch_finish!, launch_commit!, launch_step_reduction!, launch_pos_cells!,
       step_map, step_reduce, choose_lanes, radial_interactions_supported, scalar_sps_supported, ELEMENTWISE_THREADS

const ELEMENTWISE_THREADS = REDUCE_THREADS
const FULL_MASK = 0xffffffff

include("GPUTimestepDiagnostics.jl")

# Use counter-based row iteration even for the 3/9-row stencil at reach one.
# It preserves the same neighbor order as the tuple returned by cell_rows.
@inline interaction_rows(grid::CellGrid{D, R}) where {D, R} =
    GPUCellGrid.CellRows{D, R}(grid.dims[1], grid.dims[1] * grid.dims[2])

include("GPUPeriodicSearch.jl")

"""
    radial_interactions_supported(kernel, diffusion, viscosity) -> Bool

Whether the opt-in radial specialization is implemented for this combination:
2D/3D Float32/Float64 Wendland C2, zero/linear density diffusion, and
zero/artificial/laminar/SPS viscosity. SPS retains its original tensor
calculation and CPU pair orientation. Linear diffusion uses a symmetric
reciprocal-density average, changing the reference diffusion operator in
addition to rounding. This is a capability check, not an accuracy or
performance guarantee. Unsupported combinations use reference interactions.
"""
radial_interactions_supported(kernel, diffusion, viscosity) = false
function radial_interactions_supported(::SPHKernelInstance{<:WendlandC2, D, T},
                                       ::Union{ZeroDensityDiffusion, LinearDensityDiffusion, ZeroGravityLinearDensityDiffusion},
                                       ::Union{ZeroViscosity, ArtificialViscosity, Laminar, LaminarSPS}) where {D, T}
    return D in (2, 3) && T in (Float32, Float64)
end

"""
    scalar_sps_supported(kernel, viscosity) -> Bool

Whether the separate opt-in rank-one SPS stress contraction is implemented:
2D/3D Float32/Float64 with `LaminarSPS`. The original pair orientation and
closure are retained, but reassociated arithmetic can change rounding.
"""
scalar_sps_supported(kernel, viscosity) = false
function scalar_sps_supported(::SPHKernelInstance{K, D, T}, ::LaminarSPS) where {K, D, T}
    return D in (2, 3) && T in (Float32, Float64)
end

# Keep previews, the driver and standalone launches on the same capability checks.
function interaction_options(kernel, diffusion, viscosity; radial::Bool = false, scalar_sps::Bool = false)
    radial_supported = radial_interactions_supported(kernel, diffusion, viscosity)
    sps_supported = scalar_sps_supported(kernel, viscosity)
    return (; radial = radial && radial_supported, scalar_sps = scalar_sps && sps_supported,
        radial_supported, scalar_sps_supported = sps_supported)
end

#---------------------------------------------------------------
# Helpers shared by the kernels
#---------------------------------------------------------------

"""
    choose_lanes(n; target = 4 * resident threads of the device) -> K

Number of warp lanes per particle for the gather kernels: 1 for large particle
counts, more for small ones so that at least `target` threads are launched.
Always a power of two between 1 and 32.
"""
function choose_lanes(n::Integer; target::Integer = 4 * CUDA.attribute(CUDA.device(),
                                                          CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) * 2048)
    n <= 0 && return 1
    k = nextpow(2, cld(target, n))
    return clamp(k, 1, 32)
end

"""
    choose_lanes(n, ::Val{D}; target = 4 * resident threads of the device) -> K

Choose gather lanes using a smaller occupancy target than the legacy
dimension-independent selector. Three-dimensional neighborhoods use one
resident thread's worth of particle-lanes; two-dimensional neighborhoods use
one quarter, reflecting their lower candidate count.
"""
function choose_lanes(n::Integer, ::Val{D}; target::Integer = 4 * CUDA.attribute(CUDA.device(),
                                                                       CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) * 2048) where {D}
    D in (2, 3) || throw(ArgumentError("automatic GPU lane selection supports 2D and 3D, got $D"))
    scale = D == 2 ? 16 : 4
    return choose_lanes(n; target = cld(target, scale))
end

# Warp shuffle reduction of the `K` consecutive lanes of a particle. All
# lanes of the warp must execute this (no early return before it).
@inline function lanes_sum(x::T, ::Val{K}) where {T <: Real, K}
    s = K ÷ 2
    while s >= 1
        x += CUDA.shfl_down_sync(FULL_MASK, x, s, K)
        s ÷= 2
    end
    return x
end

@inline function lanes_sum(x::SVector{N, T}, ::Val{K}) where {N, T, K}
    return SVector{N, T}(ntuple(k -> lanes_sum(x[k], Val(K)), Val(N)))
end

@inline function lanes_sum(x::SMatrix{N, M, T, L}, ::Val{K}) where {N, M, T, L, K}
    return SMatrix{N, M, T, L}(ntuple(k -> lanes_sum(x[k], Val(K)), Val(L)))
end

@inline lanes_sum(x, ::Val{1}) = x

@inline function lanes_max(x, ::Val{K}) where {K}
    offset = K ÷ 2
    while offset >= 1
        x = max(x, CUDA.shfl_down_sync(FULL_MASK, x, offset, K))
        offset ÷= 2
    end
    return x
end

"""
Prescribed motion of `Moving` particles (mirrors `ProgressMotion` of the CPU
code). `motion` is a NamedTuple of device arrays indexed by group marker.
"""
@inline function apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    @inbounds if ParticleType[i] == Moving
        g = Int(GroupMarker[i])
        if motion.has[g]
            start    = motion.start[g]
            duration = motion.duration[g]
            ShouldMove = (start <= TotalTime) & (TotalTime <= start + duration)
            v = motion.velocity[g] * motion.direction[g] * ShouldMove
            Velocity[i] = v
            motion.move_particles[g] && (Position[i] += v * dt₂)
        end
    end
    return nothing
end

@inline function limit_density(ρ, ρ₀, MotionLimiter)
    return ((ρ < ρ₀) & !Bool(MotionLimiter)) ? ρ₀ : ρ
end

#---------------------------------------------------------------
# Prescribed motion of moving bodies (only launched when a case has them)
#---------------------------------------------------------------

# Only `Moving` particles change position; the cell relative form of every
# other particle is kept current by the final step and the cell list rebuild.
@inline function store_moved_pos_cell!(PosCells, i, Position, ParticleType, CellID, gridarg, H)
    PosCells === nothing && return nothing
    @inbounds ParticleType[i] == Moving && store_pos_cell!(PosCells, i, Position[i], CellID, gridarg, H)
    return nothing
end

function motion_kernel!(Position, Velocity, ParticleType, GroupMarker, motion, step,
                        PosCells, CellID, gridarg, H, n::Int32)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    dt₂       = step_dt(step) / 2
    TotalTime = step_time(step)
    apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    store_moved_pos_cell!(PosCells, i, Position, ParticleType, CellID, gridarg, H)
    return nothing
end

"""
    launch_motion!(Position, Velocity, ParticleType, GroupMarker, motion, step; pos_cells, CellID, grid, H)

Apply the first prescribed-motion half increment. With `pos_cells` the cell
relative form of the `Moving` particles is refreshed as well.
"""
function launch_motion!(Position, Velocity, ParticleType, GroupMarker, motion, step;
                        pos_cells = nothing, CellID = nothing, grid = nothing, H = nothing)
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) motion_kernel!(
        Position, Velocity, ParticleType, GroupMarker, motion, step, pos_cells, CellID, grid, H, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Fused prescribed motion and cell relative positions before DBC interactions
#---------------------------------------------------------------

@inline store_inverse_density!(::Nothing, i, density) = nothing
@inline function store_inverse_density!(cache, i, density)
    @inbounds cache[i] = inv(density)
    return nothing
end

@inline store_pressure!(::Nothing, i, pressure) = nothing
@inline function store_pressure!(cache, i, pressure)
    @inbounds cache[i] = pressure
    return nothing
end

@inline interaction_inverse_density(::Nothing, i, density) = inv(density)
@inline interaction_inverse_density(cache, i, density) = @inbounds cache[i]

@inline divide_by_density(value, ::Nothing, i, density) = value / density
@inline divide_by_density(value, cache, i, density) = value * interaction_inverse_density(cache, i, density)

function prepare_interactions_kernel!(Position, step, n::Int32,
                                      PosCells, CellID, grid, H, Velocity, ParticleType, GroupMarker, motion,
                                      InvDensity, Pressure, Density, SimConstants, boundary_mode)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        if InvDensity !== nothing || Pressure !== nothing
            ρ = Density[i]
            store_inverse_density!(InvDensity, i, ρ)
            if Pressure !== nothing
                type = boundary_mode isa UpdatedMDBC ? ParticleType[i] : Fluid
                store_pressure!(Pressure, i, particle_pressure(ρ, type, SimConstants.c₀, SimConstants.ρ₀, boundary_mode))
            end
        end
        if motion !== nothing
            apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, step_dt(step) / 2, step_time(step))
            store_moved_pos_cell!(PosCells, i, Position, ParticleType, CellID, grid, H)
        elseif PosCells !== nothing
            store_pos_cell!(PosCells, i, Position[i], CellID, grid, H)
        end
    end
    return nothing
end

"""
    launch_prepare_interactions!(Position, step; pos_cells, CellID, grid, H,
                                 Velocity, ParticleType, GroupMarker, motion,
                                 inverse_density, pressure, Density, SimConstants, boundary_mode)

Prepare cell relative positions and optionally apply prescribed motion in one
pass before DBC interactions. With `motion`, only the cell relative positions
of the `Moving` particles are refreshed (the final step and the cell list
rebuild keep the others current); without, those of every particle. No kernel
is launched when no preparation is needed. With `inverse_density` and/or
`pressure` and `Density`, refresh the interaction caches from the evaluated
state in the same pass.
"""
function launch_prepare_interactions!(Position, step;
                                      pos_cells = nothing, CellID = nothing, grid = nothing, H = nothing,
                                      Velocity = nothing, ParticleType = nothing, GroupMarker = nothing, motion = nothing,
                                      inverse_density = nothing, pressure = nothing, Density = nothing,
                                      SimConstants = nothing, boundary_mode = NoMDBC())
    inverse_density === nothing || (Density !== nothing && length(inverse_density) == length(Density) == length(Position)) ||
        throw(DimensionMismatch("Reciprocal cache and evaluated density must have one entry per particle"))
    pressure === nothing || (Density !== nothing && SimConstants !== nothing && length(pressure) == length(Density) == length(Position) &&
        (!(boundary_mode isa UpdatedMDBC) || (ParticleType !== nothing && length(ParticleType) == length(Position)))) ||
        throw(DimensionMismatch("Pressure cache, evaluated density, constants and UpdatedMDBC types must match particle count"))
    pos_cells === nothing && motion === nothing && inverse_density === nothing && pressure === nothing && return nothing
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) prepare_interactions_kernel!(
        Position, step, Int32(n), pos_cells, CellID, grid, H, Velocity, ParticleType, GroupMarker, motion,
        inverse_density, pressure, Density, SimConstants, boundary_mode)
    return nothing
end

#---------------------------------------------------------------
# Cell relative positions of every particle after a cell list rebuild (only
# with positions of a higher precision than the working precision)
#---------------------------------------------------------------

# Cell relative form of position `x` of particle `i`, for the cell the
# particle was binned into. A `nothing` destination (positions in the working
# precision, no cell relative form needed) compiles to nothing.
@inline store_pos_cell!(::Nothing, i, x, CellID, gridarg, H) = nothing
@inline function store_pos_cell!(PosCells, i, x, CellID, gridarg, H)
    grid = load_grid(gridarg)
    @inbounds begin
        c = CellID[i]
        PosCells[i] = c == Int32(1) ? eltype(PosCells)(zero(x), c) : pos_cell(x, c, grid, cell_size(grid, H))
    end
    return nothing
end

function pos_cells_kernel!(PosCells, Position, CellID, gridarg, H, step, n::Int32)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    @inbounds store_pos_cell!(PosCells, i, Position[i], CellID, gridarg, H)
    return nothing
end

"""
    launch_pos_cells!(PosCells, Position, CellID, grid, step, SimKernel)

Write the cell relative form (`PosCell`) of every position for the cell list
whose sorted cell ids are `CellID`. Needed when the positions are stored in a
higher precision than the working precision and the particles were reordered
or binned anew (`rebuild_cell_list!` with `support`). Within the time step the
final step kernel writes the form of the start-of-step positions of the next
step and the half step kernel that of the half step positions.
"""
function launch_pos_cells!(PosCells, Position, CellID, grid, step, SimKernel)
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) pos_cells_kernel!(
        PosCells, Position, CellID, grid, SimKernel.H, step, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Particle interactions (gather)
#---------------------------------------------------------------

@inline interaction_pressure(::Nothing, i, ρ, type, c₀, ρ₀, ::Nothing) = EquationOfStateGamma7(ρ, c₀, ρ₀)
@inline interaction_pressure(::Nothing, i, ρ, type, c₀, ρ₀, boundary_data) =
    is_wall(type) ? c₀^2 * (ρ - ρ₀) : EquationOfStateGamma7(ρ, c₀, ρ₀)
@inline interaction_pressure(cache, i, ρ, type, c₀, ρ₀, boundary_data) = @inbounds cache[i]

function interaction_kernel!(dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ,
                             Pairs, Position, Density,
                             Velocity::AbstractVector{SVector{D, T}}, ParticleType, SimParticles,
                             CellStart, CellID, gridarg, step,
                             SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                             ::Val{FlagKernel}, ::Val{FlagShift}, ::Val{BoundaryForces}, ::Val{K},
                             boundary_data, InvDensity, Pressure, n::Int32, ::Val{Radial} = Val(false), ::Val{ScalarSPS} = Val(false),
                             ::Val{FrozenParticles} = Val(true), ::Val{IncludeGravity} = Val(false), periodicity = nothing,
                             periodic_bounds = nothing, viscous_dt = nothing, ::Val{FastMath}=Val(false), target_indices=nothing, pair_order=nothing) where {D, T, FlagKernel, FlagShift, BoundaryForces, K, Radial, ScalarSPS, FrozenParticles, IncludeGravity, FastMath}
    step_active(step) || return nothing
    grid = load_grid(gridarg)
    t    = thread_index()
    i    = (t - Int32(1)) ÷ Int32(K) + Int32(1)
    lane = (t - Int32(1)) % Int32(K)
    valid = i <= n
    source_i = valid && target_indices !== nothing ? (@inbounds target_indices[i]) : i

    (; m₀, dx, c₀, ρ₀) = SimConstants
    (; h⁻¹, H²) = SimKernel
    s = cell_size(grid, SimKernel.H)

    dρdt  = zero(T)
    inner_diffusion = zero(T)
    diffusion_blocked = false
    pair_rate = zero(T)
    acc   = zero(SVector{D, T})
    Wsum  = zero(T)
    ∇Wsum = zero(SVector{D, T})
    ∇C    = zero(SVector{D, T})
    ∇r    = zero(T)

    slotᵢ = valid ? boundary_slot(boundary_data, ParticleType, source_i) : Int32(0)
    @inbounds if valid && (!FrozenParticles || ParticleType[source_i] != Frozen) && boundary_enabled(boundary_data, slotᵢ)
        pᵢ  = Pairs[source_i]
        vᵢ  = boundary_velocity(boundary_data, Velocity, ParticleType, source_i, slotᵢ)
        viscous_vᵢ = boundary_viscous_velocity(boundary_data, Velocity, ParticleType, source_i, slotᵢ)
        ρᵢ  = Density[source_i]
        ρᵢ⁻¹ = interaction_inverse_density(InvDensity, source_i, ρᵢ)
        typeᵢ = ParticleType[source_i]
        MLᵢ = MotionLimiterValue(T, typeᵢ)

        # Boundary particles only need the density rate; their acceleration is
        # never applied (MotionLimiter = 0). Skipping the momentum terms for them
        # is optional because the CPU code includes their acceleration in the
        # force based time step criterion. Floating bodies always need them:
        # their particle accelerations sum to the force on the body.
        forces = BoundaryForces | (MLᵢ != zero(T)) | (typeᵢ == Floating)
        Pᵢ = forces ? (viscous_dt === nothing ? interaction_pressure(Pressure,source_i,ρᵢ,typeᵢ,c₀,ρ₀,boundary_data) :
            dual_pressure(ρᵢ,SimConstants,Val(FastMath))) : zero(T)
        wallᵢ = is_wall(typeᵢ)

        c      = CellID[source_i]
        own_lo = CellStart[c] + Int32(1)
        own_hi = CellStart[c + Int32(1)]
        owner_i = pair_order === nothing ? source_i : pair_order.owners[source_i]
        anchor_i = pair_order === nothing ? c : pair_order.cells[source_i]

        for row in neighbor_rows(periodicity, grid, c, s, pair_position(pᵢ), periodic_bounds, source_i)
            jlo, jhi = row_range(grid, CellStart, c, row)
            rowcell  = c + row.off
            j = jlo + lane
            while j <= jhi
                if j != source_i && (!FrozenParticles || ParticleType[j] != Frozen)
                    xᵢⱼ  = pair_vector(Pairs, pᵢ, j, rowcell, row, s)
                    xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
                    slotⱼ = xᵢⱼ² <= H² ? boundary_slot(boundary_data, ParticleType, j) : Int32(0)
                    if SimDensityDiffusion isa FourtakasDensityDiffusion
                        diffusion_blocked |= !SimDensityDiffusion.full && xᵢⱼ² <= H² && is_wall(ParticleType[j])
                    end
                    skip_wall_pair = viscous_dt !== nothing && wallᵢ && is_wall(ParticleType[j])
                    if xᵢⱼ² <= H² && boundary_enabled(boundary_data, slotⱼ) && !skip_wall_pair
                        dᵢⱼ   = sqrt(xᵢⱼ²)
                        q     = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
                        if Radial || viscous_dt !== nothing
                            radial_factor = SPHKernels.radial_gradient_factor(SimKernel, q)
                            ∇ᵢWᵢⱼ = radial_factor * xᵢⱼ
                            inverse_distance = inv(xᵢⱼ² + SimKernel.η²)
                        else
                            ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)
                        end

                        ρⱼ   = Density[j]
                        ρⱼ⁻¹ = interaction_inverse_density(InvDensity, j, ρⱼ)
                        vⱼ   = boundary_velocity(boundary_data, Velocity, ParticleType, j, slotⱼ)
                        vᵢⱼ  = vᵢ - vⱼ
                        if viscous_dt !== nothing
                            pair_rate = max(pair_rate, dot(vᵢⱼ, xᵢⱼ) / (xᵢⱼ² + SimKernel.η²))
                        end
                        density_symmetric_term = Radial ? -radial_factor * dot(vᵢⱼ, xᵢⱼ) : dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                        # DBC walls take density from the fluid, not other walls.
                        wall_pair = wallᵢ && is_wall(ParticleType[j])
                        if !wall_pair
                            dρdt += viscous_dt === nothing ? -ρᵢ * (m₀ * ρⱼ⁻¹) * density_symmetric_term :
                                -m₀ * density_symmetric_term * (ρᵢ / ρⱼ)
                        end

                        # Reference diffusion and ordered/SPS viscosity use the CPU
                        # pair orientation. The sign flip and swaps are selects.
                        # The densities handed to the models are those of the arrays
                        # being evaluated (the predictor state in the corrector loop).
                        same_cell = pair_order === nothing ? (j >= own_lo) & (j <= own_hi) : anchor_i == pair_order.cells[j]
                        owner_j = pair_order === nothing ? j : pair_order.owners[j]
                        i_first   = same_cell ? (owner_i < owner_j) : (owner_i > owner_j)
                        sgn  = i_first ? one(T) : -one(T)
                        ia   = i_first ? source_i : j
                        ja   = i_first ? j : source_i
                        ρa   = i_first ? ρᵢ : ρⱼ
                        ρb   = i_first ? ρⱼ : ρᵢ
                        ρa⁻¹ = i_first ? ρᵢ⁻¹ : ρⱼ⁻¹
                        ρb⁻¹ = i_first ? ρⱼ⁻¹ : ρᵢ⁻¹
                        if SimDensityDiffusion isa FourtakasDensityDiffusion
                            if typeᵢ == Fluid && ParticleType[j] == Fluid
                                inner_diffusion += SPHDensityDiffusionModels.fourtakas_diffusion(
                                    SimKernel, SimConstants, xᵢⱼ, ∇ᵢWᵢⱼ, xᵢⱼ², ρᵢ, ρⱼ, ρⱼ⁻¹, Val(FastMath))
                            end
                        elseif Radial
                            D1, _ = SPHDensityDiffusionModels.radial_density_diffusion(SimDensityDiffusion,
                                SimKernel, SimConstants, xᵢⱼ, xᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹,
                                source_i, j, ParticleType, radial_factor, inverse_distance)
                            dρdt += D1
                        else
                            D1, D2 = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants,
                                                               SimParticles, sgn * xᵢⱼ, sgn * ∇ᵢWᵢⱼ, xᵢⱼ²,
                                                               ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja, ParticleType)
                            dρdt += i_first ? D1 : D2
                        end

                        if forces
                            viscous_velocity = viscous_vᵢ -
                                               boundary_viscous_velocity(boundary_data, Velocity, ParticleType, j, slotⱼ)
                            if viscous_dt !== nothing
                                dot_velocity = dot(viscous_velocity, xᵢⱼ)
                                mu = SimKernel.h * dot_velocity / (xᵢⱼ² + SimKernel.η²)
                                coefficient = dot_velocity < zero(T) ?
                                    (-SimConstants.α * c₀ * mu / ((ρᵢ + ρⱼ) / 2)) * m₀ : zero(T)
                                visc = -coefficient * ∇ᵢWᵢⱼ
                            elseif ScalarSPS
                                v1 = SPHViscosityModels.scalar_sps_viscosity(SimKernel, SimConstants,
                                    sgn * xᵢⱼ, sgn * viscous_velocity, sgn * ∇ᵢWᵢⱼ, xᵢⱼ²,
                                    ρa, ρb, ρa⁻¹, ρb⁻¹)
                                visc = sgn * v1
                            elseif Radial && !(SimViscosity isa LaminarSPS)
                                visc = SPHViscosityModels.radial_viscosity(SimViscosity, SimKernel, SimConstants,
                                    xᵢⱼ, viscous_velocity, ∇ᵢWᵢⱼ, ρᵢ, ρⱼ, inverse_distance)
                            else
                                v1, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles,
                                                          sgn * xᵢⱼ, sgn * viscous_velocity, sgn * ∇ᵢWᵢⱼ, xᵢⱼ²,
                                                          ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja)
                                visc  = sgn * v1
                            end

                            if SimViscosity isa ArtificialViscosity && (is_wall(typeᵢ) || is_wall(ParticleType[j]))
                                visc *= T(SimViscosity.boundary_factor)
                            end
                            Pⱼ = viscous_dt === nothing ? interaction_pressure(Pressure,j,ρⱼ,ParticleType[j],c₀,ρ₀,boundary_data) :
                                dual_pressure(ρⱼ,SimConstants,Val(FastMath))
                            Pfac = viscous_dt === nothing ? (Pᵢ + Pⱼ) * (ρᵢ⁻¹ * ρⱼ⁻¹) : (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
                            f_ab = InvDensity === nothing ? tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx) :
                                tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx, ρᵢ⁻¹, ρⱼ⁻¹)
                            dvdt = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ
                            acc += dvdt + visc
                        end

                        if FlagKernel
                            Wsum  += SPHKernels.Wᵢⱼ(SimKernel, q)
                            ∇Wsum += ∇ᵢWᵢⱼ
                        end

                        if FlagShift
                            MLcond = MLᵢ * MotionLimiterValue(T, ParticleType[j])
                            ∇C += (m₀ * ρᵢ⁻¹) * ∇ᵢWᵢⱼ
                            # Sign convention follows the CPU code, see
                            # https://arxiv.org/abs/2110.10076
                            ∇r += (m₀ * ρⱼ⁻¹) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
                        end
                    end
                end
                j += Int32(K)
            end
        end
    end

    if K > 1
        if viscous_dt !== nothing
            pair_rate = lanes_max(pair_rate, Val(K))
        end
        if SimDensityDiffusion isa FourtakasDensityDiffusion
            inner_diffusion = lanes_sum(inner_diffusion, Val(K))
            diffusion_blocked = lanes_sum(diffusion_blocked ? one(T) : zero(T), Val(K)) > zero(T)
        end
        dρdt = lanes_sum(dρdt, Val(K))
        acc  = lanes_sum(acc, Val(K))
        if FlagKernel
            Wsum  = lanes_sum(Wsum, Val(K))
            ∇Wsum = lanes_sum(∇Wsum, Val(K))
        end
        if FlagShift
            ∇C = lanes_sum(∇C, Val(K))
            ∇r = lanes_sum(∇r, Val(K))
        end
    end

    @inbounds if valid & (lane == Int32(0))
        if SimDensityDiffusion isa FourtakasDensityDiffusion && !diffusion_blocked
            dρdt += inner_diffusion
        end
        if IncludeGravity
            # Gravity belongs to the particle, not to each neighbor-gather lane.
            acc += ConstructGravitySVector(acc, SimConstants.g * GravityFactorValue(T, ParticleType[source_i]))
        end
        dρdtI[i]        = dρdt
        Acceleration[i] = acc
        viscous_dt === nothing || (viscous_dt[i] = pair_rate)
        if FlagKernel
            Kernel[i]         = Wsum
            KernelGradient[i] = ∇Wsum
        end
        if FlagShift
            ∇Cᵢ[i]  = ∇C
            ∇◌rᵢ[i] = ∇r
        end
    end
    return nothing
end

"""
    launch_interactions!(...; threads, lanes, boundary_forces, pos_cells, radial, scalar_sps,
                         always_inline = true, fastmath = true)

Launch the gather interaction kernel. `Position`, `Density` and
`Velocity` are the arrays of the state being evaluated (the half step arrays
for the second neighbour loop). Pressure and reciprocal densities are computed
on demand from that state, once per target and once per accepted neighbor, unless
`inverse_density` supplies a cache refreshed from exactly that state. The viscosity and diffusion models
receive the densities and reciprocals of this state as arguments, like the
CPU code. Custom models additionally get the same arrays as the NamedTuple
`SimParticles`, so nothing in the kernel can read a stale state. `grid` is a
`CellGrid` or the device vector holding it, `step` the step state (see
`GPUStepState`). `lanes` is the number of warp lanes per particle (`Val`).
With `boundary_forces = Val(false)` the momentum terms are skipped for
particles with `MotionLimiter == 0`. `pos_cells` is the `PosCell` array of
`Position` when the positions are stored in a higher precision than the
working precision (the pair vectors are then formed from it, see
`GPUCellGrid`); `nothing` forms them from `Position` directly.
`radial = Val(true)` opts into scalar Wendland contractions, direct symmetric
viscosity, and linear diffusion weighted by `(inv(Density[i]) + inv(Density[j])) / 2`.
Diffusion is evaluated directly from the target particle's viewpoint. This
changes the reference diffusion operator as well as floating-point rounding.
SPS viscosity retains the original tensor arithmetic and pair ordering.
Unsupported kernel/model combinations use the reference specialization;
the default is `Val(false)`. Standalone unsupported requests warn once per
option with the effective specialization; the driver exposes its full resolved
settings in `SimulationResult.configuration`.
`scalar_sps = Val(true)` separately opts into the rank-one SPS stress
contraction while retaining its pair orientation and closure. It can be
selected without radial continuity/diffusion, and changes rounding.
`include_gravity = Val(true)` writes total acceleration for startup/rebuild
evaluations carried into a single-neighbor predictor. The default writes only
interaction acceleration; the predictor or corrector then adds gravity.
`always_inline = true` forces helper-function inlining. Independently,
`fastmath = true` enables relaxed floating-point semantics throughout the
interaction kernel and its callees, including custom models. It may change
rounding and NaN/Inf checks, flush Float32 subnormals, and use approximate
arithmetic. Both default to `true` and can be disabled independently. Other kernels,
including mDBC corrections and pressure/reciprocal cache preparation, are not
recompiled with these options.
`target_indices` optionally maps the `target_count` outputs to source indices.
Sources and their cell IDs must share dense cell order; outputs remain in target
order. This permits periodic copies as neighbors without integrating them.
`pair_order` optionally supplies dense original owner indices and cell IDs,
preserving reference pair orientation after sorting periodic sources.
"""
function launch_interactions!(dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ,
                              Position, Density, Velocity, ParticleType,
                              CellStart, CellID, grid, step,
                              SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                              FlagKernel::Val, FlagShift::Val; threads::Integer = 128, lanes::Val = Val(1),
                              boundary_forces::Val = Val(true), pos_cells = nothing, boundary_data = nothing,
                              radial::Val = Val(false), scalar_sps::Val = Val(false), inverse_density = nothing,
                              pressure = nothing, frozen_particles::Val = Val(true), include_gravity::Val = Val(false), periodicity = nothing,
                              periodic_bounds = nothing, periodic_cell_coordinates::Bool = false, viscous_dt = nothing,
                              fast_math::Bool = false,
                              always_inline::Bool = true, fastmath::Bool = true,
                              target_indices = nothing, target_count::Integer = length(Position), pair_order = nothing)
    inverse_density === nothing || length(inverse_density) == length(Density) ||
        throw(DimensionMismatch("Reciprocal cache must have one entry per evaluated density"))
    pressure === nothing || length(pressure) == length(Density) ||
        throw(DimensionMismatch("Pressure cache must have one entry per evaluated density"))
    n = target_count
    target_indices === nothing ? (n == length(Position) || throw(DimensionMismatch("Unmapped targets must cover all sources"))) :
        (length(target_indices) == n || throw(DimensionMismatch("Target map and target count differ")))
    n == 0 && return nothing
    K = typeof(lanes).parameters[1]
    periodic_cell_coordinates && (periodicity === nothing || pos_cells === nothing) &&
        throw(ArgumentError("Periodic cell coordinates require periodicity and mixed-precision PosCell buffers"))
    Pairs = periodicity === nothing ? (pos_cells === nothing ? Position : pos_cells) :
        periodic_cell_coordinates ? PeriodicCellPairSource(eltype(Density), Position, pos_cells, grid) :
        PeriodicPairSource(eltype(Density), Position)
    SimParticles = (Position = Position, Density = Density, Velocity = Velocity,
                    Type = ParticleType)
    options = interaction_options(SimKernel, SimDensityDiffusion, SimViscosity;
        radial = typeof(radial).parameters[1], scalar_sps = typeof(scalar_sps).parameters[1])
    if typeof(radial).parameters[1] && !options.radial
        @warn "Unsupported radial interaction request; interaction_kernel! uses radial=false (reference pair arithmetic)." kernel = typeof(SimKernel) diffusion = typeof(SimDensityDiffusion) viscosity = typeof(SimViscosity) scalar_sps = options.scalar_sps maxlog = 1
    end
    if typeof(scalar_sps).parameters[1] && !options.scalar_sps
        @warn "Unsupported scalar SPS request; interaction_kernel! uses scalar_sps=false with the selected model's viscosity arithmetic." kernel = typeof(SimKernel) viscosity = typeof(SimViscosity) radial = options.radial maxlog = 1
    end
    use_radial = Val(options.radial)
    use_scalar_sps = Val(options.scalar_sps)
    @cuda always_inline=always_inline fastmath=fastmath threads=threads blocks=cld(n * K, threads) interaction_kernel!(
        dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ,
        Pairs, Position, Density, Velocity, ParticleType, SimParticles,
        CellStart, CellID, grid, step,
        SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
        FlagKernel, FlagShift, boundary_forces, lanes, boundary_data, inverse_density, pressure, Int32(n), use_radial, use_scalar_sps, frozen_particles, include_gravity, periodicity, periodic_bounds, viscous_dt, Val(fast_math), target_indices, pair_order)
    return nothing
end

#---------------------------------------------------------------
# mDBC: ghost node interpolation and density correction
#---------------------------------------------------------------

# Reference of a ghost node for `pair_vector`: the node itself in the working
# precision, or its cell relative form for the cell with global coordinates
# `cg` when the pair source holds `PosCell`s.
@inline mdbc_reference(::AbstractVector{<:SVector}, gp::SVector{D, TP}, cg, s::T) where {D, TP, T} = SVector{D, T}(gp)
@inline mdbc_reference(::AbstractVector{<:PosCell}, gp::SVector{D, TP}, cg::NTuple{D, Int32}, s::T) where {D, TP, T} =
    pos_cell(gp, cg, s)

# Accumulate the contributions of the fluid particles in the (1-based) index
# range `jlo:jhi` (every `K`-th starting at `lane`) to the ghost node `ref`,
# which lies in the stencil row `row` with `rowcell` the linear index of the
# cell of that row with the node's first coordinate (see `pair_vector`).
# Testing the fluid type before the support predicate lets the compiler
# schedule the type load independently of the distance calculation.
@inline function mdbc_range(b, A::SMatrix{DP, DP, T}, jlo::Int32, jhi::Int32, lane::Int32, ::Val{K}, ref,
                            Pairs, rowcell::Int32, row, s, Density, ParticleType, SimKernel,
                            m₀, Velocity = nothing, InvDensity = nothing) where {K, DP, T}
    (; h⁻¹, H²) = SimKernel
    j = jlo + lane
    @inbounds while j <= jhi
        xᵢⱼ  = pair_vector(Pairs, ref, j, rowcell, row, s)
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if ParticleType[j] == Fluid
            if xᵢⱼ² <= H²
                dᵢⱼ = sqrt(xᵢⱼ²)
                q   = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
                ρⱼ  = Density[j]

                Wᵢⱼ   = SPHKernels.Wᵢⱼ(SimKernel, q)
                ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                Vⱼ    = divide_by_density(m₀, InvDensity, j, ρⱼ)
                VⱼWᵢⱼ = Vⱼ * Wᵢⱼ

                b = mdbc_accumulate(b, m₀, Wᵢⱼ, ∇ᵢWᵢⱼ, VⱼWᵢⱼ, Velocity, j)

                xⱼᵢ          = -xᵢⱼ
                first_column = SVector{DP, T}(VⱼWᵢⱼ, (Vⱼ * ∇ᵢWᵢⱼ)...)
                A += first_column * SVector{DP, T}(one(T), xⱼᵢ...)'
            end
        end
        j += Int32(K)
    end
    return b, A
end

# Visit the (2R+1)^D cells around the ghost node (R the reach of the grid).
# Ghost nodes may lie outside the particle grid, so every row is range
# checked. `lg` are the local cell coordinates of the node; the row's
# `rowcell` for `pair_vector` is the (possibly virtual, never dereferenced)
# linear index of the cell of the row with the node's first coordinate.
@inline function mdbc_rows(b, A, ref, lg, grid::CellGrid{2}, CellStart, Pairs, s,
                           Density, ParticleType, SimKernel, m₀, lane, lanes::Val, Velocity = nothing, InvDensity = nothing)
    n1, n2 = grid.dims
    R   = reach(grid)
    xlo = max(lg[1] - R, Int32(0))
    xhi = min(lg[1] + R, n1 - Int32(1))
    @inbounds if xlo <= xhi
        for dy in -R:R
            ly = lg[2] + dy
            if (ly >= Int32(0)) & (ly < n2)
                row0    = Int32(1) + xlo + n1 * ly
                jlo     = CellStart[row0] + Int32(1)
                jhi     = CellStart[row0 + (xhi - xlo) + Int32(1)]
                rowcell = row0 + (lg[1] - xlo)
                b, A = mdbc_range(b, A, jlo, jhi, lane, lanes, ref, Pairs, rowcell, CellRow(Int32(0), dy, Int32(0)),
                                  s, Density, ParticleType, SimKernel, m₀, Velocity, InvDensity)
            end
        end
    end
    return b, A
end

@inline function mdbc_rows(b, A, ref, lg, grid::CellGrid{3}, CellStart, Pairs, s,
                           Density, ParticleType, SimKernel, m₀, lane, lanes::Val, Velocity = nothing, InvDensity = nothing)
    n1, n2, n3 = grid.dims
    R   = reach(grid)
    xlo = max(lg[1] - R, Int32(0))
    xhi = min(lg[1] + R, n1 - Int32(1))
    @inbounds if xlo <= xhi
        for dz in -R:R
            lz = lg[3] + dz
            if (lz >= Int32(0)) & (lz < n3)
                for dy in -R:R
                    ly = lg[2] + dy
                    if (ly >= Int32(0)) & (ly < n2)
                        row0    = Int32(1) + xlo + n1 * (ly + n2 * lz)
                        jlo     = CellStart[row0] + Int32(1)
                        jhi     = CellStart[row0 + (xhi - xlo) + Int32(1)]
                        rowcell = row0 + (lg[1] - xlo)
                        b, A = mdbc_range(b, A, jlo, jhi, lane, lanes, ref, Pairs, rowcell, CellRow(Int32(0), dy, dz),
                                          s, Density, ParticleType, SimKernel, m₀, Velocity, InvDensity)
                    end
                end
            end
        end
    end
    return b, A
end

# Thread group `g` (its `K` lanes) handles the `g`-th ghost node owner of
# `GhostIndex`, so the launch covers only boundary particles with a ghost
# node, contiguously: no half empty warps from the fluid particles that are
# interleaved with them in cell order. `Position` and `GhostPoints` are in the
# position precision `TP`, `Pairs` is the pair source (see `pair_vector`).
function mdbc_kernel!(Density::AbstractVector{T}, Position::AbstractVector{SVector{D, TP}}, Pairs,
                      GhostPoints, GhostIndex, ParticleType, CellStart, gridarg, step,
                      SimKernel, SimConstants, InvDensity, Pressure, ::Val{K}, ::Val{Compact}, nghost::Int32, periodicity, GhostNormals, periodic_bounds) where {D, T, TP, K, Compact}
    step_active(step) || return nothing
    grid = load_grid(gridarg)
    t    = thread_index()
    g    = (t - Int32(1)) ÷ Int32(K) + Int32(1)
    lane = (t - Int32(1)) % Int32(K)

    DP = D + 1
    (; m₀, ρ₀, c₀) = SimConstants
    s = cell_size(grid, SimKernel.H)

    b = zero(SVector{DP, T})
    A = zero(SMatrix{DP, DP, T, DP * DP})

    i     = Int32(1)
    gp    = zero(SVector{D, TP})
    valid = g <= nghost
    @inbounds if valid
        i  = GhostIndex[g]
        gp = GhostPoints[Compact ? g : i]
        # A translating mDBC wall keeps its ghost offset in the wall frame.
        if GhostNormals !== nothing && ParticleType[i] == Moving
            gp = Position[i] + SVector{D, TP}(GhostNormals[Compact ? g : i])
        end
        valid = ParticleType[i] != Frozen
    end

    @inbounds if valid
        b, A = mdbc_neighborhood(b, A, gp, Pairs, Position, grid, CellStart, s,
            Density, ParticleType, SimKernel, m₀, lane, Val(K), nothing, InvDensity, periodicity, periodic_bounds)
    end

    if K > 1
        b = lanes_sum(b, Val(K))
        A = lanes_sum(A, Val(K))
    end

    # Density correction (mirrors ApplyMDBCCorrection of the CPU code)
    # https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
    @inbounds if valid & (lane == Int32(0))
        ρ = Density[i]
        if abs(det(A)) >= 1e-3
            sol  = A \ b
            # in the position precision, then rounded: the two points are close
            diff = SVector{D, T}(Position[i] - gp)
            grad = SVector{D, T}(ntuple(k -> sol[k + 1], Val(D)))
            v1   = sol[1] + dot(grad, diff)
            ρ    = isnan(v1) ? ρ₀ : v1
        elseif A[1, 1] > zero(T)
            v = b[1] / A[1, 1]
            ρ = isnan(v) ? ρ₀ : v
        end
        Density[i]  = ρ
        if GhostNormals !== nothing && ParticleType[i] == Moving
            GhostPoints[Compact ? g : i] = gp
        end
        store_inverse_density!(InvDensity, i, ρ)
        store_pressure!(Pressure, i, EquationOfStateGamma7(ρ, c₀, ρ₀))
    end
    return nothing
end

"""
    launch_mdbc!(Density, Position, GhostPoints, GhostIndex, ParticleType, CellStart, grid,
                 step, SimKernel, SimConstants; threads, lanes, pos_cells)

mDBC correction of the density of the boundary particles that own a ghost
node. The launch covers the particles listed in `GhostIndex` (the ghost node
owners, see `GPUParticles`), `K` lanes each. `step` gates the kernel (see
`GPUStepState`). `pos_cells` is the `PosCell` array of `Position` when the
positions are stored in a higher precision than the working precision (the
fluid neighbours of a node are then found through it, see `GPUCellGrid`).

With `compact_ghosts = Val(true)`, `GhostPoints` is indexed by owner slot
rather than particle index; otherwise the full-particle layout is retained.
`inverse_density` must contain the current fluid reciprocals before launch;
corrected boundary entries (and `pressure` when cached) are refreshed by this kernel.
"""
function launch_mdbc!(Density, Position, GhostPoints, GhostIndex, ParticleType, CellStart, grid, step,
                      SimKernel, SimConstants; threads::Integer = 128, lanes::Val = Val(1), pos_cells = nothing,
                      compact_ghosts::Val = Val(false), inverse_density = nothing, pressure = nothing, periodicity = nothing,
                      ghost_normals = nothing, periodic_bounds = nothing)
    n = length(GhostIndex)
    n == 0 && return nothing
    K = typeof(lanes).parameters[1]
    Pairs = pos_cells === nothing ? Position : pos_cells
    @cuda threads=threads blocks=cld(n * K, threads) mdbc_kernel!(
        Density, Position, Pairs, GhostPoints, GhostIndex, ParticleType, CellStart, grid, step,
        SimKernel, SimConstants, inverse_density, pressure, lanes, compact_ghosts, Int32(n), periodicity, ghost_normals, periodic_bounds)
    return nothing
end

#---------------------------------------------------------------
include("UpdatedMDBC.jl")

# Half step (symplectic predictor) fused with density limiting, motion, and the
# cell relative form of the half step position when the positions are of a
# higher precision
#---------------------------------------------------------------

function half_step_kernel!(Positionₙ⁺, Velocityₙ⁺, ρₙ⁺,
                           Position, Velocity, Acceleration, Density, dρdtI,
                           ParticleType, GroupMarker, motion, step, SimConstants,
                           PosCellsₙ⁺, CellID, gridarg, H, boundary_mode, InvDensity, Pressure,
                           ::Val{ReusePredictors}, ::Val{FirstMotion}, ::Val{Carried}, n::Int32) where {ReusePredictors, FirstMotion, Carried}
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    dt₂       = step_dt(step) / 2
    TotalTime = step_time(step)
    FirstMotion && apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    (; g, ρ₀, c₀) = SimConstants
    T = eltype(Density)
    @inbounds begin
        type = ParticleType[i]
        if type == Frozen
            ρ = Density[i]
            Positionₙ⁺[i] = Position[i]
            Velocityₙ⁺[i] = zero(eltype(Velocity))
            ρₙ⁺[i] = ρ
            store_inverse_density!(InvDensity, i, ρ)
            store_pressure!(Pressure, i, EquationOfStateGamma7(ρ, c₀, ρ₀))
            store_pos_cell!(PosCellsₙ⁺, i, Position[i], CellID, gridarg, H)
            return nothing
        end
        ML  = MotionLimiterValue(T, type)
        acc = Acceleration[i]
        if !Carried
            acc += ConstructGravitySVector(acc, g * GravityFactorValue(T, type))
            Acceleration[i] = acc
        end
        # the increment is promoted to the precision of the position
        xₙ⁺ = Position[i]
        if !ReusePredictors || type != Fixed
            xₙ⁺ += Velocity[i] * dt₂ * ML
            Positionₙ⁺[i] = xₙ⁺
            Velocityₙ⁺[i] = Velocity[i] + acc * dt₂ * ML
        end
        ρ = Density[i] + dρdtI[i] * dt₂
        ρ = boundary_mode isa UpdatedMDBC && is_wall(type) ? Density[i] : limit_density(ρ, ρ₀, ML)
        ρₙ⁺[i] = ρ
        store_inverse_density!(InvDensity, i, ρ)
        store_pressure!(Pressure, i, EquationOfStateGamma7(ρ, c₀, ρ₀))

        store_pos_cell!(PosCellsₙ⁺, i, xₙ⁺, CellID, gridarg, H)

        apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    end
    return nothing
end

"""
    launch_half_step!(...; pos_cells, CellID, grid, SimKernel)

Symplectic predictor. With `pos_cells` (the `PosCell` array of the half step
positions, for positions stored in a higher precision than the working
precision) the kernel also writes the cell relative form of every half step
position; `CellID`, `grid` and `SimKernel` are then required.
`first_motion = true` also applies the first prescribed-motion half increment
before constructing the predictor, for single-neighbor DBC steps.
`reuse_predictors = true` uses the reorder scratch as dense predictor outputs.
Fixed position/velocity entries must be initialized after every rebuild; this
kernel leaves them untouched. Density and cell-relative preparation still cover
every particle.
With `inverse_density` and/or `pressure`, prediction also writes each predictor
reciprocal and pressure.
`carried = true` reuses total acceleration, including gravity, from the previous
corrector or a startup/rebuild interaction with `include_gravity = Val(true)`.
The default expects interaction acceleration and adds gravity once.
"""
function launch_half_step!(Positionₙ⁺, Velocityₙ⁺, ρₙ⁺,
                           Position, Velocity, Acceleration, Density, dρdtI,
                           ParticleType, GroupMarker, motion, step, SimConstants;
                           pos_cells = nothing, CellID = nothing, grid = nothing, SimKernel = nothing,
                           boundary_mode = SimpleMDBC(), first_motion::Bool = false, reuse_predictors::Bool = false,
                           inverse_density = nothing, pressure = nothing, carried::Bool = false)
    n = length(Position)
    n == 0 && return nothing
    H = SimKernel === nothing ? zero(eltype(Density)) : SimKernel.H
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) half_step_kernel!(
        Positionₙ⁺, Velocityₙ⁺, ρₙ⁺,
        Position, Velocity, Acceleration, Density, dρdtI,
        ParticleType, GroupMarker, motion, step, SimConstants,
        pos_cells, CellID, grid, H, boundary_mode, inverse_density, pressure, Val(reuse_predictors), Val(first_motion), Val(carried), Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Final step: density update, density limiting, symplectic corrector, the
# pressure for the next step and the per step reduction (time step limits and
# displacement) for the next step.
#
# The corrector advances the position with the half step velocity `Velocityₙ⁺`
# (the velocity the second neighbour loop was evaluated at) times `dt`, which
# is the scheme of the CPU `FullTimeStep`.
#---------------------------------------------------------------

"""
Per particle quantities reduced once per step: the viscous time step term,
the force based time step term and the displacement since the last half step
(used to decide when the cell list must be rebuilt). `r` and `posn` may be
of a higher precision than the rest; the result is in the working precision
of `h`.
"""
@inline function step_map(i, Position, Velocity, Acceleration, Positionₙ⁺, h, η²)
    @inbounds begin
        r = Position[i]
        v = Velocity[i]
        a = Acceleration[i]
        posn = Positionₙ⁺[i]
    end
    return step_values(r, v, a, posn, h, η²)
end

@inline function step_values(r::SVector{D, TP}, v::SVector{D, T}, a, posn::SVector{D, TP}, h::T,
                             η²::T) where {D, TP, T}
    rT   = SVector{D, T}(r)
    visc = abs(h * dot(v, rT) / (dot(rT, rT) + η²))
    dt1  = sqrt(h / norm(a))
    disp = T(norm(posn - r))   # the difference in the position precision
    return SVector{3, T}(visc, dt1, disp)
end

@inline step_reduce(a::SVector{3, T}, b::SVector{3, T}) where {T} =
    SVector{3, T}(max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))

function final_step_kernel!(Position, Velocity, Acceleration, Density, dρdtI, ρₙ⁺, Positionₙ⁺,
                            Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, step, SimKernel, SimConstants,
                            partial, ::Val{FlagShift}, boundary_mode, GhostPoints, GhostNormals, BoundarySlots,
                            PosCells, CellID, gridarg, InvDensity, NextInvDensity, NextPressure,
                            diagnostic_partial, particle_ids, damping, ::Val{CommitStep}, n::Int32) where {FlagShift, CommitStep}
    step_active(step) || return nothing
    dt = step_dt(step)
    (; g, ρ₀) = SimConstants
    T = eltype(Density)
    acc_red = SVector{3, T}(zero(T), T(Inf), zero(T))
    acc_diagnostic = diagnostic_init(diagnostic_partial)

    i = thread_index()
    stride = blockDim().x * gridDim().x
    @inbounds while i <= n
        type = ParticleType[i]
        if type == Frozen
            ρ = Density[i]
            store_inverse_density!(NextInvDensity, i, ρ)
            if NextPressure !== nothing
                store_pressure!(NextPressure, i, particle_pressure(ρ, type, SimConstants.c₀, ρ₀, boundary_mode))
            end
            i += stride
            continue
        end
        ML = MotionLimiterValue(T, type)

        # DensityEpsi! followed by LimitDensityAtBoundary! (CPU order: "07
        # Final Density", "08 Final LimitDensityAtBoundary")
        ρ    = Density[i]
        epsi = -divide_by_density(dρdtI[i], InvDensity, i, ρₙ⁺[i]) * dt
        ρ   *= (2 - epsi) / (2 + epsi)
        ρ    = boundary_mode isa UpdatedMDBC && is_wall(type) ? ρₙ⁺[i] : limit_density(ρ, ρ₀, ML)
        Density[i]  = ρ
        # The predictor reciprocal must be consumed before an aliased cache is overwritten.
        store_inverse_density!(NextInvDensity, i, ρ)
        if NextPressure !== nothing
            store_pressure!(NextPressure, i, particle_pressure(ρ, type, SimConstants.c₀, ρ₀, boundary_mode))
        end

        # FullTimeStep
        acc = Acceleration[i]
        acc += ConstructGravitySVector(acc, g * GravityFactorValue(T, type))
        Acceleration[i] = acc
        v = Velocity[i] + acc * dt * ML
        Velocity[i] = v

        # Symplectic corrector: the position moves with the half step velocity
        vₙ⁺ = Velocityₙ⁺[i]
        x = Position[i]
        if FlagShift
            D     = length(v)
            A     = 2      # Value between 1 to 6 advised
            A_FST = 0      # zero for internal flows
            A_FSM = D      # 2d, 3d val different
            A_FSC = (∇◌rᵢ[i] - A_FST) / (A_FSM - A_FST)
            δxᵢ = A_FSC < 0 ? zero(v) : -A_FSC * A * SimKernel.h * norm(vₙ⁺) * dt * ∇Cᵢ[i]
            x += (vₙ⁺ * dt + δxᵢ) * ML
        else
            x += (vₙ⁺ * dt) * ML
        end
        Position[i] = x
        v = damped_velocity(damping, x, v, type, dt)
        Velocity[i] = v
        store_pos_cell!(PosCells, i, x, CellID, gridarg, SimKernel.H)
        # Keep SimpleMDBC output and the next step's correction at the accepted pose.
        if (boundary_mode isa SimpleMDBC || boundary_mode isa UpdatedMDBC) &&
            type == Moving && GhostPoints !== nothing && GhostNormals !== nothing
            slot = BoundarySlots === nothing ? i : BoundarySlots[i]
            slot == 0 || (GhostPoints[slot] = x + GhostNormals[slot])
        end

        # Values for the time step and cell list update decision of the next step
        values = step_values(x, v, acc, Positionₙ⁺[i], SimKernel.h, SimKernel.η²)
        acc_red = step_reduce(acc_red, values)
        acc_diagnostic = diagnostic_accumulate(acc_diagnostic, values, particle_ids, i)

        i += stride
    end

    block_reduce_store!(partial, acc_red, step_reduce)
    diagnostic_store!(diagnostic_partial, acc_diagnostic)
    # A single block has completed every particle write at the reduction barrier.
    CommitStep && threadIdx().x == 1 && commit_kernel!(step)
    return nothing
end

"""
Launch the final step. Uses `red.nblocks` blocks in a grid stride loop so the
per block reduction results fit the reduction workspace; they are consumed by
`launch_finish!` (device) or `finish_reduction(red, step_reduce, init)` (host).
`step` is a `StepState` or a `HostStep` with a fixed `dt`.
`commit_step = true` commits the device step after the reduction barrier and
requires a single reduction block; multiple blocks need a separate commit launch.
With `pos_cells` (the `PosCell` array of the start-of-step positions) the
kernel also writes the cell relative form of the new positions for the cell
list whose sorted cell ids are `CellID`, so the next step needs no separate
pass; `CellID` and `grid` are then required.
`inverse_density` must hold the evaluated predictor reciprocals, including any
boundary corrections, for reuse in the final density update.
With `next_inverse_density` and/or `next_pressure`, also cache the accepted
density's reciprocal and pressure for the next step. `next_inverse_density`
may alias `inverse_density`: each predictor reciprocal is read before being
overwritten. Frozen entries cache their unchanged accepted density.
Without these destinations, the predictor caches are left unchanged.
With `diagnostics`, retain the stable `particle_ids` of both timestep limits
alongside the numerical reduction, excluding the same frozen entries.
"""
function launch_final_step!(Position, Velocity, Acceleration, Density, dρdtI, ρₙ⁺, Positionₙ⁺,
                            Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, step, SimKernel, SimConstants,
                            red::ReductionWorkspace, FlagShift::Val; boundary_mode = SimpleMDBC(),
                            ghost_points = nothing, ghost_normals = nothing, boundary_slots = nothing,
                            pos_cells = nothing, CellID = nothing, grid = nothing, commit_step::Bool = false,
                            inverse_density = nothing, next_inverse_density = nothing, next_pressure = nothing,
                            diagnostics = nothing, particle_ids = nothing, damping = ())
    commit_step && red.nblocks != 1 && throw(ArgumentError("fused commit requires one reduction block"))
    next_inverse_density === nothing || length(next_inverse_density) == length(Density) ||
        throw(DimensionMismatch("Next-state reciprocal cache must have one entry per density"))
    next_pressure === nothing || length(next_pressure) == length(Density) ||
        throw(DimensionMismatch("Next-state pressure cache must have one entry per density"))
    n = length(Position)
    n == 0 && return nothing
    if diagnostics !== nothing
        diagnostics.reduction.nblocks == red.nblocks || throw(DimensionMismatch("Timestep diagnostic reduction must match the step reduction"))
        particle_ids !== nothing && length(particle_ids) == n || throw(DimensionMismatch("Timestep diagnostics require one stable ID per particle"))
    end
    diagnostic_partial = diagnostics === nothing ? nothing : diagnostics.reduction.partial
    @cuda threads=ELEMENTWISE_THREADS blocks=red.nblocks final_step_kernel!(
        Position, Velocity, Acceleration, Density, dρdtI, ρₙ⁺, Positionₙ⁺,
        Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, step, SimKernel, SimConstants,
        red.partial, FlagShift, boundary_mode, ghost_points, ghost_normals, boundary_slots,
        pos_cells, CellID, grid, inverse_density, next_inverse_density, next_pressure,
        diagnostic_partial, particle_ids, damping, Val(commit_step), Int32(n))
    return nothing
end

"""
    launch_step_reduction!(red, Position, Velocity, Acceleration, Positionₙ⁺, SimKernel)

Per block partial results of the step reduction for a state that was not
produced by `launch_final_step!` (the initial state). Consumed by
`launch_finish!`.
Pass the same optional `diagnostics` workspace and stable `particle_ids` used
by final integration to initialize the limiter provenance too.
"""
function launch_step_reduction!(red::ReductionWorkspace{SVector{3, T}}, Position, Velocity, Acceleration,
                                Positionₙ⁺, SimKernel; diagnostics = nothing, particle_ids = nothing) where {T}
    init = SVector{3, T}(zero(T), T(Inf), zero(T))
    launch_reduce!(red, step_map, step_reduce, init, length(Position), Position, Velocity, Acceleration,
                   Positionₙ⁺, T(SimKernel.h), T(SimKernel.η²))
    if diagnostics !== nothing
        launch_timestep_diagnostic_reduction!(diagnostics, Position, Velocity, Acceleration, Positionₙ⁺, particle_ids, SimKernel)
    end
    return nothing
end

#---------------------------------------------------------------
# Device side loop control: time step from the reduction of the previous
# step (with the cell list rebuild and output time decisions) and the commit
# of a completed step. Both replace host work that needed a synchronization.
#---------------------------------------------------------------

# One block of REDUCE_THREADS threads. Combines the per block partial results
# into the time step of the next step, exactly like the host used to:
# `dt = CFL * min(dt1, h / (c₀ + visc))`, `Δx += 4 * maxdisp`. Sets the stop
# flag when the output time is reached (before computing anything) or when
# the accumulated displacement bound requires a cell list rebuild.
function finish_kernel!(step::DeviceStep{T}, partial, nblocks::Int32, CFL::T, c₀::T, h::T,
                        diagnostic_partial, diagnostic_result, stage_sampler) where {T}
    f = step.f
    s = step.i
    @inbounds begin
        s[I_STOP] == STOP_NONE || return nothing
        phase = s[I_PHASE]
        if phase == PHASE_NEED_DT && GPUStepState.output_time_reached(f[F_TIME], f[F_TOUT], s[I_CLIP_OUTPUT] != 0)
            if threadIdx().x == Int32(1)
                s[I_STOP] = STOP_OUTPUT
            end
            return nothing
        end
        # `dt` was computed before a rebuild interrupted the step: keep it.
        if phase == PHASE_DT_READY
            threadIdx().x == Int32(1) && GPUStepState.prepare_stage_samples!(stage_sampler, step)
            return nothing
        end
    end

    acc = SVector{3, T}(zero(T), T(Inf), zero(T))
    b = threadIdx().x
    @inbounds while b <= nblocks
        acc = step_reduce(acc, partial[b])
        b += blockDim().x
    end
    r = block_reduce(acc, step_reduce)
    limit = finish_diagnostic(diagnostic_partial, nblocks)

    if threadIdx().x == Int32(1)
        @inbounds begin
            visc = r[1]
            dt1  = r[2]
            dt   = Float64(CFL * min(dt1, h / (c₀ + visc)))
            if s[I_CLIP_OUTPUT] != 0
                dt = min(dt, f[F_TOUT] - f[F_TIME])
            end
            d4   = 4 * r[3]
            dx   = T(f[F_DX]) + d4
            f[F_DT]   = dt
            record_diagnostic!(diagnostic_result, limit, step, dt, CFL * dt1, CFL * (h / (c₀ + visc)), dt1 <= h / (c₀ + visc))
            f[F_DISP] = d4
            f[F_DX]   = dx
            s[I_PHASE] = PHASE_DT_READY
            if dx >= h
                s[I_STOP] = STOP_REBUILD
            else
                GPUStepState.prepare_stage_samples!(stage_sampler, step)
            end
        end
    end
    return nothing
end

"""
    launch_finish!(state, red, SimKernel, SimConstants)

Compute the time step of the next step on the device from the reduction in
`red` (see `finish_kernel!`).
With `diagnostics`, retain the selected criterion and its particle ID for
the step being started; a pending step interrupted by a rebuild keeps them.
An optional `stage_sampler` prepares motion coefficients in the same launch,
after an active timestep is ready, including the resumed rebuild path.
"""
function launch_finish!(state::StepState{T}, red::ReductionWorkspace{SVector{3, T}}, SimKernel,
                        SimConstants; diagnostics = nothing, stage_sampler = nothing) where {T}
    diagnostic_partial = diagnostics === nothing ? nothing : diagnostics.reduction.partial
    diagnostic_result = diagnostics === nothing ? nothing : diagnostics.result
    @cuda threads=REDUCE_THREADS blocks=1 finish_kernel!(state, red.partial, Int32(red.nblocks),
                                                        T(SimConstants.CFL), T(SimConstants.c₀),
                                                        T(SimKernel.h), diagnostic_partial, diagnostic_result, stage_sampler)
    return nothing
end

# Single thread. Advances the time and the iteration counter after a step ran.
function commit_kernel!(step::DeviceStep)
    f = step.f
    s = step.i
    @inbounds if s[I_STOP] == STOP_NONE
        # Assign the deadline exactly on a clipped step, including when this
        # commit is fused into the final particle kernel. Do not relabel an
        # overshooting state: every integrator has already used the clipped dt.
        time = f[F_TIME]
        dt = f[F_DT]
        f[F_TIME] = s[I_CLIP_OUTPUT] != 0 && dt >= f[F_TOUT] - time ? f[F_TOUT] : time + dt
        s[I_ITER] += Int32(1)
        s[I_PHASE] = PHASE_NEED_DT
    end
    return nothing
end

"""
    launch_commit!(state)

Commit the step that was just launched: `TotalTime += dt`, `Iteration += 1`.
"""
function launch_commit!(state::StepState)
    @cuda threads=1 blocks=1 commit_kernel!(state)
    return nothing
end

include("GPUDualSymplectic.jl")

end # module
