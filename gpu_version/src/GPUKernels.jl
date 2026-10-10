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
to `i` and `j`. For the built in models most terms are antisymmetric, but the
density diffusion term is deliberately not (`Dⱼ = -Dᵢ` is used as a cheap
approximation) and a user supplied model may not be either. To reproduce the
CPU results exactly the gather kernel reconstructs which particle of a pair
was the CPU's `i`: for pairs inside one cell the lower index, for pairs in
different cells the particle in the cell that is processed (the higher index,
because the CPU stencil only visits cells with a lower linear index). Both
particles of a pair reach the same decision, so the rule stays consistent for
any grid; it only reproduces the CPU's choice with the CPU's cells of edge
`H` (`GPUCellSubdivision = 1`).

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
the half step positions; `launch_pos_cells!` does the same for the
start-of-step positions.
"""
module GPUKernels

using CUDA
using StaticArrays
using LinearAlgebra

using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SimulationEquations
using ..SimulationGeometry
using ..GPUCellGrid
using ..GPUReductions
using ..GPUStepState
using ..SimulationMetaDataConfiguration: SimpleMDBC, UpdatedMDBC

export launch_interactions!, launch_mdbc!, launch_motion!, launch_half_step!, launch_final_step!,
       launch_inv_density!, launch_finish!, launch_commit!, launch_step_reduction!, launch_pos_cells!,
       step_map, step_reduce, choose_lanes, ELEMENTWISE_THREADS

const ELEMENTWISE_THREADS = REDUCE_THREADS
const FULL_MASK = 0xffffffff

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

function motion_kernel!(Position, Velocity, ParticleType, GroupMarker, motion, step, n::Int32)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    dt₂       = step_dt(step) / 2
    TotalTime = step_time(step)
    apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    return nothing
end

function launch_motion!(Position, Velocity, ParticleType, GroupMarker, motion, step)
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) motion_kernel!(
        Position, Velocity, ParticleType, GroupMarker, motion, step, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Reciprocal of the start-of-step density (after the mDBC correction and any
# reordering by a cell list rebuild)
#---------------------------------------------------------------

function inv_density_kernel!(InvDensity, Density, step, n::Int32,
                             PosCells, Position, CellID, grid, H, Velocity, ParticleType, GroupMarker, motion)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        if motion !== nothing
            apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, step_dt(step) / 2, step_time(step))
        end
        InvDensity[i] = inv(Density[i])
        if PosCells !== nothing
            store_pos_cell!(PosCells, i, Position[i], CellID, grid, H)
        end
    end
    return nothing
end

function launch_inv_density!(InvDensity, Density, step;
                             pos_cells = nothing, Position = nothing,
                             CellID = nothing, grid = nothing, H = nothing,
                             Velocity = nothing, ParticleType = nothing, GroupMarker = nothing, motion = nothing)
    n = length(Density)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) inv_density_kernel!(
        InvDensity, Density, step, Int32(n), pos_cells, Position, CellID, grid, H, Velocity, ParticleType, GroupMarker, motion)
    return nothing
end

#---------------------------------------------------------------
# Cell relative positions of the start-of-step state (only with positions of
# a higher precision than the working precision)
#---------------------------------------------------------------

# Cell relative form of position `x` of particle `i`, for the cell the
# particle was binned into. A `nothing` destination (positions in the working
# precision, no cell relative form needed) compiles to nothing.
@inline store_pos_cell!(::Nothing, i, x, CellID, gridarg, H) = nothing
@inline function store_pos_cell!(PosCells, i, x, CellID, gridarg, H)
    grid = load_grid(gridarg)
    @inbounds PosCells[i] = pos_cell(x, CellID[i], grid, cell_size(grid, H))
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
whose sorted cell ids are `CellID`. Launched before the neighbour loops that
evaluate the start-of-step positions when the positions are stored in a
higher precision than the working precision; the half step kernel writes the
form of the half step positions itself.
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

function interaction_kernel!(dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ,
                             Pairs, Position, Density, InvDensity, Pressure,
                             Velocity::AbstractVector{SVector{D, T}}, ParticleType, SimParticles,
                             CellStart, CellID, gridarg, step,
                             SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                             ::Val{FlagKernel}, ::Val{FlagShift}, ::Val{BoundaryForces}, ::Val{K},
                             boundary_data, n::Int32) where {D, T, FlagKernel, FlagShift, BoundaryForces, K}
    step_active(step) || return nothing
    grid = load_grid(gridarg)
    t    = thread_index()
    i    = (t - Int32(1)) ÷ Int32(K) + Int32(1)
    lane = (t - Int32(1)) % Int32(K)
    valid = i <= n

    (; m₀, dx)  = SimConstants
    (; h⁻¹, H²) = SimKernel
    s = cell_size(grid, SimKernel.H)

    dρdt  = zero(T)
    acc   = zero(SVector{D, T})
    Wsum  = zero(T)
    ∇Wsum = zero(SVector{D, T})
    ∇C    = zero(SVector{D, T})
    ∇r    = zero(T)

    @inbounds if valid && boundary_enabled(boundary_data, ParticleType, i)
        pᵢ  = Pairs[i]
        vᵢ  = boundary_velocity(boundary_data, Velocity, ParticleType, i)
        ρᵢ  = Density[i]
        ρᵢ⁻¹ = InvDensity[i]
        Pᵢ  = Pressure[i]
        typeᵢ = ParticleType[i]
        MLᵢ = MotionLimiterValue(T, typeᵢ)

        # Boundary particles only need the density rate; their acceleration is
        # never applied (MotionLimiter = 0). Skipping the momentum terms for them
        # is optional because the CPU code includes their acceleration in the
        # force based time step criterion. Floating bodies always need them:
        # their particle accelerations sum to the force on the body.
        forces = BoundaryForces | (MLᵢ != zero(T)) | (typeᵢ == Floating)
        wallᵢ = is_wall(typeᵢ)

        c      = CellID[i]
        own_lo = CellStart[c] + Int32(1)
        own_hi = CellStart[c + Int32(1)]

        for row in cell_rows(grid)
            jlo, jhi = row_range(grid, CellStart, c, row)
            rowcell  = c + row.off
            j = jlo + lane
            while j <= jhi
                if j != i
                    xᵢⱼ  = pair_vector(Pairs, pᵢ, j, rowcell, row, s)
                    xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
                    if xᵢⱼ² <= H² && boundary_enabled(boundary_data, ParticleType, j)
                        dᵢⱼ   = sqrt(xᵢⱼ²)
                        q     = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
                        ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                        ρⱼ   = Density[j]
                        ρⱼ⁻¹ = InvDensity[j]
                        vⱼ   = boundary_velocity(boundary_data, Velocity, ParticleType, j)
                        vᵢⱼ  = vᵢ - vⱼ
                        density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                        # DBC walls take density from the fluid, not other walls.
                        wall_pair = wallᵢ && is_wall(ParticleType[j])
                        if !wall_pair
                            dρdt += -ρᵢ * (m₀ * ρⱼ⁻¹) * density_symmetric_term
                        end

                        # Which particle of the pair was `i` on the CPU? Evaluate the
                        # models from that particle's point of view (branch free: the
                        # sign flip and the swaps of index and density are selects).
                        # The densities handed to the models are those of the arrays
                        # being evaluated (the predictor state in the corrector loop).
                        same_cell = (j >= own_lo) & (j <= own_hi)
                        i_first   = same_cell ? (i < j) : (i > j)
                        sgn  = i_first ? one(T) : -one(T)
                        ia   = i_first ? i : j
                        ja   = i_first ? j : i
                        ρa   = i_first ? ρᵢ : ρⱼ
                        ρb   = i_first ? ρⱼ : ρᵢ
                        ρa⁻¹ = i_first ? ρᵢ⁻¹ : ρⱼ⁻¹
                        ρb⁻¹ = i_first ? ρⱼ⁻¹ : ρᵢ⁻¹
                        D1, D2 = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants,
                                                           SimParticles, sgn * xᵢⱼ, sgn * ∇ᵢWᵢⱼ, xᵢⱼ²,
                                                           ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja, ParticleType)
                        dρdt += i_first ? D1 : D2

                        if forces
                            viscous_velocity = boundary_viscous_velocity(boundary_data, Velocity, ParticleType, i) -
                                               boundary_viscous_velocity(boundary_data, Velocity, ParticleType, j)
                            v1, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles,
                                                      sgn * xᵢⱼ, sgn * viscous_velocity, sgn * ∇ᵢWᵢⱼ, xᵢⱼ²,
                                                      ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja)
                            visc  = sgn * v1

                            Pⱼ   = Pressure[j]
                            Pfac = (Pᵢ + Pⱼ) * (ρᵢ⁻¹ * ρⱼ⁻¹)
                            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
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
        dρdtI[i]        = dρdt
        Acceleration[i] = acc
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
    launch_interactions!(...; threads, lanes, boundary_forces, pos_cells)

Launch the gather interaction kernel. `Position`, `Density`, `InvDensity`,
`Pressure` and `Velocity` are the arrays of the state being evaluated (the
half step arrays for the second neighbour loop); `InvDensity` holds the
precomputed reciprocals of `Density`. The viscosity and diffusion models
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
"""
function launch_interactions!(dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ,
                              Position, Density, InvDensity, Pressure, Velocity, ParticleType,
                              CellStart, CellID, grid, step,
                              SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                              FlagKernel::Val, FlagShift::Val; threads::Integer = 128, lanes::Val = Val(1),
                              boundary_forces::Val = Val(true), pos_cells = nothing, boundary_data = nothing)
    n = length(Position)
    n == 0 && return nothing
    K = typeof(lanes).parameters[1]
    Pairs = pos_cells === nothing ? Position : pos_cells
    SimParticles = (Position = Position, Density = Density, Velocity = Velocity, Pressure = Pressure,
                    Type = ParticleType)
    @cuda threads=threads blocks=cld(n * K, threads) interaction_kernel!(
        dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ,
        Pairs, Position, Density, InvDensity, Pressure, Velocity, ParticleType, SimParticles,
        CellStart, CellID, grid, step,
        SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
        FlagKernel, FlagShift, boundary_forces, lanes, boundary_data, Int32(n))
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
# The distance test comes first: only the roughly 30 % of the candidates
# inside the support also load `ParticleType`.
@inline function mdbc_range(b, A::SMatrix{DP, DP, T}, jlo::Int32, jhi::Int32, lane::Int32, ::Val{K}, ref,
                            Pairs, rowcell::Int32, row::CellRow, s, Density, ParticleType, SimKernel,
                            m₀, Velocity = nothing) where {K, DP, T}
    (; h⁻¹, H²) = SimKernel
    j = jlo + lane
    @inbounds while j <= jhi
        xᵢⱼ  = pair_vector(Pairs, ref, j, rowcell, row, s)
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            if ParticleType[j] == Fluid
                dᵢⱼ = sqrt(xᵢⱼ²)
                q   = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
                ρⱼ  = Density[j]

                Wᵢⱼ   = SPHKernels.Wᵢⱼ(SimKernel, q)
                ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                Vⱼ    = m₀ / ρⱼ
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
                           Density, ParticleType, SimKernel, m₀, lane, lanes::Val, Velocity = nothing)
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
                                  s, Density, ParticleType, SimKernel, m₀, Velocity)
            end
        end
    end
    return b, A
end

@inline function mdbc_rows(b, A, ref, lg, grid::CellGrid{3}, CellStart, Pairs, s,
                           Density, ParticleType, SimKernel, m₀, lane, lanes::Val, Velocity = nothing)
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
                                          s, Density, ParticleType, SimKernel, m₀, Velocity)
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
function mdbc_kernel!(Density::AbstractVector{T}, Pressure, Position::AbstractVector{SVector{D, TP}}, Pairs,
                      GhostPoints, GhostIndex, ParticleType, CellStart, gridarg, step,
                      SimKernel, SimConstants, ::Val{K}, nghost::Int32) where {D, T, TP, K}
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
        gp = GhostPoints[i]
    end

    @inbounds if valid
        cg  = cell_coords(gp, bin_scale(grid, SimKernel.H⁻¹))
        lg  = ntuple(d -> cg[d] - grid.origin[d], Val(D))
        ref = mdbc_reference(Pairs, gp, cg, s)
        b, A = mdbc_rows(b, A, ref, lg, grid, CellStart, Pairs, s, Density, ParticleType, SimKernel, m₀,
                         lane, Val(K))
    end

    if K > 1
        b = lanes_sum(b, Val(K))
        A = lanes_sum(A, Val(K))
    end

    # Density correction (mirrors ApplyMDBCCorrection of the CPU code)
    # https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
    # followed by the pressure of the corrected density (the CPU calls
    # `Pressure!` after the correction; a neighbour loop that reads the
    # corrected density must see the matching boundary pressure).
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
        Pressure[i] = EquationOfStateGamma7(ρ, c₀, ρ₀)
    end
    return nothing
end

"""
    launch_mdbc!(Density, Pressure, Position, GhostPoints, GhostIndex, ParticleType, CellStart, grid,
                 step, SimKernel, SimConstants; threads, lanes, pos_cells)

mDBC correction of the density of the boundary particles that own a ghost
node, and the pressure of the corrected density. The launch covers the
particles listed in `GhostIndex` (the ghost node owners in cell order, see
`GPUParticles`), `K` lanes each. `step` gates the kernel (see
`GPUStepState`). `pos_cells` is the `PosCell` array of `Position` when the
positions are stored in a higher precision than the working precision (the
fluid neighbours of a node are then found through it, see `GPUCellGrid`).
"""
function launch_mdbc!(Density, Pressure, Position, GhostPoints, GhostIndex, ParticleType, CellStart, grid, step,
                      SimKernel, SimConstants; threads::Integer = 128, lanes::Val = Val(1), pos_cells = nothing)
    n = length(GhostIndex)
    n == 0 && return nothing
    K = typeof(lanes).parameters[1]
    Pairs = pos_cells === nothing ? Position : pos_cells
    @cuda threads=threads blocks=cld(n * K, threads) mdbc_kernel!(
        Density, Pressure, Position, Pairs, GhostPoints, GhostIndex, ParticleType, CellStart, grid, step,
        SimKernel, SimConstants, lanes, Int32(n))
    return nothing
end

#---------------------------------------------------------------
include("UpdatedMDBC.jl")

# Half step (symplectic predictor) fused with density limiting, motion, the
# pressure of the half step density and its reciprocal, and the cell relative
# form of the half step position when the positions are of a higher precision
#---------------------------------------------------------------

function half_step_kernel!(Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, InvDensityₙ⁺, Pressure,
                           Position, Velocity, Acceleration, Density, dρdtI,
                           ParticleType, GroupMarker, motion, step, SimConstants,
                           PosCellsₙ⁺, CellID, gridarg, H, boundary_mode, ::Val{FirstMotion}, n::Int32) where {FirstMotion}
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
        ML  = MotionLimiterValue(T, type)
        acc = Acceleration[i]
        acc += ConstructGravitySVector(acc, g * GravityFactorValue(T, type))
        Acceleration[i] = acc
        # the increment is promoted to the precision of the position
        xₙ⁺ = Position[i] + Velocity[i] * dt₂ * ML
        Positionₙ⁺[i]   = xₙ⁺
        Velocityₙ⁺[i]   = Velocity[i] + acc * dt₂ * ML
        ρ = Density[i] + dρdtI[i] * dt₂
        ρ = boundary_mode isa UpdatedMDBC && is_wall(type) ? Density[i] : limit_density(ρ, ρ₀, ML)
        ρₙ⁺[i] = ρ
        InvDensityₙ⁺[i] = inv(ρ)

        store_pos_cell!(PosCellsₙ⁺, i, xₙ⁺, CellID, gridarg, H)

        apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)

        Pressure[i] = EquationOfStateGamma7(ρ, c₀, ρ₀)
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
"""
function launch_half_step!(Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, InvDensityₙ⁺, Pressure,
                           Position, Velocity, Acceleration, Density, dρdtI,
                           ParticleType, GroupMarker, motion, step, SimConstants;
                           pos_cells = nothing, CellID = nothing, grid = nothing, SimKernel = nothing,
                           boundary_mode = SimpleMDBC(), first_motion::Bool = false)
    n = length(Position)
    n == 0 && return nothing
    H = SimKernel === nothing ? zero(eltype(Density)) : SimKernel.H
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) half_step_kernel!(
        Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, InvDensityₙ⁺, Pressure,
        Position, Velocity, Acceleration, Density, dρdtI,
        ParticleType, GroupMarker, motion, step, SimConstants,
        pos_cells, CellID, grid, H, boundary_mode, Val(first_motion), Int32(n))
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

function final_step_kernel!(Position, Velocity, Acceleration, Density, Pressure, dρdtI, ρₙ⁺, Positionₙ⁺,
                            Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, step, SimKernel, SimConstants,
                            partial, ::Val{FlagShift}, boundary_mode, GhostPoints, GhostNormals, ::Val{CommitStep}, n::Int32) where {FlagShift, CommitStep}
    step_active(step) || return nothing
    dt = step_dt(step)
    (; g, ρ₀, c₀) = SimConstants
    T = eltype(Density)
    acc_red = SVector{3, T}(zero(T), T(Inf), zero(T))

    i = thread_index()
    stride = blockDim().x * gridDim().x
    @inbounds while i <= n
        type = ParticleType[i]
        ML = MotionLimiterValue(T, type)

        # DensityEpsi! followed by LimitDensityAtBoundary! (CPU order: "07
        # Final Density", "08 Final LimitDensityAtBoundary")
        ρ    = Density[i]
        epsi = -(dρdtI[i] / ρₙ⁺[i]) * dt
        ρ   *= (2 - epsi) / (2 + epsi)
        ρ    = boundary_mode isa UpdatedMDBC && is_wall(type) ? ρₙ⁺[i] : limit_density(ρ, ρ₀, ML)
        Density[i]  = ρ
        Pressure[i] = boundary_mode isa UpdatedMDBC && is_wall(type) ? c₀^2 * (ρ - ρ₀) : EquationOfStateGamma7(ρ, c₀, ρ₀)

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
        if boundary_mode isa UpdatedMDBC && type == Moving && GhostPoints !== nothing
            GhostPoints[i] = x + GhostNormals[i]
        end

        # Values for the time step and cell list update decision of the next step
        acc_red = step_reduce(acc_red, step_values(x, v, acc, Positionₙ⁺[i], SimKernel.h, SimKernel.η²))

        i += stride
    end

    block_reduce_store!(partial, acc_red, step_reduce)
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
"""
function launch_final_step!(Position, Velocity, Acceleration, Density, Pressure, dρdtI, ρₙ⁺, Positionₙ⁺,
                            Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, step, SimKernel, SimConstants,
                            red::ReductionWorkspace, FlagShift::Val; boundary_mode = SimpleMDBC(),
                            ghost_points = nothing, ghost_normals = nothing, commit_step::Bool = false)
    commit_step && red.nblocks != 1 && throw(ArgumentError("fused commit requires one reduction block"))
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=red.nblocks final_step_kernel!(
        Position, Velocity, Acceleration, Density, Pressure, dρdtI, ρₙ⁺, Positionₙ⁺,
        Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, step, SimKernel, SimConstants,
        red.partial, FlagShift, boundary_mode, ghost_points, ghost_normals, Val(commit_step), Int32(n))
    return nothing
end

"""
    launch_step_reduction!(red, Position, Velocity, Acceleration, Positionₙ⁺, SimKernel)

Per block partial results of the step reduction for a state that was not
produced by `launch_final_step!` (the initial state). Consumed by
`launch_finish!`.
"""
function launch_step_reduction!(red::ReductionWorkspace{SVector{3, T}}, Position, Velocity, Acceleration,
                                Positionₙ⁺, SimKernel) where {T}
    init = SVector{3, T}(zero(T), T(Inf), zero(T))
    launch_reduce!(red, step_map, step_reduce, init, length(Position), Position, Velocity, Acceleration,
                   Positionₙ⁺, T(SimKernel.h), T(SimKernel.η²))
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
function finish_kernel!(step::DeviceStep{T}, partial, nblocks::Int32, CFL::T, c₀::T, h::T) where {T}
    f = step.f
    s = step.i
    @inbounds begin
        s[I_STOP] == STOP_NONE || return nothing
        phase = s[I_PHASE]
        if phase == PHASE_NEED_DT && f[F_TIME] > f[F_TOUT]
            if threadIdx().x == Int32(1)
                s[I_STOP] = STOP_OUTPUT
            end
            return nothing
        end
        # `dt` was computed before a rebuild interrupted the step: keep it.
        phase == PHASE_DT_READY && return nothing
    end

    acc = SVector{3, T}(zero(T), T(Inf), zero(T))
    b = threadIdx().x
    @inbounds while b <= nblocks
        acc = step_reduce(acc, partial[b])
        b += blockDim().x
    end
    r = block_reduce(acc, step_reduce)

    if threadIdx().x == Int32(1)
        @inbounds begin
            visc = r[1]
            dt1  = r[2]
            dt   = CFL * min(dt1, h / (c₀ + visc))
            d4   = 4 * r[3]
            dx   = f[F_DX] + d4
            f[F_DT]   = dt
            f[F_DISP] = d4
            f[F_DX]   = dx
            s[I_PHASE] = PHASE_DT_READY
            if dx >= h
                s[I_STOP] = STOP_REBUILD
            end
        end
    end
    return nothing
end

"""
    launch_finish!(state, red, SimKernel, SimConstants)

Compute the time step of the next step on the device from the reduction in
`red` (see `finish_kernel!`).
"""
function launch_finish!(state::StepState{T}, red::ReductionWorkspace{SVector{3, T}}, SimKernel,
                        SimConstants) where {T}
    @cuda threads=REDUCE_THREADS blocks=1 finish_kernel!(state, red.partial, Int32(red.nblocks),
                                                        T(SimConstants.CFL), T(SimConstants.c₀),
                                                        T(SimKernel.h))
    return nothing
end

# Single thread. Advances the time and the iteration counter after a step ran.
function commit_kernel!(step::DeviceStep)
    f = step.f
    s = step.i
    @inbounds if s[I_STOP] == STOP_NONE
        f[F_TIME] += f[F_DT]
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

end # module
