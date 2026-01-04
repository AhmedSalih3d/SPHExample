# CUDA-specialized neighbor loop and simulation backend.

function SelectCudaBackend(::SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,L},
                           ::Nothing, ::LinearDensityDiffusion, ::ArtificialViscosity) where {D,T,L}
    if !CUDA.functional()
        @warn "CUDA requested but not functional; falling back to CPU simulation loop."
        return CPUBackend()
    end
    return GPUBackend()
end

SelectCudaBackend(_SimMetaData, _MotionDefinition, _SimDensityDiffusion, _SimViscosity) = CPUBackend()

function SyncBackendState!(::GPUBackend, SimParticles, SimMetaData)
    buffers = SimMetaData.CudaBuffers
    if buffers !== nothing && buffers.active
        SyncCudaStateToHost!(buffers, SimParticles)
    end
    return nothing
end

mutable struct CudaNeighborBuffers
    n_particles::Int
    host_neighbor_offsets::Vector{Int}
    host_neighbor_indices::Vector{Int}
    position
    density
    pressure
    velocity
    gravity_factor
    motion_limiter
    neighbor_offsets
    neighbor_indices
    dρdtI
    acceleration
    position_half
    velocity_half
    rho_half
    max_visc
    min_dt_force
    neighbor_dirty::Bool
    state_initialized::Bool
    active::Bool
end

function EmptyCudaNeighborBuffers()
    return CudaNeighborBuffers(
        0,
        Int[],
        Int[],
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        Int[],
        Int[],
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        true,
        false,
        false,
    )
end

function EnsureCudaNeighborBuffers!(SimMetaData, Position, Density, Pressure, Velocity,
                                    GravityFactor, MotionLimiter, neighbor_offsets,
                                    neighbor_indices, dρdtI, Acceleration; sync_state = true,
                                    sync_neighbors = true)
    n_particles = length(Position)
    offsets_len = length(neighbor_offsets)
    indices_len = length(neighbor_indices)
    buffers = SimMetaData.CudaBuffers
    needs_alloc = buffers === nothing ||
                  buffers.n_particles != n_particles ||
                  length(buffers.neighbor_offsets) != offsets_len ||
                  length(buffers.neighbor_indices) != indices_len
    if needs_alloc
        buffers = CudaNeighborBuffers(
            n_particles,
            neighbor_offsets,
            neighbor_indices,
            CuArray(Position),
            CuArray(Density),
            CuArray(Pressure),
            CuArray(Velocity),
            CuArray(GravityFactor),
            CuArray(MotionLimiter),
            CuArray(neighbor_offsets),
            CuArray(neighbor_indices),
            similar(CuArray(dρdtI)),
            similar(CuArray(Acceleration)),
            similar(CuArray(Position)),
            similar(CuArray(Velocity)),
            similar(CuArray(Density)),
            similar(CuArray(Density)),
            similar(CuArray(Density)),
            true,
            false,
            false,
        )
        SimMetaData.CudaBuffers = buffers
    end
    buffers.n_particles = n_particles
    buffers.host_neighbor_offsets = neighbor_offsets
    buffers.host_neighbor_indices = neighbor_indices

    if sync_state
        copyto!(buffers.position, Position)
        copyto!(buffers.density, Density)
        copyto!(buffers.pressure, Pressure)
        copyto!(buffers.velocity, Velocity)
        copyto!(buffers.gravity_factor, GravityFactor)
        copyto!(buffers.motion_limiter, MotionLimiter)
    end

    if sync_neighbors && (needs_alloc || buffers.neighbor_dirty)
        copyto!(buffers.neighbor_offsets, neighbor_offsets)
        copyto!(buffers.neighbor_indices, neighbor_indices)
        buffers.neighbor_dirty = false
    end

    return buffers
end

function SyncCudaStateToDevice!(buffers, Position, Density, Pressure, Velocity,
                                GravityFactor, MotionLimiter, dρdtI, Acceleration,
                                Positionₙ⁺, Velocityₙ⁺, ρₙ⁺)
    copyto!(buffers.position, Position)
    copyto!(buffers.density, Density)
    copyto!(buffers.pressure, Pressure)
    copyto!(buffers.velocity, Velocity)
    copyto!(buffers.gravity_factor, GravityFactor)
    copyto!(buffers.motion_limiter, MotionLimiter)
    copyto!(buffers.dρdtI, dρdtI)
    copyto!(buffers.acceleration, Acceleration)
    copyto!(buffers.position_half, Positionₙ⁺)
    copyto!(buffers.velocity_half, Velocityₙ⁺)
    copyto!(buffers.rho_half, ρₙ⁺)
    buffers.state_initialized = true
    return nothing
end

function SyncCudaStateToHost!(buffers, SimParticles)
    copyto!(SimParticles.Position, buffers.position)
    copyto!(SimParticles.Density, buffers.density)
    copyto!(SimParticles.Pressure, buffers.pressure)
    copyto!(SimParticles.Velocity, buffers.velocity)
    copyto!(SimParticles.Acceleration, buffers.acceleration)
    return nothing
end

function SyncCudaPositionsToHost!(buffers, Position)
    copyto!(Position, buffers.position)
    return nothing
end

function UpdateΔx!(Δx::T, posₙ⁺::CUDA.AbstractGPUArray{SVector{D, T}},
                   pos::CUDA.AbstractGPUArray{SVector{D, T}}) where {D, T<:Real}
    maxd = CUDA.mapreduce(
        (pₙ⁺, p) -> begin
            diff = pₙ⁺ - p
            sqrt(dot(diff, diff))
        end,
        max,
        posₙ⁺,
        pos,
    )
    return Δx + 4 * maxd
end

function UpdateCudaNeighborPairList!(SimMetaData, SimParticles, CellDict, ParticleRanges,
                                     NeighborCellLists)
    buffers = SimMetaData.CudaBuffers
    if buffers === nothing
        buffers = EmptyCudaNeighborBuffers()
        SimMetaData.CudaBuffers = buffers
    end

    neighbor_offsets = buffers.host_neighbor_offsets
    neighbor_indices = buffers.host_neighbor_indices
    BuildNeighborPairList!(neighbor_offsets, neighbor_indices, SimParticles.Cells,
                           CellDict, ParticleRanges, NeighborCellLists)
    buffers.neighbor_dirty = true
    return nothing
end

function NeighborLoopCuda!(buffers, SimDensityDiffusion, SimViscosity, SimKernel,
                           SimConstants; position = buffers.position,
                           density = buffers.density, pressure = buffers.pressure,
                           velocity = buffers.velocity)
    n_particles = buffers.n_particles
    threads, blocks = CudaLaunchConfig(n_particles)
    CUDA.@sync CUDA.@cuda threads=threads blocks=blocks NeighborLoopCudaKernel!(
        buffers.dρdtI,
        buffers.acceleration,
        position,
        density,
        pressure,
        velocity,
        buffers.motion_limiter,
        buffers.neighbor_offsets,
        buffers.neighbor_indices,
        SimKernel,
        SimConstants,
        SimDensityDiffusion,
        SimViscosity,
    )
    return nothing
end

@inline function CudaLaunchConfig(n_particles::Integer)
    if n_particles <= 0
        return 1, 1
    end
    threads = min(256, max(32, 1 << floor(Int, log2(min(n_particles, 256)))))
    blocks = cld(n_particles, threads)
    return threads, blocks
end

function HalfTimeStepCudaKernel!(Position, Density, Velocity, Acceleration, GravityFactor,
                                 MotionLimiter, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂, g)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        acc = Acceleration[i] + ConstructGravitySVector(Acceleration[i], g * GravityFactor[i])
        Acceleration[i] = acc
        Positionₙ⁺[i] = Position[i] + Velocity[i] * dt₂ * MotionLimiter[i]
        Velocityₙ⁺[i] = Velocity[i] + acc * dt₂ * MotionLimiter[i]
        ρₙ⁺[i] = Density[i] + dρdtI[i] * dt₂
    end
    return nothing
end

function HalfTimeStepCuda!(SimConstants, buffers, dt₂)
    threads, blocks = CudaLaunchConfig(buffers.n_particles)
    CUDA.@sync CUDA.@cuda threads=threads blocks=blocks HalfTimeStepCudaKernel!(
        buffers.position,
        buffers.density,
        buffers.velocity,
        buffers.acceleration,
        buffers.gravity_factor,
        buffers.motion_limiter,
        buffers.position_half,
        buffers.velocity_half,
        buffers.rho_half,
        buffers.dρdtI,
        dt₂,
        SimConstants.g,
    )
    return nothing
end

function FullTimeStepCudaKernel!(Position, Velocity, Acceleration, GravityFactor,
                                 MotionLimiter, dt, g, h, η², max_visc, min_dt_force)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        acc = Acceleration[i] + ConstructGravitySVector(Acceleration[i], g * GravityFactor[i])
        Acceleration[i] = acc
        limiter = MotionLimiter[i]
        vel = Velocity[i] + acc * dt * limiter
        Velocity[i] = vel
        Position[i] = Position[i] + (((vel + (vel - acc * dt * limiter)) / 2) * dt) * limiter

        r = Position[i]
        r_sq = sqrt(dot(r, r))^2
        max_visc[i] = abs(h * dot(vel, r) / (r_sq + η²))

        a_mag = norm(acc)
        min_dt_force[i] = a_mag > 0 ? sqrt(h / a_mag) : typemax(eltype(min_dt_force))
    end
    return nothing
end

function FullTimeStepCuda!(SimKernel, SimConstants, buffers, dt)
    threads, blocks = CudaLaunchConfig(buffers.n_particles)
    CUDA.@sync CUDA.@cuda threads=threads blocks=blocks FullTimeStepCudaKernel!(
        buffers.position,
        buffers.velocity,
        buffers.acceleration,
        buffers.gravity_factor,
        buffers.motion_limiter,
        dt,
        SimConstants.g,
        SimKernel.h,
        SimKernel.η²,
        buffers.max_visc,
        buffers.min_dt_force,
    )
    return nothing
end

@inbounds function SimulationLoop(::GPUBackend, SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                  SimMetaData::SimulationMetaData{Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, LMode},
                                  SimConstants, SimParticles, FullStencil,
                                  ParticleRanges, UniqueCells, CellDict,
                                  SortingScratchSpace,
                                  NeighborCellLists, dρdtI, Velocityₙ⁺,
                                  Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ,
                                  ::Nothing) where {
                                            Dimensions, FloatType, LMode,
                                            SDD<:SPHDensityDiffusion,
                                            SV<:SPHViscosity}
    @unpack Position, Density, Pressure, Velocity, Acceleration, MotionLimiter = SimParticles

    if SimMetaData.IndexCounter == 0
        SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace,
                                                    ParticleRanges, UniqueCells, CellDict)
        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
        BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView,
                                ParticleRanges, CellDict)
    end

    UpdateCudaNeighborPairList!(SimMetaData, SimParticles, CellDict, ParticleRanges,
                                NeighborCellLists)

    buffers = SimMetaData.CudaBuffers
    neighbor_offsets = buffers.host_neighbor_offsets
    neighbor_indices = buffers.host_neighbor_indices
    buffers = EnsureCudaNeighborBuffers!(SimMetaData, Position, Density, Pressure, Velocity,
                                         SimParticles.GravityFactor, MotionLimiter,
                                         neighbor_offsets, neighbor_indices, dρdtI,
                                         Acceleration; sync_state = false,
                                         sync_neighbors = true)
    if !buffers.state_initialized
        SyncCudaStateToDevice!(buffers, Position, Density, Pressure, Velocity,
                               SimParticles.GravityFactor, MotionLimiter, dρdtI,
                               Acceleration, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺)
    end
    buffers.active = true

    UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
    dt = Δt(buffers.position, buffers.velocity, buffers.acceleration, SimConstants, SimKernel)

    @no_escape begin
        dt₂ = dt * 0.5

        while SimMetaData.TotalTime <= next_output_time(SimMetaData)
            @timeit SimMetaData.HourGlass "01 Calculate IndexCounter" begin
                SimMetaData.Δx = UpdateΔx!(SimMetaData.Δx, buffers.position_half, buffers.position)
                ShouldRebuild = SimMetaData.Δx >= SimKernel.h

                if ShouldRebuild
                    SyncCudaPositionsToHost!(buffers, Position)
                    @timeit SimMetaData.HourGlass "01a Actual Calculate IndexCounter" begin
                        SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹,
                                                                    SortingScratchSpace, ParticleRanges,
                                                                    UniqueCells, CellDict)
                    end
                    SimMetaData.Δx    = zero(eltype(dρdtI))
                    UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                    BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView,
                                            ParticleRanges, CellDict)
                    UpdateCudaNeighborPairList!(SimMetaData, SimParticles, CellDict,
                                                ParticleRanges, NeighborCellLists)
                    neighbor_offsets = SimMetaData.CudaBuffers.host_neighbor_offsets
                    neighbor_indices = SimMetaData.CudaBuffers.host_neighbor_indices
                    buffers = EnsureCudaNeighborBuffers!(
                        SimMetaData, Position, Density, Pressure, Velocity,
                        SimParticles.GravityFactor, MotionLimiter, neighbor_offsets,
                        neighbor_indices, dρdtI, Acceleration; sync_state = false,
                        sync_neighbors = true,
                    )
                end
            end

            @timeit SimMetaData.HourGlass "Motion" ProgressMotion(SimParticles, dt₂, nothing, SimMetaData)

            @timeit SimMetaData.HourGlass "02 Pressure" Pressure!(buffers.pressure, buffers.density, SimConstants)
            @timeit SimMetaData.HourGlass "03 Apply MDBC before Half TimeStep" ApplyMDBCBeforeHalf!(
                SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, CellDict,
                Position, Density, SimParticles.GhostPoints, SimParticles.GhostNormals, SimParticles.Type,
            )

            @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopCuda!(
                buffers, SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
            )

            @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" HalfTimeStepCuda!(SimConstants, buffers, dt₂)

            @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary" LimitDensityAtBoundary!(
                buffers.rho_half, SimConstants.ρ₀, buffers.motion_limiter,
            )

            @timeit SimMetaData.HourGlass "Motion" ProgressMotion(SimParticles, dt₂, nothing, SimMetaData)

            @timeit SimMetaData.HourGlass "07 Pressure" Pressure!(buffers.pressure, buffers.rho_half, SimConstants)
            @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopCuda!(
                buffers, SimDensityDiffusion, SimViscosity, SimKernel, SimConstants;
                position=buffers.position_half, density=buffers.rho_half,
                pressure=buffers.pressure, velocity=buffers.velocity_half,
            )

            @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary" LimitDensityAtBoundary!(
                buffers.density, SimConstants.ρ₀, buffers.motion_limiter,
            )

            @timeit SimMetaData.HourGlass "10 Final Density" DensityEpsi!(
                buffers.density, buffers.dρdtI, buffers.rho_half, dt,
            )

            @timeit SimMetaData.HourGlass "11 Update To Final TimeStep" FullTimeStepCuda!(SimKernel, SimConstants, buffers, dt)

            @timeit SimMetaData.HourGlass "12 Update MetaData" UpdateMetaData!(SimMetaData, dt)

            @timeit SimMetaData.HourGlass "13 Update TimeStep" begin
                dt = FinalizeTimeStep(buffers.max_visc, buffers.min_dt_force, SimConstants, SimKernel)
            end
        end
    end

    return nothing
end
