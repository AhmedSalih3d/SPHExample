module SPHCUDA

export RunSimulationCUDA, CUDAAvailable

using CUDA
using LinearAlgebra
using Parameters
using StaticArrays
using TimerOutputs

using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..SimulationGeometry
using ..SimulationEquations
using ..OpenExternalPrograms
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..SPHNeighborList
using ..SPHCellList

using StructArrays: StructArray

CUDAAvailable() = CUDA.functional()

function ΔtCUDAKernel!(max_visc, min_dt_force, Position, Velocity, Acceleration, h, η²)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        r = Position[i]
        v = Velocity[i]
        a = Acceleration[i]
        r_sq = sqrt(dot(r, r))^2
        curr_visc = abs(h * dot(v, r) / (r_sq + η²))
        a_mag = norm(a)
        curr_dt_force = a_mag > 0 ? sqrt(h / a_mag) : typemax(eltype(a_mag))
        CUDA.atomic_max!(max_visc, 1, curr_visc)
        CUDA.atomic_min!(min_dt_force, 1, curr_dt_force)
    end
    return nothing
end

@inline function ΔtCUDA(Position, Velocity, Acceleration, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h, η² = SPHKernel

    scalar_type = eltype(eltype(Position))
    max_visc = CUDA.zeros(scalar_type, 1)
    min_dt_force = CUDA.fill(typemax(scalar_type), 1)

    threads = 256
    blocks = cld(length(Position), threads)
    @cuda threads=threads blocks=blocks ΔtCUDAKernel!(max_visc, min_dt_force, Position, Velocity, Acceleration, h, η²)

    global_visc = Array(max_visc)[1]
    global_dt_force = Array(min_dt_force)[1]
    dt2 = h / (c₀ + global_visc)
    return CFL * min(global_dt_force, dt2)
end

@inline function BuildNeighborCellRanges(NeighborCellLists, cell_count)
    starts = zeros(Int, cell_count + 1)
    total = 0
    @inbounds for index in 1:cell_count
        starts[index] = total + 1
        total += length(NeighborCellLists[index])
    end
    starts[cell_count + 1] = total + 1

    entries = Vector{Int}(undef, total)
    cursor = 1
    @inbounds for index in 1:cell_count
        for neighbor_index in NeighborCellLists[index]
            entries[cursor] = neighbor_index
            cursor += 1
        end
    end
    return starts, entries
end

@inline function BuildCellListIndices(Cells, CellLookup)
    indices = similar(Cells, Int)
    @inbounds for i in eachindex(Cells)
        indices[i] = CellLookupIndex(CellLookup, Cells[i], 1)
    end
    return indices
end

@inline function ComputeDensityDiffusionGPU(::ZeroDensityDiffusion, _SimKernel, _SimConstants,
                                            _Density, _MotionLimiter, _xᵢⱼ, _∇ᵢWᵢⱼ, d², _i, _j)
    return zero(d²)
end

@inline function ComputeDensityDiffusionGPU(::LinearDensityDiffusion, SimKernel, SimConstants,
                                            Density, MotionLimiter, xᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    @unpack ρ₀, m₀, c₀, δᵩ, Cb, γ, g = SimConstants
    @unpack h, η² = SimKernel

    Linear_ρ_factor = (1 / (Cb * γ)) * ρ₀
    ρᵢ = Density[i]
    ρⱼ = Density[j]

    Pᵢⱼᴴ = ρ₀ * (-g) * -xᵢⱼ[end]
    ρᵢⱼᴴ = Pᵢⱼᴴ * Linear_ρ_factor

    invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)
    ρⱼᵢ = ρⱼ - ρᵢ
    ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ) * (-xᵢⱼ) * invdᵢⱼ²η²

    MLcond = MotionLimiter[i] * MotionLimiter[j]
    Dᵢ = δᵩ * h * c₀ * (m₀ / ρⱼ) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) * MLcond

    return Dᵢ
end

@inline function ComputeViscosityGPU(::ZeroViscosity, _SimKernel, _SimConstants,
                                     _Density, _xᵢⱼ, _vᵢⱼ, _∇ᵢWᵢⱼ, _d², _i, _j)
    return zero(_xᵢⱼ)
end

@inline function ComputeViscosityGPU(::ArtificialViscosity, SimKernel, SimConstants,
                                     Density, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    @unpack m₀, α, c₀ = SimConstants
    @unpack h, η² = SimKernel

    ρᵢ = Density[i]
    ρⱼ = Density[j]

    v_dot_x = dot(vᵢⱼ, xᵢⱼ)
    if v_dot_x < 0
        ρ̄ = 0.5 * (ρᵢ + ρⱼ)
        μᵢⱼ = h * v_dot_x / (d² + η²)
        Π = -m₀ * (-α * c₀ * μᵢⱼ) / ρ̄ * ∇ᵢWᵢⱼ
        return Π
    end

    return zero(xᵢⱼ)
end

function NeighborLoopCUDAKernel!(dρdtI, Acceleration, Position, Density, Pressure, Velocity,
                                 MotionLimiter, CellListIndices, ParticleRanges, ParticleOrder,
                                 NeighborCellStarts, NeighborCellEntries, SimKernel, SimConstants,
                                 SimDensityDiffusion, SimViscosity)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(Position)
        dρdt_acc = zero(eltype(dρdtI))
        acc_acc = zero(Position[i])
        cell_list_index = CellListIndices[i]
        same_cell_start = ParticleRanges[cell_list_index]
        same_cell_end = ParticleRanges[cell_list_index + 1] - 1

        @inbounds for j in same_cell_start:same_cell_end
            j_index = ParticleOrder[j]
            if j_index != i
                xᵢⱼ = Position[i] - Position[j_index]
                xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
                if xᵢⱼ² <= SimKernel.H²
                    dᵢⱼ = sqrt(abs(xᵢⱼ²))
                    q = clamp(dᵢⱼ * SimKernel.h⁻¹, zero(eltype(dᵢⱼ)), one(eltype(dᵢⱼ)) * 2)
                    ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                    ρᵢ = Density[i]
                    ρⱼ = Density[j_index]
                    vᵢⱼ = Velocity[i] - Velocity[j_index]

                    density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                    dρdt⁺ = -ρᵢ * (SimConstants.m₀ / ρⱼ) * density_symmetric_term

                    Dᵢ = ComputeDensityDiffusionGPU(SimDensityDiffusion, SimKernel, SimConstants,
                                                    Density, MotionLimiter, xᵢⱼ, ∇ᵢWᵢⱼ,
                                                    dᵢⱼ^2, i, j_index)

                    dρdt_acc += dρdt⁺ + Dᵢ

                    Pᵢ = Pressure[i]
                    Pⱼ = Pressure[j_index]
                    Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
                    f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, SimConstants.dx)
                    dvdt⁺ = -SimConstants.m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

                    visc_term = ComputeViscosityGPU(SimViscosity, SimKernel, SimConstants,
                                                    Density, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ^2, i, j_index)

                    acc_acc += dvdt⁺ + visc_term
                end
            end
        end

        neighbor_start = NeighborCellStarts[cell_list_index]
        neighbor_end = NeighborCellStarts[cell_list_index + 1] - 1
        @inbounds for neighbor_cursor in neighbor_start:neighbor_end
            neighbor_index = NeighborCellEntries[neighbor_cursor]
            start_index = ParticleRanges[neighbor_index]
            end_index = ParticleRanges[neighbor_index + 1] - 1
            for j in start_index:end_index
                j_index = ParticleOrder[j]
                xᵢⱼ = Position[i] - Position[j_index]
                xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
                if xᵢⱼ² <= SimKernel.H²
                    dᵢⱼ = sqrt(abs(xᵢⱼ²))
                    q = clamp(dᵢⱼ * SimKernel.h⁻¹, zero(eltype(dᵢⱼ)), one(eltype(dᵢⱼ)) * 2)
                    ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                    ρᵢ = Density[i]
                    ρⱼ = Density[j_index]
                    vᵢⱼ = Velocity[i] - Velocity[j_index]

                    density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                    dρdt⁺ = -ρᵢ * (SimConstants.m₀ / ρⱼ) * density_symmetric_term

                    Dᵢ = ComputeDensityDiffusionGPU(SimDensityDiffusion, SimKernel, SimConstants,
                                                    Density, MotionLimiter, xᵢⱼ, ∇ᵢWᵢⱼ,
                                                    dᵢⱼ^2, i, j_index)

                    dρdt_acc += dρdt⁺ + Dᵢ

                    Pᵢ = Pressure[i]
                    Pⱼ = Pressure[j_index]
                    Pfac = (Pᵢ + Pⱼ) / (ρᵢ * ρⱼ)
                    f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, SimConstants.dx)
                    dvdt⁺ = -SimConstants.m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

                    visc_term = ComputeViscosityGPU(SimViscosity, SimKernel, SimConstants,
                                                    Density, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ^2, i, j_index)

                    acc_acc += dvdt⁺ + visc_term
                end
            end
        end

        dρdtI[i] = dρdt_acc
        Acceleration[i] = acc_acc
    end
    return nothing
end

function NeighborLoopPerParticleCUDA!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellLookup, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ;
                                      Position = SimParticles.Position,
                                      Density = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity,
                                      ParticleOrder) where {D,T,
                                                           B<:MDBCMode,L<:LogMode,
                                                           SDD<:SPHDensityDiffusion,
                                                           SV<:SPHViscosity}
    if !CUDAAvailable()
        error("CUDA is not available; cannot run GPU neighbor loop.")
    end

    if !(SimDensityDiffusion isa LinearDensityDiffusion || SimDensityDiffusion isa ZeroDensityDiffusion)
        error("CUDA neighbor loop supports LinearDensityDiffusion or ZeroDensityDiffusion.")
    end
    if !(SimViscosity isa ArtificialViscosity || SimViscosity isa ZeroViscosity)
        error("CUDA neighbor loop supports ArtificialViscosity or ZeroViscosity.")
    end

    cell_list_indices = BuildCellListIndices(SimParticles.Cells, CellLookup)
    cell_count = length(NeighborCellLists)
    neighbor_cell_starts, neighbor_cell_entries = BuildNeighborCellRanges(NeighborCellLists, cell_count)

    dρdtI_gpu = CuArray(dρdtI)
    acceleration_gpu = CuArray(Acceleration)
    position_gpu = CuArray(Position)
    density_gpu = CuArray(Density)
    pressure_gpu = CuArray(Pressure)
    velocity_gpu = CuArray(Velocity)
    motion_limiter_gpu = CuArray(SimParticles.MotionLimiter)
    cell_list_indices_gpu = CuArray(cell_list_indices)
    particle_ranges_gpu = CuArray(ParticleRanges)
    particle_order_gpu = CuArray(ParticleOrder)
    neighbor_cell_starts_gpu = CuArray(neighbor_cell_starts)
    neighbor_cell_entries_gpu = CuArray(neighbor_cell_entries)

    threads = 256
    blocks = cld(length(Position), threads)
    @cuda threads=threads blocks=blocks NeighborLoopCUDAKernel!(
        dρdtI_gpu, acceleration_gpu, position_gpu, density_gpu, pressure_gpu,
        velocity_gpu, motion_limiter_gpu, cell_list_indices_gpu, particle_ranges_gpu,
        particle_order_gpu, neighbor_cell_starts_gpu, neighbor_cell_entries_gpu,
        SimKernel, SimConstants, SimDensityDiffusion, SimViscosity,
    )

    CUDA.copyto!(dρdtI, dρdtI_gpu)
    CUDA.copyto!(Acceleration, acceleration_gpu)

    return nothing
end

@inbounds function SimulationLoopCUDA(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                      SimConstants, SimParticles, FullStencil,
                                      ParticleRanges, UniqueCells, CellLookup,
                                      ParticleOrder, CellOffsets,
                                      NeighborCellLists, dρdtI, Velocityₙ⁺,
                                      Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ,
                                      MotionDefinition::Union{
                                          Nothing,
                                          AbstractVector{
                                              Union{
                                                  Nothing,
                                                  MotionDetails{Dimensions, FloatType},
                                              },
                                          },
                                      }) where {
                                                Dimensions, FloatType, SMode, KMode,
                                                BMode, LMode,
                                                SDD<:SPHDensityDiffusion,
                                                SV<:SPHViscosity}
    @unpack Position, Density, Pressure, Velocity, Acceleration, MotionLimiter,
            GroupMarker = SimParticles
    ParticleType   = SimParticles.Type
    ParticleMarker = GroupMarker
    GhostPoints = hasproperty(SimParticles, :GhostPoints) ? SimParticles.GhostPoints : nothing
    GhostNormals = hasproperty(SimParticles, :GhostNormals) ? SimParticles.GhostNormals : nothing

    UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)

    position_step_gpu = CuArray(Position)
    velocity_step_gpu = CuArray(Velocity)
    acceleration_step_gpu = CuArray(Acceleration)

    dt = ΔtCUDA(position_step_gpu, velocity_step_gpu, acceleration_step_gpu, SimConstants, SimKernel)

    dt₂ = dt * 0.5

    while SimMetaData.TotalTime <= next_output_time(SimMetaData)
        @timeit SimMetaData.HourGlass "01 Calculate IndexCounter" begin
            SimMetaData.Δx = UpdateΔx!(SimMetaData.Δx, Positionₙ⁺, SimParticles.Position)
            ShouldRebuild = SimMetaData.Δx >= SimKernel.h

            if ShouldRebuild
                @timeit SimMetaData.HourGlass "01a Actual Calculate IndexCounter" begin
                    SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, ParticleRanges, UniqueCells, CellLookup, ParticleOrder, CellOffsets)
                end
                SimMetaData.Δx = zero(eltype(dρdtI))
                UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
                BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges, CellLookup)
            end
        end

        @timeit SimMetaData.HourGlass "Motion" ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

        @timeit SimMetaData.HourGlass "02 Pressure" Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
        if SimMetaData isa SimulationMetaData{Dimensions, FloatType, SMode, KMode, NoMDBC, LMode} where {SMode, KMode, LMode}
            @timeit SimMetaData.HourGlass "03 Apply MDBC before Half TimeStep" SPHCellList.ApplyMDBCBeforeHalf!(SimMetaData)
        else
            @timeit SimMetaData.HourGlass "03 Apply MDBC before Half TimeStep" SPHCellList.ApplyMDBCBeforeHalf!(
                SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, CellLookup, Position,
                Density, GhostPoints, GhostNormals, ParticleType; ParticleOrder = ParticleOrder,
            )
        end

        @timeit SimMetaData.HourGlass "04 First NeighborLoop" NeighborLoopPerParticleCUDA!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellLookup,
            NeighborCellLists, dρdtI, Acceleration, ∇Cᵢ, ∇◌rᵢ,
            ParticleOrder = ParticleOrder,
        )

        @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)

        @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary" LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)

        @timeit SimMetaData.HourGlass "Motion" ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

        @timeit SimMetaData.HourGlass "07 Pressure" Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
        @timeit SimMetaData.HourGlass "08 Second NeighborLoop" NeighborLoopPerParticleCUDA!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, ParticleRanges, CellLookup,
            NeighborCellLists, dρdtI, Acceleration, ∇Cᵢ, ∇◌rᵢ,
            ParticleOrder = ParticleOrder,
            Position = Positionₙ⁺,
            Density = ρₙ⁺,
            Velocity = Velocityₙ⁺,
        )

        @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary" LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)

        @timeit SimMetaData.HourGlass "10 Final Density" DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)

        @timeit SimMetaData.HourGlass "11 Update To Final TimeStep" FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)

        @timeit SimMetaData.HourGlass "12 Update MetaData" UpdateMetaData!(SimMetaData, dt)

        CUDA.copyto!(position_step_gpu, Positionₙ⁺)
        CUDA.copyto!(velocity_step_gpu, Velocityₙ⁺)
        CUDA.copyto!(acceleration_step_gpu, Acceleration)
        @timeit SimMetaData.HourGlass "13 Update TimeStep" dt = ΔtCUDA(position_step_gpu, velocity_step_gpu, acceleration_step_gpu, SimConstants, SimKernel)
        dt₂ = dt * 0.5
    end

    return nothing
end

function RunSimulationCUDA(;SimGeometry::Vector{Geometry{Dimensions, FloatType}},
        SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
        SimConstants::SimulationConstants,
        SimKernel::SPHKernelInstance,
        SimLogger::SimulationLogger,
        SimParticles::StructArray,
        SimViscosity::SV,
        SimDensityDiffusion::SDD,
        ParticleNormalsPath::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType,SMode,KMode,BMode,LMode,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}
    if !CUDAAvailable()
        error("CUDA is not available; cannot run RunSimulationCUDA.")
    end
    if !(SimMetaData isa SimulationMetaData{Dimensions, FloatType, NoShifting, NoKernelOutput, BMode, LMode} where {BMode, LMode})
        error("RunSimulationCUDA currently supports NoShifting and NoKernelOutput configurations.")
    end

    NumberOfPoints = length(SimParticles)

    dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimMetaData, SimParticles.Position)

    LoadMDBCNormals!(SimMetaData, SimParticles, ParticleNormalsPath)

    InitializeLog!(SimMetaData, SimLogger, SimConstants, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)

    Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)

    ParticleRanges = zeros(Int, NumberOfPoints + 1 + 1)
    UniqueCells = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
    CellLookup = InitializeCellIndexLookup(Val(Dimensions))
    FullStencil = ConstructStencil(Val(Dimensions))
    NeighborCellLists = [Int[] for _ in 1:length(UniqueCells)]
    ParticleOrder = zeros(Int, NumberOfPoints)
    CellOffsets = zeros(Int, length(ParticleRanges))

    output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    SimMetaData.OutputIterationCounter = 1
    output.enqueue_particles(SimMetaData.OutputIterationCounter)
    if SimMetaData.IndexCounter > 0
        unique_cells_view = view(UniqueCells, 1:SimMetaData.IndexCounter)
        cell_particle_counts = nothing
        cell_neighbor_counts = nothing
        if SimMetaData.ExportGridCellParticleCounts
            cell_particle_counts = ComputeCellParticleCounts(ParticleRanges, SimMetaData.IndexCounter)
            cell_neighbor_counts = ComputeCellNeighborCounts(ParticleRanges, NeighborCellLists, SimMetaData.IndexCounter)
        end
        output.enqueue_grid(
            SimMetaData.OutputIterationCounter,
            unique_cells_view,
            cell_particle_counts=cell_particle_counts,
            cell_neighbor_counts=cell_neighbor_counts,
        )
    end

    MotionDefinition = SPHCellList.GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)

    @inbounds while true
        @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoopCUDA(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
            SimConstants, SimParticles, FullStencil, ParticleRanges,
            UniqueCells, CellLookup, ParticleOrder, CellOffsets,
            NeighborCellLists, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺,
            ∇Cᵢ, ∇◌rᵢ, MotionDefinition,
        )
        push!(SimMetaData.TimeSteps, SimMetaData.CurrentTimeStep)

        LogStep!(SimMetaData, SimLogger)

        SimMetaData.OutputIterationCounter += 1

        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)

        @timeit SimMetaData.HourGlass "13 Determine Output" SPHCellList.DetermineOutput(
            Val(SimMetaData.ExportGridCellParticleCounts),
            SimMetaData,
            output,
            ParticleRanges,
            NeighborCellLists,
            UniqueCellsView,
        )

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

            show(SimMetaData.HourGlass, sortby=:name)
            show(SimMetaData.HourGlass)

            FinalizeLog!(SimMetaData, SimLogger)

            AutoOpenLogFile(SimLogger, SimMetaData)
            AutoOpenParaview(SimMetaData, output.variable_names)

            break
        end
    end
end

end
