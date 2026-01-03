using CUDA
using StaticArrays
using Adapt
using WriteVTK #To visualize results if needed
using WriteVTK: MeshCell, PolyData

function ConstructFullStencil(v::Val{d}) where d
    ranges = ntuple(_ -> -1:1, v)
    CI = CartesianIndices(ranges)
    return SVector{length(CI)}(ntuple(i -> SVector{d, Int}(Tuple(CI[i])), Val(length(CI))))
end

abstract type WendlandKernel end

struct WendlandC2{T} <: WendlandKernel
    dim::Int8
    norm::T
end

struct WendlandC2_1D{T} <: WendlandKernel
    dim::Int8
    norm::T
end

function WendlandC2(T::DataType = Float64, dim::Integer = 3)
    if dim == 1
        return WendlandC2_1D{T}(1, 5 / 4)
    elseif dim == 2
        norm = T(7 / π)
    elseif dim == 3
        norm = T(21 / 2π)
    else
        error("WendlandC2 not defined for $dim dimensions!")
    end

    return WendlandC2{T}(dim, norm)
end

WendlandC2(dim::Integer) = WendlandC2(typeof(1.0), dim)

@inline function KernelValue(kernel::WendlandC2_1D{T}, u::Real) where {T}
    if u < 1
        t1 = 1 - u
        return t1 * t1 * t1 * (1 + 3u) |> T
    else
        return zero(T)
    end
end

@inline function KernelValue(kernel::WendlandC2{T}, u::Real) where {T}
    if u < 1
        t1 = 1 - u
        t1 *= t1    # (1-u)^2
        t1 *= t1    # (1-u)^4
        return t1 * (1 + 4u) |> T
    else
        return zero(T)
    end
end

struct Particles{A, B, C, D}
    Positions::A
    Velocities::A
    Accelerations::A
    Densities::B
    Cells::C
    CellIDs::D
    Kernel::B
    KernelGradients::A
end
Adapt.@adapt_structure Particles

# Source: https://discourse.julialang.org/t/meshgrid-function-in-julia/48679/36
function Meshgrid1(x, y)
    m, n = length(x), length(y)
    Tx, Ty = eltype(x), eltype(y)
    return x' .* ones(Tx, n), ones(Ty, m)' .* y
end

function PositionsToPoints3(Positions)
    N = length(Positions)
    Points = Matrix{FloatType}(undef, 3, N)
    @inbounds for i in 1:N
        Points[1, i] = Positions[i][1]
        Points[2, i] = Positions[i][2]
        Points[3, i] = 0.0
    end
    return Points
end

function BuildCellRangesKernel!(particle_start, particle_end, CellID, count)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i == 1
        cell_id = CellID[1]
        particle_start[cell_id] = 1
    elseif i <= count
        cell_id = CellID[i]
        prev_cell = CellID[i - 1]
        if cell_id != prev_cell
            particle_end[prev_cell] = i - 1
            particle_start[cell_id] = i
        end
        if i == count
            particle_end[cell_id] = count
        end
    end
    return
end

function BuildCellRangesGPU!(particle_start, particle_end, CellID)
    count = length(CellID)
    CUDA.@sync @cuda threads=256 blocks=cld(count, 256) BuildCellRangesKernel!(
        particle_start,
        particle_end,
        CellID,
        count,
    )
    return nothing
end

function CountNeighborsKernel!(neighbor_counts, Positions, CellID, particle_start, particle_end, Nx, Ny, cutoff2, Offsets, count)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= count
        cell_id = CellID[i]
        cell_y = (cell_id - 1) ÷ Nx
        cell_x = (cell_id - 1) - cell_y * Nx
        total = 0
        for k in 1:length(Offsets)
            offset = Offsets[k]
            nx = cell_x + offset[1]
            ny = cell_y + offset[2]
            if nx < 0 || ny < 0 || nx >= Nx || ny >= Ny
                continue
            end
            neighbor_cell = nx + ny * Nx + 1
            start_idx = particle_start[neighbor_cell]
            end_idx = particle_end[neighbor_cell]
            if start_idx == 0
                continue
            end
            for j in start_idx:end_idx
                if j != i
                    r = Positions[i] - Positions[j]
                    if sum(abs2, r) <= cutoff2
                        total += 1
                    end
                end
            end
        end
        neighbor_counts[i] = total
    end
    return
end

function CountNeighborsGPU!(neighbor_counts, Positions, CellID, particle_start, particle_end, Nx, Ny, CutOff, Offsets)
    count = length(CellID)
    cutoff2 = CutOff^2
    CUDA.@sync @cuda threads=256 blocks=cld(count, 256) CountNeighborsKernel!(
        neighbor_counts,
        Positions,
        CellID,
        particle_start,
        particle_end,
        Nx,
        Ny,
        cutoff2,
        Offsets,
        count,
    )
    return nothing
end

function FillNeighborsKernel!(neighbor_indices, neighbor_offsets, Positions, CellID, particle_start, particle_end, Nx, Ny, cutoff2, Offsets, count)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= count
        cell_id = CellID[i]
        cell_y = (cell_id - 1) ÷ Nx
        cell_x = (cell_id - 1) - cell_y * Nx
        write_idx = neighbor_offsets[i] + 1
        for k in 1:length(Offsets)
            offset = Offsets[k]
            nx = cell_x + offset[1]
            ny = cell_y + offset[2]
            if nx < 0 || ny < 0 || nx >= Nx || ny >= Ny
                continue
            end
            neighbor_cell = nx + ny * Nx + 1
            start_idx = particle_start[neighbor_cell]
            end_idx = particle_end[neighbor_cell]
            if start_idx == 0
                continue
            end
            for j in start_idx:end_idx
                if j != i
                    r = Positions[i] - Positions[j]
                    if sum(abs2, r) <= cutoff2
                        neighbor_indices[write_idx] = j
                        write_idx += 1
                    end
                end
            end
        end
    end
    return
end

function FillNeighborsGPU!(neighbor_indices, neighbor_offsets, Positions, CellID, particle_start, particle_end, Nx, Ny, CutOff, Offsets)
    count = length(CellID)
    cutoff2 = CutOff^2
    CUDA.@sync @cuda threads=256 blocks=cld(count, 256) FillNeighborsKernel!(
        neighbor_indices,
        neighbor_offsets,
        Positions,
        CellID,
        particle_start,
        particle_end,
        Nx,
        Ny,
        cutoff2,
        Offsets,
        count,
    )
    return nothing
end

function WendlandKernelKernel!(kernel_values, neighbor_indices, neighbor_offsets, Positions, h, kernel, count)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= count
        start_idx = neighbor_offsets[i] + 1
        end_idx = neighbor_offsets[i + 1]
        pos_i = Positions[i]
        for idx in start_idx:end_idx
            j = neighbor_indices[idx]
            r = pos_i - Positions[j]
            q = sqrt(sum(abs2, r)) / h
            kernel_values[idx] = kernel.norm * KernelValue(kernel, q)
        end
    end
    return
end

function WendlandKernelGPU!(kernel_values, neighbor_indices, neighbor_offsets, Positions, h, kernel)
    count = length(Positions)
    CUDA.@sync @cuda threads=256 blocks=cld(count, 256) WendlandKernelKernel!(
        kernel_values,
        neighbor_indices,
        neighbor_offsets,
        Positions,
        h,
        kernel,
        count,
    )
    return nothing
end

const Dimensions = 2
const FloatType = Float64
const IntType = Int64
const WorldDimensions = SVector{Dimensions, FloatType}

κ = 2.0
Δx = 0.01
h = 1.3Δx
CutOff = κ * h
InverseCutOff = 1.0 / CutOff

x = 0:Δx:1
y = 0:Δx:1
X, Y = Meshgrid1(x, y)

Position = vec(WorldDimensions.(X, Y))
NumberOfParticles = length(Position)

Velocity = zeros(WorldDimensions, NumberOfParticles)
Acceleration = zeros(WorldDimensions, NumberOfParticles)
Density = zeros(FloatType, NumberOfParticles)

Kernel = zeros(FloatType, NumberOfParticles)
KernelGradient = zeros(WorldDimensions, NumberOfParticles)

Cells = zeros(SVector{Dimensions, IntType}, NumberOfParticles)
CellIDs = zeros(Int, NumberOfParticles)

SimParticles = Particles(
    Position,
    Velocity,
    Acceleration,
    Density,
    Cells,
    CellIDs,
    Kernel,
    KernelGradient,
)

SimParticlesGPU = adapt(CuArray, SimParticles)

# Calculate neighborlist in shaa Allah ya Rabb

@inline function ExtractCells(x, InverseCutOff::T) where T<:AbstractFloat
    @. Int(sign(x)) * unsafe_trunc(Int, muladd(abs(x), InverseCutOff, 0.5))
    # Int.(sign.(x)) .* unsafe_trunc.(Int, muladd.(abs.(x), InverseCutOff, 0.5))
end

CellsGPU = ExtractCells.(SimParticlesGPU.Positions, InverseCutOff)
CellsCPU = adapt(Array, CellsGPU)

cx = getindex.(CellsCPU, 1)
cy = getindex.(CellsCPU, 2)
Nx = maximum(cx) + 1
Ny = maximum(cy) + 1
CellCount = Nx * Ny
CellID = cx .+ cy .* Nx .+ 1 # for 1-based indexing

SortedIndices = sortperm(CellID)
SortedIndicesGPU = CuArray(SortedIndices)
CellIDGPU = CuArray(CellID[SortedIndices])

SimParticlesGPU = Particles(
    SimParticlesGPU.Positions[SortedIndicesGPU],
    SimParticlesGPU.Velocities[SortedIndicesGPU],
    SimParticlesGPU.Accelerations[SortedIndicesGPU],
    SimParticlesGPU.Densities[SortedIndicesGPU],
    CellsGPU[SortedIndicesGPU],
    CellIDGPU,
    SimParticlesGPU.Kernel[SortedIndicesGPU],
    SimParticlesGPU.KernelGradients[SortedIndicesGPU],
)

SimParticlesCPU = Particles(
    SimParticles.Positions[SortedIndices],
    SimParticles.Velocities[SortedIndices],
    SimParticles.Accelerations[SortedIndices],
    SimParticles.Densities[SortedIndices],
    CellsCPU[SortedIndices],
    CellID[SortedIndices],
    SimParticles.Kernel[SortedIndices],
    SimParticles.KernelGradients[SortedIndices],
)

particle_start = CUDA.zeros(IntType, CellCount + 1)
particle_end = CUDA.zeros(IntType, CellCount + 1)
BuildCellRangesGPU!(particle_start, particle_end, SimParticlesGPU.CellIDs)

OffsetsCPU = ConstructFullStencil(Val(Dimensions))
OffsetsGPU = CuArray(OffsetsCPU)

neighbor_counts = CUDA.zeros(IntType, NumberOfParticles)
CountNeighborsGPU!(
    neighbor_counts,
    SimParticlesGPU.Positions,
    SimParticlesGPU.CellIDs,
    particle_start,
    particle_end,
    Nx,
    Ny,
    CutOff,
    OffsetsGPU,
)

neighbor_offsets = CUDA.zeros(IntType, NumberOfParticles + 1)
neighbor_offsets[2:end] .= CUDA.cumsum(neighbor_counts)
total_neighbors = Int(Array(neighbor_offsets[end]))

neighbor_indices = CUDA.zeros(IntType, total_neighbors)
FillNeighborsGPU!(
    neighbor_indices,
    neighbor_offsets,
    SimParticlesGPU.Positions,
    SimParticlesGPU.CellIDs,
    particle_start,
    particle_end,
    Nx,
    Ny,
    CutOff,
    OffsetsGPU,
)

kernel_values = CUDA.zeros(FloatType, total_neighbors)
kernel = WendlandC2(FloatType, Dimensions)
WendlandKernelGPU!(
    kernel_values,
    neighbor_indices,
    neighbor_offsets,
    SimParticlesGPU.Positions,
    h,
    kernel,
)

neighbor_offsets_cpu = Array(neighbor_offsets)
neighbor_indices_cpu = Array(neighbor_indices)
kernel_values_cpu = Array(kernel_values)

open("particle_neighbors.txt", "w") do io
    for i in 1:NumberOfParticles
        start_idx = neighbor_offsets_cpu[i] + 1
        end_idx = neighbor_offsets_cpu[i + 1]
        if start_idx > end_idx
            println(io, i, ":")
        else
            neighbors = neighbor_indices_cpu[start_idx:end_idx]
            kernels = kernel_values_cpu[start_idx:end_idx]
            entries = join(["$(neighbors[idx])=$(kernels[idx])" for idx in eachindex(neighbors)], ", ")
            println(io, i, ": ", entries)
        end
    end
end

Points = PositionsToPoints3(SimParticlesCPU.Positions)

Verts = [MeshCell(PolyData.Verts(), collect(1:NumberOfParticles))]

vtk = vtk_grid("particles", Points, Verts)

vtk["density"] = SimParticlesCPU.Densities
vtk["velocity"] = SimParticlesCPU.Velocities'
vtk["acceleration"] = SimParticlesCPU.Accelerations'
vtk["cell"] = SimParticlesCPU.Cells
vtk["CellID"] = SimParticlesCPU.CellIDs
vtk["kernel"] = SimParticlesCPU.Kernel
vtk["kernel_gradient"] = SimParticlesCPU.KernelGradients'

vtk_save(vtk)   # writes particles.vtp
