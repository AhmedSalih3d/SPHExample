using SPHExample
using PointNeighbors
using CUDA, Adapt
using FastPow

FloatType  = Float64

# Generate grid of particles
coordinates = reshape(reinterpret(eltype(eltype(SimParticles.Position)), SimParticles.Position), 2, :)
n_particles = size(coordinates, 2)

# `FullGridCellList` requires a bounding box
min_corner = minimum(coordinates, dims=2)
max_corner = maximum(coordinates, dims=2)
search_radius = SimKernel.H
cell_list = FullGridCellList(; search_radius, min_corner, max_corner)
nhs = GridNeighborhoodSearch{2}(; search_radius, cell_list)

# Initialize the NHS to find neighbors in `coordinates` of particles in `coordinates`
initialize!(nhs, coordinates, coordinates)

# Simple example: just count the neighbors of each particle
n_neighbors = zeros(Int, n_particles)
kernel      = zeros(FloatType, n_particles)
grad_kernel = zeros(FloatType, 2, n_particles)


# Use a function for performance reasons
function count_neighbors!(n_neighbors, kernel, grad_kernel, coordinates, nhs, SimKernel)
    n_neighbors .= 0
    kernel      .= 0.0
    grad_kernel .= zero(eltype(grad_kernel))
    foreach_point_neighbor(coordinates, coordinates, nhs) do i, j, xᵢⱼ, distance
        n_neighbors[i] += 1

        q     = distance / SimKernel.H

        Wᵢⱼ   = @fastpow SPHExample.SPHKernels.Wᵢⱼ(SimKernel, q)

        ∇ᵢWᵢⱼ = @fastpow SPHExample.SPHKernels.∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

        kernel[i]         += Wᵢⱼ

        grad_kernel[1, i] += ∇ᵢWᵢⱼ[1] 
        grad_kernel[2, i] += ∇ᵢWᵢⱼ[2] 
    end
end

display(CUDA.@profile count_neighbors!(n_neighbors, kernel, grad_kernel, coordinates, nhs, SimKernel))


backend = CUDABackend()

coordinates_gpu = adapt(backend, coordinates)
nhs_gpu = adapt(backend, nhs)
n_neighbors_gpu = adapt(backend, n_neighbors)
kernel_gpu = adapt(backend, kernel)
grad_kernel_gpu = adapt(backend, grad_kernel)

display(CUDA.@profile count_neighbors!(n_neighbors_gpu, kernel_gpu, grad_kernel_gpu, coordinates_gpu, nhs_gpu, SimKernel))

using Plots

# Extract particle positions
x = coordinates[1, :]
y = coordinates[2, :]

# Compute magnitude of kernel gradients
grad_kernel_mag = sqrt.(sum(abs2, grad_kernel; dims=1))[:]
grad_kernel_gpu_cpu_mag = sqrt.(sum(abs2, grad_kernel_gpu; dims=1))[:]

# Prepare data for coloring
color_data = [
    kernel,                     # CPU Kernel
    grad_kernel_mag,            # CPU Kernel Gradient Magnitude
    kernel_gpu_cpu,             # GPU Kernel
    grad_kernel_gpu_cpu_mag     # GPU Kernel Gradient Magnitude
]

labels = [
    "CPU Kernel",
    "CPU Kernel Gradient Magnitude",
    "GPU Kernel",
    "GPU Kernel Gradient Magnitude"
]

# Create a 2x2 grid of scatter plots
plt = plot(layout = (2, 2), size=(2400, 2000), aspect_ratio = :equal)
for i in 1:4
    scatter!(
        plt[i],
        x, y,
        marker_z = color_data[i],
        colorbar = true,
        markersize = 4,  # smaller size
        label = "",
        title = labels[i],
        xlabel = "x",
        ylabel = "y"
    )
end

display(plt)