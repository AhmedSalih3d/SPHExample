module GPUMemoryBenchmark

using SPHExampleGPU, CUDA, StaticArrays, Printf
using SPHExampleGPU.SimulationLoggerConfiguration: device_buffer_bytes

include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))

function generated_wedge_case(::Type{T}, save; dx = 0.02) where {T}
    constants = SimulationConstants{T}(; dx, c₀ = 42.48576250492629, δᵩ = 0.1, CFL = 0.5)
    polygons = still_wedge_2d_polygons()
    sampled = sample_particles([ParticleRegion("Bound", polygons.tank, Fixed),
                                ParticleRegion("Fluid", polygons.water, Fluid)], constants.dx)
    geometry = [SPHGeometry{2, T}(region.positions;
        Density = region.type == Fluid ? hydrostatic_density(region.positions, constants) : constants.ρ₀,
        GroupMarker = index, Type = region.type) for (index, region) in enumerate(sampled)]
    metadata = SimulationMetaData{2, T}(SimulationName = "StillWedge", SaveLocation = save,
        VisualizeInParaview = false, OpenLogFile = false)
    return (; SimGeometry = geometry, SimMetaData = metadata, SimConstants = constants,
        SimKernel = SPHKernelInstance{2, T}(WendlandC2(); dx = constants.dx),
        SimViscosity = ArtificialViscosity(), SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = nothing)
end

function allocation_bytes(geometry::Vector{SPHGeometry{D, T}}, metadata, kernel, constants) where {D, T}
    host = AllocateDataStructures(geometry, metadata)
    gpu = upload_particles(host, metadata)
    cells = CellListWorkspace{D, position_float_type(metadata)}(length(host))
    support = GPUSupportArrays{D, T}(gpu, metadata)
    SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; support, constants)
    CUDA.synchronize()
    return (; particles = length(host), bytes = device_buffer_bytes((gpu, support, cells)))
end

function report(geometry, metadata, kernel, constants, label)
    metadata.GPUDoublePosition = true
    metadata.GPUCompactBuffers = false
    full = allocation_bytes(geometry, metadata, kernel, constants)
    GC.gc()
    CUDA.reclaim()
    metadata.GPUCompactBuffers = true
    compact = allocation_bytes(geometry, metadata, kernel, constants)
    GC.gc()
    CUDA.reclaim()
    @printf("%-28s %9d %12.4f %12.4f %9.2f\n", label, full.particles,
        full.bytes / 2.0^20, compact.bytes / 2.0^20, 100 * (1 - compact.bytes / full.bytes))
end

function main()
    println("GPU: ", CUDA.name(CUDA.device()), "; Float32 physics / Float64 positions")
    println("Live particle, reorder, support and rebuilt cell-grid buffers; excludes output, motion, reductions and allocator pool.")
    @printf("%-28s %9s %12s %12s %9s\n", "case", "particles", "full MiB", "compact MiB", "saved %")
    for dx in (0.02, 0.005)
        case = generated_wedge_case(Float32, tempdir(); dx)
        report(case.SimGeometry, case.SimMetaData, case.SimKernel, case.SimConstants, "StillWedge dx=$dx")
    end
    points = vec([SVector{3, Float32}(0.02f0 * i, 0.02f0 * j, 0.02f0 * k)
                  for i in 0:63, j in 0:63, k in 0:63])
    geometry = [SPHGeometry{3, Float32}(points; Density = 1000, GroupMarker = 1, Type = Fluid)]
    metadata = SimulationMetaData{3, Float32}(SimulationName = "Memory3D", SaveLocation = tempdir())
    kernel = SPHKernelInstance{3, Float32}(WendlandC2(); dx = 0.02f0)
    report(geometry, metadata, kernel, SimulationConstants{Float32}(dx = 0.02), "Uniform3D 64^3")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    GPUMemoryBenchmark.main()
end
