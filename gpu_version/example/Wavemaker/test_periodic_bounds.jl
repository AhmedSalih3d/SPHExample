module PeriodicSearchBoundsTests

using Test, SPHExampleGPU, CUDA, StaticArrays, Random, LinearAlgebra

const GK = SPHExampleGPU.GPUKernels
const CL = SPHExampleGPU.SPHCellList
const GRID = SPHExampleGPU.GPUCellGrid

@testset "target-oriented Fourtakas inner diffusion" begin
    for T in (Float32, Float64), D in (2, 3)
        constants = SimulationConstants{T}(D; dx = 0.02, c₀ = 20, g = 0, δᵩ = 0.1)
        kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.02))
        x = SVector{D,T}(ntuple(_ -> T(0.01), D))
        gradient = ∇Wᵢⱼ(kernel, norm(x) * kernel.h⁻¹, x)
        first, second = compute_density_diffusion(FourtakasDensityDiffusion(), kernel, constants,
            nothing, x, gradient, dot(x, x), T(1000), T(1003), inv(T(1000)), inv(T(1003)), 1, 2, [Fluid, Fluid])
        factor = constants.δᵩ * kernel.H * constants.c₀ * constants.m₀ * dot(x, gradient) / (dot(x, x) + kernel.η²)
        @test first ≈ -factor * T(3) / T(1003)
        @test second ≈ factor * T(3) / T(1000)
        @test compute_density_diffusion(FourtakasDensityDiffusion(), kernel, constants,
            nothing, x, gradient, dot(x, x), T(1000), T(1003), inv(T(1000)), inv(T(1003)), 1, 2, [Fluid, Fixed]) == (0, 0)
        gravity = SimulationConstants{T}(D; dx = 0.02, c₀ = 20, g = 9.81, δᵩ = 0.1)
        # A neighbour below the target has the reference hydrostatic density
        # increment. That contribution must vanish with the correct sign.
        density_below = T(1000) * (one(T) + T(1000) * gravity.g * x[end] * gravity.Cb⁻¹)^gravity.γ⁻¹
        hydrostatic, _ = compute_density_diffusion(FourtakasDensityDiffusion(), kernel, gravity,
            nothing, x, gradient, dot(x, x), T(1000), density_below,
            inv(T(1000)), inv(density_below), 1, 2, [Fluid, Fluid])
        @test abs(hydrostatic) <= 4eps(T) * T(1000) * abs(factor / density_below)
    end
end

function accepted_pairs(rows, points, i, starts, grid, H, T; source = GK.PeriodicPairSource(T, points))
    pairs = Tuple[]
    largest_error = 0.0
    for row in rows
        lo, hi = GK.periodic_row_range(grid, starts, row)
        for j in lo:hi
            i == j && continue
            difference = GRID.pair_vector(source, source[i], Int32(j), Int32(0), row, GRID.cell_size(grid,H))
            if source isa GK.PeriodicCellPairSource
                expected = SVector{length(points[i]),T}((points[i]-points[j])+row.shift)
                largest_error = max(largest_error,norm(difference-expected))
            end
            dot(difference, difference) <= H^2 && push!(pairs, (j, Tuple(row.shift)))
        end
    end
    source isa GK.PeriodicCellPairSource && @test largest_error <= 8eps(T)*H
    return pairs
end

if CUDA.functional()
    @testset "inner diffusion is suppressed for the complete wall-supported target" begin
        for D in (2, 3), T in (Float32, Float64), lanes in (1, 4), near_wall in (false, true)
            point(x) = SVector{D,T}(ntuple(d -> d == 1 ? T(x) : T(0.4), D))
            geometry = [SPHGeometry{D,T}([point(near_wall ? 0.49 : 1.5)]; Density = 1000, Type = Fixed, GroupMarker = 1),
                SPHGeometry{D,T}([point(0.5), point(0.52)]; Density = T[1000, 1003], Type = Fluid, GroupMarker = 2)]
            meta = SimulationMetaData{D,T}(SimulationName = "InnerDiffusion", SaveLocation = tempdir())
            gpu = upload_particles(AllocateDataStructures(geometry, meta), meta)
            sup = GPUSupportArrays{D,T}(gpu, meta)
            constants = SimulationConstants{T}(D; dx = 0.02, c₀ = 20, g = 0, δᵩ = 0.1)
            kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.02))
            cells = CellListWorkspace{D,T}(length(gpu))
            CL.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; kernel)
            GK.launch_interactions!(sup.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
                sup.∇Cᵢ, sup.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
                cells.CellStart, gpu.CellID, cells.grid_dev, HostStep(T(0.001), zero(T)),
                FourtakasDensityDiffusion(), ZeroViscosity(), kernel, constants, Val(false), Val(false); lanes = Val(lanes))
            rate = Array(sup.dρdtI)[Array(gpu.Type) .== Fluid]
            @test near_wall ? all(iszero, rate) : all(!iszero, rate)
        end
    end

    @testset "periodic bounds preserve ordered pairs with stale cell assignments" begin
        rng = MersenneTwister(20261028)
        for D in (2, 3), (T, TP) in ((Float32,Float32), (Float32,Float64), (Float64,Float64)), R in (1, 2), origin_shift in (0.0, 1e5)
            for axis in 1:D
                points = [SVector{D,TP}(ntuple(d -> origin_shift + (d == axis ? (i % 2 == 0 ? 0.02 : 0.96) + 0.01rand(rng) :
                    0.2 + 0.12rand(rng)), D)) for i in 1:64]
                periodicity = PeriodicBoundary{D}(axis; min = origin_shift, max = origin_shift + 1,
                    shift = ntuple(d -> d == mod1(axis + 1, D) ? 0.03 : 0.0, D))
                meta = SimulationMetaData{D,T}(SimulationName = "Bounds", SaveLocation = tempdir(),
                    Periodicity = periodicity, GPUDoublePosition = TP !== T)
                geometry = [SPHGeometry{D,T}(points; Density = 1000, Type = Fluid, GroupMarker = 1)]
                gpu = upload_particles(AllocateDataStructures(geometry, meta), meta)
                kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.02))
                cl = CellListWorkspace{D,TP}(length(gpu); reach = R, periodicity)
                CL.rebuild_cell_list!(gpu, cl, kernel.H⁻¹; kernel)
                original = Array(gpu.Position)
                cells, starts = Array(gpu.CellID), Array(cl.CellStart)
                bounds = CUDA.zeros(SVector{6,Float64}, 2)
                step = HostStep(T(0.001), zero(T))
                # Predictor positions stay continuous across a periodic seam
                # while their cell assignments still describe the old state.
                moved = [p + SVector{D,TP}(ntuple(d -> TP((d == axis ? 0.04 : 0.003)*sin(i + d)), D)) for (i, p) in enumerate(original)]
                copyto!(gpu.Position, moved)
                GK.launch_periodic_halo!(bounds, gpu.Position, gpu.CellID, cl.grid_dev, kernel.H, step)
                host_bounds = Array(bounds)
                s = GRID.cell_size(cl.grid, kernel.H)
                cell_points = [GRID.pos_cell(p,c,cl.grid,s) for (p,c) in zip(moved,cells)]
                cell_source = GK.PeriodicCellPairSource(T,moved,cell_points,cl.grid)
                for i in eachindex(moved)
                    rows = GK.neighbor_rows(cl.periodicity, cl.grid, cells[i], s, moved[i], host_bounds)
                    actual = accepted_pairs(rows, moved, i, starts, cl.grid, kernel.H, T)
                    old_rows = GK.neighbor_rows(cl.periodicity, cl.grid, cells[i], s)
                    previous = accepted_pairs(old_rows, moved, i, starts, cl.grid, kernel.H, T)
                    @test actual == previous
                    if TP !== T
                        cell_pairs = accepted_pairs(rows,moved,i,starts,cl.grid,kernel.H,T; source=cell_source)
                        @test cell_pairs == actual
                    end
                    oracle = Set((j, Tuple(image * cl.periodicity.translation)) for image in (0, 1, -1)
                        for j in eachindex(moved) if j != i &&
                        sum(abs2, SVector{D,T}((moved[i] - moved[j]) + image * cl.periodicity.translation)) <= kernel.H²)
                    @test Set(actual) == oracle
                end
            end
        end
    end
end

end # module PeriodicSearchBoundsTests
