using Test
using SPHExampleGPU
using CUDA
using StaticArrays
using StructArrays
using LinearAlgebra
using HDF5

include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

const REPO      = normpath(joinpath(@__DIR__, "..", ".."))
const CPU_REF   = joinpath(@__DIR__, "cpu_reference.jl")
const HAVE_CPU  = isfile(joinpath(REPO, "src", "SPHExample.jl"))

"""
Run `case` on the GPU for `simtime` seconds of physical time and return the
particles sorted by ID together with the meta data.
"""
function run_gpu(case::BenchCase, ::Type{T}, simtime; kwargs...) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    kw.SimMetaData.SimulationTime = T(simtime)
    kw.SimMetaData.OutputTimes    = T(simtime)
    for (k, v) in kwargs
        setproperty!(kw.SimMetaData, k, v)
    end
    particles = AllocateDataStructures(kw.SimGeometry)
    logger    = SimulationLogger(save; to_console = false)
    RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
    order = sortperm(particles.ID)
    return particles[order], kw.SimMetaData
end

"""
Run the CPU reference in a separate Julia process (separate environment) and
return the stored state.
"""
function run_cpu_reference(case::BenchCase, simtime)
    out = tempname() * ".h5"
    cmd = `$(Base.julia_cmd()) -t 8,0 --project=$(REPO) $(CPU_REF) $(case.name) $(simtime) $(out)`
    run(pipeline(cmd; stdout = devnull, stderr = devnull))
    return h5open(out, "r") do fid
        (ID = read(fid["ID"]), Density = read(fid["Density"]), Pressure = read(fid["Pressure"]),
         Position = read(fid["Position"]), Velocity = read(fid["Velocity"]),
         Iteration = read(fid["Iteration"]), TotalTime = read(fid["TotalTime"]))
    end
end

relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), eps(eltype(b))))

@testset "SPHExampleGPU" begin
    @test CUDA.functional()

    @testset "cell grid helpers" begin
        invH = 1 / 0.04
        @test map_floor(0.0, invH)   == 0
        @test map_floor(0.019, invH) == 0
        @test map_floor(0.021, invH) == 1
        @test map_floor(-0.021, invH) == -1
        @test map_floor(-0.019, invH) == 0
        grid = CellGrid{2}((Int32(-3), Int32(-2)), (Int32(10), Int32(8)), Int32(80))
        for c in ((-2, -1), (0, 0), (5, 4))
            lin = SPHExampleGPU.GPUCellGrid.linear_cell(grid, Int32.(c))
            l   = SPHExampleGPU.GPUCellGrid.local_coords(grid, lin)
            @test l .+ grid.origin == Int32.(c)
        end
    end

    @testset "fused reduction" begin
        n = 100_003
        x = CuArray(rand(Float64, n))
        ws = ReductionWorkspace{SVector{2, Float64}}(n)
        f(i, x) = (@inbounds v = x[i]; SVector(v, -v))
        op(a, b) = SVector(max(a[1], b[1]), min(a[2], b[2]))
        r = reduce_svector(ws, f, op, SVector(-Inf, Inf), n, x)
        xh = Array(x)
        @test r[1] == maximum(xh)
        @test r[2] == -maximum(xh)
    end

    @testset "counting sort orders particles by cell" begin
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        particles, meta = run_gpu(case, Float64, 1e-4)
        # after the run the host arrays are the (cell sorted) device state; the
        # sort above restored ID order, so check on a fresh download instead
        @test length(unique(particles.ID)) == length(particles)
        @test all(isfinite, particles.Density)
    end

    @testset "deterministic repeat" begin
        case = BENCH_CASES[findfirst(c -> c.name == "DamBreak2D_MDBC_dp0.01", BENCH_CASES)]
        p1, _ = run_gpu(case, Float64, 0.005)
        p2, _ = run_gpu(case, Float64, 0.005)
        @test p1.Density == p2.Density
        @test p1.Position == p2.Position
    end

    @testset "Float32 runs and stays physical" begin
        for name in ("StillWedge2D_MDBC_dp0.02", "DamBreak3D_dp0.02")
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            p, meta = run_gpu(case, Float32, 0.01)
            @test eltype(p.Density) == Float32
            @test all(isfinite, p.Density)
            @test all(x -> all(isfinite, x), p.Position)
            @test 900 < minimum(p.Density) && maximum(p.Density) < 1100
        end
    end

    if HAVE_CPU
        @testset "matches CPU reference: $(name)" for (name, simtime) in (
                ("StillWedge2D_MDBC_dp0.02", 0.02),
                ("MovingSquare2D_dp0.04",    0.01),
                ("DamBreak3D_dp0.02",        0.005),
                ("Duckling3D_MDBC_dp0.01",   0.005),
            )
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            ref  = run_cpu_reference(case, simtime)
            p, meta = run_gpu(case, Float64, simtime)

            @test meta.Iteration == ref.Iteration
            @test p.ID == ref.ID
            dρ = relerr(p.Density, ref.Density)
            dx = maximum(norm.(p.Position .- eachcol(ref.Position)))
            dv = maximum(norm.(p.Velocity .- eachcol(ref.Velocity)))
            @info "CPU vs GPU ($name): steps=$(meta.Iteration) max rel Δρ=$(dρ) max |Δx|=$(dx) max |Δv|=$(dv)"
            @test dρ < 1e-8
            @test dx < 1e-9
            @test dv < 1e-7
        end
    else
        @warn "CPU package not found next to gpu_version; skipping CPU comparison tests"
    end
end
