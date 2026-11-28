using CUDA, StaticArrays, Statistics, Printf
using SPHExampleGPU: SPHKernelInstance, WendlandC2
using SPHExampleGPU.SPHKernels: ∇Wᵢⱼ
using SPHExampleGPU.SimulationEquations: EquationOfStateGamma7
using SPHExampleGPU.GPUReductions: block_reduce

# One thread per particle and exactly one block. The split and resident runs
# use the same stage functions and device-side timestep and clock arithmetic.
@inline function forces!(s, kernel, i)
    @inbounds begin
        x, v, rho = s.x[i], s.v[i], s.rho[i]
        pressure_term = EquationOfStateGamma7(rho, 20f0, 1000f0) / (rho * rho)
        acceleration = zero(v)
        density_rate = 0f0
        signal = 0f0
        for j in Int32(1):Int32(length(s.x))
            r = x - s.x[j]
            r2 = sum(abs2, r)
            if 0f0 < r2 < kernel.H²
                dv = v - s.v[j]
                gradient = ∇Wᵢⱼ(kernel, sqrt(r2) * kernel.h⁻¹, r)
                rho_j = s.rho[j]
                other = EquationOfStateGamma7(rho_j, 20f0, 1000f0) / (rho_j * rho_j)
                mass = 0.4f0
                acceleration -= (mass * (pressure_term + other)) * gradient
                density_rate += mass * sum(dv .* gradient)
                signal = max(signal, abs(kernel.h * sum(dv .* r) / (r2 + kernel.η²)))
            end
        end
        s.a[i] = acceleration
        s.drho[i] = density_rate
        s.criteria[i] = SVector(sum(abs2, acceleration), signal)
    end
    return nothing
end

@inline max_components(a, b) = max.(a, b)

@inline function timestep!(s, kernel, i, step)
    @inbounds criteria = s.criteria[i]
    a2, signal = block_reduce(criteria, max_components)
    if i == 1
        dt = 0.2f0 * min(sqrt(kernel.h / sqrt(a2)), kernel.h / (20f0 + signal))
        @inbounds begin
            s.dt[step] = dt
            s.time[step + 1] = s.time[step] + Float64(dt)
        end
    end
    return nothing
end

@inline function integrate!(s, i, step)
    @inbounds begin
        dt = s.dt[step]
        s.v[i] += dt * s.a[i]
        s.x[i] += dt * s.v[i]
        s.rho[i] += dt * s.drho[i]
    end
    return nothing
end

force_kernel!(s, kernel) = forces!(s, kernel, threadIdx().x)
timestep_kernel!(s, kernel, step) = timestep!(s, kernel, threadIdx().x, step)
integrate_kernel!(s, step) = integrate!(s, threadIdx().x, step)

function resident_kernel!(s, kernel, first_step, last_step)
    i = threadIdx().x
    for step in first_step:last_step
        forces!(s, kernel, i)
        sync_threads()
        timestep!(s, kernel, i, step)
        sync_threads()
        integrate!(s, i, step)
        sync_threads()
    end
    return nothing
end

function experiment(n, steps; rounds = 21)
    n in (32, 64) || throw(ArgumentError("Expected 32 or 64 particles"))
    steps in (1, 8, 32) || throw(ArgumentError("Expected 1, 8, or 32 steps"))
    steps = Int32(steps)
    kernel = SPHKernelInstance{2,Float32}(WendlandC2(); dx = 0.02f0)
    x = [0.02f0 * SVector{2,Float32}(i % 8, i ÷ 8) for i in 0:n-1]
    v = [SVector(0.12f0 * sin(Float32(i)), 0.08f0 * cos(Float32(i))) for i in 1:n]
    rho = [1000f0 + 2f0 * sin(Float32(i)) for i in 1:n]
    s = (; x = CuArray(x), v = CuArray(v), rho = CuArray(rho), a = CuArray{SVector{2,Float32}}(undef, n),
        drho = CUDA.zeros(Float32, n), criteria = CuArray{SVector{2,Float32}}(undef, n),
        dt = CUDA.zeros(Float32, steps), time = CUDA.zeros(Float64, steps + 1))
    function reset!()
        copyto!(s.x, x)
        copyto!(s.v, v)
        copyto!(s.rho, rho)
        fill!(s.dt, NaN32)
        fill!(s.time, NaN)
        fill!(view(s.time, 1:1), 0.0)
        CUDA.synchronize()
    end
    function split!()
        for step in Int32(1):steps
            @cuda threads=n blocks=1 force_kernel!(s, kernel)
            @cuda threads=n blocks=1 timestep_kernel!(s, kernel, step)
            @cuda threads=n blocks=1 integrate_kernel!(s, step)
        end
    end
    function fused!()
        for step in Int32(1):steps
            @cuda threads=n blocks=1 resident_kernel!(s, kernel, step, step)
        end
    end
    function resident!()
        @cuda threads=n blocks=1 resident_kernel!(s, kernel, Int32(1), steps)
    end
    snapshot() = map(Array, (s.x, s.v, s.rho, s.a, s.drho, s.dt, s.time))
    reset!()
    split!()
    reference = snapshot()
    @assert all(rho -> isfinite(rho) && rho > 0, reference[3])
    @assert all(v -> all(isfinite, v), reference[1]) && all(v -> all(isfinite, v), reference[2])
    @assert all(v -> all(isfinite, v), reference[4]) && all(isfinite, reference[5])
    @assert all(dt -> isfinite(dt) && dt > 0, reference[6])
    @assert reference[7] == [0.0; accumulate(+, Float64.(reference[6]))]
    @assert reference[1] != x && reference[3] != rho
    reset!()
    fused!()
    CUDA.synchronize()
    graph = CUDA.instantiate(CUDA.capture(split!))
    fused_graph = CUDA.instantiate(CUDA.capture(fused!))
    variants = (split!, () -> CUDA.launch(graph), fused!, () -> CUDA.launch(fused_graph), resident!)
    samples = [Float64[] for _ in variants]
    for round in -4:rounds
        offset = mod(round, length(variants))
        for index in 1:length(variants)
            k = mod1(offset + index, length(variants))
            reset!()
            elapsed = @elapsed begin
                variants[k]()
                CUDA.synchronize()
            end
            @assert isequal(snapshot(), reference) "State differs for variant $k, n=$n, steps=$steps"
            round > 0 && push!(samples[k], elapsed * 1e6)
        end
    end
    @printf("%2d %2d", n, steps)
    for values in samples
        @printf(" %8.2f", median(values))
    end
    ratios = samples[4] ./ samples[5]
    @printf(" %8.3f [%5.3f, %5.3f]\n", median(ratios), quantile(ratios, 0.25), quantile(ratios, 0.75))
end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDA.functional(true) || error("CUDA is required")
    println(CUDA.name(CUDA.device()), " | Julia ", VERSION)
    println("Median warm µs/batch (21 rotating rounds), final state and timestep history bitwise checked")
    println(" n  K    split    graph    fused  fused_g resident  g_f/r [Q1, Q3]")
    for n in (32, 64), steps in (1, 8, 32)
        experiment(n, steps)
    end
end
