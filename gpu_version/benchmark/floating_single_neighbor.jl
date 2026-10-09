# Usage: julia --project=. benchmark/floating_single_neighbor.jl
# Complete warmed cylinder timesteps, including rebuilds, excluding file output.
include("floating_steps.jl")

geometry, constants = cylinder_case()
driver = SPHExampleGPU.SPHCellList
floating_mod = SPHExampleGPU.GPUFloating
println("GPU: ", CUDA.name(CUDA.device()), "; Float32 physics, Float64 positions")
symplectic_ms, symplectic_state = step_benchmark(driver, floating_mod, geometry, constants;
    scheme = SymplecticTimeStepping())
single_ms, single_state = step_benchmark(driver, floating_mod, geometry, constants;
    scheme = SingleNeighborTimeStepping())
@printf("Complete timestep speedup %.3fx; final centre difference %.6g m\n",
    symplectic_ms / single_ms,
    maximum(norm.(symplectic_state.center .- single_state.center)))
