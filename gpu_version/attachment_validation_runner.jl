using Test, SPHExampleGPU, CUDA, StaticArrays, StructArrays, LinearAlgebra, HDF5
include("test/variable_resolution.jl")
include("test/variable_resolution_solver.jl")
include("test/vresolution_logging.jl")
include("test/resolution_attachment.jl")
include("test/floating_body_updates.jl")
include("test/stationary_predictors.jl")
include("test/sorting_shadows.jl")
include("test/grid_export.jl")
include("example/VResolutionAttachedBody2d.jl")
@testset "attached tank example smoke tests" begin
    for floating in (false,true)
        result = redirect_stdout(devnull) do
            run_attached_resolution_example(;floating,duration=0.01,dx=0.05,save_location=mktempdir())
        end
        @test result.final_state.complete
        @test result.configuration.effective.variable_resolution.follow == 5
        @test result.final_state.floating === nothing ? !floating : floating
        @test all(isfinite,result.final_state.particles.Density)
    end
end
