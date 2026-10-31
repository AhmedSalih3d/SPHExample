"Configuration shared by simulation metadata and the variable-resolution backend."
module VariableResolutionConfiguration

using StaticArrays

export VariableResolutionBox, VariableResolutionConfig

"""
    VariableResolutionBox(lower, upper; ratio=2, buffer_h=2, overlap_h=2)

One stationary refinement rectangle in 2D. The fine spacing is the coarse
spacing divided by `ratio`. Buffer and overlap widths are measured in each
domain's smoothing length and the fine smoothing length, respectively.
"""
struct VariableResolutionBox
    lower::SVector{2,Float64}
    upper::SVector{2,Float64}
    ratio::Int
    buffer_h::Float64
    overlap_h::Float64
    function VariableResolutionBox(lower, upper; ratio=2, buffer_h=2, overlap_h=2)
        lo, hi = SVector{2,Float64}(lower), SVector{2,Float64}(upper)
        all(isfinite,lo) && all(isfinite,hi) && all(hi .> lo) ||
            throw(ArgumentError("refinement bounds must be finite and strictly increasing"))
        ratio isa Integer && 2 <= ratio <= 4 || throw(ArgumentError("ratio must be an integer from 2 to 4"))
        isfinite(buffer_h) && buffer_h >= 2 || throw(ArgumentError("buffer_h must be finite and at least 2"))
        isfinite(overlap_h) && overlap_h >= 2 || throw(ArgumentError("overlap_h must be finite and at least 2"))
        new(lo,hi,ratio,buffer_h,overlap_h)
    end
end

"""
    VariableResolutionConfig(region; fine_geometry, fine_constants=nothing,
        fine_kernel=nothing, particle_reserve=nothing, include_buffers=false)

Enable two-domain CUDA refinement through `SimulationMetaData.VariableResolution`.
Pass the coarse geometry, constants and kernel to `RunSimulation` as usual.
Supply the fine fluid/wall geometry explicitly, including mDBC ghost fields.
When omitted, fine constants and kernel are derived by scaling the coarse
spacing, mass, smoothing length and kernel regularization by `region.ratio`.
`particle_reserve` reserves creation slots per domain; `nothing` sizes each
pool from its spacing and the refinement area. Output normally selects owned
particles; `include_buffers=true` includes both domains' buffers as well.

The current backend supports one fixed 2D rectangle, Wendland C2, fixed Simple
mDBC walls, artificial viscosity, full Fourtakas diffusion and reference
symplectic stepping. `SimDamping` is supported on both domains.
"""
struct VariableResolutionConfig{G,C,K}
    region::VariableResolutionBox
    fine_geometry::G
    fine_constants::C
    fine_kernel::K
    particle_reserve::Union{Nothing,Int}
    include_buffers::Bool
end

function VariableResolutionConfig(region::VariableResolutionBox; fine_geometry,
        fine_constants=nothing, fine_kernel=nothing, particle_reserve=nothing, include_buffers=false)
    fine_geometry isa AbstractVector && !isempty(fine_geometry) ||
        throw(ArgumentError("fine_geometry must be a nonempty vector of SPH geometries"))
    particle_reserve === nothing || (particle_reserve isa Integer && particle_reserve > 0) ||
        throw(ArgumentError("particle_reserve must be a positive integer or nothing"))
    return VariableResolutionConfig(region,fine_geometry,fine_constants,fine_kernel,
        particle_reserve === nothing ? nothing : Int(particle_reserve),Bool(include_buffers))
end

function Base.show(io::IO,config::VariableResolutionConfig)
    print(io,"VariableResolutionConfig(region=",config.region,", fine_geometry=",
        length(config.fine_geometry)," groups, particle_reserve=",config.particle_reserve,
        ", include_buffers=",config.include_buffers,")")
end

end
