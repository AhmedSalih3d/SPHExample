"Configuration shared by simulation metadata and the variable-resolution backend."
module VariableResolutionConfiguration

using StaticArrays

export VariableResolutionBox, VariableResolutionConfig

"""
    VariableResolutionBox(lower, upper; ratio=2, buffer_h=2, overlap_h=2)

One reference refinement rectangle in 2D, stationary unless its configuration
sets `follow` to a body marker. The fine spacing is the coarse
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
        fine_kernel=nothing, particle_reserve=nothing, include_buffers=false,
        follow=nothing)

Enable two-domain CUDA refinement through `SimulationMetaData.VariableResolution`.
Pass the coarse geometry, constants and kernel to `RunSimulation` as usual.
Supply the fine fluid/wall geometry explicitly, including mDBC ghost fields.
When omitted, fine constants and kernel are derived by scaling the coarse
spacing, mass, smoothing length and kernel regularization by `region.ratio`.
`particle_reserve` reserves creation slots per domain; `nothing` sizes each
pool from its spacing and the refinement area. Output normally selects owned
particles; `include_buffers=true` includes both domains' buffers as well.

Set `follow` to a `Moving` or `Floating` body's positive `GroupMarker` to
translate and rotate the rectangle with that body. Bounds describe the body's
reference geometry coordinates. Supply the body in `fine_geometry`; an optional
coarse copy is omitted so its force and mass are counted only once. The body
must fit inside the coarse exclusion rectangle with one coarse support radius
of clearance. A followed floating body may coexist with additional `Moving`
walls prescribed through `SimMotion`, such as a coarse-domain wave piston. Prescribed bodies use
`MotionDetails`, `SimMotion` or `SimRotation`; floating bodies use
`FloatingDetails`. Attached regions require ordinary timestep schemes.
All body placement, region transforms and relative interface flux run on CUDA,
including graph execution. Discontinuous motion that outruns the buffer fails
explicitly. Output ownership uses the current transformed rectangle.

The backend supports one 2D rectangle, Wendland C2 and DBC or Simple mDBC
walls. Select ordinary symplectic, single-neighbor
or reference symplectic stepping with `SimTimeStepping`. Select any
GPU-compatible density diffusion model with `RunSimulation(SimDensityDiffusion=...)`,
as in the ordinary solver. Ordinary schemes also use the selected `SimViscosity`;
reference stepping requires artificial viscosity. `SimDamping` is supported on both domains.
Both ordinary timestep schemes reuse the solver's integration kernels. Because
refinement rebuilds the domains each step, single-neighbor stepping refreshes
the accepted derivative each step, following the ordinary rebuild contract.
"""
struct VariableResolutionConfig{G,C,K}
    region::VariableResolutionBox
    fine_geometry::G
    fine_constants::C
    fine_kernel::K
    particle_reserve::Union{Nothing,Int}
    include_buffers::Bool
    follow::Union{Nothing,Int}
end

function VariableResolutionConfig(region::VariableResolutionBox; fine_geometry,
        fine_constants=nothing, fine_kernel=nothing, particle_reserve=nothing, include_buffers=false, follow=nothing)
    fine_geometry isa AbstractVector && !isempty(fine_geometry) ||
        throw(ArgumentError("fine_geometry must be a nonempty vector of SPH geometries"))
    particle_reserve === nothing || (particle_reserve isa Integer && particle_reserve > 0) ||
        throw(ArgumentError("particle_reserve must be a positive integer or nothing"))
    follow === nothing || (follow isa Integer && !(follow isa Bool) && 0 < follow <= typemax(Int)) ||
        throw(ArgumentError("follow must be a positive body GroupMarker or nothing"))
    return VariableResolutionConfig(region,fine_geometry,fine_constants,fine_kernel,
        particle_reserve === nothing ? nothing : Int(particle_reserve),Bool(include_buffers),
        follow === nothing ? nothing : Int(follow))
end

function Base.show(io::IO,config::VariableResolutionConfig)
    print(io,"VariableResolutionConfig(region=",config.region,", fine_geometry=",
        length(config.fine_geometry)," groups, particle_reserve=",config.particle_reserve,
        ", include_buffers=",config.include_buffers,", follow=",config.follow,")")
end

end
