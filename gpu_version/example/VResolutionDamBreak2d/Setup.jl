# Definitions only. Include example/VResolutionDamBreak2d.jl to start a run.
using SPHExampleGPU
using StaticArrays: SVector

"""
    vresolution_dambreak_case(; dx = 0.01, ratio = 2, duration = 1.0, ...)

Build the 4 m by 3 m open tank and 1 m by 2 m water column from DualSPHysics
5.4 `vresolution/01_DamBreak2D`. The downstream rectangle uses spacing
`dx / ratio`. Particles are generated here; no external solver is required.
The initial lattice uses cell centres, with five wall layers outside the
physical tank surfaces. Construction does not allocate CUDA arrays or run.
"""
function vresolution_dambreak_case(; dx = 0.01, ratio = 2, duration = 1.0,
        output_interval = 0.0025, FloatType = Float32,
        region = VariableResolutionBox((2.5,-0.2),(4.1,0.6); ratio),
        save_location = joinpath(@__DIR__, "..", "..", "output", "vresolution_dambreak2d"))
    isfinite(dx) && 0 < dx <= 0.05 || throw(ArgumentError("dx must satisfy 0 < dx <= 0.05 m"))
    isapprox(1/dx,round(1/dx); rtol = 1e-8) || throw(ArgumentError("dx must divide the 1 m water width exactly, for example 0.01, 0.02, 0.025 or 0.05 m"))
    ratio == region.ratio || throw(ArgumentError("ratio must match the refinement rectangle"))
    isfinite(duration) && duration > 0 || throw(ArgumentError("duration must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))
    T = FloatType
    T in (Float32,Float64) || throw(ArgumentError("FloatType must be Float32 or Float64"))
    sound_speed = 10sqrt(2 * 9.81)
    function level_case(spacing, fine)
        constants = SimulationConstants{T}(2; dx = spacing, c₀ = sound_speed,
            γ = 1, α = 0.05, δᵩ = 0.1, CFL = 0.2)
        kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(1.5spacing), η² = T((0.1 * 1.5spacing)^2))
        # A shared lattice avoids duplicate floor/wall corner particles.
        nx, ny = round(Int,4 / spacing), round(Int,3 / spacing)
        wall = SVector{2,Float64}[]
        for j in -5:(ny-1), i in -5:(nx+4)
            if j < 0 || i < 0 || i >= nx
                push!(wall,SVector((i+0.5)*spacing,(j+0.5)*spacing))
            end
        end
        water = [SVector((i+0.5)*spacing,(j+0.5)*spacing)
            for j in 0:(round(Int,2 / spacing)-1) for i in 0:(round(Int,1 / spacing)-1)]
        if fine
            margin = region.buffer_h * kernel.h
            lower, upper = region.lower .- margin, region.upper .+ margin
            # Retain physical wall support beyond the fluid buffer.
            filter!(x -> all(lower .- kernel.H .<= x) && all(x .<= upper .+ kernel.H),wall)
            filter!(x -> all(lower .<= x) && all(x .<= upper),water)
        end
        ghost = [SVector(x[1] < 0 ? -x[1] : x[1] > 4 ? 8-x[1] : x[1],
                         x[2] < 0 ? -x[2] : x[2]) for x in wall]
        boundary = SPHGeometry{2,T}(Particles = particle_struct_array(wall,constants.ρ₀;
            GhostPoints = ghost, GhostNormals = ghost .- wall), GroupMarker = 1, Type = Fixed)
        # Inverse hydrostatic initialization for the reference's linear EOS.
        density = T[constants.ρ₀ * (1 + constants.g * max(0,2-x[2]) / constants.c₀^2) for x in water]
        fluid = SPHGeometry{2,T}(water; Density = density, GroupMarker = 2, Type = Fluid)
        return (; geometry = [boundary,fluid], constants, kernel)
    end
    coarse, fine = level_case(dx,false), level_case(dx / ratio,true)
    tank = polyline([(0.0,3.0),(0.0,0.0),(4.0,0.0),(4.0,3.0)]; thickness = 5dx, side = :right)
    water = rectangle((0.0,0.0),1.0,2.0)
    refinement = rectangle(Tuple(region.lower),region.upper[1]-region.lower[1],region.upper[2]-region.lower[2])
    return (; coarse, fine, region, simulation_time = duration, output_interval,
        save_location = abspath(save_location), shapes = (; tank,water,refinement))
end

"Run the native two-domain CUDA dam break and write its geometry and VTKHDF series."
function run_vresolution_dambreak(; particle_reserve = nothing, include_buffers = false,
        execution = :gpu, batch_steps = 32, use_graphs = true, kwargs...)
    case = vresolution_dambreak_case(; kwargs...)
    result = run_variable_resolution_simulation(;
        coarse_geometry = case.coarse.geometry, fine_geometry = case.fine.geometry,
        coarse_constants = case.coarse.constants, fine_constants = case.fine.constants,
        coarse_kernel = case.coarse.kernel, fine_kernel = case.fine.kernel,
        region = case.region, simulation_time = case.simulation_time,
        output_interval = case.output_interval, save_location = case.save_location,
        particle_reserve, include_buffers, execution, batch_steps, use_graphs)
    SavePolygonVTKHDF(joinpath(result.output,"DamBreak2D_VRes_Geometry.vtkhdf"),case.shapes)
    return result
end
