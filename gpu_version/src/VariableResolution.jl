"""Native two-level, fixed-box variable-resolution SPH on CUDA."""
module VariableResolution

using CUDA
using StaticArrays
using LinearAlgebra
using Printf
using TimerOutputs
using ..SimulationLoggerConfiguration: SimulationLogger, log_line, LogStep, LogFinal
import ..SimulationLoggerConfiguration as LOGGER
using ..SimulationGeometry
using ..SimulationConstantsConfiguration
using ..SimulationMetaDataConfiguration: DualSPHysicsSymplectic
using ..SPHKernels
using ..SPHDensityDiffusionModels: FourtakasDensityDiffusion
using ..SPHViscosityModels: ArtificialViscosity
using ..PreProcess: AllocateDataStructures
using ..SPHCellList: upload_particles, GPUSupportArrays, rebuild_cell_list!, dual_time_step
using ..GPUCellGrid
using ..GPUKernels
using ..GPUStepState: HostStep
using ..ProduceHDFVTK: SaveVTKHDF
using ..AuxiliaryFunctions: to_3d

export VariableResolutionBox, run_variable_resolution_simulation

"""
    VariableResolutionBox(lower, upper; ratio = 2, buffer_h = 2, overlap_h = 2)

One stationary, axis-aligned refinement rectangle in 2D. The fine spacing is
the coarse spacing divided by `ratio`. Buffer widths are multiples of each
domain's smoothing length; the overlap is measured with the fine smoothing
length. Domains interact through reconstructed buffer states and accumulated
mass flux, rather than direct interactions between unequal particle masses.
Only one refinement level, fixed mDBC walls and Wendland C2 are supported.
Each domain may contain one fluid group; newly created particles keep its marker.
"""
struct VariableResolutionBox
    lower::SVector{2,Float64}
    upper::SVector{2,Float64}
    ratio::Int
    buffer_h::Float64
    overlap_h::Float64
    function VariableResolutionBox(lower, upper; ratio = 2, buffer_h = 2, overlap_h = 2)
        lo, hi = SVector{2,Float64}(lower), SVector{2,Float64}(upper)
        all(isfinite, lo) && all(isfinite, hi) && all(hi .> lo) ||
            throw(ArgumentError("refinement bounds must be finite and strictly increasing"))
        ratio isa Integer && 2 <= ratio <= 4 || throw(ArgumentError("ratio must be an integer from 2 to 4"))
        isfinite(buffer_h) && buffer_h >= 2 || throw(ArgumentError("buffer_h must be finite and at least 2"))
        isfinite(overlap_h) && overlap_h >= 2 || throw(ArgumentError("overlap_h must be finite and at least 2"))
        new(lo, hi, ratio, buffer_h, overlap_h)
    end
end

@inline inside_box(x, lo, hi) = all(x .>= lo) && all(x .<= hi)

function resolution_limits(region, coarse_h, fine_h)
    overlap = region.overlap_h * fine_h
    inset = overlap + region.buffer_h * coarse_h
    all(region.upper - region.lower .> 2inset) ||
        throw(ArgumentError("refinement rectangle is too small for its coarse buffer and overlap"))
    return (; fine_inner_lo = region.lower, fine_inner_hi = region.upper,
        fine_outer_lo = region.lower .- region.buffer_h * fine_h,
        fine_outer_hi = region.upper .+ region.buffer_h * fine_h,
        coarse_outer_lo = region.lower .+ overlap, coarse_outer_hi = region.upper .- overlap,
        coarse_inner_lo = region.lower .+ inset, coarse_inner_hi = region.upper .- inset)
end

@inline function buffer_particle(x, limits, fine)
    return fine ? !inside_box(x, limits.fine_inner_lo, limits.fine_inner_hi) :
        inside_box(x, limits.coarse_outer_lo, limits.coarse_outer_hi)
end

@inline function outside_domain(x, limits, fine)
    return fine ? !inside_box(x, limits.fine_outer_lo, limits.fine_outer_hi) :
        inside_box(x, limits.coarse_inner_lo, limits.coarse_inner_hi)
end

"Midpoint quadrature on four faces; normals point out of the receiving domain."
function interface_points(limits, fine, dx)
    lo = fine ? limits.fine_outer_lo : limits.coarse_inner_lo
    hi = fine ? limits.fine_outer_hi : limits.coarse_inner_hi
    points, normals, widths = SVector{2,Float64}[], SVector{2,Float64}[], Float64[]
    for axis in 1:2, side in (-1, 1)
        tangent = 3 - axis
        count = max(1, round(Int, (hi[tangent] - lo[tangent]) / dx))
        width = (hi[tangent] - lo[tangent]) / count
        normal = SVector{2,Float64}(ntuple(d -> d == axis ? side * (fine ? 1 : -1) : 0, 2))
        for k in 1:count
            point = SVector{2,Float64}(ntuple(d -> d == axis ? (side < 0 ? lo[d] : hi[d]) : lo[d] + (k - 0.5) * width, 2))
            push!(points, point)
            push!(normals, normal)
            push!(widths, width)
        end
    end
    return (; points, normals, widths)
end

struct ResolutionDomain{P,C,S,K,F,I,PP}
    particles::P
    cells::C
    support::S
    kernel::K
    constants::F
    interface::I
    predictor::PP
    free_slots::CuVector{Int32}
    allocation::CuVector{Int32}
    next_id::CuVector{Int64}
    events::CuVector{Int64}
    fine::Bool
end

function make_resolution_domain(geometry, constants::SimulationConstants{T}, kernel, limits, fine, reserve) where {T}
    geometry isa AbstractVector && !isempty(geometry) && all(g -> g isa SPHGeometry{2,T},geometry) ||
        throw(ArgumentError("each domain needs a nonempty vector of 2D geometries matching its constants' precision"))
    geometry = SPHGeometry{2,T}[geometry...]
    fluid_markers = unique([g.GroupMarker for g in geometry if g.Type == Fluid])
    length(fluid_markers) <= 1 || throw(ArgumentError("each resolution domain supports a single fluid group"))
    fluid_marker = isempty(fluid_markers) ? UInt(2) : UInt(only(fluid_markers))
    all(g -> g.Type in (Fixed, Fluid) && g.Motion === nothing, geometry) ||
        throw(ArgumentError("variable resolution currently supports only fluid and fixed walls"))
    all(g -> g.Type != Fixed || (hasproperty(g.Particles, :GhostPoints) && hasproperty(g.Particles, :GhostNormals)), geometry) ||
        throw(ArgumentError("fixed walls require mDBC ghost points and normals"))
    for g in geometry
        all(x -> all(isfinite,x),g.Particles.Position) && all(isfinite,g.Particles.Density) ||
            throw(ArgumentError("initial particle positions and densities must be finite"))
        !hasproperty(g.Particles,:Velocity) || all(x -> all(isfinite,x),g.Particles.Velocity) ||
            throw(ArgumentError("initial particle velocities must be finite"))
        all(ρ -> 0.7constants.ρ₀ <= ρ <= 1.3constants.ρ₀,g.Particles.Density) ||
            throw(ArgumentError("initial densities must lie within 70–130% of the reference density"))
        if g.Type == Fixed
            all(x -> all(isfinite,x) && !iszero(x),g.Particles.GhostPoints) &&
                all(x -> all(isfinite,x) && !iszero(x),g.Particles.GhostNormals) ||
                throw(ArgumentError("mDBC walls require finite nonzero ghost points and normals"))
        end
    end
    reserve isa Integer && reserve > 0 || throw(ArgumentError("particle reserve must be a positive integer"))
    initial_count = sum(length(g.Particles) for g in geometry)
    initial_count > 0 || throw(ArgumentError("each domain needs initial fluid or wall particles"))
    reserve < typemax(Int32) - initial_count || throw(ArgumentError("particle capacity exceeds Int32 indexing"))
    dormant = SPHGeometry{2,T}(fill(zero(SVector{2,Float64}), reserve);
        Density = constants.ρ₀, Type = Frozen, GroupMarker = 3)
    host = AllocateDataStructures([geometry; dormant]; position_type = Float64, cells = false, kernel_output = false)
    # Odd/even namespaces give globally unique IDs across both domains and
    # across every subsequent retirement and creation, without a shared atomic.
    maximum(host.ID) <= typemax(Int64) ÷ 4 || throw(ArgumentError("particle IDs exceed the supported two-domain namespace"))
    host.ID .= 2 .* host.ID .- (fine ? 0 : 1)
    gpu = upload_particles(host; position_type = Float64, kernel_output = false)
    n = length(gpu)
    cells = CellListWorkspace{2,Float64}(n)
    # Dormant pool slots are excluded from grid bounds and all force sums.
    cells.frozen_total = reserve
    support = GPUSupportArrays{2,T}(n; position_type = Float64, shifting = false, dual_time_stepping = true)
    # Predictor neighbors are sorted independently. Keep their inputs dense and
    # scatter only the evaluated derivatives back to the accepted-state order.
    predictor = (; particles = upload_particles(host; position_type = Float64, kernel_output = false),
        cells = CellListWorkspace{2,Float64}(n),
        support = GPUSupportArrays{2,T}(n; position_type = Float64, shifting = false, dual_time_stepping = true),
        kernel, constants)
    faces = interface_points(limits, fine, Float64(constants.dx))
    interface = (; points = CuArray(faces.points), normals = CuArray(faces.normals),
        widths = CuArray(faces.widths), mass = CUDA.zeros(Float64, length(faces.points)), fluid_marker)
    return ResolutionDomain(gpu, cells, support, kernel, constants, interface, predictor,
        CUDA.zeros(Int32, n), CUDA.zeros(Int32, 2), CuArray([Int64(maximum(host.ID) + 2)]), CUDA.zeros(Int64, 4), fine)
end

include("GPUVariableResolution.jl")
include("VariableResolutionLogging.jl")

function rebuild_domain!(domain)
    # Always supply the frozen-particle state, even if a previous transfer filled
    # the last dormant slot. This is a private pool, not a public frozen output.
    domain.cells.frozen_total = 1
    rebuild_cell_list!(domain.particles, domain.cells, domain.kernel.H⁻¹;
        support = domain.support, kernel = domain.kernel, discard_acceleration = true, sort_inactive = false)
end

function domain_force_stage!(domain, position, velocity, density, pairs, time)
    (; particles, cells, support, kernel, constants) = domain
    step = HostStep(0.0, time)
    launch_pos_cells!(pairs, position, particles.CellID, cells.grid_dev, step, kernel)
    launch_mdbc!(density, position, particles.GhostPoints, particles.GhostIndex, particles.Type,
        cells.CellStart, cells.grid_dev, step, kernel, constants; pos_cells = pairs)
    launch_interactions!(support.dρdtI, particles.Acceleration, particles.Kernel, particles.KernelGradient,
        support.∇Cᵢ, support.∇◌rᵢ, position, density, velocity, particles.Type,
        cells.CellStart, particles.CellID, cells.grid_dev, step,
        FourtakasDensityDiffusion(full = true), ArtificialViscosity(), kernel, constants,
        Val(false), Val(false); boundary_forces = Val(false), pos_cells = pairs,
        viscous_dt = support.dual_state.viscous, fastmath = false)
    criteria = GPUKernels.launch_dual_criteria!(support.dual_state, velocity, particles.Acceleration, particles.Type)
    scheme = DualSPHysicsSymplectic(; smoothing_length = Float64(kernel.h), sound_speed = Float64(constants.c₀))
    return dual_time_step(criteria, kernel, constants, scheme)
end

function predictor_force_stage!(domain, time)
    p,s,half = domain.particles,domain.support,domain.predictor
    q = half.particles
    @cuda threads=128 blocks=cld(length(p),128) copy_predictor_kernel!(
        q.Position,q.Velocity,q.Density,q.Type,q.ID,q.GroupMarker,q.GhostPoints,q.GhostNormals,
        s.Positionₙ⁺,s.Velocityₙ⁺,s.ρₙ⁺,p.Type,p.GroupMarker,p.GhostPoints,p.GhostNormals,Int32(length(p)))
    rebuild_domain!(half)
    limit = domain_force_stage!(half,q.Position,q.Velocity,q.Density,half.support.PosCells,time)
    @cuda threads=128 blocks=cld(length(p),128) scatter_predictor_kernel!(
        p.Acceleration,s.dρdtI,s.ρₙ⁺,q.Acceleration,half.support.dρdtI,q.Density,q.ID,Int32(length(p)))
    return limit
end

function exchange_buffers!(target, source, limits)
    p, s, c = target.particles, source.particles, source.cells
    @cuda threads=128 blocks=cld(length(p),128) reconstruct_buffers_kernel!(
        p.Position, p.Velocity, p.Density, p.Type, s.Position, s.Velocity, s.Density, s.Type,
        c.CellStart, c.grid_dev, source.kernel, source.constants, limits, target.fine, source.fine, Int32(length(p)))
end

function transfer_particles!(target, source, limits, dt)
    p, s, c = target.particles, source.particles, source.cells
    fill!(target.free_slots, Int32(0))
    compact_nonzero!(target.cells, p.Type, target.free_slots; predicate = is_dormant)
    fill!(target.allocation, Int32(0))
    face = target.interface
    @cuda threads=128 blocks=cld(length(face.points),128) interface_flux_kernel!(
        face.points, face.normals, face.widths, face.mass,
        p.Position, p.Velocity, p.Density, p.Type, p.ID, p.GroupMarker,
        target.free_slots, target.allocation, target.next_id, target.events,
        target.cells.CellStart, target.cells.grid_dev, target.kernel,
        s.Position, s.Velocity, s.Density, s.Type, c.CellStart, c.grid_dev,
        source.kernel, source.constants, target.constants, face.fluid_marker, limits, source.fine, dt, Int32(length(face.points)))
    allocation = Array(target.allocation)
    allocation[2] == 0 || error("Variable-resolution particle reserve exhausted; increase particle_reserve (no buffer particles were silently dropped)")
    return allocation[1] > 0
end

function retire_particles!(domain, limits)
    p = domain.particles
    @cuda threads=128 blocks=cld(length(p),128) retire_particles_kernel!(
        p.Position, p.Velocity, p.Density, p.Type, domain.events, limits, domain.fine, domain.constants.ρ₀, Int32(length(p)))
end

function resolution_snapshot(domain, region, limits; include_buffers = false)
    p = domain.particles
    x, v, rho, types, ids = Array(p.Position), Array(p.Velocity), Array(p.Density), Array(p.Type), Array(p.ID)
    owned = [inside_box(position, region.lower, region.upper) == domain.fine for position in x]
    keep = findall(i -> types[i] != Frozen && (include_buffers || owned[i]), eachindex(types))
    c = domain.constants
    pressure = [Float64(c.Cb) * ((Float64(rho[i]) / c.ρ₀)^c.γ - 1) for i in keep]
    return (; position = x[keep], velocity = to_3d(v[keep]), density = rho[keep], pressure,
        type = Int8.(types[keep]), id = ids[keep], level = fill(Int8(domain.fine), length(keep)),
        spacing = fill(c.dx, length(keep)), mass = fill(c.m₀, length(keep)), owned = Int8.(owned[keep]),
        buffer = Int8[types[i] == Fluid && buffer_particle(x[i], limits, domain.fine) for i in keep])
end

"""
    run_variable_resolution_simulation(; coarse_geometry, fine_geometry,
        coarse_constants, fine_constants, coarse_kernel, fine_kernel, region,
        simulation_time = 1, output_interval = 0.0025, save_location,
        particle_reserve = nothing, include_buffers = false)

Run two native CUDA SPH domains with a common adaptive timestep, mDBC density
correction and symplectic predictor/corrector stages. Buffers exchange density
and velocity using first-order moving least squares with Shepard fallback.
Face-integrated mass flux creates particles; particles leaving each domain
are retired and their slots reused. Reserve exhaustion fails explicitly.

This initial implementation supports one fixed 2D refinement rectangle and
fixed walls. It uses MLS reconstruction instead of DualSPHysics' default
Liu–Liu reconstruction and is experimental, not a trajectory-equivalent port.
Numbered VTKHDF frames and a `.vtkhdf.series` time index contain current owned
particles (or both domains with `include_buffers=true`). `History.csv` records
particle counts, represented mass and transfer/exclusion events.
`SimulationOutput.log` uses the standard progress/ETA table and final wall-time
report, with separate initialization, solver and output timings. CUDA is
synchronized at output boundaries for complete interval timings. The driver
clock includes first-use JIT but excludes package loading and caller geometry
construction. The result exposes `log_path` and `timings`.
"""
function run_variable_resolution_simulation(; coarse_geometry, fine_geometry, coarse_constants,
        fine_constants, coarse_kernel, fine_kernel, region::VariableResolutionBox,
        simulation_time = 1.0, output_interval = 0.0025, save_location,
        particle_reserve = nothing, include_buffers = false)
    CUDA.functional() || error("variable resolution requires a functional CUDA device")
    isfinite(simulation_time) && simulation_time > 0 || throw(ArgumentError("simulation_time must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))
    for (constants, kernel) in ((coarse_constants, coarse_kernel), (fine_constants, fine_kernel))
        typeof(constants.dx) in (Float32,Float64) && typeof(kernel.h) == typeof(constants.dx) ||
            throw(ArgumentError("each domain requires matching Float32 or Float64 constants and kernel precision"))
        kernel isa SPHKernelInstance{<:WendlandC2,2} || throw(ArgumentError("variable resolution requires 2D Wendland C2 kernels"))
        positive = (constants.dx, constants.m₀, constants.c₀, constants.ρ₀, constants.Cb, constants.γ⁻¹, kernel.h)
        all(isfinite,positive) && minimum(positive) > 0 ||
            throw(ArgumentError("spacings, masses, sound speeds, reference densities, EOS coefficients and smoothing lengths must be positive and finite"))
        isapprox(constants.m₀, constants.ρ₀ * constants.dx^2; rtol = 1e-5) ||
            throw(ArgumentError("2D resolution domains require mass = reference density * spacing^2"))
        isfinite(constants.γ) && constants.γ > 0 && isfinite(constants.CFL) && constants.CFL > 0 &&
            all(v -> isfinite(v) && v >= 0,(constants.g,constants.α,constants.δᵩ)) ||
            throw(ArgumentError("gamma and CFL must be positive; gravity, viscosity and diffusion must be finite and nonnegative"))
        kernel.k == 2 || throw(ArgumentError("Wendland support radius must be 2h"))
    end
    isapprox(coarse_constants.dx / fine_constants.dx, region.ratio; rtol = 1e-5) || throw(ArgumentError("domain spacings do not match the refinement ratio"))
    isapprox(coarse_kernel.h / fine_kernel.h, region.ratio; rtol = 1e-5) || throw(ArgumentError("domain smoothing lengths do not match the refinement ratio"))
    for field in (:ρ₀, :c₀, :g, :γ, :γ⁻¹, :Cb, :α, :δᵩ, :CFL)
        getproperty(coarse_constants,field) == getproperty(fine_constants,field) || throw(ArgumentError("domain constants must agree on $field"))
    end
    output = abspath(save_location)
    isdir(output) && !isempty(readdir(output)) && throw(ArgumentError("save_location must be empty: $output"))
    limits = resolution_limits(region, Float64(coarse_kernel.h), Float64(fine_kernel.h))
    area = prod(limits.fine_outer_hi - limits.fine_outer_lo)
    coarse_reserve = particle_reserve === nothing ? max(1024, ceil(Int, 2area / coarse_constants.dx^2)) : particle_reserve
    fine_reserve = particle_reserve === nothing ? max(1024, ceil(Int, 2area / fine_constants.dx^2)) : particle_reserve
    return with_resolution_logger(output) do logger,timer
        @timeit timer "00 Initialization" begin
            settings = (; coarse_constants, fine_constants, coarse_kernel, fine_kernel,
                coarse_geometry, fine_geometry, coarse_reserve, fine_reserve, region, simulation_time, output_interval)
            initialize_resolution_log!(logger,settings)
            coarse = make_resolution_domain(coarse_geometry, coarse_constants, coarse_kernel, limits, false, coarse_reserve)
            fine = make_resolution_domain(fine_geometry, fine_constants, fine_kernel, limits, true, fine_reserve)
            domains = (coarse, fine)
            foreach(d -> retire_particles!(d, limits), domains)
            foreach(rebuild_domain!, domains)
            exchange_buffers!(coarse, fine, limits)
            exchange_buffers!(fine, coarse, limits)
            CUDA.synchronize()
        end
        log_line(logger,logger.ValuesToPrint)
        log_line(logger,logger.ValuesToPrintC)
        times, files = Float64[], String[]
        time, iterations, frame = 0.0, 0, 0
        previous_limit = Inf
        last_logged_steps = 0
        last_counts, last_mass = (0,0), 0.0
        history_path = joinpath(output, "History.csv")
        open(history_path, "w") do history
            println(history, "time,iteration,coarse_fluid,fine_fluid,represented_fluid_mass,coarse_created,fine_created,coarse_retired,fine_retired,coarse_excluded,fine_excluded,coarse_nonfinite,fine_nonfinite")
            function write_frame()
                @timeit timer "02 Output Frame" begin
                    events = map(d -> Array(d.events),domains)
                    all(e -> e[4] == 0,events) || error("Non-finite variable-resolution particle state at t=$time; coarse/fine failures=$(events[1][4])/$(events[2][4])")
                    snapshots = map(d -> resolution_snapshot(d,region,limits; include_buffers), domains)
                    fields = map(name -> vcat((getproperty(s,name) for s in snapshots)...), keys(first(snapshots)))
                    data = NamedTuple{keys(first(snapshots))}(fields)
                    file = @sprintf("DamBreak2D_VRes_%04d.vtkhdf", frame)
                    SaveVTKHDF(joinpath(output,file), to_3d(data.position),
                        ["Velocity", "Density", "Pressure", "Type", "ID", "Resolution", "Spacing", "Mass", "Owned", "IsBuffer"],
                        data.velocity, data.density, data.pressure, data.type, data.id, data.level, data.spacing, data.mass, data.owned, data.buffer)
                    counts = map(s -> count(==(Int8(Fluid)),s.type), snapshots)
                    mass = sum(data.mass[i] for i in eachindex(data.mass) if data.type[i] == Int8(Fluid) && data.owned[i] == 1; init = 0.0)
                    println(history, join((time,iterations,counts...,mass,events[1][1],events[2][1],events[1][2],events[2][2],events[1][3],events[2][3],events[1][4],events[2][4]),','))
                    flush(history)
                    push!(times,time); push!(files,file)
                    open(joinpath(output,"DamBreak2D_VRes.vtkhdf.series"),"w") do io
                        println(io,"{\"file-series-version\": \"1.0\", \"files\": [")
                        for i in eachindex(files)
                            println(io,"{\"name\": \"",files[i],"\", \"time\": ",times[i],"}",i == length(files) ? "" : ",")
                        end
                        println(io,"]}")
                    end
                    last_counts, last_mass = counts, mass
                end
                progress = (; OutputIterationCounter = frame, TotalTime = time, Iteration = iterations,
                    StepsTakenForLastOutput = last_logged_steps, SimulationTime = Float64(simulation_time))
                LogStep(logger,progress,timer)
                last_logged_steps = iterations
                frame += 1
            end
            write_frame()
            while time < simulation_time
                deadline = min(length(times) * Float64(output_interval), Float64(simulation_time))
                @timeit timer "01 Solver" begin
                    while time < deadline
                        limits_dt = map(domains) do d
                            p,s = d.particles,d.support
                            domain_force_stage!(d,p.Position,p.Velocity,p.Density,s.PosCells,time)
                        end
                        dt = min(limits_dt...,previous_limit,deadline-time)
                        isfinite(dt) && dt > 0 || error("nonpositive or nonfinite VRes timestep")
                        previous_limit = Inf
                        for d in domains
                            p,s,c = d.particles,d.support,d.constants
                            GPUKernels.launch_dual_predictor!(s.Positionₙ⁺,s.Velocityₙ⁺,s.ρₙ⁺,
                                p.Position,p.Velocity,p.Density,p.Acceleration,s.dρdtI,p.Type,dt,c)
                            previous_limit = min(previous_limit,predictor_force_stage!(d,time+dt/2))
                            GPUKernels.launch_dual_corrector!(p.Position,p.Velocity,p.Density,s.Positionₙ⁺,
                                s.ρₙ⁺,p.Acceleration,s.dρdtI,p.Type,dt,c)
                            retire_particles!(d,limits)
                            rebuild_domain!(d)
                        end
                        created_coarse = transfer_particles!(coarse,fine,limits,dt)
                        created_fine = transfer_particles!(fine,coarse,limits,dt)
                        exchange_buffers!(coarse,fine,limits)
                        exchange_buffers!(fine,coarse,limits)
                        created_coarse && rebuild_domain!(coarse)
                        created_fine && rebuild_domain!(fine)
                        time = dt == deadline-time ? deadline : time+dt
                        iterations += 1
                    end
                    CUDA.synchronize()
                end
                write_frame()
            end
        end
        return (; coarse, fine, time, iterations, output, times, files, history_path,
            particle_counts = (; coarse = last_counts[1], fine = last_counts[2]), represented_fluid_mass = last_mass,
            configuration = (; method = :two_domain_mls, region, limits, include_buffers,
                density_diffusion = :fourtakas_full, boundary = :mdbc, cell_subdivision = 1,
                coarse_spacing = coarse_constants.dx, fine_spacing = fine_constants.dx,
                coarse_mass = coarse_constants.m₀, fine_mass = fine_constants.m₀))
    end
end

end
