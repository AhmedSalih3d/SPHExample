module MDBCValidation

include(joinpath(@__DIR__, "dual_compare.jl"))
using .Test06DualComparison
using .Test06DualComparison.SphericTest06
using SPHExampleGPU
using StaticArrays: SVector, SMatrix
using LinearAlgebra: dot, norm, det
using Statistics: mean
using HDF5, CSV, TOML, Printf, CUDA

const C = Test06DualComparison
const GK = SPHExampleGPU.GPUKernels
const RESULTS = joinpath(@__DIR__, "results", "mdbc_validation")
const WORK = joinpath(@__DIR__, "runs", "mdbc_validation")

export tank_ghost_offset, square_ghost_offset, mdbc_wall_state, mdbc_pair_loads,
    mdbc_frame_loads, build_mdbc_case, analyze_mdbc_history, run_mdbc_case,
    run_pressure_traction_study, run_couette_traction_study, run_moving_square_mdbc_study

"""Reflect an exterior tank-wall particle across [0, width] × [0, height], including 45° corners."""
@inline function tank_ghost_offset(pos::SVector{2,Float64}; width::Float64 = 10.0, height::Float64 = 5.0)
    gx = pos[1] < 0.0 ? -pos[1] : (pos[1] > width ? 2.0 * width - pos[1] : pos[1])
    gy = pos[2] < 0.0 ? -pos[2] : (pos[2] > height ? 2.0 * height - pos[2] : pos[2])
    return SVector(gx, gy) - pos
end

"""Reflect an interior square particle across [-half_side, half_side]², preserving y-symmetry at corners."""
@inline function square_ghost_offset(local_pos::SVector{2,Float64}; half_side::Float64 = 0.5)
    dx_face = half_side - abs(local_pos[1])
    dy_face = half_side - abs(local_pos[2])
    sx = local_pos[1] >= 0.0 ? 1.0 : -1.0
    sy = local_pos[2] >= 0.0 ? 1.0 : -1.0
    if abs(dx_face - dy_face) <= 1e-12
        return SVector(2.0 * sx * dx_face, 2.0 * sy * dy_face)
    elseif dx_face < dy_face
        return SVector(2.0 * sx * dx_face, 0.0)
    else
        return SVector(0.0, 2.0 * sy * dy_face)
    end
end

"""Evaluate exact Float64 SimpleMDBC or UpdatedMDBC wall state from fluid neighbors around gp."""
function mdbc_wall_state(gp::SVector{2,Float64}, wall_pos::SVector{2,Float64}, wall_vel::SVector{2,Float64},
        ghost_offset::SVector{2,Float64}, wall_acc::SVector{2,Float64}, gravity_scalar::Float64,
        stored_wall_rho::Float64, fluid_cells::Dict{Tuple{Int,Int},Vector{Int}},
        position, velocity, density, kernel, constants, boundary_mode::MDBCMode;
        clamp_simple_wall_density::Bool = false)
    if boundary_mode isa NoMDBC
        p = EquationOfStateGamma7(stored_wall_rho, constants.c₀, constants.ρ₀)
        return (; active = true, density = stored_wall_rho, pressure = p, viscous_velocity = wall_vel)
    end
    A = zero(SMatrix{3,3,Float64,9})
    b_rho = zero(SVector{3,Float64})
    b_vel = zero(SVector{2,Float64})
    gx = floor(Int, gp[1] / kernel.H)
    gy = floor(Int, gp[2] / kernel.H)
    for cy in gy-1:gy+1, cx in gx-1:gx+1
        neighbors = get(fluid_cells, (cx, cy), nothing)
        neighbors === nothing && continue
        @inbounds for j in neighbors
            x_gj = gp - position[j]
            d2 = dot(x_gj, x_gj)
            if d2 <= kernel.H²
                q = sqrt(d2) * kernel.h⁻¹
                W = Wᵢⱼ(kernel, q)
                grad = ∇Wᵢⱼ(kernel, q, x_gj)
                vol = constants.m₀ / density[j]
                vw = vol * W
                b_rho += SVector(constants.m₀ * W, (constants.m₀ * grad)...)
                b_vel += vw * velocity[j]
                first_col = SVector(vw, (vol * grad)...)
                A += first_col * SVector(1.0, (-x_gj)...)'
            end
        end
    end
    if boundary_mode isa SimpleMDBC
        rho = stored_wall_rho
        if abs(det(A)) >= 1e-3
            sol = A \ b_rho
            v1 = sol[1] + dot(SVector(sol[2], sol[3]), wall_pos - gp)
            rho = isnan(v1) ? constants.ρ₀ : v1
        elseif A[1, 1] > 0.0
            v = b_rho[1] / A[1, 1]
            rho = isnan(v) ? constants.ρ₀ : v
        end
        clamp_simple_wall_density && (rho = max(constants.ρ₀, rho))
        p = EquationOfStateGamma7(rho, constants.c₀, constants.ρ₀, inv(constants.ρ₀), constants.BackgroundPressure)
        return (; active = true, density = rho, pressure = p, viscous_velocity = wall_vel)
    end
    div_pos = A[2, 2] + A[3, 3]
    active = isfinite(div_pos) && div_pos > 0.0 && A[1, 1] > 0.0
    ghost_rho = GK.updated_ghost_density(A, b_rho, constants.ρ₀, constants.dx)
    normal = ghost_offset / norm(ghost_offset)
    gravity = SVector(0.0, -gravity_scalar)
    rho, p = GK.cloned_boundary_state(ghost_rho, normal, wall_pos - gp, gravity, wall_acc, constants.c₀, constants.ρ₀)
    rho_final = active ? rho : constants.ρ₀
    p_final = (active ? p : 0.0) + constants.BackgroundPressure
    ghost_vel = active ? b_vel / A[1, 1] : zero(wall_vel)
    mirrored = active ? 2.0 * wall_vel - ghost_vel : wall_vel
    visc_vel = active && !(boundary_mode isa DBCPlus) ?
        mirrored - dot(mirrored - wall_vel, normal) * normal : wall_vel
    return (; active, density = rho_final, pressure = p_final, viscous_velocity = visc_vel)
end

"""Pressure and laminar viscous load on wall particle i from fluid particle j."""
@inline function mdbc_pair_loads(wall_pos, fluid_pos, wall_visc_vel, fluid_vel,
        wall_rho, wall_pressure, fluid_rho, kernel, constants)
    separation = wall_pos - fluid_pos
    d2 = dot(separation, separation)
    if !(0.0 < d2 < kernel.H²)
        return zero(separation), zero(separation)
    end
    q = sqrt(d2) * kernel.h⁻¹
    grad = ∇Wᵢⱼ(kernel, q, separation)
    fluid_pressure = EquationOfStateGamma7(fluid_rho, constants.c₀, constants.ρ₀, inv(constants.ρ₀), constants.BackgroundPressure)
    pressure_force = -constants.m₀^2 * ((wall_pressure + fluid_pressure) / (wall_rho * fluid_rho)) * grad
    viscosity, _ = compute_viscosity(Laminar(), kernel, constants, nothing, separation,
        wall_visc_vel - fluid_vel, grad, d2, wall_rho, fluid_rho,
        inv(wall_rho), inv(fluid_rho), 1, 2)
    return pressure_force, constants.m₀ * viscosity
end

"""Integrate square/fluid pair loads for NoMDBC, SimpleMDBC or UpdatedMDBC."""
function mdbc_frame_loads(position, velocity, density, types, markers, kernel, constants;
        boundary_mode::MDBCMode = NoMDBC(), center::SVector{2,Float64} = SVector(1.5, 2.5),
        wall_acceleration::SVector{2,Float64} = SVector(0.0, 0.0), half_side::Float64 = 0.5,
        square_marker::Int = 3, square_type::ParticleType = Moving,
        clamp_simple_wall_density::Bool = false)
    if boundary_mode isa NoMDBC && square_marker == 3 && square_type == Moving
        r = frame_loads(position, velocity, density, types, markers, kernel, constants)
        s = -0.5 * constants.ρ₀
        return merge(r, (; force_pressure = SVector(s * r.cd_pressure, s * r.cy_pressure),
            force_viscous = SVector(s * r.cd_viscous, s * r.cy_viscous)))
    end
    fluid_cells = Dict{Tuple{Int,Int},Vector{Int}}()
    cell(x) = (floor(Int, x[1] / kernel.H), floor(Int, x[2] / kernel.H))
    for j in eachindex(types)
        types[j] == Int(Fluid) || continue
        push!(get!(() -> Int[], fluid_cells, cell(position[j])), j)
    end
    pressure_force = zero(SVector{2,Float64})
    viscous_force = zero(pressure_force)
    for i in eachindex(types)
        markers[i] == square_marker && types[i] == Int(square_type) || continue
        ix, iy = cell(position[i])
        has_fluid_neighbor = false
        for cy in iy-1:iy+1, cx in ix-1:ix+1
            neighbors = get(fluid_cells, (cx, cy), nothing)
            neighbors === nothing && continue
            for j in neighbors
                d2 = sum(abs2, position[i] - position[j])
                if 0.0 < d2 < kernel.H²
                    has_fluid_neighbor = true
                    break
                end
            end
            has_fluid_neighbor && break
        end
        has_fluid_neighbor || continue
        ghost_offset = square_ghost_offset(position[i] - center; half_side)
        gp = position[i] + ghost_offset
        wstate = mdbc_wall_state(gp, position[i], velocity[i], ghost_offset, wall_acceleration,
            Float64(constants.g), density[i], fluid_cells, position, velocity, density,
            kernel, constants, boundary_mode; clamp_simple_wall_density)
        wstate.active || continue
        for cy in iy-1:iy+1, cx in ix-1:ix+1
            neighbors = get(fluid_cells, (cx, cy), nothing)
            neighbors === nothing && continue
            for j in neighbors
                pf, vf = mdbc_pair_loads(position[i], position[j], wstate.viscous_velocity, velocity[j],
                    wstate.density, wstate.pressure, density[j], kernel, constants)
                pressure_force += pf
                viscous_force += vf
            end
        end
    end
    scale = -2.0 / constants.ρ₀
    return (; cd_pressure = scale * pressure_force[1], cd_viscous = scale * viscous_force[1],
        cy_pressure = scale * pressure_force[2], cy_viscous = scale * viscous_force[2],
        force_pressure = pressure_force, force_viscous = viscous_force)
end

"""Build SPHERIC Test 06 with NoMDBC, SimpleMDBC or UpdatedMDBC on the identical lattice."""
function build_mdbc_case(; reynolds = 100, dx = 0.025, duration = 8.0, interval = 0.02,
        sound_speed = 40.0, cfl = 0.2, density_diffusion = 0.1, boundary_mode::Type{<:MDBCMode} = UpdatedMDBC,
        kernel_h_factor = sqrt(2), wall_layers = nothing,
        shifting_mode::Type{<:ShiftingMode} = NoShifting,
        motion_kind::Symbol = :integrated, legacy_timestep::Bool = false,
        save = joinpath(WORK, "case"), float_type = Float32, graph = true)
    motion_kind in (:integrated, :hermite) || throw(ArgumentError("motion_kind must be :integrated or :hermite"))
    base = build_case(; reynolds, dx, duration, interval, sound_speed, cfl, density_diffusion,
        kernel_h_factor, wall_layers, save, float_type, graph)
    T = float_type
    walls = base.geometry[1].Particles.Position
    fluid = base.geometry[2].Particles.Position
    body = base.geometry[3].Particles.Position
    geometry = if boundary_mode === NoMDBC
        base.geometry
    else
        wall_offsets = [tank_ghost_offset(p) for p in walls]
        wall_ghosts = [walls[k] + wall_offsets[k] for k in eachindex(walls)]
        center0 = SVector(1.5, 2.5)
        body_offsets = [square_ghost_offset(p - center0) for p in body]
        body_ghosts = [body[k] + body_offsets[k] for k in eachindex(body)]
        [
            SPHGeometry{2,T}(Particles = particle_struct_array(walls, base.constants.ρ₀;
                GhostPoints = wall_ghosts, GhostNormals = [SVector{2,T}(o) for o in wall_offsets]),
                GroupMarker = 1, Type = Fixed),
            SPHGeometry{2,T}(fluid; Density = base.constants.ρ₀, GroupMarker = 2, Type = Fluid),
            SPHGeometry{2,T}(Particles = particle_struct_array(body, base.constants.ρ₀;
                GhostPoints = body_ghosts, GhostNormals = [SVector{2,T}(o) for o in body_offsets]),
                GroupMarker = 3, Type = Moving),
        ]
    end
    metadata = SimulationMetaData{2,T,shifting_mode,NoKernelOutput,boundary_mode,StoreLog}(
        SimulationName = "MovingSquare", SaveLocation = abspath(save), SimulationTime = base.duration,
        OutputTimes = base.metadata.OutputTimes, VisualizeInParaview = false, OpenLogFile = false,
        ExportSingleVTKHDF = true, OutputVariables = ["Density", "Velocity", "Type", "GroupMarker"],
        GPUUseGraph = graph, GPUClipTimeStepToOutput = true, GPUDoublePosition = true,
        GPULegacyTimestep = legacy_timestep,
        GPUCellSubdivision = 1, GPUXCellSubdivision = 1, GPURadialInteractions = false,
        GPUInteractionFastMath = false, GPUPreloadInteractionTypes = false,
        GPUBoundaryForces = false, GPUOutputBufferBytes = 16 * 2^20)
    trajectory = motion_kind === :integrated ? C.integrated_trajectory() : body_trajectory()
    motion = PrescribedMotion(3; translation = trajectory)
    return merge(base, (; geometry, metadata, motion, boundary_mode, shifting_mode, motion_kind))
end

"""Reconstruct accepted-frame drag and state diagnostics for any boundary mode."""
function analyze_mdbc_history(path, case; coefficient_path, diagnostics_path)
    constants = SimulationConstants{Float64}(2; dx = case.dx, ρ₀ = case.constants.ρ₀,
        c₀ = case.constants.c₀, g = 0, ν₀ = case.constants.ν₀)
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = Float64(case.kernel.h),
        αD = Float64(case.kernel.αD), η² = Float64(case.kernel.η²))
    mode = case.boundary_mode()
    trajectory = case.motion.translation
    rows, checks = NamedTuple[], NamedTuple[]
    h5open(path, "r") do file
        root = haskey(file, "VTKHDF/Particles") ? file["VTKHDF/Particles"] : file["VTKHDF"]
        times = read(root["Steps/Values"])
        offsets = read(root["Steps/PointOffsets"])
        counts = read(root["NumberOfPoints"])
        for frame in eachindex(times)
            columns = offsets[frame]+1:offsets[frame]+counts[frame]
            position = [SVector{2,Float64}(x[1], x[2]) for x in eachcol(root["Points"][:, columns])]
            velocity = [SVector{2,Float64}(v[1], v[2]) for v in eachcol(root["PointData/Velocity"][:, columns])]
            density = Float64.(root["PointData/Density"][columns])
            types = root["PointData/Type"][columns]
            markers = root["PointData/GroupMarker"][columns]
            all(isfinite, density) && all(>(0), density) && all(x -> all(isfinite, x), position) &&
                all(v -> all(isfinite, v), velocity) || error("Nonfinite state at t=$(times[frame])")
            fluid = findall(==(Int(Fluid)), types)
            square = findall(==(3), markers)
            count(==(Int(Frozen)), types) == 0 || error("Frozen particles at t=$(times[frame])")
            length(fluid) == case.counts.fluid && length(square) == case.counts.square || error("Particle loss")
            displacement, expected_velocity, expected_acc = trajectory_state(trajectory, times[frame])
            center = SVector(1.5, 2.5) + displacement
            motion_error = norm(sum(position[square]) / length(square) - center)
            velocity_error = maximum(v -> norm(v - expected_velocity), velocity[square])
            minimum_clearance = minimum(fluid) do i
                x, y = position[i]
                square_clearance = max(abs(x - center[1]), abs(y - center[2])) - 0.5
                min(x, 10.0 - x, y, 5.0 - y, square_clearance)
            end
            loads = mdbc_frame_loads(position, velocity, density, types, markers, kernel, constants;
                boundary_mode = mode, center, wall_acceleration = expected_acc)
            push!(rows, (; time = times[frame], cd_pressure = loads.cd_pressure, cd_viscous = loads.cd_viscous,
                cy_pressure = loads.cy_pressure, cy_viscous = loads.cy_viscous))
            push!(checks, (; time = times[frame], motion_error, velocity_error,
                density_relative_max = maximum(abs, density[fluid] ./ constants.ρ₀ .- 1.0),
                max_speed = maximum(norm, velocity[fluid]), minimum_clearance,
                fluid_particles = length(fluid), square_particles = length(square)))
        end
    end
    CSV.write(diagnostics_path, checks)
    open(coefficient_path, "w") do io
        println(io, "# time Cd_pressure Cd_viscous")
        for row in rows
            @printf(io, "%.9e %.9e %.9e\n", row.time, row.cd_pressure, row.cd_viscous)
        end
    end
    CSV.write(replace(coefficient_path, ".dat" => ".csv"), rows)
    return (; rows, checks)
end

function run_mdbc_case(case; results_dir, diffusion = ZeroGravityLinearDensityDiffusion(), cleaning = nothing)
    verify_reference()
    CUDA.functional() || error("An NVIDIA CUDA GPU is required")
    save = case.metadata.SaveLocation
    mkpath(save)
    isfile(joinpath(save, "MovingSquare.vtkhdf")) && error("Refusing to overwrite an existing run: $save")
    mkpath(results_dir)
    particles = AllocateDataStructures(case.geometry, case.metadata)
    logger = SimulationLogger(save)
    result = nothing
    wall = open(joinpath(save, "console.log"), "w") do io
        redirect_stdout(io) do
            @elapsed try
                result = RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
                    SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
                    SimParticles = particles, SimViscosity = Laminar(),
                    SimDensityDiffusion = diffusion, SimTimeStepping = SymplecticTimeStepping(),
                    SimMotion = case.motion, SimDivergenceCleaning = cleaning)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
        end
    end
    result.final_state.complete && result.diagnostics.frozen_particles == 0 || error("Incomplete run")
    analysis = analyze_mdbc_history(joinpath(save, "MovingSquare.vtkhdf"), case;
        coefficient_path = joinpath(results_dir, "native.dat"),
        diagnostics_path = joinpath(results_dir, "native_state.csv"))
    info = (; reynolds = case.reynolds, dx = case.dx, n = round(Int, 1.0 / case.dx),
        sound_speed = Float64(case.constants.c₀), cfl = Float64(case.constants.CFL),
        h = Float64(case.kernel.h), wall_layers = case.layers,
        background_pressure = Float64(case.constants.BackgroundPressure),
        boundary_mode = string(nameof(case.boundary_mode)),
        shifting_mode = string(nameof(case.shifting_mode)),
        motion = string(case.motion_kind),
        divergence_cleaning = cleaning !== nothing,
        cleaning_kappa = cleaning === nothing ? 0.0 : Float64(cleaning.kappa),
        cleaning_sigma = cleaning === nothing ? 0.0 : Float64(cleaning.sigma),
        cleaning_courant = cleaning === nothing ? 0.0 : Float64(cleaning.courant),
        cleaning_absolute_pressure = cleaning === nothing ? false : cleaning.absolute_pressure,
        diffusion = string(nameof(typeof(diffusion))) * (diffusion isa FourtakasDensityDiffusion && diffusion.full ? "_full" : ""),
        density_diffusion_coefficient = Float64(case.constants.δᵩ),
        timestep_criterion = string(result.configuration.effective.timestep_criterion),
        steps = result.performance.steps, frames = result.diagnostics.output_frames,
        particles = result.diagnostics.particle_count,
        fluid_particles = case.counts.fluid, square_particles = case.counts.square, fixed_particles = case.counts.fixed,
        rebuilds = result.diagnostics.cell_list_rebuilds, readbacks = result.diagnostics.step_state_readbacks,
        wall_seconds = wall, solver_seconds = result.performance.wall_time_seconds,
        motion_error_max = maximum(x -> x.motion_error, analysis.checks),
        velocity_error_max = maximum(x -> x.velocity_error, analysis.checks),
        density_relative_max = maximum(x -> x.density_relative_max, analysis.checks),
        mach_max = maximum(x -> x.max_speed, analysis.checks) / Float64(case.constants.c₀),
        minimum_clearance = minimum(x -> x.minimum_clearance, analysis.checks),
        gpu = CUDA.name(CUDA.device()), julia = string(VERSION),
        cuda_jl = string(pkgversion(CUDA)), driver = string(CUDA.driver_version()))
    C.save_info(joinpath(results_dir, "native.toml"), info)
    return info
end

# ---------------------------------------------------------------------------
# Foundational Verification Test 1:
# Hydrostatic & Zero-Ambient Linear-Acceleration Pressure Traction on a Square
# ---------------------------------------------------------------------------

function build_submerged_square_fixture(; dx = 0.05, c0 = 40.0, g = 9.81, ax = 0.0,
        boundary_mode::Type{<:MDBCMode} = UpdatedMDBC, hydro_dbc_walls::Bool = false)
    nx, ny, ns = round.(Int, (3.0 / dx, 3.0 / dx, 1.0 / dx))
    constants = SimulationConstants{Float64}(2; dx, ρ₀ = 1000.0, g, c₀ = c0, ν₀ = 0.01, α = 0.0, δᵩ = 0.1, CFL = 0.2)
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = sqrt(2.0) * dx)
    layers = ceil(Int, kernel.H / dx) + 1
    center = SVector(1.5, 1.5)
    pressure_exact(p) = constants.ρ₀ * g * (3.0 - p[2]) - constants.ρ₀ * ax * (p[1] - center[1])
    rho_exact(p) = constants.ρ₀ * max(0.2, 1.0 + pressure_exact(p) / constants.Cb)^(1.0 / constants.γ)
    position, velocity, density = SVector{2,Float64}[], SVector{2,Float64}[], Float64[]
    types, markers = Int[], Int[]
    for j in -layers:ny+layers-1, i in -layers:nx+layers-1
        p = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        if !(0 <= i < nx && 0 <= j < ny)
            push!(position, p); push!(velocity, SVector(0.0, 0.0))
            rho_w = hydro_dbc_walls ? max(constants.ρ₀, rho_exact(p)) : constants.ρ₀
            push!(density, rho_w); push!(types, Int(Fixed)); push!(markers, 1)
        elseif ns <= i < 2ns && ns <= j < 2ns
            push!(position, p); push!(velocity, SVector(0.0, 0.0))
            rho_w = hydro_dbc_walls ? max(constants.ρ₀, rho_exact(p)) : constants.ρ₀
            push!(density, rho_w); push!(types, Int(Moving)); push!(markers, 3)
        else
            push!(position, p); push!(velocity, SVector(0.0, 0.0))
            push!(density, rho_exact(p)); push!(types, Int(Fluid)); push!(markers, 2)
        end
    end
    return (; position, velocity, density, types, markers, constants, kernel, center, boundary_mode)
end

function run_pressure_traction_study(; save_results::Bool = true)
    rows = NamedTuple[]
    # 1A: Positive hydrostatic buoyancy on submerged 1×1 square: exact Fy = +ρ0 g L² = +9810 N/m.
    exact_fy = 1000.0 * 9.81 * 1.0
    for n in (20, 40, 60)
        dx = 1.0 / n
        for (label, mode, hydro_dbc, clamp_simple) in (
                ("NoMDBC_rho0_walls", NoMDBC, false, false),
                ("NoMDBC_hydro_walls", NoMDBC, true, false),
                ("SimpleMDBC", SimpleMDBC, false, true),
                ("UpdatedMDBC", UpdatedMDBC, false, false))
            fix = build_submerged_square_fixture(; dx, g = 9.81, ax = 0.0, boundary_mode = mode, hydro_dbc_walls = hydro_dbc)
            loads = mdbc_frame_loads(fix.position, fix.velocity, fix.density, fix.types, fix.markers,
                fix.kernel, fix.constants; boundary_mode = mode(), center = fix.center,
                wall_acceleration = SVector(0.0, 0.0), clamp_simple_wall_density = clamp_simple)
            fy = loads.force_pressure[2]
            push!(rows, (; regime = "hydrostatic_positive", n, dx, mode = label,
                fx_measured = loads.force_pressure[1], fy_measured = fy,
                target_force = exact_fy, relative_error = (fy - exact_fy) / exact_fy,
                abs_relative_error = abs(fy - exact_fy) / exact_fy))
        end
    end
    # 1B: Zero-ambient linear acceleration / suction-compression field (g = 0, ax = +1 m/s²):
    # Fluid pressure p(x) = -ρ0 ax (x - 1.5) has +500 Pa on x=1.0 and -500 Pa (suction!) on x=2.0.
    # Exact fluid pressure force on the square is Fx = ∫ (-p n_x) ds = -((-500)(+1) + (+500)(-1)) = +1000 N/m
    # when p = -1000*(x-1.5), wait: at left face x=1, p = +500, outward normal into fluid is -x, force on square is +p = +500;
    # at right face x=2, p = -500, outward normal into fluid is +x, force on square is -p = +500; total Fx = +1000 N/m!
    # Note: p(x) = -ρ0 b_x (x - 1.5) corresponds to body force -b_x on fluid or wall acceleration a_w = (+1, 0) in fluid rest frame!
    exact_fx = 1000.0 * 1.0 * 1.0
    for n in (20, 40, 60)
        dx = 1.0 / n
        for (label, mode, clamp_simple) in (
                ("NoMDBC_rho0_walls", NoMDBC, false),
                ("SimpleMDBC_predictor_clamped", SimpleMDBC, true),
                ("SimpleMDBC_unclamped_stage1", SimpleMDBC, false),
                ("UpdatedMDBC", UpdatedMDBC, false))
            fix = build_submerged_square_fixture(; dx, g = 0.0, ax = 1.0, boundary_mode = mode)
            loads = mdbc_frame_loads(fix.position, fix.velocity, fix.density, fix.types, fix.markers,
                fix.kernel, fix.constants; boundary_mode = mode(), center = fix.center,
                wall_acceleration = SVector(1.0, 0.0), clamp_simple_wall_density = clamp_simple)
            fx = loads.force_pressure[1]
            push!(rows, (; regime = "zero_ambient_suction_acceleration", n, dx, mode = label,
                fx_measured = fx, fy_measured = loads.force_pressure[2],
                target_force = exact_fx, relative_error = (fx - exact_fx) / exact_fx,
                abs_relative_error = abs(fx - exact_fx) / exact_fx))
        end
    end

    # 1C: Dynamic GPU hydrostatic settling over 0.25 s at L/dp = 20 and 40.
    dynamic_rows = NamedTuple[]
    if CUDA.functional()
        for n in (20, 40), mode in (NoMDBC, SimpleMDBC, UpdatedMDBC)
            dx = 1.0 / n
            fix = build_submerged_square_fixture(; dx, g = 9.81, ax = 0.0, boundary_mode = mode,
                hydro_dbc_walls = mode === NoMDBC)
            T = Float32
            walls = [fix.position[i] for i in eachindex(fix.types) if fix.markers[i] == 1]
            fluid = [fix.position[i] for i in eachindex(fix.types) if fix.markers[i] == 2]
            fluid_rho = T[fix.density[i] for i in eachindex(fix.types) if fix.markers[i] == 2]
            body = [fix.position[i] for i in eachindex(fix.types) if fix.markers[i] == 3]
            wall_rho = T[fix.density[i] for i in eachindex(fix.types) if fix.markers[i] == 1]
            body_rho = T[fix.density[i] for i in eachindex(fix.types) if fix.markers[i] == 3]
            psa(pos, rho; kw...) = (a = particle_struct_array(pos, T(1000); kw...); a.Density .= rho; a)
            geometry = if mode === NoMDBC
                [
                    SPHGeometry{2,T}(Particles = psa(walls, wall_rho), GroupMarker = 1, Type = Fixed),
                    SPHGeometry{2,T}(Particles = psa(fluid, fluid_rho), GroupMarker = 2, Type = Fluid),
                    SPHGeometry{2,T}(Particles = psa(body, body_rho), GroupMarker = 3, Type = Fixed),
                ]
            else
                w_off = [tank_ghost_offset(p; width = 3.0, height = 3.0) for p in walls]
                b_off = [square_ghost_offset(p - fix.center) for p in body]
                [
                    SPHGeometry{2,T}(Particles = psa(walls, T.(fill(1000, length(walls)));
                        GhostPoints = walls .+ w_off, GhostNormals = [SVector{2,T}(o) for o in w_off]),
                        GroupMarker = 1, Type = Fixed),
                    SPHGeometry{2,T}(Particles = psa(fluid, fluid_rho), GroupMarker = 2, Type = Fluid),
                    SPHGeometry{2,T}(Particles = psa(body, T.(fill(1000, length(body)));
                        GhostPoints = body .+ b_off, GhostNormals = [SVector{2,T}(o) for o in b_off]),
                        GroupMarker = 3, Type = Fixed),
                ]
            end
            mktempdir() do save
                meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,mode,NoLog}(
                    SimulationName = "HydroSquare", SaveLocation = save, SimulationTime = 0.25,
                    OutputTimes = [0.25], VisualizeInParaview = false, OpenLogFile = false,
                    ExportSingleVTKHDF = true, GPUUseGraph = true, GPUClipTimeStepToOutput = true,
                    GPUDoublePosition = true, GPULegacyTimestep = false,
                    GPUCellSubdivision = 1, GPUXCellSubdivision = 1)
                particles = AllocateDataStructures(geometry, meta)
                constants = SimulationConstants{T}(2; dx, ρ₀ = 1000, g = 9.81, c₀ = 40, ν₀ = 0.02, α = 0, δᵩ = 0.1, CFL = 0.2)
                kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(sqrt(2.0) * dx))
                logger = SimulationLogger(save)
                result = try
                    RunSimulation(SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
                        SimKernel = kernel, SimLogger = logger, SimParticles = particles,
                        SimViscosity = Laminar(), SimDensityDiffusion = LinearDensityDiffusion(),
                        SimTimeStepping = SymplecticTimeStepping())
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                pos64 = [SVector{2,Float64}(p) for p in particles.Position]
                vel64 = [SVector{2,Float64}(v) for v in particles.Velocity]
                rho64 = Float64.(particles.Density)
                ty = Int.(particles.Type)
                mk = Int.(particles.GroupMarker)
                loads = mdbc_frame_loads(pos64, vel64, rho64, ty, mk, fix.kernel, fix.constants;
                    boundary_mode = mode(), center = fix.center, square_marker = 3, square_type = Fixed,
                    clamp_simple_wall_density = mode === SimpleMDBC)
                fluid_mask = particles.Type .== Fluid
                spurious_rms = sqrt(mean(sum(abs2, v) for v in vel64[fluid_mask]))
                spurious_max = maximum(norm, vel64[fluid_mask])
                fy = loads.force_pressure[2]
                push!(dynamic_rows, (; n, dx, mode = string(nameof(mode)), steps = result.performance.steps,
                    fy_measured = fy, fy_exact = exact_fy,
                    fy_relative_error = (fy - exact_fy) / exact_fy,
                    spurious_velocity_rms = spurious_rms, spurious_velocity_max = spurious_max))
            end
        end
    end
    if save_results
        mkpath(RESULTS)
        CSV.write(joinpath(RESULTS, "traction_pressure.csv"), rows)
        isempty(dynamic_rows) || CSV.write(joinpath(RESULTS, "traction_hydrostatic_dynamic.csv"), dynamic_rows)
    end
    return (; static = rows, dynamic = dynamic_rows)
end

# ---------------------------------------------------------------------------
# Foundational Verification Test 2:
# Laminar Planar Couette Viscous Wall-Traction & Galilean Invariance Test
# ---------------------------------------------------------------------------

function couette_wall_shear(n::Int, boundary_mode::Type{<:MDBCMode};
        u_top::Float64 = 1.0, nu::Float64 = 0.05, boost::SVector{2,Float64} = SVector(0.0, 0.0))
    dx = 1.0 / n
    H_gap = 1.0
    Lx = 1.0
    constants = SimulationConstants{Float64}(2; dx, ρ₀ = 1000.0, g = 0.0, c₀ = 40.0, ν₀ = nu, α = 0.0, δᵩ = 0.0, CFL = 0.2)
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = sqrt(2.0) * dx)
    layers = ceil(Int, kernel.H / dx) + 1
    # Periodic x in [0, 1): tile one periodic image layer on left and right for exact discrete operator evaluation.
    position, velocity, density = SVector{2,Float64}[], SVector{2,Float64}[], Float64[]
    types, markers, owns_primary = Int[], Int[], Bool[]
    for j in -layers:n+layers-1, i in -layers:n+layers-1
        p = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        primary = 0 <= i < n
        if j < 0
            push!(position, p); push!(velocity, boost); push!(density, 1000.0)
            push!(types, Int(Moving)); push!(markers, 1); push!(owns_primary, primary)
        elseif j >= n
            push!(position, p); push!(velocity, boost + SVector(u_top, 0.0)); push!(density, 1000.0)
            push!(types, Int(Moving)); push!(markers, 3); push!(owns_primary, primary)
        else
            push!(position, p); push!(velocity, boost + SVector(u_top * (p[2] / H_gap), 0.0)); push!(density, 1000.0)
            push!(types, Int(Fluid)); push!(markers, 2); push!(owns_primary, primary)
        end
    end
    fluid_cells = Dict{Tuple{Int,Int},Vector{Int}}()
    cell(x) = (floor(Int, x[1] / kernel.H), floor(Int, x[2] / kernel.H))
    for j in eachindex(types)
        types[j] == Int(Fluid) || continue
        push!(get!(() -> Int[], fluid_cells, cell(position[j])), j)
    end
    mode = boundary_mode()
    viscous_top = zero(SVector{2,Float64})
    viscous_bot = zero(SVector{2,Float64})
    for i in eachindex(types)
        owns_primary[i] && markers[i] in (1, 3) || continue
        ghost_offset = markers[i] == 1 ? SVector(0.0, -2.0 * position[i][2]) : SVector(0.0, 2.0 * (H_gap - position[i][2]))
        gp = position[i] + ghost_offset
        wstate = mdbc_wall_state(gp, position[i], velocity[i], ghost_offset, SVector(0.0, 0.0),
            0.0, density[i], fluid_cells, position, velocity, density, kernel, constants, mode)
        wstate.active || continue
        ix, iy = cell(position[i])
        for cy in iy-1:iy+1, cx in ix-1:ix+1
            neighbors = get(fluid_cells, (cx, cy), nothing)
            neighbors === nothing && continue
            for j in neighbors
                _, vf = mdbc_pair_loads(position[i], position[j], wstate.viscous_velocity, velocity[j],
                    wstate.density, wstate.pressure, density[j], kernel, constants)
                if markers[i] == 3
                    viscous_top += vf
                else
                    viscous_bot += vf
                end
            end
        end
    end
    # Exact shear stress opposing top wall +x motion is -Fx_top / Lx = +ρ0 * nu * u_top / H_gap.
    tau_exact = constants.ρ₀ * nu * u_top / H_gap
    tau_top = -viscous_top[1] / Lx
    tau_bot = viscous_bot[1] / Lx
    return (; tau_top, tau_bot, tau_exact, normal_top = viscous_top[2] / Lx)
end

function run_couette_traction_study(; save_results::Bool = true)
    static_rows = NamedTuple[]
    for n in (20, 40, 60), mode in (NoMDBC, SimpleMDBC, UpdatedMDBC)
        rest = couette_wall_shear(n, mode; boost = SVector(0.0, 0.0))
        boosted = couette_wall_shear(n, mode; boost = SVector(0.65, 0.40))
        push!(static_rows, (; n, dx = 1.0 / n, mode = string(nameof(mode)),
            tau_exact = rest.tau_exact,
            tau_top = rest.tau_top,
            tau_bot = rest.tau_bot,
            relative_error_top = (rest.tau_top - rest.tau_exact) / rest.tau_exact,
            abs_relative_error = abs(rest.tau_top - rest.tau_exact) / rest.tau_exact,
            galilean_drift_shear = abs(boosted.tau_top - rest.tau_top),
            galilean_drift_normal = abs(boosted.normal_top - rest.normal_top)))
    end

    dynamic_rows = NamedTuple[]
    if CUDA.functional()
        # Dynamic startup of periodic Couette channel (Lx = 1 m, H = 1 m, nu = 0.1 m²/s, t = 2.5 s => t*nu/H² = 0.25).
        # Compare against the exact unsteady analytical Fourier series at t = 2.5 s!
        u_top = 1.0
        nu = 0.1
        t_end = 2.5
        exact_u(y, t) = u_top * (y + sum((2.0 * (-1)^m / (m * π)) * exp(-(m * π)^2 * nu * t) * sin(m * π * y) for m in 1:60))
        exact_dudy_top(t) = u_top * (1.0 + 2.0 * sum((-1)^m * cos(m * π) * exp(-(m * π)^2 * nu * t) for m in 1:60))
        tau_exact_t = 1000.0 * nu * exact_dudy_top(t_end)
        for n in (20, 40), mode in (NoMDBC, SimpleMDBC, UpdatedMDBC)
            dx = 1.0 / n
            T = Float32
            constants = SimulationConstants{T}(2; dx, ρ₀ = 1000, g = 0, c₀ = 40, ν₀ = nu, α = 0, δᵩ = 0.1, CFL = 0.2)
            kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(sqrt(2.0) * dx))
            layers = ceil(Int, Float64(kernel.H) / dx) + 1
            bot_pos = [SVector((i + 0.5) * dx, (j + 0.5) * dx) for j in -layers:-1 for i in 0:n-1]
            top_pos = [SVector((i + 0.5) * dx, (j + 0.5) * dx) for j in n:n+layers-1 for i in 0:n-1]
            flu_pos = [SVector((i + 0.5) * dx, (j + 0.5) * dx) for j in 0:n-1 for i in 0:n-1]
            top_motion = MotionDetails{2,T}(Velocity = T(u_top), StartTime = -1f0, Duration = T(t_end + 2.0),
                Direction = SVector(1f0, 0f0), MoveParticles = false)
            geometry = if mode === NoMDBC
                [
                    SPHGeometry{2,T}(bot_pos; Density = 1000, GroupMarker = 1, Type = Fixed),
                    SPHGeometry{2,T}(flu_pos; Density = 1000, GroupMarker = 2, Type = Fluid),
                    SPHGeometry{2,T}(top_pos; Density = 1000, GroupMarker = 3, Type = Moving, Motion = top_motion),
                ]
            else
                bot_off = [SVector(0.0, -2.0 * p[2]) for p in bot_pos]
                top_off = [SVector(0.0, 2.0 * (1.0 - p[2])) for p in top_pos]
                [
                    SPHGeometry{2,T}(Particles = particle_struct_array(bot_pos, T(1000);
                        GhostPoints = bot_pos .+ bot_off, GhostNormals = [SVector{2,T}(o) for o in bot_off]),
                        GroupMarker = 1, Type = Fixed),
                    SPHGeometry{2,T}(flu_pos; Density = 1000, GroupMarker = 2, Type = Fluid),
                    SPHGeometry{2,T}(Particles = particle_struct_array(top_pos, T(1000);
                        GhostPoints = top_pos .+ top_off, GhostNormals = [SVector{2,T}(o) for o in top_off]),
                        GroupMarker = 3, Type = Moving, Motion = top_motion),
                ]
            end
            mktempdir() do save
                meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,mode,NoLog}(
                    SimulationName = "Couette", SaveLocation = save, SimulationTime = t_end,
                    OutputTimes = [t_end], Periodicity = PeriodicBoundary{2}(1; min = 0.0, max = 1.0),
                    GPUPeriodicImages = false, VisualizeInParaview = false, OpenLogFile = false,
                    ExportSingleVTKHDF = true, GPUUseGraph = true, GPUClipTimeStepToOutput = true,
                    GPUDoublePosition = true, GPULegacyTimestep = false,
                    GPUCellSubdivision = 1, GPUXCellSubdivision = 1)
                particles = AllocateDataStructures(geometry, meta)
                logger = SimulationLogger(save)
                result = try
                    RunSimulation(SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
                        SimKernel = kernel, SimLogger = logger, SimParticles = particles,
                        SimViscosity = Laminar(), SimDensityDiffusion = LinearDensityDiffusion(),
                        SimTimeStepping = SymplecticTimeStepping())
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                # Tile periodic images left/right across x in [0, 1) to evaluate top wall shear in Float64.
                c64 = SimulationConstants{Float64}(2; dx, ρ₀ = 1000.0, g = 0.0, c₀ = 40.0, ν₀ = nu)
                k64 = SPHKernelInstance{2,Float64}(WendlandC2(); h = sqrt(2.0) * dx)
                pos_all, vel_all, rho_all, ty_all, mk_all, primary_all = SVector{2,Float64}[], SVector{2,Float64}[], Float64[], Int[], Int[], Bool[]
                for shift in (-1.0, 0.0, 1.0)
                    for idx in eachindex(particles.Type)
                        p = SVector{2,Float64}(particles.Position[idx]) + SVector(shift, 0.0)
                        (-0.3 <= p[1] <= 1.3) || continue
                        push!(pos_all, p)
                        push!(vel_all, SVector{2,Float64}(particles.Velocity[idx]))
                        push!(rho_all, Float64(particles.Density[idx]))
                        push!(ty_all, Int(particles.Type[idx]))
                        push!(mk_all, Int(particles.GroupMarker[idx]))
                        push!(primary_all, shift == 0.0)
                    end
                end
                fcells = Dict{Tuple{Int,Int},Vector{Int}}()
                cell(x) = (floor(Int, x[1] / k64.H), floor(Int, x[2] / k64.H))
                for j in eachindex(ty_all)
                    ty_all[j] == Int(Fluid) || continue
                    push!(get!(() -> Int[], fcells, cell(pos_all[j])), j)
                end
                vforce_top = zero(SVector{2,Float64})
                for i in eachindex(ty_all)
                    primary_all[i] && mk_all[i] == 3 || continue
                    goff = SVector(0.0, 2.0 * (1.0 - pos_all[i][2]))
                    gp = pos_all[i] + goff
                    ws = mdbc_wall_state(gp, pos_all[i], vel_all[i], goff, SVector(0.0, 0.0),
                        0.0, rho_all[i], fcells, pos_all, vel_all, rho_all, k64, c64, mode())
                    ws.active || continue
                    ix, iy = cell(pos_all[i])
                    for cy in iy-1:iy+1, cx in ix-1:ix+1
                        nbrs = get(fcells, (cx, cy), nothing)
                        nbrs === nothing && continue
                        for j in nbrs
                            _, vf = mdbc_pair_loads(pos_all[i], pos_all[j], ws.viscous_velocity, vel_all[j],
                                ws.density, ws.pressure, rho_all[j], k64, c64)
                            vforce_top += vf
                        end
                    end
                end
                tau_top = -vforce_top[1] / 1.0
                fmask = findall(==(Fluid), particles.Type)
                u_err_l2 = sqrt(mean((Float64(particles.Velocity[idx][1]) - exact_u(Float64(particles.Position[idx][2]), t_end))^2 for idx in fmask)) / u_top
                push!(dynamic_rows, (; n, dx, mode = string(nameof(mode)), steps = result.performance.steps,
                    tau_exact = tau_exact_t, tau_top,
                    tau_relative_error = (tau_top - tau_exact_t) / tau_exact_t,
                    abs_tau_relative_error = abs(tau_top - tau_exact_t) / tau_exact_t,
                    velocity_profile_nrmse = u_err_l2))
            end
        end
    end
    if save_results
        mkpath(RESULTS)
        CSV.write(joinpath(RESULTS, "traction_couette_static.csv"), static_rows)
        isempty(dynamic_rows) || CSV.write(joinpath(RESULTS, "traction_couette_dynamic.csv"), dynamic_rows)
    end
    return (; static = static_rows, dynamic = dynamic_rows)
end

# ---------------------------------------------------------------------------
# SPHERIC Test 06 Moving-Square Boundary & Spatial-Resolution Study
# ---------------------------------------------------------------------------

function moving_square_variants()
    variants = NamedTuple[]
    # 1. All-Re matched boundary & shifting comparison at L/dp = 40, c0 = 40 m/s (consistent integrated velocity).
    for re in (50, 100, 150), (tag, mode, shift) in (
            ("dbc", NoMDBC, NoShifting),
            ("simple_mdbc", SimpleMDBC, NoShifting),
            ("updated_mdbc", UpdatedMDBC, NoShifting),
            ("dbc_shift", NoMDBC, PlanarShifting),
            ("simple_mdbc_shift", SimpleMDBC, PlanarShifting),
            ("updated_mdbc_shift", UpdatedMDBC, PlanarShifting))
        push!(variants, (; id = "re$(re)_n40_c40_$(tag)_integrated",
            reynolds = re, n = 40, sound_speed = 40, boundary_mode = mode,
            shifting_mode = shift, motion_kind = :integrated, p_bg = 0.0))
    end
    # 2. Motion and closed-tank positive-ambient-pressure controls at Re = 100, L/dp = 40, c0 = 40 m/s.
    push!(variants, (; id = "re100_n40_c40_updated_mdbc_hermite",
        reynolds = 100, n = 40, sound_speed = 40, boundary_mode = UpdatedMDBC,
        shifting_mode = NoShifting, motion_kind = :hermite, p_bg = 0.0))
    push!(variants, (; id = "re100_n40_c40_updated_mdbc_pbg2000_integrated",
        reynolds = 100, n = 40, sound_speed = 40, boundary_mode = UpdatedMDBC,
        shifting_mode = NoShifting, motion_kind = :integrated, p_bg = 2000.0))
    push!(variants, (; id = "re100_n40_c40_updated_mdbc_pbg2000_shift_integrated",
        reynolds = 100, n = 40, sound_speed = 40, boundary_mode = UpdatedMDBC,
        shifting_mode = PlanarShifting, motion_kind = :integrated, p_bg = 2000.0))
    # 3. Sound-speed 160 m/s boundary comparison at Re = 100 and Re = 150, L/dp = 40.
    for (tag, mode, shift) in (
            ("dbc", NoMDBC, NoShifting),
            ("simple_mdbc", SimpleMDBC, NoShifting),
            ("updated_mdbc", UpdatedMDBC, NoShifting),
            ("updated_mdbc_shift", UpdatedMDBC, PlanarShifting))
        push!(variants, (; id = "re100_n40_c160_$(tag)_integrated",
            reynolds = 100, n = 40, sound_speed = 160, boundary_mode = mode,
            shifting_mode = shift, motion_kind = :integrated, p_bg = 0.0))
    end
    for (tag, mode) in (("dbc", NoMDBC), ("simple_mdbc", SimpleMDBC))
        push!(variants, (; id = "re150_n40_c160_$(tag)_integrated",
            reynolds = 150, n = 40, sound_speed = 160, boundary_mode = mode,
            shifting_mode = NoShifting, motion_kind = :integrated, p_bg = 0.0))
    end
    # 4. Spatial resolution study across L/dp = 20, 40, 60 at Re = 100, c0 = 40 m/s (n = 40 covered above).
    for n in (20, 60), (tag, mode, shift) in (
            ("dbc", NoMDBC, NoShifting),
            ("simple_mdbc", SimpleMDBC, NoShifting),
            ("updated_mdbc", UpdatedMDBC, NoShifting),
            ("updated_mdbc_shift", UpdatedMDBC, PlanarShifting))
        push!(variants, (; id = "re100_n$(n)_c40_$(tag)_integrated",
            reynolds = 100, n, sound_speed = 40, boundary_mode = mode,
            shifting_mode = shift, motion_kind = :integrated, p_bg = 0.0))
    end
    return variants
end

function summarize_moving_square()
    records = NamedTuple[]
    for v in moving_square_variants()
        dir = joinpath(RESULTS, v.id)
        isfile(joinpath(dir, "native.toml")) || continue
        data = CSV.File(joinpath(dir, "native.csv"))
        reference = read_reference(joinpath(REFERENCE_DIR, @sprintf("Force_Re%03d.dat", v.reynolds)))
        t, p, vis = Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous)
        metrics, _ = C.drag_metrics(t, p, vis, reference)
        total, _ = C.drag_metrics(t, p + vis, vis, hcat(reference[:, 1], reference[:, 2] + reference[:, 3], reference[:, 3]))
        info = TOML.parsefile(joinpath(dir, "native.toml"))
        push!(records, (; id = v.id, reynolds = v.reynolds, n = v.n, dx = 1.0 / v.n,
            sound_speed = v.sound_speed, boundary_mode = info["boundary_mode"],
            shifting_mode = get(info, "shifting_mode", string(nameof(v.shifting_mode))),
            motion = info["motion"], p_bg = v.p_bg,
            timestep_criterion = info["timestep_criterion"],
            pressure_raw_nrmse = metrics["pressure_raw_nrmse"],
            pressure_binned_nrmse = metrics["pressure_nrmse"],
            viscous_raw_nrmse = metrics["viscous_raw_nrmse"],
            viscous_binned_nrmse = metrics["viscous_nrmse"],
            total_raw_nrmse = total["pressure_raw_nrmse"],
            total_binned_nrmse = total["pressure_nrmse"],
            pressure_noise_rms = metrics["pressure_highpass_rms"],
            viscous_noise_rms = metrics["viscous_highpass_rms"],
            pressure_late_bias = metrics["pressure_late_bias"],
            viscous_late_bias = metrics["viscous_late_bias"],
            density_relative_max = info["density_relative_max"],
            mach_max = info["mach_max"],
            minimum_clearance = info["minimum_clearance"],
            motion_error_max = info["motion_error_max"],
            velocity_error_max = info["velocity_error_max"],
            steps = info["steps"], frames = info["frames"], particles = info["particles"],
            rebuilds = info["rebuilds"], readbacks = info["readbacks"],
            solver_seconds = info["solver_seconds"]))
    end
    isempty(records) || CSV.write(joinpath(RESULTS, "summary.csv"), records)
    return records
end

function run_moving_square_mdbc_study(; filter_fn = _ -> true)
    verify_reference()
    mkpath(RESULTS); mkpath(WORK)
    C.save_info(joinpath(RESULTS, "settings.toml"), (;
        source_commit = "f992471c",
        duration = 8.0,
        output_interval = 0.02,
        cfl = 0.2,
        timestep_criterion = "pairwise_signal (GPULegacyTimestep = false)",
        precision = "Float32 physics, Float64 positions and clock",
        gpu = CUDA.name(CUDA.device()),
        julia = string(VERSION),
        cuda_jl = string(pkgversion(CUDA)),
        driver = string(CUDA.driver_version()),
        notes = "Matched moving-square boundary, shifting, sound-speed and spatial-convergence study with exact mDBC pair-load reconstruction"))
    for v in moving_square_variants()
        filter_fn(v) || continue
        dir = mkpath(joinpath(RESULTS, v.id))
        isfile(joinpath(dir, "native.toml")) && continue
        println("Running ", v.id, " ..."); flush(stdout)
        case = build_mdbc_case(; reynolds = v.reynolds, dx = 1.0 / v.n,
            sound_speed = Float64(v.sound_speed), boundary_mode = v.boundary_mode,
            shifting_mode = v.shifting_mode, motion_kind = v.motion_kind,
            legacy_timestep = false, save = joinpath(WORK, v.id))
        if v.p_bg > 0.0
            rho_init = Float32(1000.0 * (1.0 + v.p_bg / Float64(case.constants.Cb))^(1.0 / 7.0))
            case.geometry[2].Particles.Density .= rho_init
        end
        info = run_mdbc_case(case; results_dir = dir)
        info.timestep_criterion == "pairwise_signal" || error("Unexpected timestep criterion: $(info.timestep_criterion)")
        info.frames == 401 || error("Expected 401 frames, got $(info.frames)")
        summarize_moving_square()
        @printf("Completed %s in %.2f s (%d steps, min clearance %.4f m)\n",
            v.id, info.solver_seconds, info.steps, info.minimum_clearance)
        flush(stdout)
        # Clean up large ignored VTKHDF history once verified metrics/CSVs are written so laptop disk stays clean.
        vtk_path = joinpath(WORK, v.id, "MovingSquare.vtkhdf")
        isfile(vtk_path) && rm(vtk_path; force = true)
        GC.gc(true); CUDA.reclaim()
    end
    return summarize_moving_square()
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "all" : first(ARGS)
    if mode in ("all", "traction")
        MDBCValidation.run_pressure_traction_study()
        MDBCValidation.run_couette_traction_study()
    end
    if mode in ("all", "square")
        MDBCValidation.run_moving_square_mdbc_study()
    end
end
