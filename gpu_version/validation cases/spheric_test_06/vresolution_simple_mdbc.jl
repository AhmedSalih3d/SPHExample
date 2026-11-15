module VResolutionSimpleMDBC

include(joinpath(@__DIR__, "mdbc_validation.jl"))
using .MDBCValidation
using .MDBCValidation.Test06DualComparison
using .MDBCValidation.Test06DualComparison.SphericTest06
using SPHExampleGPU
using StaticArrays: SVector
using LinearAlgebra: dot, norm
using HDF5, CSV, TOML, Printf, CUDA

const C = MDBCValidation.Test06DualComparison
const VR = SPHExampleGPU.VariableResolution
const RESULTS = joinpath(@__DIR__, "results", "vresolution_simple_mdbc")
const WORK = joinpath(@__DIR__, "runs", "vresolution_simple_mdbc")

export build_vres_simple_mdbc_case, analyze_vres_history, run_vres_case,
    vres_variants, summarize_vres, run_vres_study

"""
    build_vres_simple_mdbc_case(; reynolds = 100, coarse_n = 20, ratio = 4, ...)

Build SPHERIC Test 06 with body-following `VariableResolutionConfig` (`follow = 3`)
and `SimpleMDBC` around the moving unit square `[1, 2] × [2, 3]`.
`box_margin` is the owned fine-domain half-margin around the unit square faces.
At `t = 8 s`, the square's right face reaches `x = 9.47354 m`, leaving `0.52646 m`
to the right tank wall `x = 10 m`; choosing `box_margin` so `box_margin + H_fine < 0.52`
keeps the entire moving fine domain + outer buffer strictly inside `(0, 10) × (0, 5)` m
throughout `t ∈ [0, 8] s` while satisfying the `2 H_coarse + H_fine` body clearance.
"""
function build_vres_simple_mdbc_case(; reynolds::Int = 100, coarse_n::Int = 25, ratio::Int = 4,
        box_margin::Union{Nothing,Float64} = nothing,
        wake_margin::Union{Nothing,Float64} = nothing,
        side_margin::Union{Nothing,Float64} = nothing,
        front_margin::Union{Nothing,Float64} = nothing,
        buffer_h::Union{Nothing,Float64} = nothing,
        overlap_h::Union{Nothing,Float64} = nothing,
        background_pressure::Float64 = 0.0,
        shell_only::Bool = true, max_steps_per_sync::Int = 128,
        duration::Float64 = 8.0, interval::Float64 = 0.02, sound_speed::Float64 = 40.0,
        cfl::Float64 = 0.2, boundary_mode::Type{<:MDBCMode} = SimpleMDBC,
        shifting_mode::Type{<:ShiftingMode} = NoShifting,
        motion_kind::Symbol = :integrated, legacy_timestep::Bool = false,
        save = joinpath(WORK, "case"), float_type = Float32, graph::Bool = true)
    reynolds in (50, 100, 150) || throw(ArgumentError("reynolds must be 50, 100 or 150"))
    boundary_mode in (NoMDBC, SimpleMDBC) || throw(ArgumentError("VariableResolution supports NoMDBC or SimpleMDBC"))
    shifting_mode in (NoShifting, PlanarShifting) || throw(ArgumentError("VariableResolution supports NoShifting or PlanarShifting"))
    ratio in (2, 3, 4) || throw(ArgumentError("ratio must be 2, 3 or 4"))
    motion_kind in (:integrated, :hermite) || throw(ArgumentError("motion_kind must be :integrated or :hermite"))

    dx_c = 1.0 / coarse_n
    fine_n = coarse_n * ratio
    dx_f = 1.0 / fine_n
    nx_c, ny_c = round.(Int, (10.0 / dx_c, 5.0 / dx_c))
    all(isapprox(n * dx_c, L; atol = 1e-10) for (n, L) in ((nx_c, 10.0), (ny_c, 5.0))) ||
        throw(ArgumentError("coarse_n must divide tank dimensions"))

    T = float_type
    coarse_constants = SimulationConstants{T}(2; dx = dx_c, ρ₀ = 1000, g = 0, c₀ = sound_speed,
        ν₀ = 1 / reynolds, α = 0, δᵩ = 0.1, CFL = cfl, BackgroundPressure = T(background_pressure))
    coarse_kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(sqrt(2.0) * dx_c))
    fine_constants = SimulationConstants{T}(2; dx = dx_f, ρ₀ = 1000, g = 0, c₀ = sound_speed,
        ν₀ = 1 / reynolds, α = 0, δᵩ = 0.1, CFL = cfl, BackgroundPressure = T(background_pressure))
    fine_kernel = SPHKernelInstance{2,T}(WendlandC2();
        h = T(Float64(coarse_kernel.h) / ratio),
        η² = T(Float64(coarse_kernel.η²) / ratio^2),
        αD = T(Float64(coarse_kernel.αD) * ratio^2))

    h_c = Float64(coarse_kernel.h)
    h_f = Float64(fine_kernel.h)
    H_c = Float64(coarse_kernel.H)
    H_f = Float64(fine_kernel.H)
    bh = buffer_h !== nothing ? Float64(buffer_h) : (coarse_n * ratio <= 40 ? 2.85 : 3.2)
    oh = overlap_h !== nothing ? Float64(overlap_h) : (coarse_n * ratio <= 40 ? 2.0 : (coarse_n <= 20 ? 2.5 : 3.0))
    min_clear = oh * h_f + (bh + 2.0) * h_c - 0.5 * dx_f + 0.008
    max_front = 0.5264 - bh * h_f - 0.008
    m_front = front_margin !== nothing ? Float64(front_margin) :
        (box_margin !== nothing ? Float64(box_margin) : min(max_front, max(min_clear + 0.004, 0.42)))
    m_wake = wake_margin !== nothing ? Float64(wake_margin) :
        (box_margin !== nothing ? Float64(box_margin) : max(m_front, min(0.70, 1.0 - bh * h_f - 0.12)))
    m_side = side_margin !== nothing ? Float64(side_margin) :
        (box_margin !== nothing ? Float64(box_margin) : max(m_front, 0.55))

    all(>=(min_clear - 0.005), (m_front, m_wake, m_side)) &&
        m_front <= max_front + 0.005 && m_wake < 1.0 - bh * h_f - 0.01 && m_side < 2.0 - bh * h_f - 0.01 ||
        throw(ArgumentError("margins (wake=$m_wake, side=$m_side, front=$m_front) violate body clearance ($min_clear) or tank walls"))

    center0 = SVector(1.5, 2.5)
    region = VariableResolutionBox((1.0 - m_wake, 2.0 - m_side), (2.0 + m_front, 3.0 + m_side);
        ratio, buffer_h = bh, overlap_h = oh)
    limits = VR.resolution_limits(region, h_c, h_f)

    # 1. Coarse domain: outer tank walls + coarse fluid outside coarse_inner
    layers_c = ceil(Int, H_c / dx_c) + 1
    walls_c = SVector{2,Float64}[]
    fluid_c = SVector{2,Float64}[]
    for j in -layers_c:ny_c+layers_c-1, i in -layers_c:nx_c+layers_c-1
        p = SVector((i + 0.5) * dx_c, (j + 0.5) * dx_c)
        if !(0 <= i < nx_c && 0 <= j < ny_c)
            push!(walls_c, p)
        elseif !(1.0 <= p[1] <= 2.0 && 2.0 <= p[2] <= 3.0) &&
                !VR.inside_box(p, limits.coarse_inner_lo, limits.coarse_inner_hi)
            push!(fluid_c, p)
        end
    end

    # 2. Fine domain: fine fluid inside fine_outer (outside [1,2]×[2,3]) + fine square
    layers_f = ceil(Int, H_f / dx_f) + 1
    i_lo = max(0, floor(Int, limits.fine_outer_lo[1] / dx_f) - 1)
    i_hi = min(10 * fine_n - 1, ceil(Int, limits.fine_outer_hi[1] / dx_f) + 1)
    j_lo = max(0, floor(Int, limits.fine_outer_lo[2] / dx_f) - 1)
    j_hi = min(5 * fine_n - 1, ceil(Int, limits.fine_outer_hi[2] / dx_f) + 1)
    fluid_f = SVector{2,Float64}[]
    body_f = SVector{2,Float64}[]
    for j in j_lo:j_hi, i in i_lo:i_hi
        p = SVector((i + 0.5) * dx_f, (j + 0.5) * dx_f)
        if 1.0 < p[1] < 2.0 && 2.0 < p[2] < 3.0
            local_p = p - center0
            if !shell_only || max(abs(local_p[1]), abs(local_p[2])) >= 0.5 - layers_f * dx_f - 1e-12
                push!(body_f, p)
            end
        elseif VR.inside_box(p, limits.fine_outer_lo, limits.fine_outer_hi)
            push!(fluid_f, p)
        end
    end

    coarse_geometry = if boundary_mode === NoMDBC
        [
            SPHGeometry{2,T}(walls_c; Density = coarse_constants.ρ₀, GroupMarker = 1, Type = Fixed),
            SPHGeometry{2,T}(fluid_c; Density = coarse_constants.ρ₀, GroupMarker = 2, Type = Fluid),
        ]
    else
        w_off = [tank_ghost_offset(p) for p in walls_c]
        [
            SPHGeometry{2,T}(Particles = particle_struct_array(walls_c, coarse_constants.ρ₀;
                GhostPoints = walls_c .+ w_off, GhostNormals = [SVector{2,T}(o) for o in w_off]),
                GroupMarker = 1, Type = Fixed),
            SPHGeometry{2,T}(fluid_c; Density = coarse_constants.ρ₀, GroupMarker = 2, Type = Fluid),
        ]
    end

    fine_geometry = if boundary_mode === NoMDBC
        [
            SPHGeometry{2,T}(fluid_f; Density = fine_constants.ρ₀, GroupMarker = 2, Type = Fluid),
            SPHGeometry{2,T}(body_f; Density = fine_constants.ρ₀, GroupMarker = 3, Type = Moving),
        ]
    else
        b_off = [square_ghost_offset(p - center0) for p in body_f]
        [
            SPHGeometry{2,T}(fluid_f; Density = fine_constants.ρ₀, GroupMarker = 2, Type = Fluid),
            SPHGeometry{2,T}(Particles = particle_struct_array(body_f, fine_constants.ρ₀;
                GhostPoints = body_f .+ b_off, GhostNormals = [SVector{2,T}(o) for o in b_off]),
                GroupMarker = 3, Type = Moving),
        ]
    end

    vres_config = VariableResolutionConfig(region; fine_geometry, fine_constants, fine_kernel, follow = 3)
    times = collect(interval:interval:duration)
    isempty(times) || isapprox(last(times), duration; atol = 1e-12, rtol = 0) || push!(times, duration)
    times[end] = duration

    metadata = SimulationMetaData{2,T,shifting_mode,NoKernelOutput,boundary_mode,StoreLog}(
        SimulationName = "MovingSquare", SaveLocation = abspath(save), SimulationTime = duration,
        OutputTimes = times, VariableResolution = vres_config,
        VisualizeInParaview = false, OpenLogFile = false,
        ExportSingleVTKHDF = true,
        OutputVariables = ["Density", "Velocity", "Type", "GroupMarker", "Resolution", "Owned"],
        GPUUseGraph = graph, GPUMaxStepsPerSync = max_steps_per_sync,
        GPUClipTimeStepToOutput = true, GPUDoublePosition = true,
        GPULegacyTimestep = legacy_timestep,
        GPUCellSubdivision = 1, GPUXCellSubdivision = 1, GPURadialInteractions = false,
        GPUInteractionFastMath = false, GPUPreloadInteractionTypes = false,
        GPUBoundaryForces = false, GPUOutputBufferBytes = 16 * 2^20)

    trajectory = motion_kind === :integrated ? C.integrated_trajectory() : body_trajectory()
    motion = PrescribedMotion(3; translation = trajectory)
    return (; coarse_constants, coarse_kernel, fine_constants, fine_kernel,
        coarse_geometry, fine_geometry, vres_config, metadata, motion,
        reynolds, coarse_n, fine_n, ratio, dx_coarse = dx_c, dx_fine = dx_f,
        margin = m_front, wake_margin = m_wake, side_margin = m_side, front_margin = m_front,
        buffer_h = bh, overlap_h = oh, background_pressure,
        duration, interval, boundary_mode, shifting_mode, motion_kind, shell_only,
        initial_counts = (; coarse_fixed = length(walls_c), coarse_fluid = length(fluid_c),
            fine_fluid = length(fluid_f), fine_square = length(body_f)))
end

"""Reconstruct square/fluid pair loads on the fine domain (`Resolution == 1`) from VTKHDF."""
function analyze_vres_history(path, case; coefficient_path, diagnostics_path)
    c64 = SimulationConstants{Float64}(2; dx = case.dx_fine, ρ₀ = case.fine_constants.ρ₀,
        c₀ = case.fine_constants.c₀, g = 0.0, ν₀ = case.fine_constants.ν₀,
        BackgroundPressure = Float64(case.fine_constants.BackgroundPressure))
    k64 = SPHKernelInstance{2,Float64}(WendlandC2(); h = Float64(case.fine_kernel.h),
        αD = Float64(case.fine_kernel.αD), η² = Float64(case.fine_kernel.η²))
    mode = case.boundary_mode()
    trajectory = case.motion.translation
    rows, checks = NamedTuple[], NamedTuple[]
    h5open(path, "r") do file
        root = haskey(file, "VTKHDF/Particles") ? file["VTKHDF/Particles"] : file["VTKHDF"]
        times = read(root["Steps/Values"])
        offsets = read(root["Steps/PointOffsets"])
        counts = read(root["NumberOfPoints"])
        for frame in eachindex(times)
            cols = offsets[frame]+1:offsets[frame]+counts[frame]
            pos_all = [SVector{2,Float64}(x[1], x[2]) for x in eachcol(root["Points"][:, cols])]
            vel_all = [SVector{2,Float64}(v[1], v[2]) for v in eachcol(root["PointData/Velocity"][:, cols])]
            rho_all = Float64.(root["PointData/Density"][cols])
            ty_all = Int.(root["PointData/Type"][cols])
            mk_all = Int.(root["PointData/GroupMarker"][cols])
            res_all = Int.(root["PointData/Resolution"][cols])
            all(isfinite, rho_all) && all(>(0), rho_all) && all(x -> all(isfinite, x), pos_all) &&
                all(v -> all(isfinite, v), vel_all) || error("Nonfinite state at t=$(times[frame])")
            count(==(Int(Frozen)), ty_all) == 0 || error("Frozen particles exported at t=$(times[frame])")

            fine_idx = findall(==(1), res_all)
            pos_f = pos_all[fine_idx]
            vel_f = vel_all[fine_idx]
            rho_f = rho_all[fine_idx]
            ty_f = ty_all[fine_idx]
            mk_f = mk_all[fine_idx]

            square_f = findall(==(3), mk_f)
            length(square_f) == case.initial_counts.fine_square || error("Square particle loss at t=$(times[frame])")
            fluid_all = findall(==(Int(Fluid)), ty_all)
            fluid_f = findall(==(Int(Fluid)), ty_f)

            displacement, expected_velocity, expected_acc = trajectory_state(trajectory, times[frame])
            center = SVector(1.5, 2.5) + displacement
            motion_error = norm(sum(pos_f[square_f]) / length(square_f) - center)
            velocity_error = maximum(v -> norm(v - expected_velocity), vel_f[square_f])
            minimum_clearance = minimum(fluid_all) do i
                x, y = pos_all[i]
                square_clearance = max(abs(x - center[1]), abs(y - center[2])) - 0.5
                min(x, 10.0 - x, y, 5.0 - y, square_clearance)
            end
            loads = mdbc_frame_loads(pos_f, vel_f, rho_f, ty_f, mk_f, k64, c64;
                boundary_mode = mode, center, wall_acceleration = expected_acc)
            push!(rows, (; time = times[frame], cd_pressure = loads.cd_pressure, cd_viscous = loads.cd_viscous,
                cy_pressure = loads.cy_pressure, cy_viscous = loads.cy_viscous))
            push!(checks, (; time = times[frame], motion_error, velocity_error,
                density_relative_max = maximum(abs, rho_all[fluid_all] ./ c64.ρ₀ .- 1.0),
                max_speed = maximum(norm, vel_all[fluid_all]), minimum_clearance,
                coarse_fluid = count(i -> ty_all[i] == Int(Fluid) && res_all[i] == 0, eachindex(ty_all)),
                fine_fluid = length(fluid_f), square_particles = length(square_f)))
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

function run_vres_case(case; results_dir, scheme = SymplecticTimeStepping())
    verify_reference()
    CUDA.functional() || error("An NVIDIA CUDA GPU is required")
    save = case.metadata.SaveLocation
    mkpath(save)
    isfile(joinpath(save, "MovingSquare.vtkhdf")) && error("Refusing to overwrite an existing run: $save")
    mkpath(results_dir)
    particles = AllocateDataStructures(case.coarse_geometry, case.metadata)
    logger = SimulationLogger(save)
    result = nothing
    wall = open(joinpath(save, "console.log"), "w") do io
        redirect_stdout(io) do
            @elapsed try
                result = RunSimulation(SimGeometry = case.coarse_geometry, SimMetaData = case.metadata,
                    SimConstants = case.coarse_constants, SimKernel = case.coarse_kernel, SimLogger = logger,
                    SimParticles = particles, SimViscosity = Laminar(),
                    SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
                    SimTimeStepping = scheme, SimMotion = case.motion)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
        end
    end
    result.final_state.complete || error("Incomplete variable-resolution run")
    hist = CSV.File(joinpath(save, "MovingSquare_History.csv"))
    max_excluded = max(maximum(hist.coarse_excluded), maximum(hist.fine_excluded))
    max_nonfinite = max(maximum(hist.coarse_nonfinite), maximum(hist.fine_nonfinite))
    max_nonfinite == 0 || error("Nonfinite variable-resolution event")
    mass_initial = Float64(first(hist.represented_fluid_mass))
    mass_drift_rel = maximum(abs.(Float64.(hist.represented_fluid_mass) .- mass_initial)) / mass_initial

    analysis = analyze_vres_history(joinpath(save, "MovingSquare.vtkhdf"), case;
        coefficient_path = joinpath(results_dir, "native.dat"),
        diagnostics_path = joinpath(results_dir, "native_state.csv"))
    cp(joinpath(save, "MovingSquare_History.csv"), joinpath(results_dir, "resolution_history.csv"); force = true)

    info = (; reynolds = case.reynolds, coarse_n = case.coarse_n, fine_n = case.fine_n, ratio = case.ratio,
        dx_coarse = case.dx_coarse, dx_fine = case.dx_fine, box_margin = case.margin,
        wake_margin = case.wake_margin, side_margin = case.side_margin, front_margin = case.front_margin,
        buffer_h = case.buffer_h, overlap_h = case.overlap_h,
        background_pressure = case.background_pressure,
        sound_speed = Float64(case.coarse_constants.c₀), cfl = Float64(case.coarse_constants.CFL),
        boundary_mode = string(nameof(case.boundary_mode)),
        shifting_mode = string(nameof(case.shifting_mode)),
        scheme = string(nameof(typeof(scheme))),
        motion = string(case.motion_kind), shell_only = case.shell_only,
        steps = result.performance.steps, frames = result.diagnostics.output_frames,
        captured_graphs = result.diagnostics.captured_graphs,
        readbacks = result.diagnostics.step_state_readbacks,
        coarse_created = Int(last(hist.coarse_created)), fine_created = Int(last(hist.fine_created)),
        coarse_retired = Int(last(hist.coarse_retired)), fine_retired = Int(last(hist.fine_retired)),
        excluded_density = Int(max_excluded), mass_drift_relative_max = mass_drift_rel,
        wall_seconds = wall, solver_seconds = result.performance.wall_time_seconds,
        motion_error_max = maximum(x -> x.motion_error, analysis.checks),
        velocity_error_max = maximum(x -> x.velocity_error, analysis.checks),
        density_relative_max = maximum(x -> x.density_relative_max, analysis.checks),
        mach_max = maximum(x -> x.max_speed, analysis.checks) / Float64(case.coarse_constants.c₀),
        minimum_clearance = minimum(x -> x.minimum_clearance, analysis.checks),
        gpu = CUDA.name(CUDA.device()), julia = string(VERSION),
        cuda_jl = string(pkgversion(CUDA)), driver = string(CUDA.driver_version()))
    C.save_info(joinpath(results_dir, "native.toml"), info)
    return info
end

function vres_variants()
    return [
        # 1. Interface baseline checks: L/dp = 20 -> 40 at Re = 50 and Re = 100 (c0 = 40 m/s)
        (; id = "re50_c20_r2_c40_simple_mdbc", reynolds = 50, coarse_n = 20, ratio = 2, sound_speed = 40.0, boundary_mode = SimpleMDBC),
        (; id = "re100_c20_r2_c40_simple_mdbc", reynolds = 100, coarse_n = 20, ratio = 2, sound_speed = 40.0, boundary_mode = SimpleMDBC),
        # 2. Boundary-layer spatial refinement at c0 = 40 m/s: L/dp = 20 -> 80 and 30 -> 120 (dx_fine = 12.5 mm and 8.33 mm)
        (; id = "re100_c20_r4_c40_simple_mdbc", reynolds = 100, coarse_n = 20, ratio = 4, sound_speed = 40.0, boundary_mode = SimpleMDBC),
        (; id = "re150_c20_r4_c40_simple_mdbc", reynolds = 150, coarse_n = 20, ratio = 4, sound_speed = 40.0, boundary_mode = SimpleMDBC),
        (; id = "re150_c30_r4_c40_simple_mdbc", reynolds = 150, coarse_n = 30, ratio = 4, sound_speed = 40.0, boundary_mode = SimpleMDBC),
        # 3. Stiff acoustic + boundary-layer-resolved hero cases at c0 = 160 m/s:
        #    Re = 100 (20 -> 80) and Re = 150 (20 -> 80 and 25 -> 100, dx_fine = 12.5 mm and 10.0 mm)
        (; id = "re100_c20_r4_c160_simple_mdbc", reynolds = 100, coarse_n = 20, ratio = 4, sound_speed = 160.0, boundary_mode = SimpleMDBC),
        (; id = "re150_c20_r4_c160_simple_mdbc", reynolds = 150, coarse_n = 20, ratio = 4, sound_speed = 160.0, boundary_mode = SimpleMDBC),
        (; id = "re150_c25_r4_c160_simple_mdbc", reynolds = 150, coarse_n = 25, ratio = 4, sound_speed = 160.0, boundary_mode = SimpleMDBC),
    ]
end

function summarize_vres()
    records = NamedTuple[]
    for v in vres_variants()
        dir = joinpath(RESULTS, v.id)
        isfile(joinpath(dir, "native.toml")) || continue
        data = CSV.File(joinpath(dir, "native.csv"))
        reference = read_reference(joinpath(REFERENCE_DIR, @sprintf("Force_Re%03d.dat", v.reynolds)))
        t, p, vis = Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous)
        metrics, _ = C.drag_metrics(t, p, vis, reference)
        total, _ = C.drag_metrics(t, p + vis, vis, hcat(reference[:, 1], reference[:, 2] + reference[:, 3], reference[:, 3]))
        info = TOML.parsefile(joinpath(dir, "native.toml"))
        push!(records, (; id = v.id, reynolds = v.reynolds,
            coarse_n = Int(info["coarse_n"]), fine_n = Int(info["fine_n"]), ratio = Int(info["ratio"]),
            dx_coarse = Float64(info["dx_coarse"]), dx_fine = Float64(info["dx_fine"]),
            box_margin = Float64(info["box_margin"]),
            sound_speed = Int(round(Float64(info["sound_speed"]))),
            boundary_mode = info["boundary_mode"], motion = info["motion"],
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
            density_relative_max = Float64(info["density_relative_max"]),
            mach_max = Float64(info["mach_max"]),
            minimum_clearance = Float64(info["minimum_clearance"]),
            motion_error_max = Float64(info["motion_error_max"]),
            velocity_error_max = Float64(info["velocity_error_max"]),
            excluded_density = Int(info["excluded_density"]),
            mass_drift_relative_max = Float64(info["mass_drift_relative_max"]),
            coarse_created = Int(info["coarse_created"]),
            fine_created = Int(info["fine_created"]),
            steps = Int(info["steps"]), frames = Int(info["frames"]),
            captured_graphs = Int(info["captured_graphs"]),
            readbacks = Int(info["readbacks"]),
            solver_seconds = Float64(info["solver_seconds"])))
    end
    isempty(records) || CSV.write(joinpath(RESULTS, "summary.csv"), records)
    return records
end

function run_vres_study(; filter_fn = _ -> true)
    verify_reference()
    mkpath(RESULTS); mkpath(WORK)
    C.save_info(joinpath(RESULTS, "settings.toml"), (;
        source_commit = "c634d9d",
        duration = 8.0,
        output_interval = 0.02,
        cfl = 0.2,
        timestep_criterion = "pairwise_signal (GPULegacyTimestep = false)",
        motion = "Consistent integrated official velocity trajectory",
        boundary = "SimpleMDBC with body-following VariableResolutionConfig(follow = 3)",
        precision = "Float32 physics, Float64 positions and clock",
        gpu = CUDA.name(CUDA.device()),
        julia = string(VERSION),
        cuda_jl = string(pkgversion(CUDA)),
        driver = string(CUDA.driver_version())))
    for v in vres_variants()
        filter_fn(v) || continue
        dir = mkpath(joinpath(RESULTS, v.id))
        isfile(joinpath(dir, "native.toml")) && continue
        println("Running ", v.id, " ..."); flush(stdout)
        case = build_vres_simple_mdbc_case(; reynolds = v.reynolds, coarse_n = v.coarse_n,
            ratio = v.ratio, sound_speed = v.sound_speed, boundary_mode = v.boundary_mode,
            motion_kind = :integrated, legacy_timestep = false, save = joinpath(WORK, v.id))
        info = run_vres_case(case; results_dir = dir)
        info.frames == 401 || error("Expected 401 frames, got $(info.frames)")
        summarize_vres()
        @printf("Completed %s in %.2f s (%d steps, min clearance %.5f m, excluded=%d, mass_drift=%.2e)\n",
            v.id, info.solver_seconds, info.steps, info.minimum_clearance, info.excluded_density, info.mass_drift_relative_max)
        flush(stdout)
        vtk_path = joinpath(WORK, v.id, "MovingSquare.vtkhdf")
        isfile(vtk_path) && rm(vtk_path; force = true)
        GC.gc(true); CUDA.reclaim()
    end
    return summarize_vres()
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    VResolutionSimpleMDBC.run_vres_study()
end
