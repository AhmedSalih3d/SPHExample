module RegularizedMDBCExperiment

using SPHExampleGPU, CSV, HDF5, StaticArrays, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "mdbc_validation.jl"))

const M = MDBCValidation
const GK = SPHExampleGPU.GPUKernels
const RUNS = joinpath(@__DIR__, "runs", "regularized_mdbc_experiment")
const FROZEN_TIMES = (0.76, 0.78, 0.80, 0.82, 0.86)
const BOUNDARY_MODES = Dict("dbc" => NoMDBC, "simple" => SimpleMDBC,
    "regularized" => RegularizedMDBC, "impedance" => DBCPlusImpedance)

function experiment_case(mode, dir; n = 80, cfl = 0.2, duration = 0.9, shifting_mode = NoShifting)
    case = M.build_mdbc_case(; dx = 1 / n, duration, interval = 0.02,
        sound_speed = 160.0, cfl, kernel_h_factor = 1.2sqrt(2), boundary_mode = mode,
        legacy_timestep = true, motion_kind = :integrated, shifting_mode, save = joinpath(dir, "state"))
    motion = PrescribedMotion(3; translation = M.C.SphericTest06.body_trajectory(motion_kind = :integrated))
    push!(case.metadata.OutputVariables, "Pressure")
    return merge(case, (; motion))
end

function run_experiment(command, study)
    mode = BOUNDARY_MODES[command]
    n = study == "coarse" ? 40 : 80
    cfl = startswith(study, "half-step") ? 0.1 : 0.2
    shifting_mode = startswith(study, "half-step-planar") ? PlanarShifting : NoShifting
    diffusion = if study == "half-step-planar-zero"
        ZeroDensityDiffusion()
    elseif study == "half-step-planar-zero-gravity"
        ZeroGravityLinearDensityDiffusion()
    else
        LinearDensityDiffusion()
    end
    duration = study == "full" ? 8.0 : 0.9
    name = command == "simple" ? "control" : command
    suffix = study == "startup" ? "" : "_$study"
    dir = joinpath(RUNS, "$(name)_n$n$suffix")
    case = experiment_case(mode, dir; n, cfl, duration, shifting_mode)
    info = M.run_mdbc_case(case; results_dir = dir, diffusion,
        compare_stored = mode !== DBCPlusImpedance)
    history = CSV.File(joinpath(dir, "native.csv"))
    length(history) == info.frames == round(Int, duration / 0.02) + 1 &&
        first(history.time) == 0.0 && last(history.time) == duration || error("Incomplete output time grid")
    println("Completed $(info.boundary_mode): $(info.steps) steps, $(info.frames) frames. Results: $dir")
end

function nearby_fluid(point, cells, position, kernel)
    ix, iy = M.mdbc_cell(point, kernel)
    return [j for cy in iy-1:iy+1 for cx in ix-1:ix+1 for j in get(cells, (cx, cy), Int[])
        if sum(abs2, position[j] - point) < kernel.H²]
end

function frozen_wall_response(wall, ghost, positions, densities, stored, kernel, constants)
    offsets = positions .- Ref(ghost)
    columns = map(offsets, densities) do x, rho
        q = norm(x) * kernel.h⁻¹
        weight, grad = Wᵢⱼ(kernel, q), ∇Wᵢⱼ(kernel, q, -x)
        return (constants.m₀ / rho) * SVector(weight, grad...)
    end
    matrix = sum(col * SVector(1.0, x...)' for (col, x) in zip(columns, offsets))
    offset = wall - ghost
    is_mls = abs(det(matrix)) >= 1e-3
    shape = is_mls ? matrix' \ SVector(1.0, offset...) : SVector(inv(matrix[1, 1]), 0.0, 0.0)
    # Keep volumes and the sample geometry fixed, as in the earlier audit.
    # Probe both signs of the worst direction for SimpleMDBC, because the
    # regularized map is nonlinear. This is a directional sensitivity.
    epsilon = 0.001
    perturbation = [epsilon * sign(dot(shape, col)) for col in columns]
    function wall_densities(values)
        rhs = sum(col * rho for (col, rho) in zip(columns, values))
        stats = GK.mdbc_reconstruction_rhs(Val(true), Val(3), constants.ρ₀)
        for (rho, col, x) in zip(values, columns, offsets)
            stats = GK.accumulate_regularized_mdbc(stats, rho, col, x, kernel.h⁻¹)
        end
        return SVector(
            GK.mdbc_wall_density(matrix, rhs, offset, stored, constants.ρ₀, kernel.h),
            GK.mdbc_wall_density(matrix, stats, offset, stored, constants.ρ₀, kernel.h))
    end
    rho = wall_densities(densities)
    plus, minus = wall_densities(densities + perturbation), wall_densities(densities - perturbation)
    all(isfinite, rho) && all(>(0), rho) && all(isfinite, plus) && all(isfinite, minus) ||
        error("Invalid frozen wall density at $wall")
    gain = max.(abs.(plus - rho), abs.(minus - rho)) / epsilon
    low, high = extrema(densities)
    overshoot = max.(0.0, low .- rho, rho .- high)
    return (; is_mls, rho, gain, overshoot)
end

function audit_frozen_startup()
    control = joinpath(RUNS, "control_n80")
    case = experiment_case(SimpleMDBC, control)
    constants = SimulationConstants{Float64}(2; dx = case.dx, ρ₀ = case.constants.ρ₀,
        c₀ = case.constants.c₀, g = 0, ν₀ = case.constants.ν₀)
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = Float64(case.kernel.h),
        αD = Float64(case.kernel.αD), η² = Float64(case.kernel.η²))
    saved = CSV.File(joinpath(control, "native.csv"))
    forces, neighborhoods = NamedTuple[], NamedTuple[]
    h5open(joinpath(control, "state", "MovingSquare.vtkhdf"), "r") do file
        root = haskey(file, "VTKHDF/Particles") ? file["VTKHDF/Particles"] : file["VTKHDF"]
        times = read(root["Steps/Values"])
        offsets = read(root["Steps/PointOffsets"])
        counts = read(root["NumberOfPoints"])
        for time in FROZEN_TIMES
            frame = findfirst(==(time), times)
            frame === nothing && error("Missing frozen output at t=$time")
            columns = offsets[frame]+1:offsets[frame]+counts[frame]
            (; position, velocity, density, types, markers) = M.read_mdbc_frame(root, columns)
            displacement, _, wall_acceleration = trajectory_state(case.motion.translation, time)
            center = SVector(1.5, 2.5) + displacement
            loads(mode) = M.mdbc_frame_loads(position, velocity, density, types, markers, kernel, constants;
                boundary_mode = mode, center, wall_acceleration)
            stored, simple, regularized = loads(NoMDBC()), loads(SimpleMDBC()), loads(RegularizedMDBC())
            isapprox(simple.cd_pressure, saved.cd_pressure[frame]; atol = 2e-8) ||
                error("SimpleMDBC frozen pressure load mismatch at t=$time")
            isapprox(simple.cd_viscous, saved.cd_viscous[frame]; atol = 2e-8) ||
                error("SimpleMDBC frozen viscous load mismatch at t=$time")
            push!(forces, (; time, stored_pressure_cd = stored.cd_pressure, simple_pressure_cd = simple.cd_pressure,
                regularized_pressure_cd = regularized.cd_pressure, stored_viscous_cd = stored.cd_viscous,
                simple_viscous_cd = simple.cd_viscous, regularized_viscous_cd = regularized.cd_viscous))

            cells = M.mdbc_fluid_cells(position, types, kernel)
            by_face = Dict(face => NamedTuple[] for face in ("front", "rear", "side"))
            for i in eachindex(markers)
                markers[i] == 3 && types[i] == Int(Moving) || continue
                wall = position[i]
                isempty(nearby_fluid(wall, cells, position, kernel)) && continue
                local_pos = wall - center
                ghost = wall + M.square_ghost_offset(local_pos)
                neighbors = nearby_fluid(ghost, cells, position, kernel)
                isempty(neighbors) && error("Missing ghost support at t=$time, wall $wall")
                face = abs(local_pos[1]) < abs(local_pos[2]) ? "side" : (local_pos[1] > 0 ? "front" : "rear")
                stat = frozen_wall_response(wall, ghost, position[neighbors], density[neighbors], density[i], kernel, constants)
                push!(by_face[face], stat)
            end
            for face in ("front", "rear", "side")
                selected = by_face[face]
                push!(neighborhoods, (; time, face, owners = length(selected),
                    mls_owners = count(r -> r.is_mls, selected),
                    simple_overshoot_max = maximum(r -> r.overshoot[1], selected),
                    regularized_overshoot_max = maximum(r -> r.overshoot[2], selected),
                    simple_sensitivity_median = median(r.gain[1] for r in selected),
                    regularized_sensitivity_median = median(r.gain[2] for r in selected),
                    simple_sensitivity_max = maximum(r -> r.gain[1], selected),
                    regularized_sensitivity_max = maximum(r -> r.gain[2], selected)))
            end
        end
    end
    CSV.write(joinpath(RUNS, "frozen_forces.csv"), forces)
    CSV.write(joinpath(RUNS, "frozen_neighborhoods.csv"), neighborhoods)
    # Reuses the saved states if the control was produced without stored-wall diagnostics.
    if !hasproperty(CSV.File(joinpath(control, "native_state.csv")), :cd_pressure_stored)
        M.analyze_mdbc_history(joinpath(control, "state", "MovingSquare.vtkhdf"), case;
            coefficient_path = joinpath(control, "reconstructed.dat"),
            diagnostics_path = joinpath(control, "stored_wall_state.csv"), compare_stored = true)
    end
    println("Wrote frozen force and directional sensitivity comparisons to $RUNS")
end

function main(args)
    args == ["frozen"] && return audit_frozen_startup()
    length(args) in (1, 2) && haskey(BOUNDARY_MODES, first(args)) ||
        error("Usage: regularized_mdbc_experiment.jl dbc|simple|regularized|impedance [startup|full|half-step|half-step-planar|half-step-planar-zero|half-step-planar-zero-gravity|coarse], or frozen")
    study = length(args) == 2 ? args[2] : "startup"
    study in ("startup", "full", "half-step", "half-step-planar", "half-step-planar-zero", "half-step-planar-zero-gravity", "coarse") ||
        error("Unknown experiment: $study")
    return run_experiment(first(args), study)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    RegularizedMDBCExperiment.main(ARGS)
end
