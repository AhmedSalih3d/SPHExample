include("case.jl")
using .SphericTest06
using SPHExampleGPU, StaticArrays, LinearAlgebra, HDF5, CSV, Statistics

const RUNS = joinpath(@__DIR__, "runs")
const REFINEMENT = "matched_timestep_refinement"
const OUT = joinpath(@__DIR__, "results", "fine_spike_replay")

function face_loads(root, columns, case, kernel, constants, time)
    types = root["PointData/Type"][columns]
    markers = root["PointData/GroupMarker"][columns]
    position = root["Points"][:, columns]
    velocity = root["PointData/Velocity"][:, columns]
    density = Float64.(root["PointData/Density"][columns])
    displacement, wall_velocity, _ = trajectory_state(case.motion.translation, time)
    center = case.body_center + displacement
    xy(a, i) = SVector{2,Float64}(a[1, i], a[2, i])
    cell(x) = (floor(Int, x[1] / kernel.H), floor(Int, x[2] / kernel.H))
    cells = Dict{Tuple{Int,Int},Vector{Int}}()
    for j in eachindex(types)
        types[j] == Int(Fluid) || continue
        x = xy(position, j)
        maximum(abs, x - center) <= 0.5 + kernel.H + 1e-10 || continue
        push!(get!(() -> Int[], cells, cell(x)), j)
    end
    front = rear = side = 0.0
    fill = Float64[]
    max_approach = 0.0
    max_rear_pressure = -Inf
    for i in eachindex(types)
        markers[i] == 3 && types[i] == Int(Moving) || continue
        wall = xy(position, i)
        local_pos = wall - center
        face_weight = isapprox(abs(local_pos[1]), abs(local_pos[2]); atol = 1e-10, rtol = 0) ?
            0.5 : Float64(abs(local_pos[1]) > abs(local_pos[2]))
        central_rear = abs(local_pos[1] + 0.5 - case.dx / 2) < 1e-9 && abs(local_pos[2]) <= 0.125
        ix, iy = cell(wall)
        fx = support = 0.0
        for z in iy-1:iy+1, x in ix-1:ix+1
            neighbors = get(cells, (x,z), nothing)
            neighbors === nothing && continue
            for j in neighbors
                water = xy(position, j)
                d2 = sum(abs2, wall - water)
                0 < d2 < kernel.H² || continue
                pressure, _ = SphericTest06.pair_loads(wall, water, xy(velocity, i), xy(velocity, j),
                    density[i], density[j], kernel, constants)
                fx += pressure[1]
                if central_rear
                    support += constants.m₀ / density[j] * Wᵢⱼ(kernel, sqrt(d2) * kernel.h⁻¹)
                    max_approach = max(max_approach, velocity[1, j] - wall_velocity[1])
                    max_rear_pressure = max(max_rear_pressure, EquationOfStateGamma7(density[j], constants.c₀, constants.ρ₀))
                end
            end
        end
        central_rear && push!(fill, support)
        if local_pos[1] > 0
            front += face_weight * fx
        else
            rear += face_weight * fx
        end
        side += (1 - face_weight) * fx
    end
    length(fill) == 40 || error("Unexpected number of central rear probes")
    scale = -2 / constants.ρ₀
    return (; time, pressure_cd = scale * (front + rear + side),
        front_cd = scale * front, rear_cd = scale * rear, side_cd = scale * side,
        rear_supported = count(>(0), fill), rear_median_support = median(fill),
        max_rear_approach_mps = max_approach,
        peak_rear_fluid_pressure_kpa = isfinite(max_rear_pressure) ? max_rear_pressure/1000 : NaN)
end

function main(args)
    args in (String[], ["archived"], ["control"], ["manual"], ["no-shifting"], ["dual-shifting"]) ||
        error("Unknown replay analysis mode")
    unshifted = args == ["no-shifting"]
    dual = args == ["dual-shifting"]
    history = unshifted || dual
    archived = args == ["archived"]
    control = args == ["control"]
    manual = args == ["manual"]
    baseline = control ? REFINEMENT : joinpath(REFINEMENT, "fine_time")
    baseline_path = joinpath(RUNS, baseline, "Re100_n160", "MovingSquare.vtkhdf")
    replay = unshifted ? "fine_spike_no_shifting" : dual ? "fine_spike_dual_shifting" : "fine_spike_replay"
    path = archived || control ? baseline_path : joinpath(RUNS, replay, "MovingSquare.vtkhdf")
    reference = history ? nothing : CSV.File(joinpath(@__DIR__, "results", baseline, "Re100_n160.csv"))
    if manual
        path = joinpath(RUNS, "coefh", "Re100_n160_Test8", "MovingSquare.vtkhdf")
        reference = CSV.File(joinpath(@__DIR__, "results", "coefh", "Re100_n160.csv"))
    end
    case = build_case(; dx = 1/160, duration = 0.9, cfl = 0.1, sound_speed = 160.0,
        kernel_h_factor = 2sqrt(2), shifting = unshifted ? NoShifting : dual ? DualSPHysicsShifting : PlanarShifting, motion_kind = :integrated)
    constants = SimulationConstants{Float64}(2; dx = case.dx, ρ₀ = case.constants.ρ₀,
        c₀ = case.constants.c₀, g = case.constants.g, ν₀ = case.constants.ν₀)
    kernel = SPHKernelInstance{2,Float64}(case.kernel.kernel; h = Float64(case.kernel.h),
        k = Float64(case.kernel.k), αD = Float64(case.kernel.αD), η² = Float64(case.kernel.η²))
    rows = NamedTuple[]
    matched = 0
    h5open(path, "r") do file
        root = haskey(file, "VTKHDF/Particles") ? file["VTKHDF/Particles"] : file["VTKHDF"]
        times = read(root["Steps/Values"])
        offsets = read(root["Steps/PointOffsets"])
        counts = read(root["NumberOfPoints"])
        indices = findall(t -> manual ? 0.6 <= t <= 0.7 : 0.759999 <= t <= 0.86002, times)
        length(indices) == (archived || control || manual ? 6 : 101) || error("Unexpected diagnostic frame count")
        for frame in (history ? eachindex(times) : indices)
            columns = offsets[frame]+1:offsets[frame]+counts[frame]
            row = face_loads(root, columns, case, kernel, constants, times[frame])
            i = isnothing(reference) ? nothing : findfirst(==(row.time), reference.time)
            if i !== nothing
                isapprox(row.pressure_cd, reference.cd_pressure[i]; atol = 1e-8, rtol = 0) ||
                    error("Replay disagrees with archive at $(row.time): $(row.pressure_cd) vs $(reference.cd_pressure[i])")
                matched += 1
            end
            push!(rows, row)
        end
    end
    history || matched == 6 || error("Expected six matched original output frames")
    mkpath(OUT)
    filename = manual ? "manual_test8_faces.csv" : control ? "larger_timestep_faces.csv" :
        archived ? "archived_faces.csv" : "fine_faces.csv"
    if history
        filename = unshifted ? "no_shifting_history.csv" : "dual_shifting_history.csv"
    end
    CSV.write(joinpath(OUT, filename), rows)
    println("Wrote $(length(rows)) face-load measurements to $filename.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
