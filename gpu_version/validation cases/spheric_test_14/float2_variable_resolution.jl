# Run from gpu_version. The fixed box encloses the full Float 2 heave envelope.
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA, Statistics

function build_fixed_refinement_case(release; dx = 0.025, half_width = 2.4, kernel_h_factor = 2.,
        box = VariableResolutionBox((-0.60,-0.60,-0.625),(0.60,0.60,0.50)),
        duration = 4.5, pause_time = 0.0, snapshot_interval = 0.1, output_variables = ["Type","Velocity","Pressure"],
        batch_steps = 8, use_graphs = true, save_location)
    coarse = build_full_case(release; dx, half_width, kernel_h_factor, duration, pause_time,
        sponge_width = half_width < 1.2 ? 0.20 : 0.50, output_interval = snapshot_interval,
        output_variables = [output_variables; "Resolution"; "Spacing"; "Mass"], save_location)
    fine_dx = dx / box.ratio
    fine_constants = SimulationConstants{Float32}(3; dx=fine_dx, ρ₀=coarse.constants.ρ₀, g=coarse.constants.g,
        c₀=coarse.constants.c₀, CFL=coarse.constants.CFL, ν₀=coarse.constants.ν₀, δᵩ=coarse.constants.δᵩ)
    fine_kernel = SPHKernelInstance{3,Float32}(coarse.kernel.kernel; h=Float32(kernel_h_factor*fine_dx))
    limits = SPHExampleGPU.VariableResolution.resolution_limits(box,Float64(coarse.kernel.h),Float64(fine_kernel.h))
    sampling_box = (;lower=limits.fine_outer_lo,upper=limits.fine_outer_hi)
    fine_geometry, _ = sample_geometry(release,fine_constants,fine_kernel; dx=fine_dx,half_width,depth=coarse.depth,
        wall_top_z=0.25,pause_time,boundary_mode=UpdatedMDBC,sampling_box)
    filter!(g -> !isempty(g.Particles),fine_geometry)
    # The floating group exists only on the fine level, where its physical
    # mass is distributed over the complete fine sampling of the hull.
    coarse_geometry = coarse.geometry[1:2]
    coarse.metadata.VariableResolution = VariableResolutionConfig(box; fine_geometry,fine_constants,fine_kernel)
    coarse.metadata.GPUUseGraph = use_graphs
    coarse.metadata.GPUMaxStepsPerSync = batch_steps
    return merge(coarse,(;geometry=coarse_geometry,fine_dx))
end

function run_float2_refinement(; smoke=false,held=false,compact=false,duration=nothing)
    nominal = last(RELEASES)
    release = held ? merge(nominal,(;offset=0.0)) : nominal
    tag = smoke ? (held ? "float2_vr_smoke_held" : "float2_vr_smoke_release") :
        (held ? "float2_vr_held" : compact ? "float2_vr_compact_release" : "float2_vr_release")
    default_duration = smoke ? 0.025 : held ? 1.5 : 4.5
    duration = duration === nothing ? default_duration : Float64(duration)
    isfinite(duration) && duration >= 0.005 && isapprox(duration/0.005,round(duration/0.005);atol=1e-10,rtol=0) ||
        throw(ArgumentError("duration must be a positive multiple of the 5 ms body sampling interval"))
    if duration != default_duration
        tag *= "_$(round(Int,1000duration))ms"
    end
    output = joinpath(@__DIR__,"results",tag)
    save_location = joinpath(@__DIR__,"runs",tag,release.name)
    pause_time = held ? duration + 1 : 0.0
    case = build_fixed_refinement_case(release; dx=smoke ? 0.05 : 0.025,
        half_width=smoke ? 0.8 : (held || compact) ? 1.2 : 2.4, kernel_h_factor=smoke ? 1.0 : 2.0,
        duration,pause_time,save_location)
    # Use the same floating-point times for coincident particle and body writes.
    times = collect(0.005:0.005:duration)
    times[end] = duration
    snapshots = times[20:20:end]
    case.metadata.OutputTimes = isempty(snapshots) ? [duration] : snapshots
    state_sampling = (;times,floating_history=true)
    completed = run_case(release,case;output,state_sampling,save_plot=!held && !smoke)
    cp(joinpath(save_location,"FreeHeave_History.csv"),joinpath(output,release.name*"_resolution.csv");force=true)
    if held
        rows = CSV.File(joinpath(output,release.name*"_body.csv"))
        center = getproperty(rows,Symbol("Center:2"))
        force = getproperty(rows,Symbol("Force:2"))
        all(==(first(center)),center) || error("Float moved during the hold")
        weight = release.mass*Float64(case.constants.g)
        tail = rows.Time .>= max(0.005,duration-min(0.5,0.2duration))
        println("Held vertical fluid load ",mean(force[tail])," N over ",first(rows.Time[tail]),"–",last(rows.Time[tail]),
            " s. Body weight ",weight," N.")
    end
    return completed
end

if abspath(PROGRAM_FILE) == @__FILE__
    positional = filter(!in(("--smoke","--held","--compact")),ARGS)
    length(positional) <= 1 || error("Usage: float2_variable_resolution.jl [--held] [--smoke] [--compact] [duration_seconds]")
    duration = isempty(positional) ? nothing : parse(Float64,only(positional))
    run_float2_refinement(;smoke="--smoke" in ARGS,held="--held" in ARGS,compact="--compact" in ARGS,duration)
end
