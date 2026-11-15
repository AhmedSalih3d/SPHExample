module Test06DivergenceCleaning

include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison
using SPHExampleGPU, CUDA, CSV, TOML, Statistics, LinearAlgebra, Printf, SHA
const C = Test06DualComparison
const DC = SPHExampleGPU.GPUDivergenceCleaning
const DR = SPHExampleGPU.SPHCellList
const WORK = joinpath(@__DIR__,"runs","divergence_cleaning")
const RESULTS = joinpath(@__DIR__,"results","divergence_cleaning")

configuration(re,variant) = merge(C.config("re$(re)_$(variant)"; reynolds=re),(; cfl=0.1))
cleaning(variant) = variant == "off" ? nothing : DivergenceCleaning(;
    kappa=endswith(variant,"k10") ? 10 : 40,pressure_mode=startswith(variant,"positive") ? :positive : :absolute)

function analyze_variant(re,variant)
    id = "re$(re)_$(variant)"
    directory = joinpath(RESULTS,id)
    data = CSV.File(joinpath(directory,"native.csv"))
    reference = C.read_reference(joinpath(C.REFERENCE_DIR,@sprintf("Force_Re%03d.dat",re)))
    metrics,spectra = C.drag_metrics(Float64.(data.time),Float64.(data.cd_pressure),Float64.(data.cd_viscous),reference)
    CSV.write(joinpath(directory,"spectra.csv"),spectra)
    info = TOML.parsefile(joinpath(directory,"native.toml"))
    baseline = CSV.File(joinpath(RESULTS,"re$(re)_off","native.csv"))
    bp,bv,bt = Float64.(baseline.cd_pressure),Float64.(baseline.cd_viscous),Float64.(baseline.time)
    base_metrics,_ = C.drag_metrics(bt,bp,bv,reference)
    p,v,t = Float64.(data.cd_pressure),Float64.(data.cd_viscous),Float64.(data.time)
    state = CSV.File(joinpath(directory,"native_state.csv"))
    return (; case=id,reynolds=re,variant,kappa=variant == "off" ? 0 : endswith(variant,"k10") ? 10 : 40,
        pressure_mode=variant == "off" ? "disabled" : startswith(variant,"positive") ? "positive" : "absolute",
        pressure_noise_rms=metrics["pressure_highpass_rms"],viscous_noise_rms=metrics["viscous_highpass_rms"],
        pressure_noise_reduction=1-metrics["pressure_highpass_rms"]/base_metrics["pressure_highpass_rms"],
        pressure_raw_nrmse=metrics["pressure_raw_nrmse"],viscous_raw_nrmse=metrics["viscous_raw_nrmse"],
        pressure_binned_nrmse=metrics["pressure_nrmse"],viscous_binned_nrmse=metrics["viscous_nrmse"],
        pressure_mean_change_central=C.bin_average(t,p,2.,6.5)/C.bin_average(bt,bp,2.,6.5)-1,
        viscous_mean_change_central=C.bin_average(t,v,2.,6.5)/C.bin_average(bt,bv,2.,6.5)-1,
        pressure_mean_change_late=C.bin_average(t,p,4.,8.)/C.bin_average(bt,bp,4.,8.)-1,
        maximum_density_deviation=maximum(state.density_relative_max),maximum_mach=maximum(state.max_speed)/40,
        minimum_solid_clearance=minimum(state.minimum_clearance),maximum_motion_error=maximum(state.motion_error),
        steps=info["steps"],frames=info["frames"],particles=info["particles"],rebuilds=info["rebuilds"],
        readbacks=info["readbacks"],wall_seconds=info["wall_seconds"],psi_rms=get(info,"psi_rms",0.))
end

function study()
    C.verify_reference()
    mkpath(RESULTS)
    settings = Dict("paper_doi"=>"10.1016/j.compfluid.2025.106638","kappa"=>40.,"sigma"=>1.,"cleaning_courant"=>0.1,
        "cfl_both_variants"=>0.1,"default_pressure_extension"=>"abs(p)","pressure_control"=>"max(p,0)","wall_psi"=>0.,"initial_psi"=>0.,
        "sound_speed"=>40.,"dp"=>0.025,"duration"=>8.,"output_interval"=>0.02,
        "motion"=>"original Hermite curve retained to isolate cleaning","edge_damping"=>false,
        "library_base_commit"=>strip(read(`git rev-parse HEAD`,String)),"julia"=>string(VERSION),
        "gpu"=>CUDA.name(CUDA.device()),"cuda_jl"=>string(pkgversion(CUDA)),"cuda_driver"=>string(CUDA.driver_version()))
    if !isfile(joinpath(RESULTS,"settings.toml"))
        open(joinpath(RESULTS,"settings.toml"),"w") do io
            TOML.print(io,settings; sorted=true)
        end
    end
    rows = NamedTuple[]
    for variant in ("off","positive","on"), re in (50,100,150)
        id = "re$(re)_$(variant)"
        results = mkpath(joinpath(RESULTS,id))
        if !isfile(joinpath(results,"native.toml"))
            println("Divergence cleaning study: ",id); flush(stdout)
            info = C.native_run(configuration(re,variant),joinpath(WORK,id); results,cleaning=cleaning(variant))
            info.frames == 401 && info.particles == 84864 || error("Incomplete cleaning study")
            C.save_info(joinpath(results,"native.toml"),info)
            GC.gc(true); CUDA.reclaim()
        end
        push!(rows,analyze_variant(re,variant))
        CSV.write(joinpath(RESULTS,"summary.csv"),rows)
    end
    for variant in ("positive_k10","k10")
        re = 100
        id = "re$(re)_$(variant)"
        results = mkpath(joinpath(RESULTS,id))
        if !isfile(joinpath(results,"native.toml"))
            println("Divergence cleaning sensitivity: ",id); flush(stdout)
            info = C.native_run(configuration(re,variant),joinpath(WORK,id); results,cleaning=cleaning(variant))
            info.frames == 401 && info.particles == 84864 || error("Incomplete sensitivity run")
            C.save_info(joinpath(results,"native.toml"),info)
        end
        push!(rows,analyze_variant(re,variant))
    end
    CSV.write(joinpath(RESULTS,"summary.csv"),rows)
    return rows
end

function timing_study(; repeats=3)
    directory = mkpath(joinpath(WORK,"timing"))
    # Every variant gets a full-duration warmup in this process.
    for variant in ("off","on")
        warm = mktempdir(directory; prefix="warm_$(variant)_")
        println("Cleaning timing warmup: ",variant); flush(stdout)
        C.native_run(configuration(100,variant),warm; analyze=false,cleaning=cleaning(variant))
    end
    rows = NamedTuple[]
    states = Dict{String,Matrix{Float64}}()
    for repetition in 1:repeats
        variants = isodd(repetition) ? ("off","on") : ("on","off")
        for variant in variants
            run_directory = mktempdir(directory; prefix="pair$(repetition)_$(variant)_")
            println("Cleaning timing pair ",repetition,": ",variant); flush(stdout)
            info = C.native_run(configuration(100,variant),run_directory; analyze=false,cleaning=cleaning(variant))
            data = CSV.File(joinpath(run_directory,"final.csv"))
            state = hcat(Float64.(data.x),Float64.(data.y),Float64.(data.u),Float64.(data.v),Float64.(data.rho))
            old = get!(states,variant,state)
            discrepancy = maximum(abs,state-old)
            discrepancy == 0 || error("Repeated $variant states differ")
            push!(rows,(; repetition,variant,wall_seconds=info.wall_seconds,solver_seconds=info.solver_seconds,
                steps=info.steps,particles=info.particles,frames=info.frames,rebuilds=info.rebuilds,readbacks=info.readbacks,
                psi_rms=get(info,:psi_rms,0.),repeat_state_max_difference=discrepancy))
            CSV.write(joinpath(RESULTS,"timings.csv"),rows)
        end
    end
    return rows
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        Test06DivergenceCleaning.study()
    elseif first(ARGS) == "timing"
        Test06DivergenceCleaning.timing_study()
    else
        error("Expected no arguments or timing")
    end
end
