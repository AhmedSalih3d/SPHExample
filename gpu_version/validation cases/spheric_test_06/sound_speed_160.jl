module Test06Sound160

include(joinpath(@__DIR__,"divergence_cleaning.jl"))
using .Test06DivergenceCleaning
using SPHExampleGPU, CUDA, CSV, TOML, Printf
const E = Test06DivergenceCleaning
const C = E.C
const WORK = joinpath(@__DIR__,"runs","c160_comparison")
const RESULTS = joinpath(@__DIR__,"results","c160_comparison")
const VARIANTS = ("baseline40","standard160","step_control40","cleaning_control160","cleaning160","cleaning_k10_160","cleaning_k40_160")

function parameters(variant)
    return variant == "baseline40" ? (; speed=40,cfl=0.2,cleaning=nothing) :
        variant == "standard160" ? (; speed=160,cfl=0.2,cleaning=nothing) :
        variant == "step_control40" ? (; speed=40,cfl=0.05,cleaning=nothing) :
        variant == "cleaning_control160" ? (; speed=160,cfl=0.1,cleaning=nothing) :
        variant == "cleaning160" ? (; speed=160,cfl=0.1,cleaning=DivergenceCleaning(kappa=160)) :
        variant == "cleaning_k10_160" ? (; speed=160,cfl=0.1,cleaning=DivergenceCleaning(kappa=10)) :
        variant == "cleaning_k40_160" ? (; speed=160,cfl=0.1,cleaning=DivergenceCleaning(kappa=40)) :
        throw(ArgumentError("Unknown c160 variant: $variant"))
end

function data_directory(re,variant)
    if variant == "baseline40" || (variant == "standard160" && re == 100)
        return joinpath(C.RESULTS,"re$(re)_c$(parameters(variant).speed)")
    end
    return joinpath(RESULTS,"re$(re)_$(variant)")
end

function analyze(re,variant,solver)
    directory = data_directory(re,variant)
    data = CSV.File(joinpath(directory,solver*".csv"))
    reference = C.read_reference(joinpath(C.REFERENCE_DIR,@sprintf("Force_Re%03d.dat",re)))
    metrics,spectra = C.drag_metrics(Float64.(data.time),Float64.(data.cd_pressure),Float64.(data.cd_viscous),reference)
    info = TOML.parsefile(joinpath(directory,solver*".toml"))
    archive = variant == "baseline40" || (variant == "standard160" && re == 100)
    settings = parameters(variant)
    return (; reynolds=re,variant,solver,sound_speed=settings.speed,cfl=settings.cfl,
        pressure_noise_rms=metrics["pressure_highpass_rms"],viscous_noise_rms=metrics["viscous_highpass_rms"],
        pressure_raw_nrmse=metrics["pressure_raw_nrmse"],viscous_raw_nrmse=metrics["viscous_raw_nrmse"],
        pressure_binned_nrmse=metrics["pressure_nrmse"],viscous_binned_nrmse=metrics["viscous_nrmse"],
        pressure_dominant_hz=metrics["pressure_dominant_hz"],steps=info["steps"],frames=info["frames"],
        particles=info["particles"],mean_timestep=8/info["steps"],wall_seconds=info["wall_seconds"],
        source=archive ? "published archive 9302adb" : "current run",
        data_path=replace(relpath(directory,@__DIR__),'\\'=>'/'))
end

function summarize()
    rows = NamedTuple[]
    for variant in VARIANTS, re in (50,100,150), solver in ("native","dual")
        startswith(variant,"cleaning_k") && re != 100 && continue
        solver == "dual" && !(variant in ("baseline40","standard160","step_control40")) && continue
        directory = data_directory(re,variant)
        isfile(joinpath(directory,solver*".toml")) || continue
        push!(rows,analyze(re,variant,solver))
    end
    CSV.write(joinpath(RESULTS,"summary.csv"),rows)
    return rows
end

function study(; solvers=("native","dual"))
    C.verify_reference()
    mkpath(RESULTS)
    open(joinpath(RESULTS,"settings.toml"),"w") do io
        TOML.print(io,Dict("dp"=>0.025,"duration"=>8.,"output_interval"=>0.02,"reynolds"=>[50,100,150],
            "baseline_cfl"=>0.2,"c40_small_step_cfl"=>0.05,"cleaning_comparison_cfl"=>0.1,
            "cleaning_kappa"=>160.,"cleaning_sigma"=>1.,"cleaning_courant"=>0.1,"pressure_mode"=>"absolute",
            "dual_cleaning"=>"stock executable, no divergence-cleaning modification",
            "motion"=>"original Hermite retained","gpu"=>CUDA.name(CUDA.device()),
            "julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),"cuda_driver"=>string(CUDA.driver_version()),
            "archive_commit"=>"9302adb263233d6501f16fb4f0cd77e8dcccfce4",
            "step_control"=>"equal nominal acoustic cap: 0.05*h/40 = 0.2*h/160; actual adaptive steps reported"); sorted=true)
    end
    for variant in VARIANTS[2:end], re in (50,100,150), solver in solvers
        startswith(variant,"cleaning_k") && re != 100 && continue
        solver == "dual" && !(variant in ("standard160","step_control40")) && continue
        variant == "standard160" && re == 100 && continue
        results = mkpath(data_directory(re,variant))
        isfile(joinpath(results,solver*".toml")) && continue
        id = "re$(re)_$(variant)"
        println("c0 = 160 comparison: ",id," / ",solver); flush(stdout)
        directory = joinpath(WORK,id,solver)
        settings = parameters(variant)
        if solver == "native"
            config = merge(C.config(id; reynolds=re,sound_speed=settings.speed),(; cfl=settings.cfl))
            info = C.native_run(config,directory; results,cleaning=settings.cleaning)
        else
            prefix = C.prepare_dual(reynolds=re,sound_speed=settings.speed,cfl=settings.cfl)
            C.verify_geometry(prefix,40)
            info = C.dual_run(prefix,directory; reynolds=re)
            C.extract_dual(directory,results)
        end
        info.frames == 401 && info.particles == 84864 || error("Incomplete c160 case")
        C.save_info(joinpath(results,solver*".toml"),info)
        summarize()
        GC.gc(true); CUDA.reclaim()
    end
    return summarize()
end

function export_fields()
    for re in (50,100,150)
        directory = joinpath(re == 100 ? C.WORK : WORK,re == 100 ? "re100_c160" : "re$(re)_standard160","dual")
        path = joinpath(directory,"fields_0400.csv")
        isfile(path) && continue
        C.logged(`$(C.PART) -dirdata $(joinpath(directory,"data")) -files:400 -savecsv $(joinpath(directory,"fields")) -onlytype:-all,+fluid,+moving -vars:-all,+idp,+vel,+rhop,+type,+mk`,joinpath(directory,"fields.log"))
    end
end

function timing_study(; repeats=3)
    directory = mkpath(joinpath(WORK,"timing"))
    config = C.config("re100_c160"; sound_speed=160)
    prefix = C.prepare_dual(sound_speed=160)
    C.verify_geometry(prefix,40)
    for solver in ("native","dual")
        println("c160 timing warmup: ",solver); flush(stdout)
        warm = mktempdir(directory; prefix="warm_$(solver)_")
        if solver == "native"
            C.native_run(config,warm; analyze=false)
        else
            C.dual_run(prefix,warm; postprocess=false)
        end
    end
    records = NamedTuple[]
    states = Dict{String,Matrix{Float64}}()
    for repetition in 1:repeats
        order = isodd(repetition) ? ("native","dual") : ("dual","native")
        for (slot,solver) in enumerate(order)
            println("c160 timing pair ",repetition,": ",solver); flush(stdout)
            GC.gc(true); CUDA.reclaim()
            run_directory = mktempdir(directory; prefix="pair$(repetition)_$(solver)_")
            info = solver == "native" ? C.native_run(config,run_directory; analyze=false) :
                C.dual_run(prefix,run_directory; postprocess=false)
            if solver == "native"
                data = CSV.File(joinpath(run_directory,"final.csv"))
                state = hcat(Float64.(data.x),Float64.(data.y),Float64.(data.u),Float64.(data.v),Float64.(data.rho))
            else
                data = C.dual_final_state(run_directory)
                state = hcat([Float64[getproperty(row,Symbol(key)) for row in data] for key in
                    ("Pos.x [m]","Pos.z [m]","Vel.x [m/s]","Vel.z [m/s]","Rhop [kg/m^3]")]...)
            end
            previous = get!(states,solver,copy(state))
            size(state) == size(previous) == (84864,5) || error("Repeat particle count changed")
            info.frames == 401 && info.particles == 84864 || error("Incomplete timing run")
            discrepancy = maximum(abs,state-previous)
            isfinite(discrepancy) || error("Nonfinite repeated state")
            push!(records,(; solver,repetition,order=slot,wall_seconds=info.wall_seconds,
                solver_seconds=info.solver_seconds,steps=info.steps,frames=info.frames,particles=info.particles,
                rebuilds=get(info,:rebuilds,missing),readbacks=get(info,:readbacks,missing),
                repeat_position_max=maximum(abs,state[:,1:2]-previous[:,1:2]),
                repeat_velocity_max=maximum(abs,state[:,3:4]-previous[:,3:4]),
                repeat_density_max=maximum(abs,state[:,5]-previous[:,5])))
            CSV.write(joinpath(RESULTS,"timings.csv"),records)
        end
    end
    for solver in ("native","dual")
        group = filter(r -> r.solver == solver,records)
        length(unique(r.steps for r in group)) == 1 || error("Repeat step count changed for $solver")
        solver == "dual" && continue
        length(unique(r.rebuilds for r in group)) == length(unique(r.readbacks for r in group)) == 1 ||
            error("Repeat rebuild or readback count changed")
    end
    return records
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        Test06Sound160.study()
    elseif first(ARGS) == "native"
        Test06Sound160.study(solvers=("native",))
    elseif first(ARGS) == "dual"
        Test06Sound160.study(solvers=("dual",))
    elseif first(ARGS) == "fields"
        Test06Sound160.export_fields()
    elseif first(ARGS) == "timing"
        Test06Sound160.timing_study()
    else
        error("Expected no arguments, native, dual, fields or timing")
    end
end
