module Test06EdgeDamping

include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison
using SPHExampleGPU, StaticArrays, CSV, TOML, Statistics, LinearAlgebra, CUDA, Printf, SHA

const C = Test06DualComparison
const WORK = joinpath(@__DIR__,"runs","edge_damping")
const RESULTS = joinpath(@__DIR__,"results","edge_damping")
const SETTINGS = (; width = 0.5,overlimit = 0.1)

function edge_zone(; width = SETTINGS.width,rate = 10.0,overlimit = SETTINGS.overlimit)
    # Validate using the same bounds as the XML writer.
    C.edge_damping_xml((; width,rate,overlimit))
    return DampingBox(((width,width),(10-width,5-width)),((0.,0.),(10.,5.));
        overlimit,rate,factors = (1.,1.),directions = :all)
end

function analyze_variant(reynolds, rate)
    name = "re$(reynolds)_beta$(rate)"
    results = joinpath(RESULTS,name)
    reference = C.read_reference(joinpath(C.REFERENCE_DIR,@sprintf("Force_Re%03d.dat",reynolds)))
    rows, spectra = NamedTuple[], NamedTuple[]
    for solver in ("native","dual")
        data = CSV.File(joinpath(results,solver*".csv"))
        baseline = CSV.File(joinpath(C.RESULTS,"re$(reynolds)_c40",solver*".csv"))
        t,p,v = Float64.(data.time),Float64.(data.cd_pressure),Float64.(data.cd_viscous)
        bt,bp,bv = Float64.(baseline.time),Float64.(baseline.cd_pressure),Float64.(baseline.cd_viscous)
        metrics, spectrum = C.drag_metrics(t,p,v,reference)
        base_metrics,_ = C.drag_metrics(bt,bp,bv,reference)
        info = TOML.parsefile(joinpath(results,solver*".toml"))
        mean_change(value,base,left,right) = C.bin_average(t,value,left,right)/C.bin_average(bt,base,left,right)-1
        push!(rows,(; reynolds,rate,solver,width = SETTINGS.width,
            pressure_noise_rms = metrics["pressure_highpass_rms"],baseline_pressure_noise_rms = base_metrics["pressure_highpass_rms"],
            pressure_noise_reduction = 1-metrics["pressure_highpass_rms"]/base_metrics["pressure_highpass_rms"],
            viscous_noise_rms = metrics["viscous_highpass_rms"],
            pressure_raw_nrmse = metrics["pressure_raw_nrmse"],viscous_raw_nrmse = metrics["viscous_raw_nrmse"],
            pressure_binned_nrmse = metrics["pressure_nrmse"],viscous_binned_nrmse = metrics["viscous_nrmse"],
            pressure_late_bias = metrics["pressure_late_bias"],viscous_late_bias = metrics["viscous_late_bias"],
            pressure_mean_change_central = mean_change(p,bp,2.,6.5),viscous_mean_change_central = mean_change(v,bv,2.,6.5),
            pressure_mean_change_late = mean_change(p,bp,4.,8.),viscous_mean_change_late = mean_change(v,bv,4.,8.),
            steps = info["steps"],frames = info["frames"],particles = info["particles"],wall_seconds = info["wall_seconds"]))
        append!(spectra,[(; solver,row...) for row in spectrum])
    end
    CSV.write(joinpath(results,"metrics.csv"),rows)
    CSV.write(joinpath(results,"spectra.csv"),spectra)
    return rows
end

function run_variant(reynolds,rate)
    name = "re$(reynolds)_beta$(rate)"
    results = mkpath(joinpath(RESULTS,name))
    configuration = C.config(name; reynolds)
    damping = (; SETTINGS...,rate = Float64(rate))
    for solver in ("native","dual")
        isfile(joinpath(results,solver*".toml")) && continue
        directory = joinpath(WORK,name,solver)
        println("Outer-edge damping: ",name," / ",solver); flush(stdout)
        if solver == "native"
            result = C.native_run(configuration,directory; results,damping = edge_zone(; damping...))
        else
            prefix = C.prepare_dual(; reynolds,damping)
            C.verify_geometry(prefix,40)
            result = C.dual_run(prefix,directory; reynolds)
            C.extract_dual(directory,results)
            run_log = read(joinpath(directory,"Run.out"),String)
            occursin("Damping configuration:\nDamping zone_0 (type: Box):",replace(run_log,"\r\n"=>"\n")) ||
                error("Dual run did not report an active damping box")
        end
        result.frames == 401 && result.particles == 84864 || error("Incomplete or changed-count damping run")
        C.save_info(joinpath(results,solver*".toml"),result)
        GC.gc(true); CUDA.reclaim()
    end
    return analyze_variant(reynolds,rate)
end

function study()
    C.verify_reference()
    mkpath(RESULTS)
    configuration = Dict{String,Any}("width"=>SETTINGS.width,"overlimit"=>SETTINGS.overlimit,
        "rates"=>[10.,100.],"edges"=>["left","right","bottom","top"],
        "corner_rule"=>"maximum profile, not sum","velocity_components"=>"both physical components",
        "native_profile"=>"DampingBox, 2D","dual_profile"=>"dampingbox, x/z plane",
        "dp"=>0.025,"sound_speed"=>40.,"duration"=>8.,"output_interval"=>0.02,
        "motion"=>"original Hermite curve, retained to isolate damping",
        "baseline_commit"=>"9302adb263233d6501f16fb4f0cd77e8dcccfce4",
        "library_commit"=>strip(read(`git rev-parse HEAD`,String)),
        "gpu"=>CUDA.name(CUDA.device()),"julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
        "dual_version"=>"5.4.355 (08-04-2025)","dual_executable_sha256"=>bytes2hex(sha256(read(C.DUAL))))
    open(joinpath(RESULTS,"settings.toml"),"w") do io
        TOML.print(io,configuration; sorted = true)
    end
    rows = NamedTuple[]
    for rate in (10,100), reynolds in (50,100,150)
        append!(rows,run_variant(reynolds,rate))
        CSV.write(joinpath(RESULTS,"summary.csv"),rows)
    end
    return rows
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Test06EdgeDamping.study()
end
