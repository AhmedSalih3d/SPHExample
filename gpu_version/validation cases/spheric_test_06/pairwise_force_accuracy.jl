module PairwiseForceAccuracy
include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison, SPHExampleGPU, CUDA, CSV, TOML, Printf
const C = Test06DualComparison
const RESULTS = joinpath(@__DIR__,"results","pairwise_timestep")
const WORK = joinpath(@__DIR__,"runs","pairwise_timestep")

function summarize()
    records = NamedTuple[]
    for speed in (40,160), re in (50,100,150), criterion in ("legacy","pairwise")
        directory = joinpath(RESULTS,"re$(re)_c$(speed)_$(criterion)")
        isfile(joinpath(directory,"native.toml")) || continue
        data = CSV.File(joinpath(directory,"native.csv"))
        reference = C.read_reference(joinpath(C.REFERENCE_DIR,@sprintf("Force_Re%03d.dat",re)))
        t,p,v = Float64.(data.time),Float64.(data.cd_pressure),Float64.(data.cd_viscous)
        metrics,_ = C.drag_metrics(t,p,v,reference)
        total,_ = C.drag_metrics(t,p+v,v,hcat(reference[:,1],reference[:,2]+reference[:,3],reference[:,3]))
        info = TOML.parsefile(joinpath(directory,"native.toml"))
        push!(records,(; reynolds=re,sound_speed=speed,criterion,
            pressure_raw_nrmse=metrics["pressure_raw_nrmse"],pressure_binned_nrmse=metrics["pressure_nrmse"],
            viscous_raw_nrmse=metrics["viscous_raw_nrmse"],viscous_binned_nrmse=metrics["viscous_nrmse"],
            total_raw_nrmse=total["pressure_raw_nrmse"],total_binned_nrmse=total["pressure_nrmse"],
            pressure_noise_rms=metrics["pressure_highpass_rms"],viscous_noise_rms=metrics["viscous_highpass_rms"],
            total_noise_rms=total["pressure_highpass_rms"],steps=info["steps"],frames=info["frames"],
            particles=info["particles"],solver_seconds=info["solver_seconds"]))
    end
    CSV.write(joinpath(RESULTS,"summary.csv"),records)
    return records
end

function study()
    C.verify_reference()
    mkpath(RESULTS); mkpath(WORK)
    C.save_info(joinpath(RESULTS,"settings.toml"),(; source_commit="fc4012c4",duration=8.,dp=0.025,
        output_interval=0.02,cfl=0.2,reynolds=[50,100,150],sound_speeds=[40,160],
        motion="Original official position/velocity Hermite trajectory",boundary="DBC / NoMDBC",
        diffusion="ZeroGravityLinearDensityDiffusion",viscosity="Laminar",cleaning=false,damping=false,
        precision="Float32 physics, Float64 positions and clock",gpu=CUDA.name(CUDA.device()),
        julia=string(VERSION),cuda_jl=string(pkgversion(CUDA)),driver=string(CUDA.driver_version()),
        force_analysis="Independent Float64 square/fluid pair loads at 401 accepted output frames",
        raw_error="Piecewise-linear time-integrated relative L2 error on [0.1,8] s",
        binned_error="Relative L2 error of 0.1 s bin averages on [0.1,8] s",
        noise="Residual minus 0.5 s centered box average, RMS on [1.5,6.5] s; output Nyquist 25 Hz",
        timing="Single force-validation runs; includes compilation on first variants; not a paired speed study"))
    for speed in (40,160), re in (50,100,150), criterion in ("pairwise","legacy")
        id = "re$(re)_c$(speed)_$(criterion)"
        directory = mkpath(joinpath(RESULTS,id))
        info_path = joinpath(directory,"native.toml")
        isfile(info_path) && continue
        println("Force validation: ",id); flush(stdout)
        config = merge(C.config(id;reynolds=re,sound_speed=speed),(;legacy_timestep=criterion == "legacy"))
        info = C.native_run(config,joinpath(WORK,id);results=directory)
        info.timestep_criterion == (criterion == "legacy" ? "legacy_absolute_position" : "pairwise_signal") || error("Wrong timestep criterion")
        info.frames == 401 && info.particles == 84864 || error("Incomplete validation")
        C.save_info(info_path,info)
        summarize()
        println("Completed ",id,"; ",info.steps," steps"); flush(stdout)
        GC.gc(true); CUDA.reclaim()
    end
    return summarize()
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    PairwiseForceAccuracy.study()
end
