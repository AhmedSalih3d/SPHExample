include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison, SPHExampleGPU, CSV, TOML
const C = Test06DualComparison
const OUTPUT = mkpath(joinpath(@__DIR__,"results","timestep_scheme_control"))
open(joinpath(OUTPUT,"settings.toml"),"w") do io
    TOML.print(io,Dict("reynolds"=>100,"kinematic_viscosity"=>0.01,"dp"=>0.025,"cfl"=>0.2,
        "scheme"=>"DualSPHysicsSymplectic with current-stage retries and no minimum-step override",
        "viscosity"=>"Laminar","density_diffusion"=>"Fourtakas inner, delta=0.1",
        "sound_speeds"=>[40,160],"motions"=>["hermite","integrated"],
        "particle_precision"=>"Float32","position_and_clock_precision"=>"Float64",
        "fast_math"=>false,"graph"=>false,"lanes"=>1,"edge_damping"=>false,"divergence_cleaning"=>false); sorted=true)
end
records = NamedTuple[]
for speed in (40,160), motion in (:hermite,:integrated)
    id = "re100_c$(speed)_$(motion)"
    directory = mkpath(joinpath(OUTPUT,id))
    info_path = joinpath(directory,"native.toml")
    config = merge(C.config(id; sound_speed=speed,motion_kind=motion,diffusion=2),
        (; scheme=DualSPHysicsSymplectic(respect_stability_limits=true,fast_math=false),graph=false,lanes=1))
    if !isfile(info_path)
        println("Neighbor-based timestep/integrator control: ",id); flush(stdout)
        info = C.native_run(config,joinpath(@__DIR__,"runs","timestep_scheme_control",id); results=directory)
        C.save_info(info_path,info)
    end
    audit = TOML.parsefile(info_path)
    audit["accepted_limit_ratio_max"] <= 1 || error("An accepted step exceeded a computed current-stage limit")
    audit["frames"] == 401 && audit["particles"] == 84864 || error("Incomplete timestep control")
    data = CSV.File(joinpath(directory,"native.csv"))
    reference = C.read_reference(joinpath(C.REFERENCE_DIR,"Force_Re100.dat"))
    metrics,_ = C.drag_metrics(Float64.(data.time),Float64.(data.cd_pressure),Float64.(data.cd_viscous),reference)
    push!(records,(; c0=speed,motion=String(motion),pressure_noise_rms=metrics["pressure_highpass_rms"],
        pressure_raw_nrmse=metrics["pressure_raw_nrmse"],viscous_raw_nrmse=metrics["viscous_raw_nrmse"]))
    CSV.write(joinpath(OUTPUT,"summary.csv"),records)
end
