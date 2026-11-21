include(joinpath(@__DIR__,"case.jl"))
include(joinpath(@__DIR__,"planar_case.jl"))
using CUDA

function boundary_study()
    root = joinpath(@__DIR__,"results","mdbc_study")
    mkpath(root)
    settings = (;dx=0.0125,half_width=8.,depth=0.8,duration=4.5,output_interval=0.005,
        kernel_h_factor=2.,sound_speed=25.,cfl=0.2,viscosity="Laminar, nu=1e-6 m2/s",
        body="Cartesian 2D, draft-matched mass per unit span, heave-only",
        boundaries=["NoMDBC","SimpleMDBC"],diffusion=["LinearDensityDiffusion","FourtakasDensityDiffusion"],
        gpu=CUDA.name(CUDA.device()),julia=string(VERSION),cuda_jl=string(pkgversion(CUDA)))
    open(joinpath(root,"settings.toml"),"w") do io
        TOML.print(io,Dict(string(k)=>v for (k,v) in pairs(settings)))
    end
    for diffusion in (LinearDensityDiffusion(),FourtakasDensityDiffusion()), mode in (NoMDBC,SimpleMDBC), release in RELEASES
        variant = "$(nameof(mode))_$(nameof(typeof(diffusion)))"
        output = joinpath(root,variant)
        info = joinpath(output,release.name*".toml")
        isfile(info) && continue
        directory = joinpath(@__DIR__,"runs","mdbc_study",variant,release.name)
        println("Running ",variant," / ",release.name);flush(stdout)
        case = build_planar_case(release;boundary_mode=mode,diffusion,save_location=directory)
        completed = run_case(release,case;output,vertical_axis=2)
        record = TOML.parsefile(info)
        record["boundary_mode"] = string(nameof(mode))
        record["density_diffusion"] = string(nameof(typeof(diffusion)))
        open(info,"w") do io
            TOML.print(io,record)
        end
        write_heave_state(merge(release,(;title=release.title*" | 2D | $(nameof(mode))")),completed;
            layout="stacked",screenshot=true,planar=true)
        println("Completed ",variant," / ",release.name);flush(stdout)
        GC.gc(true);CUDA.reclaim()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    boundary_study()
end
