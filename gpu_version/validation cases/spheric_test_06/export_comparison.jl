include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison
using SPHExampleGPU, StaticArrays, CSV, TOML, Printf, CUDA

function export_comparison()
    C = Test06DualComparison
    for re in (50,100,150)
        directory = joinpath(C.WORK,"re$(re)_c40","dual")
        output = joinpath(directory,"fields_0400.csv")
        if !isfile(output)
            C.logged(`$(C.PART) -dirdata $(joinpath(directory,"data")) -files:400 -savecsv $(joinpath(directory,"fields")) -onlytype:-all,+fluid,+moving -vars:-all,+idp,+vel,+rhop,+type,+mk`,joinpath(directory,"fields.log"))
        end
    end
    # Independently reproduce the vendor force tool on two compact snapshots.
    checks = NamedTuple[]
    directory = joinpath(C.WORK,"re100_c40","dual")
    table = CSV.File(joinpath(C.RESULTS,"re100_c40","dual.csv"))
    for frame in (100,400)
        time = Float64(table.time[frame+1])
        center = 1.5+first(trajectory_state(C.body_trajectory(),time))[1]
        stem = joinpath(directory,"check$(frame)")
        file = stem*"_"*lpad(frame,4,'0')*".csv"
        if !isfile(file)
            C.logged(`$(C.PART) -dirdata $(joinpath(directory,"data")) -files:$frame -savecsv $stem -onlypos:$(center-0.65):-1:1.85:$(center+0.65):1:3.15 -onlytype:-all,+fluid,+moving -vars:-all,+vel,+rhop,+type`,stem*".log")
        end
        rows = collect(C.read_dual_csv(file))
        position = [SVector(row[Symbol("Pos.x [m]")],row[Symbol("Pos.z [m]")]) for row in rows]
        velocity = [SVector(row[Symbol("Vel.x [m/s]")],row[Symbol("Vel.z [m/s]")]) for row in rows]
        density = Float64[row[Symbol("Rhop [kg/m^3]")] for row in rows]
        types = [Int(row.Type == 1 ? Moving : Fluid) for row in rows]
        markers = [row.Type == 1 ? 3 : 2 for row in rows]
        constants = SimulationConstants{Float64}(2; dx = 0.025,c₀ = 40,ν₀ = 0.01,g = 0)
        h = sqrt(2)*0.025
        kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h,η² = (0.1h)^2)
        loads = C.frame_loads(position,velocity,density,types,markers,kernel,constants)
        push!(checks,(; time,frame,pressure_tool = table.cd_pressure[frame+1],pressure_pairs = loads.cd_pressure,
            viscous_tool = table.cd_viscous[frame+1],viscous_pairs = loads.cd_viscous,
            square_particles = count(==(3),markers)))
    end
    CSV.write(joinpath(C.RESULTS,"force_crosscheck.csv"),checks)
    environment = Dict("julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
        "gpu"=>CUDA.name(CUDA.device()),"driver"=>string(CUDA.driver_version()),
        "nvidia_driver"=>readchomp(`nvidia-smi --query-gpu=driver_version --format=csv,noheader`),
        "dual_version"=>"5.4.355 (08-04-2025)","gencase_version"=>"5.4.354.01 (07-04-2025)",
        "force_tool_version"=>"5.4.266.02 (07-05-2025)",
        "dual_executable_sha256"=>bytes2hex(C.sha256(read(C.DUAL))),
        "source_commit"=>strip(read(`git rev-parse HEAD`,String)),"platform"=>string(Sys.KERNEL),
        "particles"=>84864,"fluid"=>78400,"square"=>1600,"fixed"=>4864,
        "dp"=>0.025,"h"=>sqrt(2)*0.025,"rho0"=>1000.,"cfl"=>0.2,
        "duration"=>8.,"output_interval"=>0.02,"motion_file_interval"=>0.0001,
        "density_diffusion"=>0.1,"native_eta_squared"=>(0.01sqrt(2)*0.025)^2,
        "dual_eta_squared"=>(0.1sqrt(2)*0.025)^2)
    open(joinpath(C.RESULTS,"environment.toml"),"w") do io
        TOML.print(io,environment; sorted = true)
    end
    println(checks)
end

export_comparison()
