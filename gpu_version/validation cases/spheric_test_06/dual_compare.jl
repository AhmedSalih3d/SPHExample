module Test06DualComparison

include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
using SPHExampleGPU
using StaticArrays: SVector
using CSV, TOML, Printf, Statistics, LinearAlgebra, SHA, CUDA
include(joinpath(@__DIR__, "noise_metrics.jl"))

const ROOT = get(ENV,"DUALSPHYSICS_ROOT",raw"W:\DualSPHysics_v5.4")
const BIN = joinpath(ROOT,"bin",Sys.iswindows() ? "windows" : "linux")
const SUFFIX = Sys.iswindows() ? "_win64.exe" : "_linux64"
const DUAL = joinpath(BIN,"DualSPHysics5.4"*SUFFIX)
const GEN = joinpath(BIN,"GenCase"*SUFFIX)
const FORCES = joinpath(BIN,"ComputeForces"*SUFFIX)
const PART = joinpath(BIN,"PartVTK"*SUFFIX)
const WORK = joinpath(@__DIR__, "runs", "comparison")
const RESULTS = joinpath(@__DIR__, "results", "comparison")

function logged(command, path)
    mkpath(dirname(path))
    open(path, "w") do io
        run(pipeline(command; stdout = io, stderr = io))
    end
end

"""Integrate the official velocity table, removing rounded-position derivative jitter."""
function integrated_trajectory()
    data = read_reference(joinpath(REFERENCE_DIR, "Motion_Body.dat"))
    displacement = zeros(size(data,1))
    for i in 2:length(displacement)
        displacement[i] = displacement[i-1] + (data[i-1,3]+data[i,3])*(data[i,1]-data[i-1,1])/2
    end
    return SampledTrajectory(data[:,1], [SVector(x,0.) for x in displacement];
        velocities = [SVector(v,0.) for v in data[:,3]])
end

function prepare_dual(; n = 40, reynolds = 100, sound_speed = 40, duration = 8.0, interval = 0.02,
        motion_kind = :hermite, diffusion = 1)
    name = "re$(reynolds)_n$(n)_c$(sound_speed)_$(motion_kind)_ddt$(diffusion)_t$(duration)_out$(interval)_centered_v2"
    directory = mkpath(joinpath(WORK, "definitions", name))
    prefix = joinpath(directory, "Case")
    isfile(prefix * ".bi4") && return prefix
    dx = 1/n
    case = build_case(; dx, sound_speed, duration, interval)
    trajectory = motion_kind == :hermite ? body_trajectory() : integrated_trajectory()
    motion_path = joinpath(directory, "motion.txt")
    # Dense common samples make Dual's linear file interpolation a controlled
    # approximation of the same curve used by Julia (0.1 ms sample interval).
    open(motion_path, "w") do io
        for time in 0:0.0001:8.1
            offset = first(trajectory_state(trajectory, time))
            @printf(io, "%.10f %.15e\n", time, offset[1])
        end
    end
    box(x,z,w,h) = "<drawbox><boxfill>solid</boxfill><point x=\"$(x+dx/2)\" y=\"-1\" z=\"$(z+dx/2)\"/><size x=\"$(w-dx)\" y=\"2\" z=\"$(h-dx)\"/></drawbox>"
    l = case.layers*dx
    walls = box(-l,-l,10+2l,l)*box(-l,5,10+2l,l)*box(-l,0,l,5)*box(10,0,l,5)
    fluid = box(0,0,1,5)*box(2,0,8,5)*box(1,0,1,2)*box(1,3,1,2)
    xml = """
    <?xml version="1.0" encoding="UTF-8"?>
    <case><casedef><constantsdef>
      <gravity x="0" y="0" z="0"/><rhop0 value="1000"/><rhopgradient value="1"/>
      <hswl value="0" auto="false"/><gamma value="7"/>
      <speedsystem value="1" auto="false"/><coefsound value="$sound_speed"/>
      <speedsound value="$sound_speed" auto="false"/><b value="$(1000sound_speed^2/7)" auto="false"/>
      <coefh value="1"/><cflnumber value="0.2"/>
    </constantsdef><mkconfig boundcount="240" fluidcount="9"/>
    <geometry><definition dp="$dx"><pointref x="$(dx/2)" y="0" z="$(dx/2)"/>
      <pointmin x="$(-l-dx)" y="0" z="$(-l-dx)"/><pointmax x="$(10+l+dx)" y="0" z="$(5+l+dx)"/>
    </definition><commands><mainlist><setshapemode>actual | dp | bound</setshapemode>
      <setdrawmode mode="full"/><setmkbound mk="0"/>$walls
      <setmkbound mk="10"/>$(box(1,2,1,1))
      <setmkfluid mk="0"/>$fluid
    </mainlist></commands></geometry>
    <motion><objreal ref="10"><begin mov="1" start="0"/>
      <mvfile id="1" duration="8.1"><file name="$(replace(motion_path,'\\'=>'/'))" fields="2" fieldtime="0" fieldx="1"/></mvfile>
    </objreal></motion></casedef><execution><parameters>
      <parameter key="SavePosDouble" value="1"/><parameter key="StepAlgorithm" value="2"/>
      <parameter key="Kernel" value="2"/><parameter key="ViscoTreatment" value="3"/>
      <parameter key="Visco" value="$(1/reynolds)"/><parameter key="ViscoBoundFactor" value="1"/>
      <parameter key="DensityDT" value="$diffusion"/><parameter key="DensityDTvalue" value="0.1"/>
      <parameter key="Shifting" value="0"/><parameter key="RigidAlgorithm" value="1"/>
      <parameter key="CoefDtMin" value="0.01"/><parameter key="DtIni" value="$(0.2sqrt(2)*dx/sound_speed)"/>
      <parameter key="TimeMax" value="$duration"/><parameter key="TimeOut" value="$interval"/>
      <parameter key="PartsOutMax" value="0.001"/><parameter key="RhopOutMin" value="700"/>
      <parameter key="RhopOutMax" value="1300"/>
    </parameters></execution></case>
    """
    definition = joinpath(directory,"Case_Def")
    write(definition*".xml", xml)
    logged(`$GEN $definition $prefix -save:all`, joinpath(directory,"gencase.log"))
    return prefix
end

function read_dual_csv(path)
    lines = readlines(path)
    header = findfirst(line -> startswith(line,"Pos.x"), lines)
    header === nothing && (header = findfirst(line -> startswith(line,"Part;Time"), lines))
    header === nothing && error("CSV header not found: $path")
    return CSV.File(IOBuffer(join(lines[header:end],'\n')); delim = ';', normalizenames = false)
end

function verify_geometry(prefix, n)
    initial = prefix*"_initial.csv"
    isfile(initial) || logged(`$PART -filedata $(prefix*".bi4") -savecsv $initial -vars:-all,+idp,+mk,+type,+rhop,+vel`,prefix*"_initial.log")
    rows = read_dual_csv(initial)
    case = build_case(dx = 1/n)
    for (type,group) in ((0,1),(1,3),(3,2))
        selected = filter(row -> getproperty(row,:Type) == type, collect(rows))
        nodes = Set((round(Int,row[Symbol("Pos.x [m]")]*n-0.5),round(Int,row[Symbol("Pos.z [m]")]*n-0.5)) for row in selected)
        target = Set((round(Int,p[1]*n-0.5),round(Int,p[2]*n-0.5)) for p in case.geometry[group].Particles.Position)
        nodes == target && length(selected) == length(target) || error("GenCase lattice differs for type $type: $(length(selected)) vs $(length(target))")
        maximum(max(abs(row[Symbol("Pos.x [m]")]*n-0.5-round(row[Symbol("Pos.x [m]")]*n-0.5)),
            abs(row[Symbol("Pos.z [m]")]*n-0.5-round(row[Symbol("Pos.z [m]")]*n-0.5))) for row in selected)/n < 2e-6 ||
            error("GenCase coordinate residual exceeds CSV precision tolerance")
    end
    return true
end

function dual_run(prefix, directory; postprocess = true, reynolds = 100)
    isfile(joinpath(directory,"Run.csv")) && error("Refusing to overwrite $directory")
    mkpath(directory)
    wall = @elapsed logged(`$DUAL -gpu $prefix $directory -dirdataout data -svres -svtimers:0`, joinpath(directory,"console.log"))
    lines = readlines(joinpath(directory,"Run.csv"))
    fields = Dict(zip(split(lines[1],';'), split(lines[2],';')))
    number(key) = parse(Float64,replace(fields[key],","=>""))
    number("PartsOut") == 0 || error("DualSPHysics lost particles")
    result = (; wall_seconds = wall, solver_seconds = number("TSimul"), steps = Int(number("Steps")),
        particles = Int(number("Np")), frames = Int(number("PartFiles")))
    if postprocess
        for (label, viscosity) in (("pressure",0), ("total",1/reynolds))
            logged(`$FORCES -dirdata $(joinpath(directory,"data")) -onlymk:20 -viscolam:$viscosity:1 -threads:8 -savecsv $(joinpath(directory,label))`, joinpath(directory,label*".log"))
        end
    end
    return result
end

function config(id; reynolds = 100, sound_speed = 40, motion_kind = :hermite,
        diffusion = 1, eta = :native, float_type = Float32)
    return (; id, reynolds, sound_speed, motion_kind, diffusion, eta, float_type, n = 40)
end

function native_case(configuration, directory)
    (; reynolds, sound_speed, motion_kind, diffusion, eta, float_type, n) = configuration
    case = build_case(; reynolds, sound_speed, dx = 1/n, save = directory, float_type)
    trajectory = motion_kind == :hermite ? body_trajectory() : integrated_trajectory()
    kernel = eta == :native ? case.kernel : SPHKernelInstance{2,float_type}(WendlandC2();
        h = case.kernel.h, η² = float_type((0.1case.kernel.h)^2))
    return merge(case,(; kernel, motion = PrescribedMotion(3; translation = trajectory)))
end

function native_run(configuration, directory; analyze = true, results = joinpath(RESULTS,configuration.id))
    mkpath(directory)
    isfile(joinpath(directory,"MovingSquare.vtkhdf")) && error("Refusing to overwrite $directory")
    case = native_case(configuration,directory)
    particles = AllocateDataStructures(case.geometry,case.metadata)
    logger = SimulationLogger(directory)
    diffusion = configuration.diffusion == 0 ? ZeroDensityDiffusion() :
        configuration.diffusion == 2 ? FourtakasDensityDiffusion() : ZeroGravityLinearDensityDiffusion()
    result = nothing
    wall = open(joinpath(directory,"console.log"),"w") do io
        redirect_stdout(io) do
            @elapsed try
                result = RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
                    SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
                    SimParticles = particles, SimViscosity = Laminar(), SimDensityDiffusion = diffusion,
                    SimTimeStepping = SymplecticTimeStepping(), SimMotion = case.motion)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
        end
    end
    result.final_state.complete && result.diagnostics.frozen_particles == 0 || error("Incomplete native run")
    if analyze
        mkpath(results)
        analyze_history(joinpath(directory,"MovingSquare.vtkhdf"),case;
            coefficient_path = joinpath(results,"native.dat"), diagnostics_path = joinpath(results,"native_state.csv"),
            trajectory = case.motion.translation)
    end
    order = sortperm(particles.ID)
    analyze || CSV.write(joinpath(directory,"final.csv"),[(; id = particles.ID[i],x = particles.Position[i][1],
        y = particles.Position[i][2],u = particles.Velocity[i][1],v = particles.Velocity[i][2],rho = particles.Density[i]) for i in order])
    return (; wall_seconds = wall, solver_seconds = result.performance.wall_time_seconds,
        steps = result.performance.steps, particles = result.diagnostics.particle_count,
        frames = result.diagnostics.output_frames, rebuilds = result.diagnostics.cell_list_rebuilds,
        readbacks = result.diagnostics.step_state_readbacks)
end

function dual_final_state(directory)
    path = joinpath(directory,"final_0400.csv")
    isfile(path) || logged(`$PART -dirdata $(joinpath(directory,"data")) -files:400 -savecsv $(joinpath(directory,"final")) -vars:-all,+idp,+vel,+rhop,+type,+mk`,joinpath(directory,"final.log"))
    return sort(collect(read_dual_csv(path)); by = row -> row.Idp)
end

function timing_study(; repeats = 3, aligned = false)
    repeats >= 2 || error("At least two pairs are required")
    records = NamedTuple[]
    state_reference = Dict{Tuple{Int,String},Matrix{Float64}}()
    timing_group = aligned ? "timings_aligned" : "timings"
    mkpath(joinpath(WORK,timing_group))
    warm = mktempdir(joinpath(WORK,timing_group); prefix = "warm_native_")
    println("Full-duration native timing-process warmup"); flush(stdout)
    warm_configuration = aligned ? config("re100_dual_operators"; eta = :dual,diffusion = 2) : config("re50_c40"; reynolds = 50)
    native_run(warm_configuration,warm; analyze = false)
    for reynolds in (aligned ? (100,) : (50,100,150))
        configuration = aligned ? config("re100_dual_operators"; eta = :dual,diffusion = 2) : config("re$(reynolds)_c40"; reynolds)
        prefix = prepare_dual(; reynolds)
        verify_geometry(prefix,40)
        for repetition in 1:repeats
            order = isodd(repetition) ? ("native","dual") : ("dual","native")
            for (slot,solver) in enumerate(order)
                directory = joinpath(WORK,timing_group,"re$(reynolds)_$(solver)_$(repetition)")
                info_path = joinpath(directory,"timing.toml")
                if isfile(info_path)
                    info = TOML.parsefile(info_path)
                else
                    println("Timing Re=",reynolds," / ",solver," / pair ",repetition); flush(stdout)
                    GC.gc(true); CUDA.reclaim()
                    result = solver == "native" ? native_run(configuration,directory; analyze = false) :
                        dual_run(prefix,directory; postprocess = false,reynolds)
                    save_info(info_path,result)
                    info = TOML.parsefile(info_path)
                end
                if solver == "native"
                    data = CSV.File(joinpath(directory,"final.csv"))
                    state = hcat(data.x,data.y,data.u,data.v,data.rho)
                else
                    data = dual_final_state(directory)
                    state = hcat([Float64[getproperty(row,Symbol(key)) for row in data] for key in
                        ("Pos.x [m]","Pos.z [m]","Vel.x [m/s]","Vel.z [m/s]","Rhop [kg/m^3]")]...)
                end
                previous = get!(state_reference,(reynolds,solver),copy(state))
                size(state) == size(previous) == (84864,5) || error("Repeat particle count changed")
                push!(records,(; reynolds,solver,repetition,order = slot,wall_seconds = info["wall_seconds"],
                    solver_seconds = info["solver_seconds"],steps = info["steps"],frames = info["frames"],particles = info["particles"],
                    rebuilds = get(info,"rebuilds",missing),readbacks = get(info,"readbacks",missing),
                    repeat_position_max = maximum(abs,state[:,1:2]-previous[:,1:2]),
                    repeat_velocity_max = maximum(abs,state[:,3:4]-previous[:,3:4]),
                    repeat_density_max = maximum(abs,state[:,5]-previous[:,5])))
                CSV.write(joinpath(RESULTS,timing_group*".csv"),records)
            end
        end
    end
    return records
end

function extract_dual(directory, results)
    pressure, total = read_dual_csv(joinpath(directory,"pressure.csv")), read_dual_csv(joinpath(directory,"total.csv"))
    time = Float64.(getproperty(pressure,Symbol("Time [s]")))
    force_p = Float64.(getproperty(pressure,Symbol("ForceFluid.x [N/m]")))
    force_t = Float64.(getproperty(total,Symbol("ForceFluid.x [N/m]")))
    count = Int.(getproperty(pressure,:Np))
    all(==(1600),count) || error("The force selection is not the 1600-particle square")
    all(isfinite,time) && all(isfinite,force_p) && all(isfinite,force_t) && all(>(0),diff(time)) ||
        error("Invalid or nonfinite Dual force history")
    mkpath(results)
    CSV.write(joinpath(results,"dual.csv"),[(; time = time[i],cd_pressure = -2force_p[i]/1000,
        cd_viscous = -2(force_t[i]-force_p[i])/1000) for i in eachindex(time)])
end

function analyze_pair(configuration)
    directory = joinpath(RESULTS,configuration.id)
    reference = read_reference(joinpath(REFERENCE_DIR,@sprintf("Force_Re%03d.dat",configuration.reynolds)))
    summary = NamedTuple[]
    spectra = NamedTuple[]
    for solver in ("native","dual")
        path = joinpath(directory,solver*".csv")
        isfile(path) || continue
        data = CSV.File(path)
        metrics, spectrum = drag_metrics(Float64.(data.time),Float64.(data.cd_pressure),Float64.(data.cd_viscous),reference)
        info = TOML.parsefile(joinpath(directory,solver*".toml"))
        push!(summary,(; case = configuration.id, solver, reynolds = configuration.reynolds,
            sound_speed = configuration.sound_speed, motion = String(configuration.motion_kind), diffusion = configuration.diffusion,
            pressure_nrmse = metrics["pressure_nrmse"],viscous_nrmse = metrics["viscous_nrmse"],
            pressure_raw_nrmse = metrics["pressure_raw_nrmse"],viscous_raw_nrmse = metrics["viscous_raw_nrmse"],
            pressure_noise_rms = metrics["pressure_highpass_rms"],viscous_noise_rms = metrics["viscous_highpass_rms"],
            pressure_dominant_hz = metrics["pressure_dominant_hz"],viscous_dominant_hz = metrics["viscous_dominant_hz"],
            pressure_late_bias = metrics["pressure_late_bias"],viscous_late_bias = metrics["viscous_late_bias"],
            steps = info["steps"],frames = info["frames"],wall_seconds = info["wall_seconds"]))
        append!(spectra,[(; solver,row...) for row in spectrum])
    end
    CSV.write(joinpath(directory,"metrics.csv"),summary)
    CSV.write(joinpath(directory,"spectra.csv"),spectra)
    return summary
end

function save_info(path,result)
    open(path,"w") do io
        TOML.print(io,Dict(string(k)=>v for (k,v) in pairs(result)); sorted = true)
    end
end

function run_configuration(configuration; solvers = ("native","dual"))
    results = mkpath(joinpath(RESULTS,configuration.id))
    for solver in solvers
        isfile(joinpath(results,solver*".toml")) && continue
        directory = joinpath(WORK,configuration.id,solver)
        println("Running ",configuration.id," / ",solver); flush(stdout)
        if solver == "native"
            result = native_run(configuration,directory; results)
        else
            prefix = prepare_dual(; n = configuration.n,reynolds = configuration.reynolds,
                sound_speed = configuration.sound_speed,motion_kind = configuration.motion_kind,diffusion = configuration.diffusion)
            verify_geometry(prefix,configuration.n)
            result = dual_run(prefix,directory; reynolds = configuration.reynolds)
            extract_dual(directory,results)
        end
        save_info(joinpath(results,solver*".toml"),result)
        GC.gc(true); CUDA.reclaim()
    end
    return analyze_pair(configuration)
end

function study()
    verify_reference()
    all(isfile,(DUAL,GEN,FORCES,PART)) || error("Installed DualSPHysics tools not found")
    configurations = [config("re$(re)_c40"; reynolds = re) for re in (50,100,150)]
    append!(configurations,[config("re100_c$speed"; sound_speed = speed) for speed in (80,160)])
    push!(configurations,config("re100_integrated"; motion_kind = :integrated))
    push!(configurations,config("re100_no_ddt"; diffusion = 0))
    rows = NamedTuple[]
    for configuration in configurations
        append!(rows,run_configuration(configuration))
        CSV.write(joinpath(RESULTS,"summary.csv"),rows)
    end
    for configuration in (config("re100_dual_eta"; eta = :dual),
            config("re100_dual_operators"; eta = :dual,diffusion = 2),
            config("re100_float64"; float_type = Float64))
        append!(rows,run_configuration(configuration; solvers = ("native",)))
        CSV.write(joinpath(RESULTS,"summary.csv"),rows)
    end
    return rows
end

function pilot()
    prefix = prepare_dual(; duration = 0.1, interval = 0.02)
    verify_geometry(prefix,40)
    directory = joinpath(WORK,"pilot_centered")
    result = dual_run(prefix,directory)
    println(result)
    logged(`$PART -filedata $(prefix*".bi4") -savecsv $(joinpath(directory,"initial")) -vars:-all,+idp,+mk,+type,+rhop,+vel`, joinpath(directory,"initial.log"))
    println("Pilot: ",directory)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        Test06DualComparison.study()
    elseif first(ARGS) == "timing"
        Test06DualComparison.timing_study()
    elseif first(ARGS) == "timing_aligned"
        Test06DualComparison.timing_study(aligned = true)
    else
        Test06DualComparison.pilot()
    end
end
