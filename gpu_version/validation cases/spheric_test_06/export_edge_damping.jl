include(joinpath(@__DIR__,"edge_damping.jl"))
using .Test06EdgeDamping
using SPHExampleGPU, CSV, TOML, LinearAlgebra

function export_edge_damping()
    E = Test06EdgeDamping
    C = E.C
    data = collect(CSV.File(joinpath(E.RESULTS,"summary.csv")))
    length(data) == 12 || error("All twelve damping runs must complete before export")
    center = 1.5+first(trajectory_state(C.body_trajectory(),8.))[1]
    open(joinpath(E.RESULTS,"geometry.toml"),"w") do io
        TOML.print(io,Dict("square_final_center_x"=>center,"square_to_sponge_final_gap"=>9.5-center-0.5,
            "kernel_support_radius"=>2sqrt(2)*0.025,"initial_sponge_fluid_particles"=>22400,
            "initial_fluid_particles"=>78400); sorted = true)
    end
    checks = NamedTuple[]
    for rate in (10,100), re in (50,100,150)
        id = "re$(re)_beta$(rate)"
        directory = joinpath(E.WORK,id,"dual")
        results = joinpath(E.RESULTS,id)
        lines = readlines(joinpath(directory,"Run.out"))
        start = findfirst(==("Damping configuration:"),lines)
        start === nothing && error("Missing actual damping configuration for $id")
        configuration = join(rstrip.(lines[start:start+7]),'\n')*"\n"
        for text in ("(type: Box)","[top, bottom, left, right]","(0.5,-1,0.5)-(9.5,1,4.5)",
                "(0,-1,0)-(10,1,5)","Overlimit.....: 0.100000","ReduMax.......: $rate","Factorxyz.....: (1,0,1)")
            occursin(text,configuration) || error("Unexpected Dual damping configuration: $text")
        end
        write(joinpath(results,"dual_damping_configuration.txt"),configuration)
        write(joinpath(results,"dual_damping.xml"),C.edge_damping_xml((; E.SETTINGS...,rate = Float64(rate))))
        # RunPARTs records counts at every output, independently of the force tool.
        parts = CSV.File(joinpath(directory,"RunPARTs.csv"); delim = ';',types = String,comment = "#",ignoreemptyrows = true)
        number(row,key) = parse(Float64,replace(getproperty(row,Symbol(key)),","=>""))
        length(parts) == 401 || error("Missing Dual output records")
        all(row -> number(row,"NpSave") == number(row,"NpSim") == number(row,"NpNormal") == 84864 && number(row,"NpOut") == 0,parts) ||
            error("Dual output lost or changed particle types")
        native = CSV.File(joinpath(results,"native_state.csv"))
        length(native) == 401 && all(isfinite,native.density_relative_max) && all(isfinite,native.max_speed) || error("Invalid native state history")
        push!(checks,(; reynolds = re,rate,native_frames = length(native),dual_frames = length(parts),
            native_fluid_count = minimum(native.fluid_particles),native_square_count = minimum(native.square_particles),
            dual_particles = 84864,dual_particles_out = 0,
            native_maximum_density_deviation = maximum(native.density_relative_max),
            native_maximum_mach = maximum(native.max_speed)/40,
            native_minimum_solid_clearance = minimum(native.minimum_clearance),
            native_maximum_motion_error = maximum(native.motion_error),native_maximum_velocity_error = maximum(native.velocity_error)))
        if rate == 100
            path = joinpath(directory,"fields_0400.csv")
            isfile(path) || C.logged(`$(C.PART) -dirdata $(joinpath(directory,"data")) -files:400 -savecsv $(joinpath(directory,"fields")) -onlytype:-all,+fluid,+moving -vars:-all,+idp,+vel,+rhop,+type,+mk`,joinpath(directory,"fields.log"))
            fields = C.read_dual_csv(path)
            count(==(3),fields.Type) == 78400 && count(==(1),fields.Type) == 1600 || error("Changed Dual final-state counts")
            all(isfinite,getproperty(fields,Symbol("Rhop [kg/m^3]"))) || error("Nonfinite Dual final density")
        end
    end
    CSV.write(joinpath(E.RESULTS,"state_checks.csv"),checks)
    return checks
end

export_edge_damping()
