include(joinpath(@__DIR__,"sound_speed_160.jl"))
using .Test06Sound160, SPHExampleGPU, CUDA, HDF5, CSV, StaticArrays, LinearAlgebra
const S = Test06Sound160
const GK = SPHExampleGPU.GPUKernels

function audit_snapshots()
    records = NamedTuple[]
    for speed in (40,160), re in (50,100,150)
        println("Timestep snapshot audit: Re=",re," c0=",speed); flush(stdout)
        variant = speed == 40 ? "baseline40" : "standard160"
        run_directory = speed == 40 || re == 100 ? joinpath(S.C.WORK,"re$(re)_c$(speed)","native") :
            joinpath(S.WORK,"re$(re)_standard160","native")
        case = S.C.build_case(reynolds=re,sound_speed=speed,float_type=Float64)
        host = AllocateDataStructures(case.geometry,case.metadata)
        gpu = upload_particles(host,case.metadata)
        support = GPUSupportArrays{2,Float64}(gpu,case.metadata)
        cells = CellListWorkspace{2,Float64}(length(gpu))
        pair_rate = CUDA.zeros(Float64,length(gpu))
        kernel,constants = case.kernel,case.constants
        h5open(joinpath(run_directory,"MovingSquare.vtkhdf"),"r") do file
            root = file["VTKHDF"]
            for frame in 1:10:length(root["Steps/Values"])
                offset,count = root["Steps/PointOffsets"][frame],root["NumberOfPoints"][frame]
                columns = offset+1:offset+count
                copyto!(gpu.Position,[SVector{2,Float64}(x[1],x[2]) for x in eachcol(root["Points"][:,columns])])
                copyto!(gpu.Velocity,[SVector{2,Float64}(x[1],x[2]) for x in eachcol(root["PointData/Velocity"][:,columns])])
                copyto!(gpu.Density,Float64.(root["PointData/Density"][columns]))
                copyto!(gpu.Type,ParticleType.(root["PointData/Type"][columns]))
                S.E.DR.rebuild_cell_list!(gpu,cells,kernel.H⁻¹; support,kernel)
                GK.launch_interactions!(support.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,support.∇Cᵢ,support.∇◌rᵢ,
                    gpu.Position,gpu.Density,gpu.Velocity,gpu.Type,cells.CellStart,gpu.CellID,cells.grid_dev,HostStep(0.,0.),
                    ZeroGravityLinearDensityDiffusion(),Laminar(),kernel,constants,Val(false),Val(false);
                    threads=128,lanes=Val(4),boundary_forces=Val(false),viscous_dt=pair_rate,fastmath=false)
                x,v,rho = Array(gpu.Position),Array(gpu.Velocity),Array(gpu.Density)
                legacy = maximum(abs(kernel.h*dot(a,b)/(dot(b,b)+kernel.η²)) for (a,b) in zip(v,x))
                pair = maximum(Array(pair_rate))
                vmax = maximum(norm,v)
                cmax = maximum(speed*(r/1000)^3 for r in rho)
                push!(records,(; reynolds=re,c0=speed,time=Float64(root["Steps/Values"][frame]),
                    nominal_acoustic_cap=0.2kernel.h/speed,legacy_acoustic_cap=0.2kernel.h/(speed+legacy),
                    pair_acoustic_cap=0.2kernel.h/(speed+kernel.h*pair),
                    dual_style_cap=0.2kernel.h/(max(speed,10vmax)+kernel.h*pair),
                    local_sound_pair_cap=0.2kernel.h/(cmax+kernel.h*pair),maximum_speed=vmax,
                    maximum_local_sound_speed=cmax,pair_compression_rate=pair))
            end
        end
        GC.gc(true); CUDA.reclaim()
    end
    directory = mkpath(joinpath(@__DIR__,"results","timestep_audit"))
    CSV.write(joinpath(directory,"snapshot_limits.csv"),records)
end
function write_scale_estimates()
    h = sqrt(2)*0.025
    eta = 0.01h
    velocity = SVector(1.,0.)
    near,shifted = SVector(eta,0.),SVector(100+eta,0.)
    near_rate = GK.step_values(near,velocity,zero(velocity),near,h,eta^2)[1]
    shifted_rate = GK.step_values(shifted,velocity,zero(velocity),shifted,h,eta^2)[1]
    rows = [(; c0=speed,reynolds=re,h,acoustic_dt_cfl02=0.2h/speed,
        h2_over_8nu_diagnostic=h^2/(8/re),legacy_dt_near_origin=0.2h/(speed+near_rate),
        legacy_dt_shifted_100m=0.2h/(speed+shifted_rate)) for speed in (40,160) for re in (50,100,150)]
    directory = mkpath(joinpath(@__DIR__,"results","timestep_audit"))
    CSV.write(joinpath(directory,"origin_and_viscosity.csv"),rows)
end

write_scale_estimates()
if isempty(ARGS)
    audit_snapshots()
elseif ARGS != ["scales"]
    error("Expected no arguments or scales")
end
