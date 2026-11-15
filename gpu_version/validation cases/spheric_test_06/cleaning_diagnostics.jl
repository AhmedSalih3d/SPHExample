include(joinpath(@__DIR__,"divergence_cleaning.jl"))
using .Test06DivergenceCleaning
using SPHExampleGPU, CUDA, HDF5, CSV, StaticArrays, Statistics, LinearAlgebra

function divergence_diagnostics()
    E = Test06DivergenceCleaning
    C,DC,DR = E.C,E.DC,E.DR
    rows = NamedTuple[]
    # Evaluate the raw velocity-difference divergence in Float64. This is an
    # offline operator measurement, with zero psi and no density diffusion.
    for variant in ("off","on"), re in (50,100,150)
        id = "re$(re)_$(variant)"
        println("Divergence diagnostics: ",id); flush(stdout)
        case = C.build_case(reynolds=re,float_type=Float64)
        source_kernel = C.build_case(reynolds=re).kernel
        kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h=Float64(source_kernel.h),
            αD=Float64(source_kernel.αD),η²=Float64(source_kernel.η²))
        host = AllocateDataStructures(case.geometry,case.metadata)
        gpu = upload_particles(host,case.metadata)
        sup = GPUSupportArrays{2,Float64}(gpu,case.metadata)
        cl = CellListWorkspace{2,Float64}(length(gpu))
        scalar = DC.CleaningWorkspace(DivergenceCleaning(sigma=0),length(gpu),Float64)
        # c0=1 makes the returned scalar source equal -div(u); physical
        # pressure/acceleration is discarded and the recorded state is untouched.
        constants = SimulationConstants{Float64}(2; dx=0.025,c₀=1,g=0)
        ctx = (; gpu,sup,cl,SimKernel=kernel,SimConstants=constants,
            SimDensityDiffusion=ZeroDensityDiffusion(),SimViscosity=ZeroViscosity(),
            FlagKernel=Val(false),FlagShift=Val(false),threads=128,lanes=Val(4),bforces=Val(false),
            divergence_cleaning=scalar,interaction_fastmath=false)
        local_rows = NamedTuple[]
        h5open(joinpath(E.WORK,id,"MovingSquare.vtkhdf"),"r") do file
            root = file["VTKHDF"]
            times = read(root["Steps/Values"])
            offsets = read(root["Steps/PointOffsets"])
            counts = read(root["NumberOfPoints"])
            for frame in 1:10:length(times)
                columns = offsets[frame]+1:offsets[frame]+counts[frame]
                positions = [SVector{2,Float64}(x[1],x[2]) for x in eachcol(root["Points"][:,columns])]
                velocities = [SVector{2,Float64}(x[1],x[2]) for x in eachcol(root["PointData/Velocity"][:,columns])]
                copyto!(gpu.Position,positions)
                copyto!(gpu.Velocity,velocities)
                copyto!(gpu.Density,Float64.(root["PointData/Density"][columns]))
                copyto!(gpu.Type,ParticleType.(root["PointData/Type"][columns]))
                DR.rebuild_cell_list!(gpu,cl,kernel.H⁻¹; support=sup,kernel)
                DR.launch_solver_interactions!(ctx,gpu.Position,gpu.Density,gpu.Velocity,sup.PosCells,HostStep(0.,times[frame]))
                fluid = Array(gpu.Type) .== Fluid
                divergence = -Array(scalar.derivative)[fluid]
                volume = constants.m₀ ./ Array(gpu.Density)[fluid]
                row = (; case=id,reynolds=re,variant,time=times[frame],
                    divergence_rms=sqrt(mean(abs2,divergence)),
                    divergence_volume_rms=sqrt(sum(volume .* abs2.(divergence))/sum(volume)),
                    divergence_abs_max=maximum(abs,divergence))
                push!(local_rows,row); push!(rows,row)
            end
        end
        CSV.write(joinpath(E.RESULTS,id,"divergence.csv"),local_rows)
        GC.gc(true); CUDA.reclaim()
    end
    CSV.write(joinpath(E.RESULTS,"divergence.csv"),rows)
    return rows
end

divergence_diagnostics()
