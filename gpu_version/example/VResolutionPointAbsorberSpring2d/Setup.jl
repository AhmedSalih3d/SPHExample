# Definitions only. The sibling entry point runs the full 06_Special case.
using SPHExampleGPU, StaticArrays, CSV
using LinearAlgebra: normalize

const POINT_ABSORBER_DATA = joinpath(@__DIR__,"..","data","vresolution_special")

"Second-order regular piston motion, with the reference's smooth one-period ramp."
function point_absorber_piston_state(t;depth=0.4,height=0.1,period=1.2,gravity=9.81)
    omega = 2pi/period
    k = omega^2/gravity
    for _ in 1:50
        k -= (gravity*k*tanh(k*depth)-omega^2)/(gravity*tanh(k*depth)+gravity*k*depth*sech(k*depth)^2)
    end
    transfer = 2sinh(k*depth)^2/(sinh(k*depth)*cosh(k*depth)+k*depth)
    a1 = height/(2transfer)
    a2 = height^2/(32depth)*(3cosh(k*depth)/sinh(k*depth)^3-2/transfer)
    s = clamp(t/period,0.0,1.0)
    ramp,rate = s^2*(6-8s+3s^2),0 < t < period ? 12s*(1-s)^2/period : 0.0
    displacement = a1*sin(omega*t)+a2*sin(2omega*t)
    velocity = a1*omega*cos(omega*t)+2a2*omega*cos(2omega*t)
    return ramp*displacement,rate*displacement+ramp*velocity
end

"""
    point_absorber_spring_case(; dx=0.01, ratio=2, duration=10,
        output_interval=0.05, FloatType=Float32, ...)

Build `vresolution/06_Special/CasePointAbsorberSpringVRes` using the ordinary
Julia solver settings: gamma 7, Wendland h=1.5dp, artificial viscosity 0.05,
linear density diffusion 0.1, CFL 0.2, no shifting and DBC. The 0.3 by 0.16 m
body has density 500 kg/m³, heave-only motion and a 100 N/m, 500 Ns/m spring.
The coarse piston drives 0.1 m high, 1.2 s waves in 0.4 m water. Construction
does not run either solver. Existing DualSPHysics output is bundled separately.
"""
function point_absorber_spring_case(;dx=0.01,ratio=2,duration=10.0,output_interval=0.05,FloatType=Float32,
        h_coefficient=1.5,viscosity=0.05,diffusion=0.1,CFL=0.2,
        stiffness=100.0,damping=500.0,rest_length=0.325,
        region=VariableResolutionBox((1.7,0.2),(2.3,0.6);ratio),
        save_location=joinpath(@__DIR__,"..","..","output","vresolution_point_absorber_spring2d"))
    T = FloatType
    T in (Float32,Float64) || throw(ArgumentError("FloatType must be Float32 or Float64"))
    isfinite(dx) && 0 < dx <= 0.02 || throw(ArgumentError("dx must satisfy 0 < dx <= 0.02; enlarge region when using coarse test spacings"))
    isfinite(duration) && duration > 0 && isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("duration and output_interval must be positive and finite"))
    ratio == region.ratio || throw(ArgumentError("ratio must match the refinement region"))
    depth,gravity = 0.4,9.81
    sound_speed = 20sqrt(gravity*depth)
    anchor,attachment = SVector(2.0,0.0),SVector(2.0,0.325)
    spring = LinearSpring{2,T}(;Anchor=anchor,Attachment=attachment,Stiffness=stiffness,Damping=damping,RestLength=rest_length)
    constraints = FloatingConstraints{2,T}(;TranslationLocks=(true,false),RotationLocks=(true,))
    bed(x) = x <= 2.5 ? 0.0 : (x-2.5)*0.6/6.7
    function level(spacing,fine)
        constants = SimulationConstants{T}(2;dx=spacing,c₀=sound_speed,γ=7,g=gravity,α=viscosity,δᵩ=diffusion,CFL)
        h = h_coefficient*spacing
        kernel = SPHKernelInstance{2,T}(WendlandC2();h=T(h),η²=T((0.1h)^2))
        if !fine && isapprox(spacing,0.01;rtol=1e-10)
            # Preserve the already generated coarse sloping skin; no GenCase call.
            fixed = [SVector(Float64(r.x),Float64(r.y)) for r in CSV.File(joinpath(POINT_ABSORBER_DATA,"BoundarySection.csv"))]
        else
            fixed = SVector{2,Float64}[]
            for i in ceil(Int,-0.2/spacing):floor(Int,9.2/spacing), layer in 0:2
                x = i*spacing
                normal = x <= 2.5 ? SVector(0.0,1.0) : normalize(SVector(-0.6/6.7,1.0))
                push!(fixed,SVector(x,bed(x))-layer*spacing*normal)
            end
        end
        lo = fine ? region.lower .- region.buffer_h*kernel.h : SVector(0.0,0.0)
        hi = fine ? region.upper .+ region.buffer_h*kernel.h : SVector(9.2,depth)
        water = SVector{2,Float64}[]
        for i in max(1,ceil(Int,lo[1]/spacing)):floor(Int,hi[1]/spacing), j in max(1,ceil(Int,lo[2]/spacing)):(ceil(Int,depth/spacing)-1)
            x,y = i*spacing,j*spacing
            y <= hi[2] && y >= bed(x)+spacing-1e-10 || continue
            1.85-1e-10 <= x <= 2.15+1e-10 && 0.32-1e-10 <= y <= 0.48+1e-10 && continue
            push!(water,SVector(x,y))
        end
        boundary = SPHGeometry{2,T}(fixed;Density=constants.ρ₀,Type=Fixed,GroupMarker=1)
        fluid = SPHGeometry{2,T}(water;Density=hydrostatic_density(water,constants;water_level=depth),Type=Fluid,GroupMarker=2)
        geometry = [boundary,fluid]
        if fine
            nx,ny = round(Int,0.3/spacing),round(Int,0.16/spacing)
            isapprox(nx*spacing,0.3;atol=1e-10) && isapprox(ny*spacing,0.16;atol=1e-10) ||
                throw(ArgumentError("fine spacing must divide the body's 0.3 m width and 0.16 m height"))
            body = [SVector(1.85+i*spacing,0.32+j*spacing) for i in 0:nx for j in 0:ny]
            push!(geometry,SPHGeometry{2,T}(body;Density=constants.ρ₀,Type=Floating,GroupMarker=50,
                Floating=FloatingDetails{T}(;RelativeWeight=T(0.5),Constraints=constraints,Springs=(spring,))))
        else
            nx,ny = round(Int,0.05/spacing),round(Int,0.6/spacing)
            piston = [SVector(-i*spacing,j*spacing) for i in 0:nx for j in 1:ny]
            push!(geometry,SPHGeometry{2,T}(piston;Density=constants.ρ₀,Type=Moving,GroupMarker=10))
        end
        return (;geometry,constants,kernel)
    end
    coarse,fine = level(dx,false),level(dx/ratio,true)
    # Exact derivatives of the sampled displacement give smooth wall velocities.
    times = collect(range(0.0,duration+0.002;step=0.002))
    states = point_absorber_piston_state.(times)
    trajectory = SampledTrajectory(times,[(s[1],0.0) for s in states];velocities=[(s[2],0.0) for s in states])
    motion = PrescribedMotion(10;translation=trajectory)
    return (;coarse,fine,region,motion,spring,simulation_time=Float64(duration),output_interval=Float64(output_interval),
        save_location=abspath(save_location),depth,sound_speed)
end

function point_absorber_geometry(case,heave=0.0)
    shift = SVector(0.0,heave)
    lower = case.region.lower+shift
    tank = polyline([(-0.2,0.0),(2.5,0.0),(9.2,0.6)];thickness=0.02,side=:right)
    body = rectangle((1.85,0.32+heave),0.3,0.16)
    patch = rectangle(Tuple(lower),case.region.upper[1]-case.region.lower[1],case.region.upper[2]-case.region.lower[2])
    spring = polyline([Tuple(case.spring.Anchor),Tuple(case.spring.Attachment+shift)];thickness=0.003)
    return (;tank,body,refinement=patch,spring)
end

function point_absorber_outputs(case,result)
    body_rows = collect(CSV.File(joinpath(case.save_location,"PointAbsorberSpring2D_Floating.csv")))
    spring_rows = collect(CSV.File(joinpath(case.save_location,"PointAbsorberSpring2D_Springs.csv")))
    times = Float64[r.Time for r in body_rows]
    function interpolate(values,time)
        length(times) == 1 && return first(values)
        i = clamp(searchsortedlast(times,time),1,length(times)-1)
        weight = (time-times[i])/(times[i+1]-times[i])
        return (1-weight)*values[i]+weight*values[i+1]
    end
    native_y = Float64[getproperty(r,Symbol("Center:1")) for r in body_rows]
    native_v = Float64[getproperty(r,Symbol("Velocity:1")) for r in body_rows]
    native_f = Float64[getproperty(r,Symbol("Force:1")) for r in spring_rows]
    open(joinpath(case.save_location,"PointAbsorberSpring2D_Comparison.csv"),"w") do io
        println(io,"time,native_center_y,reference_center_y,native_velocity_y,reference_velocity_y,native_spring_force_y,reference_spring_force_y")
        for r in CSV.File(joinpath(POINT_ABSORBER_DATA,"FloatingReference.csv"))
            first(times) <= r.time <= last(times) || continue
            println(io,join((r.time,interpolate(native_y,r.time),r.center_y,interpolate(native_v,r.time),r.velocity_y,
                interpolate(native_f,r.time),r.spring_force_y),','))
        end
    end
    files = String[]
    for (i,r) in enumerate(body_rows)
        filename = "PointAbsorberSpring2D_Link_"*lpad(string(i),6,'0')*".vtkhdf"
        SavePolygonVTKHDF(joinpath(case.save_location,filename),point_absorber_geometry(case,getproperty(r,Symbol("Center:1"))-0.4))
        push!(files,filename)
    end
    open(joinpath(case.save_location,"PointAbsorberSpring2D_Link.vtkhdf.series"),"w") do io
        println(io,"{\"file-series-version\":\"1.0\",\"files\":[")
        for i in eachindex(files)
            println(io,"{\"name\":\"",files[i],"\",\"time\":",times[i],"}",i == lastindex(files) ? "" : ",")
        end
        println(io,"]}")
    end
    return result
end

"Run the native spring/floater case, then export the link animation and existing-reference comparison."
function run_point_absorber_spring(;particle_reserve=40000,metadata_options=(;),kwargs...)
    any(k -> k in (:SimulationName,:SaveLocation,:VariableResolution),keys(metadata_options)) &&
        throw(ArgumentError("set case identity/location/refinement through the case keywords; metadata_options is for GPU/output controls"))
    case = point_absorber_spring_case(;kwargs...)
    isdir(case.save_location) && !isempty(readdir(case.save_location)) &&
        throw(ArgumentError("save_location must be empty: $(case.save_location)"))
    mkpath(case.save_location)
    T = typeof(case.coarse.constants.dx)
    refinement = VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,
        fine_constants=case.fine.constants,fine_kernel=case.fine.kernel,follow=50,particle_reserve)
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        merge((;SimulationName="PointAbsorberSpring2D",SaveLocation=case.save_location,SimulationTime=case.simulation_time,
            OutputTimes=case.output_interval,VariableResolution=refinement,GPUUseGraph=true,GPUClipTimeStepToOutput=true,
            VisualizeInParaview=false,OpenLogFile=false,
            OutputVariables=["Velocity","Density","Pressure","Type","ID","GroupMarker","Resolution","Spacing","Mass","Owned","IsBuffer"]),metadata_options)...)
    logger = SimulationLogger(case.save_location)
    particles = AllocateDataStructures(case.coarse.geometry,metadata)
    SavePolygonVTKHDF(joinpath(case.save_location,"PointAbsorberSpring2D_Geometry.vtkhdf"),point_absorber_geometry(case))
    result = try
        RunSimulation(;SimGeometry=case.coarse.geometry,SimConstants=case.coarse.constants,SimKernel=case.coarse.kernel,
            SimMetaData=metadata,SimParticles=particles,SimLogger=logger,SimMotion=case.motion,
            SimViscosity=ArtificialViscosity(),SimDensityDiffusion=LinearDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping())
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    return point_absorber_outputs(case,result)
end
