include("AuxillaryFunctions.jl");   using .AuxillaryFunctions
include("PreProcess.jl");           using .PreProcess
include("PostProcess.jl");          using .PostProcess
include("TimeStepping.jl");         using .TimeStepping
include("SimulationEquations.jl");  using .SimulationEquations

using CSV
using DataFrames
using Printf
using StaticArrays
using CellListMap
using LinearAlgebra

function RunSimulation(SaveLocation="E:/SecondApproach/Results")
    foreach(rm, filter(endswith(".vtp"), readdir(SaveLocation,join=true)))

    ### Play with code
    # FLUID_CSV = "./input/3D_DamBreak_Fluid_3LAYERS.csv"
    # BOUND_CSV = "./input/3D_DamBreak_Boundary_3LAYERS.csv"
    # ρ₀  = 1000
    # dx  = 0.0085#0.02
    # H   = sqrt(3)*dx#sqrt(2)*dx
    # m₀  = ρ₀*dx*dx*dx
    # mᵢ  = mⱼ = m₀
    # αD  = 21/(16*π*H^3) #(7/(4*π*H^2))
    # α   = 0.01
    # g   = 9.81
    # c₀  = sqrt(g*0.3)*20#sqrt(9.81*2)*20#81#85.89
    # γ   = 7
    # dt  = 1e-5
    # δᵩ  = 0.1
    # CFL = 0.2

    FLUID_CSV = "./input/FluidPoints_Dp0.02.csv"
    BOUND_CSV = "./input/BoundaryPoints_Dp0.02.csv"
    ρ₀  = 1000
    dx  = 0.02
    H   = sqrt(2)*dx
    m₀  = ρ₀*dx*dx
    mᵢ  = mⱼ = m₀
    αD  = (7/(4*π*H^2))
    α   = 0.01
    g   = 9.81
    c₀  = sqrt(g*2)*20
    γ   = 7
    dt  = 1e-5
    δᵩ  = 0.1
    CFL = 0.2

    # Load in the fluid and boundary particles. Return these points and both data frames
    points,DF_FLUID,DF_BOUND    = LoadParticlesFromCSV(FLUID_CSV,BOUND_CSV)

    # Use the data frame to construct vectors which inf
    GravityFactor = [Int64(-1) .+ 0*collect(1:size(DF_FLUID,1));Int64(1) .+ 0*collect(1:size(DF_BOUND,1))]
    MotionLimiter = [Int64(1)  .+ 0*collect(1:size(DF_FLUID,1));Int64(0) .+ 0*collect(1:size(DF_BOUND,1))]

    BoundaryBool  = .!Bool.(MotionLimiter)


    # Initialize arrays
    density  = Array([DF_FLUID.Rhop;DF_BOUND.Rhop])
    velocity = zeros(SVector{3,Float64},length(points))
    acceleration = zeros(SVector{3,Float64},length(points))
    create_vtp_file(SaveLocation*"/PlayAround_"*lpad("0",4,"0"),points,density.*0,acceleration.*0,density,acceleration,velocity)

    # Generate normals for boundary particles
    system_boundary  = InPlaceNeighborList(x=points[BoundaryBool], cutoff=2*H, parallel=false)
    list_boundary    = neighborlist!(system_boundary)
    WiINormals,_     = ∑ⱼWᵢⱼ(list_boundary,points[BoundaryBool],αD,H)
    WgINormals,_     = ∑ⱼ∇ᵢWᵢⱼ(list_boundary,points[BoundaryBool],αD,H)
    WgINormals     .*= -1 .* 2dx ./ replace!(norm.(WgINormals),0.0=>1)
    create_vtp_file(SaveLocation*"/Normals",points[BoundaryBool],WiINormals,WgINormals,WiINormals*0,WgINormals*0,WgINormals*0)

    system  = InPlaceNeighborList(x=points, cutoff=2*H, parallel=true)
    for big_iter = 1:200001
        update!(system,points)
        list = neighborlist!(system)

        WiI,_   = ∑ⱼWᵢⱼ(list,points,αD,H)
        WgI,WgL = ∑ⱼ∇ᵢWᵢⱼ(list,points,αD,H)

        dρdtI,_ = ∂ρᵢ∂tDDT(list,points,H,m₀,δᵩ,c₀,γ,g,ρ₀,density,velocity,WgL,MotionLimiter)

        viscI,_ = ∂Πᵢⱼ∂t(list,points,H,density,α,velocity,c₀,m₀,WgL)
        dvdtI,_ = ∂vᵢ∂t(system,points,m₀,density,WgL,c₀,γ,ρ₀)
        # We add gravity as a final step for the i particles, not the L ones, since we do not split the contribution, that is unphysical!
        dvdtI .= map((x,y)->x+y*SVector(0,g,0),dvdtI+viscI,GravityFactor)


        density_n_half  = density  .+ dρdtI * (dt/2)
        density_n_half[(density_n_half .< ρ₀) .* BoundaryBool] .= ρ₀

        velocity_n_half = velocity .+ dvdtI * (dt/2) .* MotionLimiter
        points_n_half   = points   .+ velocity_n_half * (dt/2) .* MotionLimiter

        dρdtI_n_half,_ = ∂ρᵢ∂tDDT(list,points_n_half,H,m₀,δᵩ,c₀,γ,g,ρ₀,density_n_half,velocity_n_half,WgL,MotionLimiter)


        viscI_n_half,_ = ∂Πᵢⱼ∂t(list,points_n_half,H,density_n_half,α,velocity_n_half,c₀,m₀,WgL)
        dvdtI_n_half,_ = ∂vᵢ∂t(system,points_n_half,m₀,density_n_half,WgL,c₀,γ,ρ₀)
        dvdtI_n_half  .= map((x,y)->x+y*SVector(0,g,0),dvdtI_n_half+viscI_n_half,GravityFactor) 


        epsi = -( dρdtI_n_half ./ density_n_half)*dt

        density_new   = density  .* (2 .- epsi)./(2 .+ epsi)
        density_new[(density_new .< ρ₀) .* BoundaryBool] .= ρ₀

        velocity_new  = velocity .+ dvdtI_n_half * dt .* MotionLimiter
        points_new    = points   .+ ((velocity_new .+ velocity)/2) * dt .* MotionLimiter

        density      = density_new
        velocity     = velocity_new
        points       = points_new
        acceleration = dvdtI_n_half

        # Automatic time stepping probably does not work in non-vicous sim
        dt = Δt(acceleration,points,velocity,c₀,H,CFL)

        @printf "Iteration %i | dt = %.5e \n" big_iter dt
        if big_iter % 50 == 0
            create_vtp_file(SaveLocation*"/PlayAround_"*lpad(big_iter,4,"0"),points,WiI,WgI,density,acceleration,velocity)
        end
    end
end

RunSimulation()
