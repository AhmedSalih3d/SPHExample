using SPHExample
using CSV
using DataFrames
using Printf
using StaticArrays
using CellListMap
using LinearAlgebra
using TimerOutputs
using Parameters
import ProgressMeter: Progress, next!
using Formatting
using StructArrays
using LoopVectorization

include("../src/ProduceVTP.jl")

"""
    RunSimulation(;SimulationMetaData::SimulationMetaData, SimulationConstants::SimulationConstants)

Run a Smoothed Particle Hydrodynamics (SPH) simulation using specified metadata and simulation constants.

This function initializes the simulation environment, loads particle data, and runs the simulation iteratively until the maximum number of iterations is reached. It outputs simulation results at specified intervals.

## Arguments
- `SimulationMetaData::SimulationMetaData`: A struct containing metadata for the simulation, including the simulation name, save location, maximum iterations, output iteration frequency, and other settings.
- `SimulationConstants::SimulationConstants`: A struct containing constants used in the simulation, such as reference density, initial particle distance, smoothing length, initial mass, normalization constant for kernel, artificial viscosity alpha value, gravity, speed of sound, gamma for the pressure equation of state, initial time step, coefficient for density diffusion, and CFL number.

## Variable Explanation
- `FLUID_CSV`: Path to CSV file containing fluid particles. See "input" folder for examples.
- `BOUND_CSV`: Path to CSV file containing boundary particles. See "input" folder for examples.
- `ρ₀`: Reference density.
- `dx`: Initial particle distance. See "dp" in CSV files. For 3D simulations, a typical value might be 0.0085.
- `H`: Smoothing length.
- `m₀`: Initial mass, calculated as reference density multiplied by initial particle distance to the power of simulation dimensions.
- `mᵢ = mⱼ = m₀`: All particles have the same mass.
- `αD`: Normalization constant for the kernel.
- `α`: Artificial viscosity alpha value.
- `g`: Gravity (positive value).
- `c₀`: Speed of sound, which must be 10 times the highest velocity in the simulation.
- `γ`: Gamma, most commonly 7 for water, used in the pressure equation of state.
- `dt`: Initial time step.
- `δᵩ`: Coefficient for density diffusion, typically 0.1.
- `CFL`: CFL number for the simulation.

## Example
```julia
#See SimulationMetaData and SimulationConstants for all possible inputs
SimMetaData  = SimulationMetaData(SimulationName="MySimulation", SaveLocation=raw"path/to/results", MaxIterations=101)
SimConstants = SimulationConstants{SimMetaData.FloatType, SimMetaData.IntType}()
RunSimulation(
    FluidCSV = "./input/FluidPoints_Dp0.02.csv",
    BoundCSV = "./input/BoundaryPoints_Dp0.02.csv",
    SimulationMetaData = SimMetaData,
    SimulationConstants = SimConstants
)
```
"""
function RunSimulation(;FluidCSV::String,
                        BoundCSV::String,
                        SimMetaData::SimulationMetaData{FloatType},
                        SimConstants::SimulationConstants,
) where FloatType
    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, MaxIterations, OutputIteration, SilentOutput, ThreadsCPU = SimMetaData;

    # Unpack simulation constants
    @unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants

    # Load in the fluid and boundary particles. Return these points and both data frames
    points, density_fluid, density_bound  = LoadParticlesFromCSV(FloatType, FluidCSV,BoundCSV)

    # Read this as "GravityFactor * g", so -1 means negative acceleration for fluid particles
    # 1 means boundary particles push back against gravity
    GravityFactor            = [-ones(size(density_fluid,1)) ; zeros(size(density_bound,1))]

    # MotionLimiter is what allows fluid particles to move, while not letting the velocity of boundary
    # particles change
    MotionLimiter = [ ones(size(density_fluid,1)) ; zeros(size(density_bound,1))]

    # Based on MotionLimiter we assess which particles are boundary particles
    BoundaryBool  = .!Bool.(MotionLimiter)

    # Preallocate simulation arrays
    SizeOfParticlesI1 = (length(points),)
    TypeOfParticleI3  = eltype(points)

    Density           = deepcopy([density_fluid;density_bound])

    Kernel            = zeros(FloatType,         SizeOfParticlesI1)
    KernelL           = zeros(FloatType,         SizeOfParticlesI1)

    dρdtI             = zeros(FloatType,         SizeOfParticlesI1)

    # dvdtIˣ            = zeros(FloatType,  SizeOfParticlesI1)
    # dvdtIʸ            = zeros(FloatType,  SizeOfParticlesI1)
    # dvdtIᶻ            = zeros(FloatType,  SizeOfParticlesI1)
    # dvdtI             = StructArray{TypeOfParticleI3}(( dvdtIˣ, dvdtIʸ, dvdtIᶻ))

    # dvdtLˣ            = zeros(FloatType,  SizeOfParticlesI1)
    # dvdtLʸ            = zeros(FloatType,  SizeOfParticlesI1)
    # dvdtLᶻ            = zeros(FloatType,  SizeOfParticlesI1)
    # dvdtL             = StructArray{TypeOfParticleI3}(( dvdtLˣ, dvdtLʸ, dvdtLᶻ))

    ρₙ⁺               = zeros(FloatType,         SizeOfParticlesI1)

    Positionₙ⁺ˣ       = zeros(FloatType,  SizeOfParticlesI1)
    Positionₙ⁺ʸ       = zeros(FloatType,  SizeOfParticlesI1)
    Positionₙ⁺ᶻ       = zeros(FloatType,  SizeOfParticlesI1)
    Positionₙ⁺        = StructArray{TypeOfParticleI3}(( Positionₙ⁺ˣ, Positionₙ⁺ʸ, Positionₙ⁺ᶻ))

  
    dρdtIₙ⁺           = zeros(FloatType,         SizeOfParticlesI1)

    # KernelGradientˣ   = zeros(FloatType,  SizeOfParticlesI1)
    # KernelGradientʸ   = zeros(FloatType,  SizeOfParticlesI1)
    # KernelGradientᶻ   = zeros(FloatType,  SizeOfParticlesI1)
    # KernelGradient    = StructArray{TypeOfParticleI3}(( KernelGradientˣ, KernelGradientʸ, KernelGradientᶻ))

    # KernelGradientLˣ  = zeros(FloatType,  SizeOfParticlesI1)
    # KernelGradientLʸ  = zeros(FloatType,  SizeOfParticlesI1)
    # KernelGradientLᶻ  = zeros(FloatType,  SizeOfParticlesI1)
    # KernelGradientL   = StructArray{TypeOfParticleI3}(( KernelGradientLˣ, KernelGradientLʸ, KernelGradientLᶻ))

    Accelerationˣ     = zeros(FloatType,  SizeOfParticlesI1)
    Accelerationʸ     = zeros(FloatType,  SizeOfParticlesI1)
    Accelerationᶻ     = zeros(FloatType,  SizeOfParticlesI1)
    Acceleration      = StructArray{TypeOfParticleI3}(( Accelerationˣ, Accelerationʸ, Accelerationᶻ))

    # Velocityˣ         = zeros(FloatType,  SizeOfParticlesI1)
    # Velocityʸ         = zeros(FloatType,  SizeOfParticlesI1)
    # Velocityᶻ         = zeros(FloatType,  SizeOfParticlesI1)
    # Velocity          = StructArray{TypeOfParticleI3}(( Velocityˣ, Velocityʸ, Velocityᶻ))
  
    # Velocityₙ⁺ˣ        = zeros(FloatType,  SizeOfParticlesI1)
    # Velocityₙ⁺ʸ        = zeros(FloatType,  SizeOfParticlesI1)
    # Velocityₙ⁺ᶻ        = zeros(FloatType,  SizeOfParticlesI1)
    # Velocityₙ⁺         = StructArray{TypeOfParticleI3}(( Velocityₙ⁺ˣ, Velocityₙ⁺ʸ, Velocityₙ⁺ᶻ))

    Positionˣ          = getindex.(points,1)
    Positionʸ          = getindex.(points,2)
    Positionᶻ          = getindex.(points,3)
    Position           = DimensionalData(Positionˣ,Positionʸ,Positionᶻ)

    KernelGradient     = DimensionalData{3,FloatType}(length(points))
    KernelGradientL    = DimensionalData{3,FloatType}(length(points))
    xᵢⱼ                = DimensionalData{3,FloatType}(length(points))
    Acceleration       = DimensionalData{3,FloatType}(length(points))
    Velocity           = DimensionalData{3,FloatType}(length(points))
    dvdtI              = DimensionalData{3,FloatType}(length(points))
    dvdtL              = DimensionalData{3,FloatType}(length(points))
    Velocityₙ⁺         = DimensionalData{3,FloatType}(length(points))
    Positionₙ⁺         = DimensionalData{3,FloatType}(length(points))
 
    drhopLp            = zeros(FloatType,         SizeOfParticlesI1)
    drhopLn            = zeros(FloatType,         SizeOfParticlesI1) 
          
    Pressureᵢ          = zeros(FloatType,         SizeOfParticlesI1)

    # Initialize the system system.nb.list
    # The result from CellListMap using neighborlist! is a vector of tuples, (i index, j index, d eucledian distance between particles)
    # By using I, J, D as below, through the StructArray composition, it is possible to do as close as possible to an in-place transfer
    # of information. Using I, J and D vectors allows for parallezation using @tturbo from LoopVectorization.jl. 
    I                 = zeros(Int64,   SizeOfParticlesI1)
    J                 = zeros(Int64,   SizeOfParticlesI1)
    D                 = zeros(Float64, SizeOfParticlesI1)
    list_me           = StructArray{Tuple{Int64,Int64,Float64}}((I,J,D))

    system  = InPlaceNeighborList(x=Position.V, cutoff=2*h, parallel=true)

    # Save the initial particle layout with dummy values
    # create_vtp_file(SimMetaData,SimConstants,Position.V; Kernel, KernelGradient.V, Density, Acceleration)
    PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(SimMetaData.Iteration,6,"0") * ".vtp", Position.V, ["Kernel", "KernelGradient", "Density", "Pressure", "Acceleration" , "Velocity"], Kernel, KernelGradient.V, Density, Pressureᵢ, Acceleration.V, Velocity.V)

    # Define Progress spec for displaying simulation results
    show_vals(x) = [(:(Iteration),format(FormatExpr("{1:d}"), x.Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"),x.TotalTime))]

    @inbounds for SimMetaData.Iteration = 1:MaxIterations
        # Be sure to update and retrieve the updated neighbour list at each time step
        @timeit HourGlass "0 | Update Neighbour system.nb.list" begin
            update!(system,Position.V)
            neighborlist!(system)
            resize!(list_me, system.nb.n)
            list_me .= system.nb.list
        end
        
        @timeit HourGlass "0 | Reset arrays to zero and resize L arrays" begin
            # Resize L based values (interactions between all particles i and j) based on length of neighborsystem.nb.list
            ResizeBuffers!(KernelL, KernelGradientL, dvdtL, xᵢⱼ, drhopLp, drhopLn; N = system.nb.n)
            # Clean up arrays, Vector{T} and Vector{SVector{3,T}}
            ResetArrays!(Kernel, dρdtI,dρdtIₙ⁺,KernelGradient.V,dvdtI.V, Acceleration.V, drhopLp, drhopLn)
        end

         # Here we calculate the distances between particles, output the kernel gradient value for each particle and also the kernel gradient value
        # based on the pair-to-pair interaction system.nb.list, for use in later calculations.
        # Other functions follow a similar format, with the "I" and "L" ending
        @timeit HourGlass "1 | Update xᵢⱼ, kernel values and kernel gradient" begin
            # updatexᵢⱼ!(xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ, I, J, Positionˣ, Positionʸ, Positionᶻ)
            updatexᵢⱼ!(xᵢⱼ, Position, I, J)
            
            # Here we output the kernel value for each particle. Note that KernelL is list of interactions, while Kernel is the value for each actual particle
            ∑ⱼWᵢⱼ!(Kernel, KernelL, I, J, D, SimConstants)
            # ∑ⱼ∇ᵢWᵢⱼ!(KernelGradientˣ,KernelGradientʸ,KernelGradientᶻ,KernelGradientLˣ,KernelGradientLʸ,KernelGradientLᶻ, I, J, D, xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ, SimConstants)
            ∑ⱼ∇ᵢWᵢⱼ!(KernelGradient,KernelGradientL, I, J, D, xᵢⱼ, SimConstants)
        end

        # Then we calculate the density derivative at time step "n"
        # @timeit HourGlass "2| DDT" ∂ρᵢ∂tDDT!(dρdtI, I, J, D, xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ,Density, Velocityˣ, Velocityʸ, Velocityᶻ,KernelGradientLˣ,KernelGradientLʸ,KernelGradientLᶻ,MotionLimiter,drhopLp,drhopLn, SimConstants)
        @timeit HourGlass "2| DDT" ∂ρᵢ∂tDDT!(dρdtI, I, J, D, xᵢⱼ,Density, Velocity,KernelGradientL,MotionLimiter,drhopLp,drhopLn, SimConstants)

        # # We calculate viscosity contribution and momentum equation at time step "n"
        @timeit HourGlass "2| Pressure" Pressure!(Pressureᵢ, Density, SimConstants)
        # @timeit HourGlass "2| Artificial Viscosity Momentum Equation" ArtificialViscosityMomentumEquation!(I,J,D, dvdtIˣ, dvdtIʸ, dvdtIᶻ, dvdtLˣ, dvdtLʸ, dvdtLᶻ,Density,KernelGradientLˣ,KernelGradientLʸ,KernelGradientLᶻ,xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ, Velocityˣ, Velocityʸ, Velocityᶻ, Pressureᵢ, GravityFactor, SimConstants)
        @timeit HourGlass "2| Artificial Viscosity Momentum Equation" ArtificialViscosityMomentumEquation!(I,J,D, dvdtI, dvdtL,Density,KernelGradientL, xᵢⱼ, Velocity, Pressureᵢ, GravityFactor, SimConstants)

        # # Based on the density derivative at "n", we calculate "n+½"
        @timeit HourGlass "2| ρₙ⁺" @. ρₙ⁺  = Density  + dρdtI * (dt/2)
        # # We make sure to limit the density of boundary particles in such a way that they cannot produce suction
        @timeit HourGlass "2| LimitDensityAtBoundary!(ρₙ⁺)" LimitDensityAtBoundary!(ρₙ⁺,BoundaryBool,ρ₀)

        # # We now calculate velocity and position at "n+½"
        @timeit HourGlass "2| vₙ⁺"        @. Velocityₙ⁺.V   = Velocity.V   + dvdtI.V * (dt/2) * MotionLimiter
        @timeit HourGlass "2| Positionₙ⁺" @. Positionₙ⁺.V   = Position.V   + Velocityₙ⁺.V * (dt/2)   * MotionLimiter
        @timeit HourGlass "2| updatexᵢⱼ!" updatexᵢⱼ!(xᵢⱼ, Positionₙ⁺, I, J)
        
        # # # Density derivative at "n+½" - Note that we keep the kernel gradient values calculated at "n" for simplicity
        ResetArrays!(drhopLp, drhopLn)
        @timeit HourGlass "2| DDT2" ∂ρᵢ∂tDDT!(dρdtIₙ⁺, I, J, D, xᵢⱼ,ρₙ⁺, Velocityₙ⁺,KernelGradientL,MotionLimiter,drhopLp,drhopLn, SimConstants)

        # # # Viscous contribution and momentum equation at "n+½"
        @timeit HourGlass "2| Pressure2" Pressure!(Pressureᵢ, ρₙ⁺, SimConstants)
        @timeit HourGlass "2| Artificial Viscosity Momentum Equation2" ArtificialViscosityMomentumEquation!(I,J,D, Acceleration, dvdtL, ρₙ⁺,KernelGradientL, xᵢⱼ, Velocityₙ⁺, Pressureᵢ, GravityFactor, SimConstants)

        # # Factor for properly time stepping the density to "n+1" - We use the symplectic scheme as done in DualSPHysics
        @timeit HourGlass "2| DensityEpsi!"  DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt)

        # # Clamp boundary particles minimum density to avoid suction
        @timeit HourGlass "2| LimitDensityAtBoundary!(Density)" LimitDensityAtBoundary!(Density,BoundaryBool,ρ₀)

        # # # Update Velocity in-place and then use the updated value for Position
        @timeit HourGlass "2| Velocity" @. Velocity.V += Acceleration.V * dt * MotionLimiter
        @timeit HourGlass "2| Position" @. Position.V += ((Velocity.V + (Velocity.V - Acceleration.V * dt * MotionLimiter)) / 2) * dt * MotionLimiter

        # Automatic time stepping control
        @timeit HourGlass "3| Calculating time step" begin
            dt =  Δt(Position.V, Velocity.V, Acceleration.V,SimConstants)
            SimMetaData.CurrentTimeStep = dt
            SimMetaData.TotalTime      += dt
        end
        
        # OutVTP is based on a well-developed Julia package, WriteVTK, while CustomVTP is based on my hand-rolled solution.
        # CustomVTP is about 10% faster, but does not mean much in this case.
        if SimMetaData.Iteration % SimMetaData.OutputIteration == 0
            #@timeit HourGlass "4| OutputVTP" OutputVTP(SimMetaData,SimConstants,Position; Kernel, KernelGradient, Density, Acceleration, Velocity)
            #@timeit HourGlass "4| OutputVTP" OutputVTP(SimMetaData,SimConstants,Position.V; Kernel, KernelGradient.V)
            # @timeit HourGlass "4| CustomVTP" PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * lpad(SimMetaData.Iteration,6,"0") * ".vtp", Position, ["Kernel", "KernelGradient", "Density", "Pressure", "Acceleration" , "Velocity"], Kernel, KernelGradient, Density, Pressureᵢ, Acceleration, Velocity)
            @timeit HourGlass "4| CustomVTP" PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(SimMetaData.Iteration,6,"0") * ".vtp", Position.V, ["Kernel", "KernelGradient", "Density", "Pressure", "Acceleration" , "Velocity"], Kernel, KernelGradient.V, Density, Pressureᵢ, Acceleration.V, Velocity.V)
        end

        next!(SimMetaData.ProgressSpecification; showvalues = show_vals(SimMetaData))
    end
    
    # # Print the timings in the default way
    show(HourGlass,sortby=:name)
    show(HourGlass)
    disable_timer!(HourGlass)

    return nothing
end

# Initialize Simulation
begin
    T = Float64
    SimMetaData  = SimulationMetaData{T}(
                                    SimulationName="MySimulation", 
                                    SaveLocation=raw"E:\SecondApproach\Results", 
                                    MaxIterations=10001,
                                    OutputIteration=50,
    )
    # Initialze the constants to use
    SimConstants = SimulationConstants{T}()
    # Clean up folder before running (remember to make folder before hand!)
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))

    # And here we run the function - enjoy!
    RunSimulation(
        FluidCSV     = "./input/FluidPoints_Dp0.02.csv",
        BoundCSV     = "./input/BoundaryPoints_Dp0.02.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstants
    )
end
