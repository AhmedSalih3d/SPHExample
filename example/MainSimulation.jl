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
                        SimulationMetaData::SimulationMetaData,
                        SimulationConstants::SimulationConstants
)

    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, MaxIterations, OutputIteration, SilentOutput, ThreadsCPU, FloatType, IntType = SimulationMetaData;

    # Unpack simulation constants
    @unpack ρ₀, dx, H, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimulationConstants

    # Load in the fluid and boundary particles. Return these points and both data frames
    points,DF_FLUID,DF_BOUND    = LoadParticlesFromCSV(FluidCSV,BoundCSV)

    # Generate simulation data results array
    FinalResults = SimulationDataResults{3,FloatType}(NumberOfParticles = length(points))
    @unpack Kernel, KernelGradient, Density, Position, Acceleration, Velocity = FinalResults
    # Initialize Arrays
    Position .= deepcopy(points)
    Density  .= Array([DF_FLUID.Rhop;DF_BOUND.Rhop])

    GravityContribution = SVector(0.0,g,0.0)

    # Read this as "GravityFactor * g", so -1 means negative acceleration for fluid particles
    # 1 means boundary particles push back against gravity
    GravityFactor = [-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))]
    GravityContributionArray = map((x)->x * GravityContribution,GravityFactor) 

    # MotionLimiter is what allows fluid particles to move, while not letting the velocity of boundary
    # particles change
    MotionLimiter = [ ones(size(DF_FLUID,1)) ; zeros(size(DF_BOUND,1))]

    # Based on MotionLimiter we assess which particles are boundary particles
    BoundaryBool  = .!Bool.(MotionLimiter)

    # Save the initial particle layout with dummy values
    create_vtp_file(SimulationMetaData,SimulationConstants,FinalResults)

    # Preallocate simulation arrays
    dρdtI           = zeros(eltype(Density), size(Density))
    dvdtI           = zeros(eltype(Velocity), size(Velocity))

    ρₙ⁺             = zeros(eltype(Density), size(Density))
    vₙ⁺             = zeros(eltype(Position), size(Position))
    Positionₙ⁺      = zeros(eltype(Position), size(Position))

    dρdtIₙ⁺         = zeros(eltype(Density), size(Density))

    
    xᵢⱼ             = zeros(eltype(Position), size(Position))
    KernelGradientL = zeros(eltype(Position), size(Position))
    drhopLp         = zeros(eltype(Density), size(Density))
    drhopLn         = zeros(eltype(Density), size(Density))

    Pressureᵢ       = zeros(eltype(Density), size(Density))

    # Initialize the system list
    system  = InPlaceNeighborList(x=Position, cutoff=2*H, parallel=true)

    # Define Progress spec
    @inbounds for SimulationMetaData.Iteration = 1:MaxIterations
        # Be sure to update and retrieve the updated neighbour list at each time step
        @timeit HourGlass "0 | Update Neighbour List" begin
            update!(system,Position)
            list = neighborlist!(system)
        end
        
        @timeit HourGlass "0 | Reset arrays to zero and resize L arrays" begin
            # Clean up arrays
            ResetArrays!(Kernel,KernelGradient,dρdtI,dρdtIₙ⁺, dvdtI, Acceleration)
            # Resize KernelGradientL based on length of neighborlist
            ResizeBuffers!(KernelGradientL, xᵢⱼ, drhopLp, drhopLn; N = length(list))
        end

        @timeit HourGlass "1 | Update xᵢⱼ, kernel values and kernel gradient" begin
            updatexᵢⱼ!(xᵢⱼ, list, Position)
            # Here we output the kernel value for each particle
            ∑ⱼWᵢⱼ!(Kernel, list, SimulationConstants)
            # Here we output the kernel gradient value for each particle and also the kernel gradient value
            # based on the pair-to-pair interaction list, for use in later calculations.
            # Other functions follow a similar format, with the "I" and "L" ending
            ∑ⱼ∇ᵢWᵢⱼ!(KernelGradient, KernelGradientL, list, xᵢⱼ, SimulationConstants)
        end

        # Then we calculate the density derivative at time step "n"
        @timeit HourGlass "2| DDT" ∂ρᵢ∂tDDT!(dρdtI,list,xᵢⱼ,Density,Velocity,KernelGradientL,MotionLimiter,drhopLp,drhopLn, SimulationConstants)

        # We calculate viscosity contribution and momentum equation at time step "n"
        @timeit HourGlass "2| Pressure" map!(x -> Pressure(x, c₀, γ, ρ₀), Pressureᵢ, Density)
        @timeit HourGlass "2| ∂vᵢ∂t!"   ∂vᵢ∂t!(dvdtI, list, Density, KernelGradientL,Pressureᵢ, SimulationConstants)
        @timeit HourGlass "2| ∂Πᵢⱼ∂t!"  ∂Πᵢⱼ∂t!(dvdtI, list, xᵢⱼ ,Density,Velocity,KernelGradientL, SimulationConstants)
        @timeit HourGlass "2| Gravity"  dvdtI   .+=    GravityContributionArray

        # Based on the density derivative at "n", we calculate "n+½"
        @timeit HourGlass "2| ρₙ⁺" @. ρₙ⁺  = Density  + dρdtI * (dt/2)
        # We make sure to limit the density of boundary particles in such a way that they cannot produce suction
        @timeit HourGlass "2| LimitDensityAtBoundary!(ρₙ⁺)" LimitDensityAtBoundary!(ρₙ⁺,BoundaryBool,ρ₀)

        # We now calculate velocity and position at "n+½"
        @timeit HourGlass "2| vₙ⁺" @. vₙ⁺          = Velocity   + dvdtI * (dt/2) * MotionLimiter
        @timeit HourGlass "2| Positionₙ⁺" @. Positionₙ⁺   = Position   + vₙ⁺ * (dt/2)   * MotionLimiter
        @timeit HourGlass "2| updatexᵢⱼ!" updatexᵢⱼ!(xᵢⱼ, list, Positionₙ⁺)

        # Density derivative at "n+½" - Note that we keep the kernel gradient values calculated at "n" for simplicity
        @timeit HourGlass "2| DDT2" ∂ρᵢ∂tDDT!(dρdtIₙ⁺,list,xᵢⱼ,ρₙ⁺,vₙ⁺,KernelGradientL,MotionLimiter, drhopLp, drhopLn, SimulationConstants)

        # Viscous contribution and momentum equation at "n+½"
        @timeit HourGlass "2| Pressure2" map!(x -> Pressure(x, c₀, γ, ρ₀), Pressureᵢ, ρₙ⁺)
        @timeit HourGlass "2| ∂vᵢ∂t!2" ∂vᵢ∂t!(Acceleration, list, ρₙ⁺, KernelGradientL, Pressureᵢ, SimulationConstants) 
        @timeit HourGlass "2| ∂Πᵢⱼ∂t!2" ∂Πᵢⱼ∂t!(Acceleration,list, xᵢⱼ ,ρₙ⁺,vₙ⁺, KernelGradientL, SimulationConstants)
        @timeit HourGlass "2| Acceleration2" Acceleration .+= GravityContributionArray

        # Factor for properly time stepping the density to "n+1" - We use the symplectic scheme as done in DualSPHysics
        @timeit HourGlass "2| DensityEpsi!" DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt)

        # Clamp boundary particles minimum density to avoid suction
        @timeit HourGlass "2| LimitDensityAtBoundary!(Density)" LimitDensityAtBoundary!(Density,BoundaryBool,ρ₀)

        # Update Velocity in-place and then use the updated value for Position
        @timeit HourGlass "2| Velocity" @. Velocity += Acceleration * dt * MotionLimiter
        @timeit HourGlass "2| Position" @. Position += ((Velocity + (Velocity - Acceleration * dt * MotionLimiter)) / 2) * dt * MotionLimiter

        # Automatic time stepping control
        @timeit HourGlass "3| Calculating time step" begin
            dt = Δt(FinalResults,SimulationConstants)
            SimulationMetaData.CurrentTimeStep = dt
            SimulationMetaData.TotalTime      += dt
        end
        
        @timeit HourGlass "4| OutputVTP" OutputVTP(SimulationMetaData,SimulationConstants,FinalResults)

        next!(SimulationMetaData.ProgressSpecification; showvalues = [(:(SimulationMetaData.Iteration),format(FormatExpr("{1:d}"),SimulationMetaData.Iteration)), (:(SimulationMetaData.TotalTime),format(FormatExpr("{1:3.3f}"),SimulationMetaData.TotalTime))])
    end

    # Print the timings in the default way
    show(HourGlass,sortby=:name)
    show(HourGlass)
    disable_timer!(HourGlass)
end

# Initialize SimulationMetaData
begin
    SimMetaData  = SimulationMetaData(
                                    SimulationName="MySimulation", 
                                    SaveLocation=raw"E:\SecondApproach\Results", 
                                    MaxIterations=10001
    )
    # Initialze the constants to use
    SimConstants = SimulationConstants{SimMetaData.FloatType, SimMetaData.IntType}()
    # Clean up folder before running (remember to make folder before hand!)
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))
    # And here we run the function - enjoy!
    @profview RunSimulation(
        FluidCSV = "./input/FluidPoints_Dp0.02.csv",
        BoundCSV = "./input/BoundaryPoints_Dp0.02.csv",
        SimulationMetaData = SimMetaData,
        SimulationConstants = SimConstants
    )
end
