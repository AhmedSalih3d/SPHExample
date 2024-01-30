using SPHExample
using CSV
using DataFrames
using Printf
using StaticArrays
using CellListMap
using LinearAlgebra
using TimerOutputs
using Parameters
import ProgressMeter: Progress, next!, @showprogress

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

    # Functions to avoid temporary array in epsi calculation later on
    F_Epsi(DensityDerivative, DensityValue, TimeStepValue)      =  @. -( DensityDerivative / DensityValue) * TimeStepValue
    F_EpsiFinal(DensityDerivative, DensityValue, TimeStepValue) =  @.  ( 2 - F_Epsi(DensityDerivative, DensityValue, TimeStepValue)) /  (2 + F_Epsi(DensityDerivative, DensityValue, TimeStepValue))   

    # Preallocate simulation arrays
    dρdtI           = zeros(eltype(Density), size(Density))
    dvdtI           = zeros(eltype(Velocity), size(Velocity))

    ρₙ⁺             = zeros(eltype(Density), size(Density))
    vₙ⁺             = zeros(eltype(Position), size(Position))
    Positionₙ⁺      = zeros(eltype(Position), size(Position))

    dρdtIₙ⁺         = zeros(eltype(Density), size(Density))

    # Initialize the system list
    system  = InPlaceNeighborList(x=Position, cutoff=2*H, parallel=true)

    # Define Progress spec
    for SimulationMetaData.Iteration = 1:MaxIterations
        # Be sure to update and retrieve the updated neighbour list at each time step
        update!(system,Position)
        list = neighborlist!(system)

        # Clean up arrays
        ResetArray(Kernel,KernelGradient,dρdtI,dρdtIₙ⁺, dvdtI, Acceleration)

        # Here we output the kernel value for each particle
        ∑ⱼWᵢⱼ!(Kernel, list, SimulationConstants)

        # Here we output the kernel gradient value for each particle and also the kernel gradient value
        # based on the pair-to-pair interaction list, for use in later calculations.
        # Other functions follow a similar format, with the "I" and "L" ending
        KernelGradientL = ∑ⱼ∇ᵢWᵢⱼ!(KernelGradient, list,Position,SimulationConstants)

        # Then we calculate the density derivative at time step "n"
        ∂ρᵢ∂tDDT!(dρdtI,list,Position,Density,Velocity,KernelGradientL,MotionLimiter, SimulationConstants)

        # We calculate viscosity contribution and momentum equation at time step "n"
        ∂vᵢ∂t!(dvdtI, list, Density, KernelGradientL, SimulationConstants)
        ∂Πᵢⱼ∂t!(dvdtI, list,Position,Density,Velocity,KernelGradientL, SimulationConstants)
        dvdtI   .+=    GravityContributionArray

        # Based on the density derivative at "n", we calculate "n+½"
        @. ρₙ⁺  = Density  + dρdtI * (dt/2)
        # We make sure to limit the density of boundary particles in such a way that they cannot produce suction
        @. ρₙ⁺[(ρₙ⁺ < ρ₀) * BoundaryBool] = ρ₀

        # We now calculate velocity and position at "n+½"
        @. vₙ⁺          = Velocity   + dvdtI * (dt/2) * MotionLimiter
        @. Positionₙ⁺   = Position   + vₙ⁺ * (dt/2)   * MotionLimiter

        # Density derivative at "n+½" - Note that we keep the kernel gradient values calculated at "n" for simplicity
        ∂ρᵢ∂tDDT!(dρdtIₙ⁺,list,Positionₙ⁺,ρₙ⁺,vₙ⁺,KernelGradientL,MotionLimiter, SimulationConstants)

        # Viscous contribution and momentum equation at "n+½"
        ∂vᵢ∂t!(Acceleration, list, ρₙ⁺, KernelGradientL, SimulationConstants) 
        ∂Πᵢⱼ∂t!(Acceleration,list,Positionₙ⁺,ρₙ⁺,vₙ⁺, KernelGradientL, SimulationConstants)
        Acceleration .+= GravityContributionArray

        # Factor for properly time stepping the density to "n+1" - We use the symplectic scheme as done in DualSPHysics
        println(@allocated DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt))

        # Clamp boundary particles minimum density to avoid suction
        #clamp!(Density[BoundaryBool], ρ₀,2ρ₀) #Never going to hit the high unless breaking sim
        @. Density[(Density < ρ₀) * BoundaryBool] = ρ₀

        # Update Velocity in-place and then use the updated value for Position
        @. Velocity += Acceleration * dt * MotionLimiter
        @. Position += ((Velocity + (Velocity - Acceleration * dt * MotionLimiter)) / 2) * dt * MotionLimiter

        # Automatic time stepping control
        dt = Δt(Acceleration,Position,Velocity,SimulationConstants)
        SimulationMetaData.CurrentTimeStep = dt
        SimulationMetaData.TotalTime      += dt
        
        OutputVTP(SimulationMetaData,SimulationConstants,FinalResults)

        next!(SimulationMetaData.ProgressSpecification; showvalues = [(:(SimulationMetaData.Iteration),SimulationMetaData.Iteration), (:(SimulationMetaData.TotalTime),SimulationMetaData.TotalTime)])
    end
end

# Initialize SimulationMetaData
SimMetaData  = SimulationMetaData(
                                  SimulationName="MySimulation", 
                                  SaveLocation=raw"E:\SecondApproach\Results", 
                                  MaxIterations=1001
)
# Initialze the constants to use
SimConstants = SimulationConstants{SimMetaData.FloatType, SimMetaData.IntType}()
# Clean up folder before running (remember to make folder before hand!)
foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))
# And here we run the function - enjoy!
RunSimulation(
    FluidCSV = "./input/FluidPoints_Dp0.02.csv",
    BoundCSV = "./input/BoundaryPoints_Dp0.02.csv",
    SimulationMetaData = SimMetaData,
    SimulationConstants = SimConstants
)
