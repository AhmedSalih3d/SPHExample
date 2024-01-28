using Revise

using SPHExample
using CSV
using DataFrames
using Printf
using StaticArrays
using CellListMap
using LinearAlgebra
using TimerOutputs
using Parameters

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


    # Read this as "GravityFactor * g", so -1 means negative acceleration for fluid particles
    # 1 means boundary particles push back against gravity
    GravityFactor = [-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))]

    # MotionLimiter is what allows fluid particles to move, while not letting the velocity of boundary
    # particles change
    MotionLimiter = [ ones(size(DF_FLUID,1)) ; zeros(size(DF_BOUND,1))]

    # Based on MotionLimiter we assess which particles are boundary particles
    BoundaryBool  = .!Bool.(MotionLimiter)

    # Save the initial particle layout with dummy values
    create_vtp_file(SimulationMetaData,SimulationConstants,FinalResults)

    # Initialize the system list
    system  = InPlaceNeighborList(x=Position, cutoff=2*H, parallel=true)
    for SimulationMetaData.Iteration = 1:MaxIterations
        # Be sure to update and retrieve the updated neighbour list at each time step
        update!(system,Position)
        list = neighborlist!(system)

        # Here we output the kernel value for each particle
        Kernel,_   = ∑ⱼWᵢⱼ(list,Position,SimulationConstants)
        # Here we output the kernel gradient value for each particle and also the kernel gradient value
        # based on the pair-to-pair interaction list, for use in later calculations.
        # Other functions follow a similar format, with the "I" and "L" ending
        KernelGradient,WgL = ∑ⱼ∇ᵢWᵢⱼ(list,Position,SimulationConstants)

        # Then we calculate the density derivative at time step "n"
        dρdtI,_ = ∂ρᵢ∂tDDT(list,Position,Density,Velocity,WgL,MotionLimiter, SimulationConstants)

        # We calculate viscosity contribution and momentum equation at time step "n"
        viscI,_ = ∂Πᵢⱼ∂t(list,Position,Density,Velocity,WgL, SimulationConstants)
        dvdtI,_ = ∂vᵢ∂t(list,Position, Density, WgL, SimulationConstants)
        # We add gravity as a final step for the i particles, not the L ones, since we do not split the contribution, that is unphysical!
        # So please be careful with using "L" results directly in some cases
        dvdtI .= map((x,y)->x+y*SVector(0,g,0),dvdtI+viscI,GravityFactor)


        # Based on the density derivative at "n", we calculate "n+½"
        density_n_half  = Density  .+ dρdtI * (dt/2)
        # We make sure to limit the density of boundary particles in such a way that they cannot produce suction
        density_n_half[(density_n_half .< ρ₀) .* BoundaryBool] .= ρ₀

        # We now calculate velocity and position at "n+½"
        velocity_n_half = Velocity   .+ dvdtI * (dt/2) .* MotionLimiter
        points_n_half   = Position   .+ velocity_n_half * (dt/2) .* MotionLimiter

        # Density derivative at "n+½" - Note that we keep the kernel gradient values calculated at "n" for simplicity
        dρdtI_n_half,_ = ∂ρᵢ∂tDDT(list,points_n_half,density_n_half,velocity_n_half,WgL,MotionLimiter, SimulationConstants)

        # Viscous contribution and momentum equation at "n+½"
        viscI_n_half,_ = ∂Πᵢⱼ∂t(list,points_n_half,density_n_half,velocity_n_half, WgL, SimulationConstants)
        dvdtI_n_half,_ = ∂vᵢ∂t(list,points_n_half,density_n_half, WgL, SimulationConstants)
        dvdtI_n_half  .= map((x,y)->x+y*SVector(0,g,0),dvdtI_n_half+viscI_n_half,GravityFactor) 

        # Factor for properly time stepping the density to "n+1" - We use the symplectic scheme as done in DualSPHysics
        epsi = -( dρdtI_n_half ./ density_n_half)*dt

        # Finally we update all values to their next time step, "n+1"
        density_new   = Density  .* (2 .- epsi)./(2 .+ epsi)
        density_new[(density_new .< ρ₀) .* BoundaryBool] .= ρ₀
        velocity_new  = Velocity .+ dvdtI_n_half * dt .* MotionLimiter
        points_new    = Position   .+ ((velocity_new .+ Velocity)/2) * dt .* MotionLimiter

        # And for clarity updating the values in our simulation is done explicitly here
        Density      .= density_new
        Velocity     .= velocity_new
        points       .= points_new
        Acceleration .= dvdtI_n_half
        Position     .= points_new

        # Automatic time stepping control
        dt = Δt(Acceleration,points,Velocity,c₀,H,CFL)
        SimulationMetaData.CurrentTimeStep = dt
        
        OutputVTP(SimulationMetaData,SimulationConstants,FinalResults)
    end
end

# Initialize SimulationMetaData
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
RunSimulation(
    FluidCSV = "./input/FluidPoints_Dp0.02.csv",
    BoundCSV = "./input/BoundaryPoints_Dp0.02.csv",
    SimulationMetaData = SimMetaData,
    SimulationConstants = SimConstants
)
