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

include("../../src/ProduceVTP.jl")

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
                        SimMetaData::SimulationMetaData{Dimensions, FloatType},
                        SimConstants::SimulationConstants,
) where {Dimensions,FloatType}
    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;

    # Unpack simulation constants
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants

    # Load in the fluid and boundary particles. Return these points and both data frames
    # @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
    @inline points, density_fluid, density_bound  = LoadParticlesFromCSV(Dimensions,FloatType, FluidCSV,BoundCSV)

    # Read this as "GravityFactor * g", so -1 means negative acceleration for fluid particles
    GravityFactor            = [-ones(size(density_fluid,1)) ; zeros(size(density_bound,1))]

    # MotionLimiter is what allows fluid particles to move, while not letting the velocity of boundary
    # particles change
    MotionLimiter = [ ones(size(density_fluid,1)) ; zeros(size(density_bound,1))]


    # Based on MotionLimiter we assess which particles are boundary particles
    BoundaryBool  = .!Bool.(MotionLimiter)

    # Preallocate simulation arrays
    NumberOfPoints = length(points)

    Density           = deepcopy([density_fluid;density_bound])
    Kernel            = zeros(FloatType, NumberOfPoints)
    KernelL           = zeros(FloatType, NumberOfPoints)
    dρdtI             = zeros(FloatType, NumberOfPoints)
    ρₙ⁺               = zeros(FloatType, NumberOfPoints)
    dρdtIₙ⁺           = zeros(FloatType, NumberOfPoints)

    drhopLp            = zeros(FloatType, NumberOfPoints)
    drhopLn            = zeros(FloatType, NumberOfPoints) 
    Pressureᵢ          = zeros(FloatType, NumberOfPoints)

    Position           = DimensionalData(points.vectors...)

    KernelGradient     = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    KernelGradientL    = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    xᵢⱼ                = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    Acceleration       = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    Velocity           = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    dvdtI              = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    dvdtL              = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    Velocityₙ⁺         = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    Positionₙ⁺         = DimensionalData{Dimensions,FloatType}(NumberOfPoints)

    # Initialize the system system.nb.list
    # The result from CellListMap using neighborlist! is a vector of tuples, (i index, j index, d eucledian distance between particles)
    # By using I, J, D as below, through the StructArray composition, it is possible to do as close as possible to an in-place transfer
    # of information. Using I, J and D vectors allows for parallezation using @tturbo from LoopVectorization.jl.

    I                 = zeros(Int64,   NumberOfPoints)
    J                 = zeros(Int64,   NumberOfPoints)
    D                 = zeros(Float64, NumberOfPoints)
    list_me           = StructArray{Tuple{Int64,Int64,Float64}}((I,J,D))

    system          = InPlaceNeighborList(x=Position.V, cutoff=2*h*1)

    to_3d(vec_2d) = length(first(vec_2d)) == 2 ? [SVector(v..., 0.0) for v in vec_2d] : nothing
    # Save the initial particle layout with dummy values
    @timeit HourGlass "6.1 outputting savefile" PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(0, 6,"0") * ".vtp", to_3d(Position.V), ["Kernel", "KernelGradient", "Density", "Pressure", "Acceleration" , "Velocity"], Kernel, KernelGradient.V, Density, Pressureᵢ, Acceleration.V, Velocity.V)

    # Define Progress spec for displaying simulation results
    generate_showvalues(Iteration, TotalTime) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime))]

    OutputCounter = 0.0
    OutputIterationCounter = 0
    @inbounds while true
        # Be sure to update and retrieve the updated neighbour list at each time step
        @timeit HourGlass "0.1 update particle positions"           CellListMap.update!(system,Position.V)
        @timeit HourGlass "0.2 extract updated neighborlist"        neighborlist!(system)
        @timeit HourGlass "0.3 resize split neighborlist"           resize!(list_me, system.nb.n)
        @timeit HourGlass "0.4 update values of split neighborlist" list_me .= system.nb.list

        # @timeit HourGlass "Step 0.2 | Reset arrays to zero and resize L arrays" begin
        # Resize L based values (interactions between all particles i and j) based on length of neighborsystem.nb.list
        @timeit HourGlass "0.5 resize calculation buffers" ResizeBuffers!(KernelL, KernelGradientL, dvdtL, xᵢⱼ, drhopLp, drhopLn; N = system.nb.n)
        # Clean up arrays, Vector{T} and Vector{SVector{3,T}}
        @timeit HourGlass "0.6 reset calculation buffers"  ResetArrays!(Kernel, dρdtI,dρdtIₙ⁺,KernelGradient.V,dvdtI.V, Acceleration.V, drhopLp, drhopLn)

        # Here we calculate the distances between particles, output the kernel gradient value for each particle and also the kernel gradient value
        # based on the pair-to-pair interaction system.nb.list, for use in later calculations.
        # Other functions follow a similar format, with the "I" and "L" ending
        # @timeit HourGlass "Step 1 | Update xᵢⱼ, kernel values and kernel gradient" begin
        @timeit HourGlass "1.1 calculate xᵢⱼ" updatexᵢⱼ!(xᵢⱼ, Position, I, J)
        # Here we output the kernel and kernel gradient value for each particle. Note that KernelL is list of interactions, while Kernel is the value for each actual particle. Similar naming for other variables
        @timeit HourGlass "1.2 calculate kernel and kernel gradient" ∑ⱼWᵢⱼ!∑ⱼ∇ᵢWᵢⱼ!(KernelGradient,KernelGradientL, Kernel, KernelL, I, J, D, xᵢⱼ, SimConstants)

        # @timeit HourGlass "Step 2 | Simulation Equations to update values, preparing for n+1/2" begin
        @timeit HourGlass "2.1 DDT" ∂ρᵢ∂t!(dρdtI, I, J, D, xᵢⱼ, Density, Velocity,KernelGradientL,drhopLp,drhopLn, SimConstants, MotionLimiter)
        # We calculate viscosity contribution and momentum equation at time step "n"
        @timeit HourGlass "2.2 Pressure" Pressure!(Pressureᵢ, Density, SimConstants)
        @timeit HourGlass "2.3 Artificial Viscosity Momentum Equation" ArtificialViscosityMomentumEquation!(I,J,D, dvdtI, dvdtL,Density,KernelGradientL, xᵢⱼ, Velocity, Pressureᵢ, GravityFactor, SimConstants)

        @timeit HourGlass "2.1 half step velocity" @. Velocityₙ⁺.V   = Velocity.V   + dvdtI.V * (dt/2) * MotionLimiter
        @timeit HourGlass "2.2 half step position" @. Positionₙ⁺.V   = Position.V   + Velocityₙ⁺.V * (dt/2)   * MotionLimiter
       
        # Based on the density derivative at "n", we calculate "n+½"
        @timeit HourGlass "2.3 half step density"  @. ρₙ⁺  = Density  + dρdtI * (dt/2) 
        # density of boundary particles in such a way that they cannot produce suction
        @timeit HourGlass "2.4 half step limit density at boundary" LimitDensityAtBoundary!(ρₙ⁺,BoundaryBool,ρ₀)

        # Even though particles have moved slightly, we do not update xᵢⱼ or kernel values!

        # Density derivative at "n+½" - Note that we keep the kernel gradient values calculated at "n" for simplicity
        @timeit HourGlass "3.1 reset L arrays for density diffusion"   ResetArrays!(drhopLp, drhopLn)
        @timeit HourGlass "3.2 DDT"                                    ∂ρᵢ∂t!(dρdtIₙ⁺, I, J, D, xᵢⱼ, ρₙ⁺, Velocityₙ⁺, KernelGradientL, drhopLp,drhopLn, SimConstants, MotionLimiter)
        # Viscous contribution and momentum equation at "n+½"
        @timeit HourGlass "3.3 Pressure"                               Pressure!(Pressureᵢ, ρₙ⁺, SimConstants)
        @timeit HourGlass "3.4 Artificial Viscosity Momentum Equation" ArtificialViscosityMomentumEquation!(I,J,D, Acceleration, dvdtL, ρₙ⁺,KernelGradientL, xᵢⱼ, Velocityₙ⁺, Pressureᵢ, GravityFactor, SimConstants)
        # Factor for properly time stepping the density to "n+1" - We use the symplectic scheme as done in DualSPHysics
        @timeit HourGlass "4.1 DensityEpsi!"  DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt)
        # Clamp boundary particles minimum density to avoid suction
        @timeit HourGlass "4.2 LimitDensityAtBoundary!(Density)" LimitDensityAtBoundary!(Density,BoundaryBool,ρ₀)
        # Update Velocity in-place and then use the updated value for Position
        @timeit HourGlass "4.3 final velocity" @. Velocity.V += Acceleration.V * dt * MotionLimiter
        @timeit HourGlass "4.4 final position" @. Position.V += ((Velocity.V + (Velocity.V - Acceleration.V * dt * MotionLimiter)) / 2) * dt * MotionLimiter
        
        @timeit HourGlass "5.1 calculate dt"       dt = Δt(Position.V, Velocity.V, Acceleration.V,SimConstants)
        @timeit HourGlass "5.1 +1 iteration"       SimMetaData.Iteration      += 1
        @timeit HourGlass "5.2 set new dt"         SimMetaData.CurrentTimeStep = dt
        @timeit HourGlass "5.3 increment total dt" SimMetaData.TotalTime      += dt
        
        OutputCounter += dt
        if OutputCounter >= SimMetaData.OutputEach
            OutputCounter = 0.0
            OutputIterationCounter += 1
            # OutVTP is based on a well-developed Julia package, WriteVTK, while CustomVTP is based on my hand-rolled solution.
            # CustomVTP is about 10% faster, but does not mean much in this case.
            @timeit HourGlass "6.1 outputting savefile" PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(OutputIterationCounter,6,"0") * ".vtp", to_3d(Position.V), ["Kernel", "KernelGradient", "Density", "Pressure", "Acceleration" , "Velocity"], Kernel, to_3d(KernelGradient.V), Density, Pressureᵢ, Acceleration.V, Velocity.V)
        end

        @timeit HourGlass "6.2 updating progress bar" next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime))

        if SimMetaData.TotalTime >= SimMetaData.SimulationTime + 1e-3
            break
        end
    end

    # Print the timings in the default way
    disable_timer!(HourGlass)
    show(HourGlass,sortby=:name)
    show(HourGlass)

    return nothing
end

# Initialize Simulation
begin
    # Actual run
    D          = 2
    T          = Float64
    SimMetaData  = SimulationMetaData{D, T}(
                                    SimulationName="MySimulation", 
                                    SaveLocation=raw"E:\SecondApproach\Results", 
                                    SimulationTime=0.5, #seconds
                                    OutputEach=0.02,  #seconds
    )
    # Initialze the constants to use
    SimConstants = SimulationConstants{T}(
        dx = 0.02,
        h  = 1*sqrt(2)*0.02,
        c₀ = 88.14487860902641,
        α  = 0.02
    )
    # Clean up folder before running (remember to make folder before hand!)
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))

    # And here we run the function - enjoy!
    RunSimulation(
        FluidCSV     = "./input/DSPH_DamBreak_Fluid_Dp0.02.csv",
        BoundCSV     = "./input/DSPH_DamBreak_Boundary_Dp0.02.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstants
    )
end
