# SPH Example

An example of how to write a relatively simple SPH (Smoothed Particle Hydrodynamics) dam-break simulation in Julia. Feel free to star the package if it helped you and please do let me know if you ended up using this to learn SPH yourself, used it in your teaching etc. It is really a motivation booster to hear!

If for some reason you end up pointing to this repository due to you writing a thesis, paper etc., I would love to know!

## Description

The purpose of this code is to serve as an introduction to the SPH (Smoothed Particle Hydrodynamics) method. I have been working in SPH for quite a few years now and noticed that although great software packages exist in this field, it was difficult to find a "simple" example of setting up an SPH solver.

To fill this "void" I decided to go about writing one and learning the necessary steps to do so. The choice of language was Julia, since I've been part of this community for some years and really believe in the concept behind Julia. Also it is a language with a syntax complexity level of around Python, so it would still serve as good inspiration for others wanting to write their own dam break code to look through this.

Key-elements of the code are:

- Weakly Compressible SPH
  - Density varies about ~1% in time, for numerical reasons and the pressure equation is based on the density
- Single threaded approach
  - For simplicity since I consider this a teaching example. Commercial scale SPH should always be done on GPU's anyways.
- Dynamic Boundary Condition (as in DualSPHysics)
  - DualSPHysics is one of the most well-known SPH packages. Thought it would be a good idea to show how one implements this boundary condition.
- Density Diffusion
  - Necessary to produce a non-noise density field, which is important for the momentum equation, since pressure is a function of density in weakly compressible SPH
- Wendland Quintic Kernel (as in DualSPHysics)
  - One of the simpler kernels which does not require tensile correction to be applied.

*Please* remember that this code is not made to be performant. It is made to teach and showcase one way to code a relatively simple SPH dam break.

## Getting Started

### Introduction
The package is structured into "input" and "src". "input" contains some pre-generated particle layouts from DualSPHysics in .csv format. These paths are specified in the "RunSimulation" function, not as an input but as something which should be changed inside the function and recompiled. The "src" package contains all the code files used in this process. An overview of these is shown;

* MainSimulation
  * This script is run to start the simulation
* PreProcess
  * Function for loading in the files in "input"
* PostProcess
  * Function to output .vtp files
* AuxillaryFunctions
  * Not used, but provided as a service to extract all particle ids based on a neighbour list
* TimeStepping
  * Some simple time stepping controls
* SimulatioNEquations
  * All SPH related physics functions

A few key-packages are used and automatically imported. Listed here for your convenience:

* CellListMap
  * A package which allows to return an array of tuples consisting of (particle i, particle j, distance between particle i and particle j), which significantly simplifies and speeds up the calculation process.
* WriteVTK
  * A package which outputs results calculated in this package to the .vtp format. Remember to view results in Paraview you have to select something other than "Solid Color" and for example "Point Gaussian" instead of "Surface" repesentation.

Without these, this example would have been significantly harder to write - and of course thank you to the Julia eco-system as a whole. 

### Executing program

In "MainSimulation.jl"  you will find:

```julia
RunSimulation(SaveLocation="E:/SecondApproach/Results",SimulationName="DamBreak")
```

You have to make the save location before hand for the code to work. Then you can call the code using the following syntax from above. Inside the "RunSimulation" function you will find hard coded:

```julia
    ### VARIABLE EXPLANATION
    # FLUID_CSV = PATH TO FLUID PARTICLES, SEE "input" FOLDER
    # BOUND_CSV = PATH TO BOUNDARY PARTICLES, SEE "input" FOLDER
    # ρ₀  = REFERENCE DENSITY
    # dx  = INITIAL PARTICLE DISTANCE, SEE "dp" IN CSV FILES, FOR 3D SIM: 0.0085
    # H   = SMOOTHING LENGTH
    # m₀  = INITIAL MASS (REFERENCE DENSITY * DX^(SIMULATION DIMENSIONS))
    # mᵢ  = mⱼ = m₀ | ALL PARTICLES HAVE THE SAME MASS, ALWAYS
    # αD  = NORMALIZATION CONSTANT FOR KERNEL
    # α   = ARTIFICIAL VISCOSITY ALPHA VALUE
    # g   = GRAVITY (POSITIVE!)
    # c₀  = SPEED OF SOUND, MUST BE 10X HIGHEST VELOCITY IN SIMULATION
    # γ   = GAMMA, MOST COMMONLY 7 FOR WATER, USE FOR PRESSURE EQUATION OF STATE
    # dt  = INITIAL TIME STEP
    # δᵩ  = 0.1 | COEFFICIENT FOR DENSITY DIFFUSION, SHOULD ALWAYS BE 0.1
    # CFL = CFL NUMBER

    ### 2D Dam Break
    FLUID_CSV = "./input/FluidPoints_Dp0.02.csv"
    BOUND_CSV = "./input/BoundaryPoints_Dp0.02.csv"
    ρ₀  = 1000
    dx  = 0.02
    H   = 1.2*sqrt(2)*dx
    m₀  = ρ₀*dx*dx #mᵢ  = mⱼ = m₀
    αD  = (7/(4*π*H^2))
    α   = 0.01
    g   = 9.81
    c₀  = sqrt(g*2)*20
    γ   = 7
    dt  = 1e-5
    δᵩ  = 0.1
    CFL = 0.2
```

These parameters have been tested and should work out of the box. Other than these inputs you should not have to change anything else. 

## Help

Any questions about the code feel free to post an issue on this repository. Please do understand it might take some time for me to respond back.

## Authors

Written by Ahmed Salih [@AhmedSalih3D](https://github.com/AhmedSalih3d)

## Version History

* 0.1 Release Version
* 0.2 I expect some feedback would be given, which leads to a 0.2 update of the package

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

DualSPHysics (https://dual.sphysics.org/) was a great inspiration for this code.

Thank you to the general Julia eco-system and especially Leandro Martínez (https://github.com/lmiq) who is also the author of CellListMap.jl. He was a great help in understanding how to best use the neighbour-list algorithm. 