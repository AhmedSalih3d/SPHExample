# SPH Example

This package serves as an example of how to write a relatively simple SPH (Smooehd Particle Hydrodynamics) simulation in Julia. A few examples are provided out of the box; still wedge and dam break. Custom particle distributions can be loaded in as well. 

Please consider 🌟 the package if it has been useful for you. I would love to know if you have used it to learn SPH, for your teaching and more, it really motivates me!

If you are using this package for an academic project or a scientific paper, please do cite the project - and feel free to reach out too!


Below are some examples of what the code can run: 

The code can produce a 2D dam-break ([@DamBreak2D-Video](https://www.youtube.com/watch?v=7kDVjZkc_TI)):

![plot](./images/2d_dambreak.png)

Or if you are really patient (1+ day to calculate) a 3D case ([@DamBreak3D-Video](https://www.youtube.com/watch?v=_2e6LopvIe8))::

![plot](./images/3d_dambreak.png)

## Description

The purpose of this code is to serve as an introduction to the SPH (Smoothed Particle Hydrodynamics) method. I have been working in SPH for quite a few years now and noticed that although great software packages exist in this field, it was difficult to find a "simple" example of setting up an SPH solver.

To fill this "void" I decided to go about writing one and learning the necessary steps to do so. The choice of language was Julia, since I've been part of this community for some years and really believe in the concept behind Julia. It is a language with a syntax complexity level of around Python/Matlab. I think it can serve as an inspiration even if you are not exactly familiar with the Julia language in the beginning.

Key-elements of the code are:

- Weakly Compressible SPH
  - Density varies about ~1% in time, for numerical reasons and the pressure equation is based on the density. This ensured by enforcing a Mach Number of 0.1, by artificially setting the speed of sound of the fluid (c₀) to ten times the highest velocity in the flow (manual input) 
- Multi-threaded approach
  - Multi-threading has been added. It is achieved using `ChunkSplitters.jl` and code can easily be run in sequential form by removing the `@threads` in the `NeighborLoop` function in `src/SPHCellList.jl`.
- Dynamic Boundary Condition (as in DualSPHysics)
  - DualSPHysics is one of the most well-known SPH packages. This is one of the simplest and most elegant boundary conditions.
- Density Diffusion
  - Necessary to produce a non-noise density field, which is important for the momentum equation, since pressure is a function of density in weakly compressible SPH. The formulation of Fourtakas et. al. 2019 is utilized right now.
- Wendland Quintic Kernel (as in DualSPHysics)
  - One of the simpler kernels which does not require tensile correction to be applied.

*Please* remember that the main aim of this code is not to be performant. It is made to teach and showcase one way to code a relatively simple SPH code. *Unofficially*  I have benchmarked this code up against DualSPHysics similar cases and found that for 2D simulations this code is on par with DualSPHysics. 

## Getting Started

### Introduction
The package is structured into "input", "example" and "src". "input" contains some pre-generated particle layouts from DualSPHysics in .csv format. 

The "src" package contains all the code files used in this process. An overview of these is shown;

* PreProcess
  * Function for loading in the files in "input". 
* PostProcess
  * Function to output .vtp files. "ProduceVTP.jl" is a hand-rolled custom solution for those interested in that.
* AuxillaryFunctions
  * To store some small, repeatedly used smaller functions
* TimeStepping
  * Some simple time stepping controls
* SimulationConstantsConfigurations
  * The interface for stating the most relevant simulation constants is found here
* SimulationMetaDataConfiguration
  * The interface for the meta data associated with a simulation, such as total time, save location etc.
* SimulationEquations
  * SPH physics functions
* SPHCellList
  * Holds the main code for the custom neighbor finding algorithm which I've made for this project
* SPHExample
  * The "glue" package file exporting all the functions, to allow for `using SPHExample`.  

### Installation

It is recommended to `git clone ..` into a folder and following the instructions in `Executing program`. The `using Pkg; Pkg.add(url="https://github.com/AhmedSalih3d/SPHExample")` should also work if one is simply interesting in installing and using some package functionality. 

Then you can open one of the files in `example` such as `example\still_wedge.jl` and modify the `ComputerInteractions!` function which determines particle interaction to your liking or `SimulationLoop` which controls the time-stepping. You also have access to the whole `RunSimulation` functionality and can adapt that to your needs. 

Since this is the preferred workflow and the way to gain most out of this package, please use the `git clone ..` approach. 

## Help

Any questions about the code feel free to post an issue on this repository. Please do understand it might take some time for me to respond back.

## Authors

Written by Ahmed Salih [@AhmedSalih3D](https://github.com/AhmedSalih3d)

## Version History

* (main) Version 0.4 | Complete rewrite, letting go of `LoopVectorization.jl` and `CellListMap.jl` to only code exactly what is needed and improve performance. 
* Version 0.3 | A highly optimized version for CPU, with extremely few allocations after the initial array allocation. Only the neighbour search and saving of data allocates memory now. Recommend to use this. (Not provided anymore)
* Version 0.2        | A cleaned up version of the original release version data allocates memory now. (Not provided anymore)
* Version 0.1        | Initial release version

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

DualSPHysics (https://dual.sphysics.org/) was a great inspiration for this code.

Thank you to the general Julia eco-system and especially Leandro Martínez (https://github.com/lmiq) who is also the author of CellListMap.jl. He was a great help in understanding how to best use the neighbour-list algorithm. Leandro was also a big part of reviewing the whole code base and suggesting/showing potential optimizations. 

Thank you for PharmCat (https://github.com/PharmCat) for his suggestions in pull requests and providing of code, which I later implemented into the library. Much appreciated. 
