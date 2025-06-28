# SPH Example

A Julia implementation of a basic Smoothed Particle Hydrodynamics (SPH) solver. The code is primarily written for educational purposes and contains several ready to run examples such as 2D and 3D dam breaks or wedge impacts. Custom particle setups can also be loaded from the `input/` folder.

Please consider giving the project a :star: if it is useful in your work. Citation is appreciated for academic use and feel free to reach out with feedback or questions.

The code can produce a 2D dam break ([@DamBreak2D-Video](https://www.youtube.com/watch?v=7kDVjZkc_TI)):

https://github.com/user-attachments/assets/a0070389-e2a5-4bf8-9eda-e40364eea7ce

Or, if you are really patient (1+ day to calculate), a 3D case ([@DamBreak3D-Video](https://www.youtube.com/watch?v=_2e6LopvIe8)):

https://github.com/user-attachments/assets/a38aaf39-3cf3-4041-b983-03f6107de8b9

## Description


The project demonstrates how to assemble a small SPH solver with Julia. It focuses on clarity rather than ultimate performance. Unofficial benchmarks suggest that for 2D cases the CPU runtime is comparable to DualSPHysics. Key features include:

- **Weakly compressible formulation** – density varies ~1 % and pressure is a function of density.
- **Multi-threaded execution** – achieved by spawning the neighbour loop.
- **Dynamic boundary condition** – inspired by DualSPHysics.
- **Density diffusion** – based on Fourtakas et al. 2019 to reduce pressure noise.
- **Wendland quintic kernel** – simple and stable without tensile corrections.

## Folder Structure

```
.
├── example/          # Ready to run simulations
├── input/            # Pre-generated particle layouts (.csv)
├── src/              # Package source code
├── images/           # Images used in this README
├── Project.toml      # Package dependencies
└── Manifest.toml     # Exact dependency versions
```

Example scripts live in `example/` and read geometry from `input/`. The solver code itself is in `src/`:

```
src/
├── AuxiliaryFunctions.jl            # Small helper utilities
├── OpenExternalPrograms.jl          # Convenience wrappers for logs and ParaView
├── PreProcess.jl                    # Load inputs and allocate arrays
├── ProduceHDFVTK.jl                 # Write simulation data in HDF5/VTK format
├── SPHCellList.jl                   # Custom neighbour search and time stepping
├── SPHDensityDiffusionModels.jl     # Density diffusion implementations
├── SPHExample.jl                    # Glue module re-exporting all functions
├── SPHKernels.jl                    # SPH kernel definitions
├── SPHViscosityModels.jl            # Viscosity models such as Laminar or SPS
├── SimulationConstantsConfiguration.jl  # User-facing solver parameters
├── SimulationEquations.jl           # Core SPH physics equations
├── SimulationGeometry.jl            # Domain and geometry definitions
├── SimulationLoggerConfiguration.jl # Logging helpers for timer outputs
├── SimulationMetaDataConfiguration.jl  # Metadata such as run time and output path
└── TimeStepping.jl                  # Controls for Δt and CFL condition
```

## Getting Started

### Installation

The easiest way to experiment with the code is to clone the repository and activate it in Julia:

```julia
using Pkg
Pkg.activate("/path/to/SPHExample")
Pkg.instantiate()
```

Alternatively, install it directly:

```julia
using Pkg
Pkg.add(url="https://github.com/AhmedSalih3d/SPHExample")
```

### Running an Example

Open one of the files in `example/`, for instance `example/StillWedgeMDBC.jl`, and adjust the simulation parameters or the `ComputerInteractions!` function. Run the script to start the simulation. Results are written in `hdfvtk` format which can be loaded with ParaView 5.12 or newer.

## Help

Questions or issues can be posted on the GitHub issue tracker. Response times may vary but all feedback is welcome.

## Authors

Written by Ahmed Salih ([AhmedSalih3d](https://github.com/AhmedSalih3d)).

## Version History

| Version | Description |
|---------|-------------|
| 0.6.10 | Implemented concepts of tests, aim is to understand allocations and run time |
| 0.6.9  | Specify output times via `OutputTimes` (float or vector). |
| 0.6.8  | Select which variables are written to `vtkhdf` files. |
| 0.6.7  | Introduced mDBC boundary conditions and other improvements allowing particles to interact with boundaries. |
| 0.6.6  | Added neighbour grid visualisation in ParaView for debugging. |
| 0.6.5  | Linearised density diffusion, optional single-file output and performance improvements. |
| 0.6.4  | Revised geometry configuration interface and added time-step plot. |
| 0.6.3  | Added automatic log visualisation and `CloseHDFVTKManually` helper. |
| 0.6.2  | Added automatic ParaView visualisation support. |
| 0.6    | Major rewrite with solver setup changes and moving object support. |
| 0.5    | Logging and `hdfvtk` output added. |
| 0.4    | Complete rewrite focusing on custom cell lists. |
| 0.3    | Highly optimised CPU version with minimal allocations. |
| 0.2    | Cleanup of initial release. |
| 0.1    | Initial release. |

## License

This project is licensed under the MIT License – see [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- [DualSPHysics](https://dual.sphysics.org/) for inspiration.
- Many thanks to the Julia community and especially [Leandro Martínez](https://github.com/lmiq) for guidance on neighbour-list algorithms.
- Thanks to [PharmCat](https://github.com/PharmCat) for suggestions and code contributions.

[![Star History](https://api.star-history.com/svg?repos=AhmedSalih3d/SPHExample)](https://star-history.com/#AhmedSalih3d/SPHExample)

