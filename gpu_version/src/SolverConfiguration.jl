"""Selection of the pressure formulation, independently of WCSPH time stepping."""
module SolverConfiguration

export SPHSolver, WCSPH, IISPH

abstract type SPHSolver end

"""`WCSPH()` selects the existing equation-of-state solver (the default)."""
struct WCSPH <: SPHSolver end

"""
    IISPH(; relative_density_tolerance=1e-3, min_iterations=2,
          max_iterations=1000, relaxation=0.1, cfl=0.25,
          max_timestep=Inf, warm_start=0.5)

Matrix-free implicit incompressible SPH with projected relaxed Jacobi pressure
iterations and symplectic Euler integration. `relative_density_tolerance` bounds
the maximum linearized density constraint residual divided by rest density.
Underfilled free-surface particles may have zero pressure; at positive pressure
both compression and expansion residuals are checked. The tolerance is not a
bound on the nonlinear summation density after particle advection.

`cfl` controls velocity and acceleration timestep limits; explicit laminar
viscosity also limits the step. `max_timestep` is an optional physical-time cap.
`warm_start` multiplies the previous accepted pressure before each solve. Failure
to converge throws before velocities or positions are advanced.

The initial GPU implementation supports uniform-resolution 2D/3D fluid and
stationary fixed walls, NoMDBC, NoShifting, NoKernelOutput, ZeroDensityDiffusion,
and ZeroViscosity or Laminar. Fixed walls must sample the solid volume at the
fluid resolution, with volume `m₀/ρ₀` per sample and enough layers to fill the
kernel support. Surface-only wall sampling is unsupported. The conservative
default relaxation accommodates the package's `h=2dx` neighbour stencil.
Pressure is stored independently of density. Pressure iteration and output run
directly and synchronously; CUDA graphs and asynchronous output are deferred.
"""
struct IISPH <: SPHSolver
    relative_density_tolerance::Float64
    min_iterations::Int
    max_iterations::Int
    relaxation::Float64
    cfl::Float64
    max_timestep::Float64
    warm_start::Float64

    function IISPH(; relative_density_tolerance=1e-3, min_iterations=2,
                   max_iterations=1000, relaxation=0.1, cfl=0.25,
                   max_timestep=Inf, warm_start=0.5)
        isfinite(relative_density_tolerance) && 0 < relative_density_tolerance < 1 ||
            throw(ArgumentError("relative_density_tolerance must be finite and between zero and one"))
        min_iterations isa Integer && max_iterations isa Integer && 1 <= min_iterations <= max_iterations ||
            throw(ArgumentError("IISPH requires 1 <= min_iterations <= max_iterations"))
        isfinite(relaxation) && 0 < relaxation <= 1 || throw(ArgumentError("relaxation must be in (0, 1]"))
        isfinite(cfl) && 0 < cfl <= 1 || throw(ArgumentError("cfl must be in (0, 1]"))
        max_timestep > 0 && !isnan(max_timestep) || throw(ArgumentError("max_timestep must be positive"))
        isfinite(warm_start) && 0 <= warm_start <= 1 || throw(ArgumentError("warm_start must be in [0, 1]"))
        new(relative_density_tolerance, min_iterations, max_iterations, relaxation, cfl, max_timestep, warm_start)
    end
end

end
