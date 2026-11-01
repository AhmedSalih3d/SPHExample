# DualSPHysics vresolution/06_Special: a heaving box on a damped linear spring.
include(joinpath(@__DIR__,"VResolutionPointAbsorberSpring2d","Setup.jl"))

# Our ordinary solver settings; geometry, wave and link dimensions follow the
# existing case. This runs only SPHExampleGPU and reads bundled reference data.
run_point_absorber_spring(
    dx=0.01,
    ratio=2,
    duration=10.0,
    output_interval=0.05,
    h_coefficient=1.5,
    viscosity=0.05,
    diffusion=0.1,
    CFL=0.2,
    stiffness=100.0,
    damping=500.0,
    rest_length=0.325,
)
