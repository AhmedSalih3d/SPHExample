# Recirculating flow over an L-shaped barrier, matching the case pictured in
# DualSPHysics' "CaseRZ_Flow2D" video. Requires an NVIDIA GPU.
# julia --project=gpu_version -t 2,0 gpu_version/example/RZFlow2d.jl
include(joinpath(@__DIR__, "RZFlow2d", "FlowSetup.jl"))

let
    case = rz_flow_case()
    mkpath(case.metadata.SaveLocation)
    particles = AllocateDataStructures(case.geometry, case.metadata)
    logger = SimulationLogger(case.metadata.SaveLocation)
    RunSimulation(
        SimGeometry = case.geometry,
        SimMetaData = case.metadata,
        SimConstants = case.constants,
        SimKernel = case.kernel,
        SimLogger = logger,
        SimParticles = particles,
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = ZeroDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        SimDamping = case.relaxation,
    )
end
