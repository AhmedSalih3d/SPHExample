using SPHExampleGPU

"""
    run_impinging_jet(; spacing=0.0005, duration=0.1, output_interval=0.0002, ...)

Run the DualSPHysics-inspired 2D impinging jet. Defaults retain the upstream
geometry, jet speed, sound speed, CFL and output cadence. See ImpingingJet2d.md
for the current boundary-model differences and a smaller smoke run.
"""
function run_impinging_jet(; spacing=0.0005,duration=0.1,output_interval=0.0002,
        save_location=joinpath(@__DIR__,"..","output","impinging_jet"),use_graph=true)
    T = Float32
    flow = ImpingingJet{T}(Spacing=spacing,Duration=duration)
    geometry = impinging_jet_geometry(flow)
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName="ImpingingJet2D",SaveLocation=save_location,SimulationTime=duration,
        OutputTimes=output_interval,VisualizeInParaview=false,OpenLogFile=false,
        ExportSingleVTKHDF=true,GPUDoublePosition=false,GPUUseGraph=use_graph,
        GPUClipTimeStepToOutput=true)
    constants = SimulationConstants{T}(2;dx=spacing,g=9.81,c₀=230,CFL=0.1,ν₀=1e-6)
    kernel = SPHKernelInstance{2,T}(WendlandC2();h=T(sqrt(2)*spacing))
    mkpath(save_location)
    logger = SimulationLogger(save_location)
    try
        return RunSimulation(SimGeometry=geometry,SimMetaData=metadata,
            SimParticles=AllocateDataStructures(geometry,metadata),SimConstants=constants,
            SimKernel=kernel,SimLogger=logger,SimViscosity=LaminarSPS(),
            SimDensityDiffusion=FourtakasDensityDiffusion(full=true),SimTimeStepping=SymplecticTimeStepping(),
            SimOpenBoundaries=flow)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_impinging_jet()
end
