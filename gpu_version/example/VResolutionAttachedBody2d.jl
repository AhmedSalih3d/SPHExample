using SPHExampleGPU, StaticArrays

"""
    run_attached_resolution_example(; floating=true, duration=0.5, dx=0.04,
        save_location=joinpath(@__DIR__, "output", "AttachedResolution"))

A refined patch travels and turns with a square body in a DBC tank. Select
`floating=false` for prescribed translation/rotation instead of fluid-driven
rigid-body motion. Geometry is uploaded once; both resolution domains and the
body advance on the GPU between output frames.
"""
function run_attached_resolution_example(;floating=true,duration=0.5,dx=0.04,
        save_location=joinpath(@__DIR__,"output","AttachedResolution"))
    T = Float32
    marker = 5
    function geometry(spacing)
        fluid = [SVector(x,y) for x in -1.5:spacing:1.5 for y in -0.9:spacing:0.3
            if !(abs(x) <= 0.16 && abs(y+0.2) <= 0.16)]
        wall = [SVector(x,y) for x in (-1.5-3spacing):spacing:(1.5+3spacing)
            for y in (-0.9-3spacing):spacing:0.6 if x < -1.5 || x > 1.5 || y < -0.9]
        square = [SVector(x,y) for x in -0.12:spacing:0.12 for y in -0.32:spacing:-0.08]
        return [SPHGeometry{2,T}(wall;Density=T(1000),Type=Fixed,GroupMarker=1),
            SPHGeometry{2,T}(fluid;Density=T(1000),Type=Fluid,GroupMarker=2),
            SPHGeometry{2,T}(square;Density=T(1000),Type=floating ? Floating : Moving,GroupMarker=marker,
                Floating=floating ? FloatingDetails{T}(;RelativeWeight=T(0.5)) : nothing)]
    end
    coarse,fine = geometry(dx),geometry(dx/2)
    config = VariableResolutionConfig(VariableResolutionBox((-0.7,-0.85),(0.7,0.55));
        fine_geometry=fine,follow=marker,particle_reserve=20000)
    constants = SimulationConstants{T}(2;dx,c₀=25,g=9.81)
    kernel = SPHKernelInstance{2,T}(WendlandC2();h=T(1.3dx))
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        SimulationName=floating ? "FloatingRefinement" : "MovingRefinement",SaveLocation=save_location,
        SimulationTime=duration,OutputTimes=0.01,VariableResolution=config,
        GPUUseGraph=true,GPUClipTimeStepToOutput=true,VisualizeInParaview=false,OpenLogFile=false,
        OutputVariables=["Velocity","Density","Pressure","Type","GroupMarker","Resolution","Spacing","Owned"])
    motion = floating ? nothing : PrescribedMotion(marker;center=(0,-0.2),
        translation=MotionSequence([KinematicSegment(10;velocity=(0.3,0.0))]),
        rotation=MotionSequence([KinematicSegment(10;velocity=0.4)]))
    particles = AllocateDataStructures(coarse,metadata)
    logger = SimulationLogger(save_location)
    try
        return RunSimulation(;SimGeometry=coarse,SimConstants=constants,SimKernel=kernel,
            SimMetaData=metadata,SimParticles=particles,SimLogger=logger,SimMotion=motion,
            SimTimeStepping=SymplecticTimeStepping(),SimViscosity=ArtificialViscosity(),
            SimDensityDiffusion=LinearDensityDiffusion())
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_attached_resolution_example()
