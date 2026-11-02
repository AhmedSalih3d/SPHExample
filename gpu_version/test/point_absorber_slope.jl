module PointAbsorberSlopeTests
using Test, SPHExampleGPU, StaticArrays, LinearAlgebra, CUDA
let path = joinpath(@__DIR__, "..", "example", "VResolutionPointAbsorberSpring2d.jl")
    source = replace(read(path, String), r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n")
    include_string(@__MODULE__, source, path)
end

"A small closed DBC tank isolates the sloped-bed placement from waves and refinement."
function run_static_slope(aligned; duration = 0.5, directory = mktempdir())
    T = Float32
    spacing, depth, width = 0.02, 0.4, 3.0
    slope = 0.6 / 6.7
    bed(x) = slope*x
    normal = normalize(SVector(-slope, 1.0))
    nx = round(Int, width/spacing)
    fixed = [SVector(i*spacing, bed(i*spacing)) - layer*spacing*normal
        for i in -3:nx+3 for layer in 0:2]
    for side in (0, 1), layer in 0:2
        x = side == 0 ? -layer*spacing : width+layer*spacing
        for j in 0:ceil(Int, (depth+0.1-bed(x))/spacing)
            push!(fixed, SVector(x, bed(x)+j*spacing))
        end
    end
    unique!(fixed)
    water = if aligned
        # Translate the example's actual slope sampler into this small tank.
        point_absorber_water(spacing, SVector(2.5+spacing,0.0), SVector(2.5+width-spacing,depth);
            exclude_body=false) .- Ref(SVector(2.5,0.0))
    else
        [SVector(i*spacing,j*spacing) for i in 1:nx-1 for j in 1:ceil(Int,depth/spacing)-1
            if j*spacing >= bed(i*spacing)+spacing-1e-10]
    end
    constants = SimulationConstants{T}(2; dx=spacing, c₀=20sqrt(9.81depth), α=0.05, δᵩ=0.1, CFL=0.2)
    h = 1.2sqrt(2)*spacing
    kernel = SPHKernelInstance{2,T}(WendlandC2(); h=T(h), η²=T((0.1h)^2))
    geometry = [SPHGeometry{2,T}(fixed; Density=constants.ρ₀, Type=Fixed, GroupMarker=1),
        SPHGeometry{2,T}(water; Density=hydrostatic_density(water,constants;water_level=depth), Type=Fluid, GroupMarker=2)]
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(;
        SimulationName=aligned ? "AlignedSlope" : "CartesianSlope", SaveLocation=directory,
        SimulationTime=duration, OutputTimes=duration/5, ExportSingleVTKHDF=true, ExportGridCells=false,
        GPUDoublePosition=true, GPUInteractionFastMath=false, GPULanesPerParticle=1,
        GPUBoundaryForces=false, GPUUseGraph=true, GPUClipTimeStepToOutput=true,
        VisualizeInParaview=false, OpenLogFile=false,
        OutputVariables=["Velocity","Density","Pressure","Type","ID"])
    particles = AllocateDataStructures(geometry,metadata)
    mkpath(directory)
    logger = SimulationLogger(directory)
    result = try
        redirect_stdout(devnull) do
            RunSimulation(; SimGeometry=geometry, SimConstants=constants, SimKernel=kernel,
                SimParticles=particles, SimMetaData=metadata, SimLogger=logger,
                SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
                SimTimeStepping=SymplecticTimeStepping())
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    p = result.final_state.particles
    ids = findall(i -> p.Type[i] == Fixed && 0.4 < p.Position[i][1] < 2.6 &&
        abs(p.Position[i][2]-bed(p.Position[i][1])) < 1e-7, eachindex(p.Type))
    sort!(ids; by=i -> p.Position[i][1])
    pressure = [Float64(particle_pressure(p.Density[i],Fixed,constants.c₀,constants.ρ₀)) for i in ids]
    hydrostatic = [1000*9.81*(depth-p.Position[i][2]) for i in ids]
    error = pressure-hydrostatic
    return (; result, metadata, pressure, hydrostatic, x=[p.Position[i][1] for i in ids],
        rms_error=sqrt(sum(abs2,error)/length(error)), max_error=maximum(abs,error),
        fluid_count=length(water), max_speed=maximum(norm,p.Velocity))
end

if CUDA.functional()
    @testset "DBC slope placement suppresses periodic pressure spikes" begin
        original = run_static_slope(false)
        aligned = run_static_slope(true)
        for case in (original, aligned)
            @test case.result.final_state.complete
            @test case.result.diagnostics.frozen_particles == 0
            @test count(==(Fluid),case.result.final_state.particles.Type) == case.fluid_count
            @test all(isfinite,case.result.final_state.particles.Density)
        end
        @test aligned.rms_error < 0.35original.rms_error
        @test aligned.max_error < 0.35original.max_error
        @test maximum(aligned.pressure) < 2maximum(aligned.hydrostatic)
        @test aligned.max_speed < 0.05
        println("Static DBC slope pressure error (Pa): ", (; original_rms=original.rms_error,
            aligned_rms=aligned.rms_error, original_max=original.max_error, aligned_max=aligned.max_error))
    end
end
end
