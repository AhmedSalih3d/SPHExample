# Run from gpu_version:
# julia --project=. -t 2,0 "validation cases/spheric_test_04/manual_case.jl"
using SPHExampleGPU
using StaticArrays: SVector
using LinearAlgebra: norm

function run_ripple_case(; oscillation=1, dx=0.005, duration=8.46, frames=85,
        slot=1, overwrite=true, visualize=false, boundary=SimpleMDBC,
        coefh=2.0, sound_speed=10.0, viscosity=1e-6)
    oscillation in 1:3 || throw(ArgumentError("Choose oscillation 1, 2, or 3"))
    slot in 1:3 || throw(ArgumentError("Choose run slot 1, 2, or 3"))
    1 <= frames <= 200 || throw(ArgumentError("Choose 1 to 200 output intervals"))
    all(x -> isfinite(x) && x > 0, (dx, duration, coefh, sound_speed, viscosity)) ||
        throw(ArgumentError("Spacing, duration, coefh, sound speed and viscosity must be positive and finite"))
    boundary in (NoMDBC, SimpleMDBC) || throw(ArgumentError("Choose NoMDBC or SimpleMDBC"))

    length_x, height, wavelength, radius = 1.6, 0.5, 0.22, 0.19
    nx, ny = round(Int, length_x / dx), round(Int, height / dx)
    isapprox(nx*dx, length_x) && isapprox(ny*dx, height) ||
        throw(ArgumentError("dx must divide both 1.6 m and 0.5 m"))
    nx * ny <= 500_000 || throw(ArgumentError("This manual case allows at most 500,000 fluid lattice sites"))
    period = 8.46
    amplitude = (0.131, 0.196, 0.294)[oscillation]
    T = Float32
    constants = SimulationConstants{T}(2; dx, ρ₀=1000, g=0, c₀=sound_speed, ν₀=viscosity, CFL=0.2)
    kernel = SPHKernelInstance{2,T}(WendlandC2(); h=T(coefh*sqrt(2)*dx))
    layers = ceil(Int, kernel.H / dx) + 1
    nx * (ny + 2layers) <= 500_000 || throw(ArgumentError("This manual case allows at most 500,000 total lattice sites"))

    # Each half ripple is a circular arc tangent to the flat bed at its trough.
    fluid = SVector{2,Float64}[]
    walls = similar(fluid)
    offsets = similar(fluid)
    for i in 0:nx-1
        x = (i+0.5)*dx
        from_center = x - length_x/2
        from_crest = mod(from_center + wavelength/2, wavelength) - wavelength/2
        on_bed = abs(from_center) < 1.5*wavelength
        u = on_bed ? wavelength/2 - abs(from_crest) : 0.0
        arc = sqrt(radius^2-u^2)
        bed = radius-arc
        slope = iszero(from_crest) ? 0.0 : -sign(from_crest)*u/arc
        normal = SVector(-slope, 1.0)
        normal /= norm(normal)
        for j in 0:layers-1
            distance = (j+0.5)*dx
            push!(walls, SVector(x, bed-distance), SVector(x, height+distance))
            push!(offsets, 2distance*normal[2]*normal, SVector(0.0, -2distance))
        end
        for j in 0:ny-1
            push!(fluid, SVector(x, bed+(j+0.5)*dx))
            last(fluid)[2] < height || pop!(fluid)
        end
    end

    walls_geometry = if boundary === SimpleMDBC
        wrapped_ghosts = [SVector(mod(p[1]+o[1], length_x), p[2]+o[2]) for (p,o) in zip(walls,offsets)]
        particles = particle_struct_array(walls, constants.ρ₀;
            GhostPoints=wrapped_ghosts, GhostNormals=SVector{2,T}.(offsets))
        SPHGeometry{2,T}(; Particles=particles, GroupMarker=1, Type=Fixed)
    else
        SPHGeometry{2,T}(walls; Density=constants.ρ₀, GroupMarker=1, Type=Fixed)
    end
    water = SPHGeometry{2,T}(fluid; Density=constants.ρ₀, GroupMarker=2, Type=Fluid)
    geometry = [walls_geometry, water]

    # Uniform sinusoidal acceleration gives a sine-phase far-field velocity
    # in the inviscid limit, starting from rest.
    ω = 2π/period
    force_times = range(0.0, duration; length=max(2,ceil(Int,400duration/period)+1))
    forcing = ExternalForce{2,T}(2; times=collect(force_times),
        linear_acceleration=[SVector(amplitude*ω*cos(ω*t), 0.0) for t in force_times])

    run_root = joinpath(@__DIR__, "runs")
    save = joinpath(run_root, "osc$(oscillation)_slot$(slot)")
    for path in (run_root, save)
        islink(path) && error("Output directory cannot be a symbolic link: $path")
        isdir(path) || mkpath(path)
    end
    realpath(save) == joinpath(realpath(run_root), basename(save)) || error("Run folder escapes runs/")
    owned = ("RippledBed.vtkhdf", "RippledBed_SingleVTKHDFStateFile.py", "SimulationOutput.log", "settings.txt")
    lock = joinpath(save, ".running")
    mkdir(lock)
    try
        for name in owned
            path = joinpath(save, name)
            islink(path) && error("Output cannot be a symbolic link: $path")
            if ispath(path)
                isfile(path) || error("Expected a regular output file: $path")
                overwrite || error("Run slot already contains output. Choose another slot or enable overwrite.")
            end
        end
        times = collect(range(duration/frames, duration; length=frames))
        metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,boundary,StoreLog}(;
            SimulationName="RippledBed", SaveLocation=save, SimulationTime=duration, OutputTimes=times,
            OutputVariables=["Velocity", "Pressure", "Type"], ExportSingleVTKHDF=true,
            Periodicity=PeriodicBoundary{2}(1; min=0, max=length_x),
            GPUDoublePosition=true, GPUBoundaryForces=false, OpenLogFile=false,
            VisualizeInParaview=visualize)
        particles = AllocateDataStructures(geometry, metadata)
        logger = SimulationLogger(save)
        result = try
            RunSimulation(; SimGeometry=geometry, SimMetaData=metadata, SimConstants=constants,
                SimKernel=kernel, SimLogger=logger, SimParticles=particles, SimViscosity=Laminar(),
                SimDensityDiffusion=ZeroGravityLinearDensityDiffusion(),
                SimTimeStepping=SymplecticTimeStepping(), SimExternalForces=forcing)
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        result.termination_reason === :completed && result.diagnostics.frozen_particles == 0 ||
            error("Run terminated early or froze particles. See $save")
        open(joinpath(save, "settings.txt"), "w") do io
            println(io, (; oscillation, slot, boundary, dx, duration, frames, coefh, sound_speed,
                viscosity, period, amplitude, fluid=length(fluid), walls=length(walls)))
        end
        println("Results saved in ", save)
        return result
    finally
        rm(lock)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ripple_case(;
        oscillation=1,       # 1, 2, or 3 from the benchmark table
        dx=0.005,           # metres
        duration=8.46,       # seconds
        frames=85,           # output intervals plus the initial state
        slot=1,              # 1, 2, or 3 for each oscillation
        overwrite=true,      # reuse this slot; false protects existing output
        visualize=false,     # true opens ParaView after the run
        boundary=SimpleMDBC, # NoMDBC selects DBC
        coefh=2.0,
        sound_speed=10.0,    # m/s
        viscosity=1e-6,      # m²/s
    )
end
