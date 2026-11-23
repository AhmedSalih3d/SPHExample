# Run from gpu_version:
# julia --project=. -t 2,0 "validation cases/spheric_test_04/manual_case.jl"
using SPHExampleGPU
using SPHExampleGPU.OpenExternalPrograms: AutoOpenParaview
using StaticArrays: SVector
using LinearAlgebra: norm

let
    # Edit these settings, then execute the whole file in Julia.
    Dimensions = 2
    FloatType = Float32
    oscillation = 1        # 1, 2, or 3 from the benchmark table
    dx = 0.0025             # Particle spacing in metres
    duration = 8.46        # One oscillation, in seconds
    frames = 850            # Output intervals, plus the initial state
    slot = 1               # Three reusable output slots per oscillation
    overwrite = true       # Replace the solver output in this slot
    visualize = true       # Open ParaView when the run finishes
    boundary = SimpleMDBC      # NoMDBC selects DBC
    coefh = 2.0
    sound_speed = 10.0     # m/s
    viscosity = 1e-6       # m?/s

    oscillation in 1:3 || throw(ArgumentError("Choose oscillation 1, 2, or 3"))
    slot in 1:3 || throw(ArgumentError("Choose run slot 1, 2, or 3"))
    frames > 0 || throw(ArgumentError("Choose at least one output interval"))
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
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx,
        ρ₀ = 1000,
        g = 0,
        c₀ = sound_speed,
        ν₀ = viscosity,
        CFL = 0.2,
    )
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2(); h = FloatType(coefh * sqrt(2) * dx))
    layers = ceil(Int, SimKernel.H / dx) + 1
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

    FixedBoundary = if boundary === SimpleMDBC
        wrapped_ghosts = [SVector(mod(p[1]+o[1], length_x), p[2]+o[2]) for (p,o) in zip(walls,offsets)]
        particles = particle_struct_array(walls, SimConstants.ρ₀;
            GhostPoints=wrapped_ghosts, GhostNormals=SVector{Dimensions,FloatType}.(offsets))
        SPHGeometry{Dimensions,FloatType}(; Particles=particles, GroupMarker=1, Type=Fixed)
    else
        SPHGeometry{Dimensions,FloatType}(walls; Density=SimConstants.ρ₀, GroupMarker=1, Type=Fixed)
    end
    Water = SPHGeometry{Dimensions,FloatType}(fluid; Density=SimConstants.ρ₀, GroupMarker=2, Type=Fluid)
    SimulationGeometry = [FixedBoundary, Water]

    # Uniform sinusoidal acceleration gives a sine-phase far-field velocity
    # in the inviscid limit, starting from rest.
    ω = 2π/period
    force_times = range(0.0, duration; length=max(2,ceil(Int,400duration/period)+1))
    SimExternalForces = ExternalForce{Dimensions,FloatType}(2; times=collect(force_times),
        linear_acceleration=[SVector(amplitude*ω*cos(ω*t), 0.0) for t in force_times])

    # Select All in the editor uses the source path. A plain REPL paste uses its working directory.
    case_dir = @__DIR__
    if !isfile(joinpath(case_dir, "manual_case.jl"))
        for candidate in (joinpath(pwd(), "validation cases", "spheric_test_04"),
                          joinpath(pwd(), "gpu_version", "validation cases", "spheric_test_04"))
            isfile(joinpath(candidate, "manual_case.jl")) || continue
            case_dir = candidate
            break
        end
    end
    isfile(joinpath(case_dir, "manual_case.jl")) || error("Start Julia from gpu_version or its repository root")
    run_root = joinpath(case_dir, "runs")
    save = joinpath(run_root, "osc$(oscillation)_slot$(slot)")
    for path in (run_root, save)
        islink(path) && error("Output directory cannot be a symbolic link: $path")
        isdir(path) || mkpath(path)
    end
    realpath(save) == joinpath(realpath(run_root), basename(save)) || error("Run folder escapes runs/")
    owned = ("RippledBed.vtkhdf", "RippledBed_SingleVTKHDFStateFile.py", "SimulationOutput.log", "settings.txt")
    owned = (owned..., "circulation.csv", "circulation_reference.csv",
             "circulation_paraview.pvsm", "circulation_paraview.png")
    python = Sys.which("python")
    pvpython = Sys.which("pvpython")
    isnothing(python) && error("Case 04 requires Python with NumPy, SciPy and h5py on PATH")
    isnothing(pvpython) && error("Case 04 requires ParaView's pvpython on PATH")
    run(`$python -c "import h5py, numpy, scipy.spatial, scipy.ndimage"`)
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
        SimMetaData = SimulationMetaData{
            Dimensions, FloatType, PlanarShifting, NoKernelOutput, boundary, StoreLog,
        }(;
            SimulationName = "RippledBed",
            SaveLocation = save,
            SimulationTime = duration,
            OutputTimes = times,
            OutputVariables = ["Velocity", "Density", "Pressure", "Type"],
            ExportSingleVTKHDF = true,
            Periodicity = PeriodicBoundary{Dimensions}(1; min = 0, max = length_x),
            GPUDoublePosition = true,
            GPUBoundaryForces = false,
            OpenLogFile = false,
            VisualizeInParaview = false,
        )
        SimViscosity = Laminar()
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion()
        SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)
        SimLogger = SimulationLogger(save)
        result = try
            RunSimulation(;
                SimGeometry = SimulationGeometry,
                SimMetaData = SimMetaData,
                SimConstants = SimConstants,
                SimKernel = SimKernel,
                SimLogger = SimLogger,
                SimParticles = SimParticles,
                SimViscosity = SimViscosity,
                SimDensityDiffusion = SimDensityDiffusion,
                SimTimeStepping = SymplecticTimeStepping(),
                SimExternalForces = SimExternalForces,
            )
        finally
            isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
        end
        result.termination_reason === :completed && result.diagnostics.frozen_particles == 0 ||
            error("Run terminated early or froze particles. See $save")
        speed_max = ceil(1.25 * amplitude; digits = 2)
        run_summary = join([
            "RippledBed | SPHERIC Test 04 | osc$(oscillation), slot $(slot)",
            "Particle resolution: $(1000dx) mm | wavelength/dx = $(round(wavelength/dx; digits=1)) | $(length(fluid)) fluid, $(length(walls)) wall",
            "Parameters: $(boundary), symplectic, planar shifting, coefh=$(coefh), c0=$(sound_speed) m/s, nu=$(viscosity) m^2/s, CFL=0.2",
            "Forcing: U=$(amplitude) m/s, T=$(period) s | duration=$(duration) s, output intervals=$(frames)",
            "Variables: velocity magnitude (0-$(speed_max) m/s), pressure, particle type",
            "Validation: vorticity and vortex circulation from saved fluid velocity",
        ], "\n")
        write(joinpath(save, "settings.txt"), run_summary * "\n")
        run(`$python $(joinpath(case_dir, "compare_circulation.py")) $save $oscillation $dx`)
        AutoOpenParaview(SimMetaData, SimConstants, SimMetaData.OutputVariables;
            color_variable="Velocity", velocity_range=(0.0, speed_max), run_annotation=run_summary)
        loader = joinpath(save, "RippledBed_SingleVTKHDFStateFile.py")
        open(loader, "a") do io
            println(io)
            write(io, read(joinpath(case_dir, "paraview_circulation.py"), String))
            println(io, "add_circulation_chart(directory, layout1, renderView1, $oscillation, ", repr(string(boundary)), ")")
        end
        run(`$pvpython --force-offscreen-rendering $loader`)
        if visualize
            paraview = Sys.which("paraview")
            isnothing(paraview) && error("Add paraview to PATH to open the saved comparison")
            run(`$paraview --state=$(joinpath(save, "circulation_paraview.pvsm"))`; wait=false)
        end
        println("Results saved in ", save)
        result
    finally
        rm(lock)
    end
end
