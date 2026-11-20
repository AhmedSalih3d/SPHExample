# Cartesian vertical section of SPHERIC Test 14.
# julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case_2d.jl"
include(joinpath(@__DIR__, "case.jl"))

let
    Dimensions = 2
    FloatType = Float32

    release_name = "float1_down" # Also float1_up or float2_down.
    release = only(filter(case -> case.name == release_name, RELEASES))
    dx = 0.0125
    depth = 0.8
    half_width = 8.0
    wall_top_z = 0.25
    sponge_width = 1.0
    sponge_rate = 12.0
    duration = 4.5
    output_interval = 0.01
    visualize_in_paraview = true

    kernel_family = WendlandC2()
    kernel_h_factor = 1.3
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx, ρ₀ = 1000.0, g = 9.81, c₀ = 25.0, ν₀ = 1e-6, CFL = 0.2, δᵩ = 0.1)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(kernel_family; h = FloatType(kernel_h_factor * dx))
    SimViscosity = Laminar()
    SimDensityDiffusion = LinearDensityDiffusion()
    SimTimeStepping = SymplecticTimeStepping()

    run_name = "planar_dp$(dx)_half$(half_width)_t$(duration)"
    save_location = joinpath(@__DIR__, "runs", run_name, release_name)
    result_location = joinpath(@__DIR__, "results", run_name)
    nx, nz = round(Int, 2half_width / dx), round(Int, depth / dx)
    isapprox(nx * dx, 2half_width) && isapprox(nz * dx, depth) ||
        throw(ArgumentError("Spacing must divide tank width and water depth"))
    0 < sponge_width < half_width || throw(ArgumentError("Damping width must fit inside the tank"))

    # Sample the water and the central float section on one lattice.
    equilibrium_center_z = release.cog - release.draft
    base = release.offset - release.draft
    layers = ceil(Int, SimKernel.H / dx)
    top = ceil(Int, (depth + max(wall_top_z, base + release.height)) / dx)
    wall_top = ceil(Int, (depth + wall_top_z) / dx)
    wall = SVector{2,Float64}[]
    water, body = similar(wall), similar(wall)
    for iz in -layers:top, ix in -layers:nx+layers-1
        x, z = (ix + 0.5 - nx/2) * dx, (iz + 0.5) * dx - depth
        point = SVector(x, z)
        if iz < 0 || !(0 <= ix < nx)
            iz <= wall_top && push!(wall, point)
        elseif in_float(SVector(x, 0.0, z), release, base)
            push!(body, point)
        elseif iz < nz
            push!(water, point)
        end
    end
    isempty(body) && error("No body points at this spacing")

    # The planar SPH equations use mass per unit span. Match the nominal
    # draft using the submerged section area instead of the 3D displaced volume.
    core = release.radius - release.corner
    area = 2core * release.corner + π * release.corner^2 / 2 +
        2release.radius * (release.draft - release.corner)
    line_mass = SimConstants.ρ₀ * area
    floating = FloatingDetails{FloatType}(
        RelativeWeight = line_mass / (SimConstants.m₀ * length(body)),
        Center = (0.0, equilibrium_center_z + release.offset),
        Constraints = FloatingConstraints{Dimensions,FloatType}(
            TranslationLocks = (true, false), RotationLocks = (true,)),
    )
    density(points) = hydrostatic_density(points, SimConstants; water_level = 0.0)
    Tank = SPHGeometry{Dimensions,FloatType}(wall; Density = density(wall), GroupMarker = 1, Type = Fixed)
    Water = SPHGeometry{Dimensions,FloatType}(water; Density = density(water), GroupMarker = 2, Type = Fluid)
    Float = SPHGeometry{Dimensions,FloatType}(body;
        Density = density(body), GroupMarker = 3, Type = Floating, Floating = floating)
    SimDamping = (
        DampingZone((half_width - sponge_width, 0.0), (half_width, 0.0); rate = sponge_rate),
        DampingZone((-half_width + sponge_width, 0.0), (-half_width, 0.0); rate = sponge_rate),
    )
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        SimulationName = "FreeHeave", SaveLocation = save_location, SimulationTime = duration,
        OutputTimes = output_interval, OutputVariables = ["Type", "Velocity", "Pressure"],
        ExportSingleVTKHDF = true, VisualizeInParaview = false, OpenLogFile = false,
        GPUDoublePosition = true, GPUClipTimeStepToOutput = true)
    case = (; geometry = [Tank, Water, Float], metadata = SimMetaData, constants = SimConstants,
        kernel = SimKernel, damping = SimDamping, equilibrium_center_z, dx, half_width, depth, duration,
        viscosity = SimViscosity, diffusion = SimDensityDiffusion, stepping = SimTimeStepping,
        line_mass)
    completed = run_case(release, case; output = result_location, vertical_axis = 2)
    planar_release = merge(release, (; title = release.title * " | Cartesian 2D"))
    state = write_heave_state(planar_release, completed; layout = "stacked", screenshot = true, planar = true)
    println("ParaView launcher saved to ", state)
    if visualize_in_paraview
        paraview = Sys.which("paraview")
        if paraview === nothing
            @warn "Load $state using File > Load State in ParaView."
        else
            run(`$paraview --state=$state`; wait = false)
        end
    end
end
