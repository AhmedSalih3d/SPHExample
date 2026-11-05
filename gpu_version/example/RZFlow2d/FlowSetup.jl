using SPHExampleGPU
using StaticArrays: SVector

"""
    rz_flow_case(; dx = 0.005, duration = 9.0, visualize = true, save_location)

Build the tank, L-shaped barrier and submerged return-flow zone of
DualSPHysics `wavecoupling/04_RZ_Flow2D`, in metres. Reference x/z map to x/y.
This function allocates host geometry only; it does not start a simulation.
"""
function rz_flow_case(; dx = 0.005, duration = 9.0, visualize = true,
        save_location = joinpath(@__DIR__, "..", "..", "output", "rz_flow_2d"))
    isfinite(dx) && 0 < dx <= 0.02 ||
        throw(ArgumentError("dx must be finite and in (0, 0.02]"))
    isfinite(duration) && duration > 0 ||
        throw(ArgumentError("duration must be finite and positive"))
    # Keep the lattice aligned with every reference face, including 4 cm walls.
    faces = (0.2, 0.24, 0.45, 0.5, 0.8, 1.05, 1.3)
    all(x -> isapprox(round(x / dx) * dx, x; atol = 1e-10, rtol = 0), faces) ||
        throw(ArgumentError("dx must divide all reference geometry dimensions"))
    n(x) = round(Int, x / dx)
    walls = SVector{2,Float64}[]
    water = SVector{2,Float64}[]
    for j in 0:n(0.8), i in 0:n(1.3)
        point = SVector(i * dx, j * dx)
        outer = i == 0 || i == n(1.3) || j == 0 || j == n(0.8)
        barrier = (n(0.2) <= i <= n(0.24) && n(0.2) <= j <= n(0.5)) ||
            (n(0.2) <= i <= n(1.05) && n(0.2) <= j <= n(0.24))
        if outer || barrier
            push!(walls, point)
        # Exclude the fillbox upper plane, leaving one spacing of clearance.
        elseif j < n(0.45)
            push!(water, point)
        end
    end

    water_bottom, water_level = extrema(last, water)
    constants = SimulationConstants{Float32}(2;
        dx, ρ₀ = 1000, g = 9.81,
        c₀ = 20sqrt(9.81 * (water_level - water_bottom)), α = 0.05, CFL = 0.2)
    density(points) = hydrostatic_density(points, constants; water_level)
    geometry = [
        SPHGeometry{2,Float32}(walls; Density = density(walls),
            GroupMarker = 1, Type = Fixed),
        SPHGeometry{2,Float32}(water; Density = density(water),
            GroupMarker = 2, Type = Fluid),
    ]
    relaxation = UniformFlowZone((0.6, 0.0), (0.9, 0.2);
        direction = (-1.0, 0.0),
        times = (0.0, 1.0, 3.0, 6.0, 8.0),
        speeds = (0.0, 0.3, 0.4, 0.7, 0.2),
        coefdt = 1000, psi = 0.7, beta = 3)
    metadata = SimulationMetaData{2,Float32,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "CaseRZ_Flow2D",
        SaveLocation = abspath(save_location),
        SimulationTime = duration,
        OutputTimes = 0.025,
        VisualizeInParaview = visualize,
        OpenLogFile = visualize,
        ExportSingleVTKHDF = true,
        GPUCellSubdivision = 1,
        GPUDoublePosition = true,
        GPUClipTimeStepToOutput = true,
    )
    kernel = SPHKernelInstance{2,Float32}(WendlandC2(); h = Float32(sqrt(2) * dx))
    return (; geometry, constants, metadata, kernel, relaxation)
end
