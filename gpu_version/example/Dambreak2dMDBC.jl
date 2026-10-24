# 2D dam-break particles are generated in this file, then the GPU solver runs.
# Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version gpu_version/example/Dambreak2dMDBC.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
using SPHExampleGPU
using Meshes
using StructArrays
using StaticArrays: SVector

function run_dambreak_2d()
    Dimensions = 2
    FloatType = Float32

    SimConstantsDambreak = SimulationConstants{FloatType}(
        dx = 0.01,
        c₀ = 88.14487860902641,
        δᵩ = 0.1,
        CFL = 0.5,
        α = 0.01,
    )

    # Create SPHGeometry instances
    # Sample particles here; the generator script supplies shape definitions.
    # The commented CSV inputs below are the DualSPHysics Dp0.02 files: set
    # dx = 0.02 when using them. Compare generated and CSV runs at the same dx;
    # at dx = 0.01 the flow develops cavities that dx = 0.02 smooths out.
    polygons = dam_break_2d_polygons()
    regions = [
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, SimConstantsDambreak.dx)
    positions(name) = only(r.positions for r in sampled if r.name == name)
    water_level = maximum(last, positions("Fluid"))

    wall_positions = positions("Bound")
    dx = SimConstantsDambreak.dx
    mirror_x(x) = x < dx / 2 ? dx - x : x > 4 - dx / 2 ? 8 - dx - x : x
    ghosts =
        [SVector(mirror_x(x[1]), x[2] < dx / 2 ? dx - x[2] : x[2]) for x in wall_positions]

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Bound"), SimConstantsDambreak.ρ₀;
            GhostPoints = ghosts,
            GhostNormals = ghosts .- wall_positions,
        ),
        # CSVFile     = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_Bound_ThreeLayers.csv",
        GroupMarker = 1,
        Type = Fixed,   # Using the enum value Fixed
        Motion = nothing,
    )

    Water = SPHGeometry{Dimensions, FloatType}(
        Particles = StructArray((
            Position = positions("Fluid"),
            Density = hydrostatic_density(
                positions("Fluid"),
                SimConstantsDambreak;
                water_level,
            ),
        )),
        # CSVFile     = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_Fluid_ThreeLayers.csv",
        GroupMarker = 2,
        Type = Fluid,   # Using the enum value Fluid
        Motion = nothing,
    )

    # Collect the SPHGeometry instances into a vector
    SimulationGeometry = [FixedBoundary; Water]

    SimMetaDataDambreak = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName = "DamBreak2D",
        SaveLocation = "C:/TestSimulations/DamBreak2D_MDBC_GPU/",
        SimulationTime = 2,
        OutputTimes = collect(0.01:0.01:2),
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OpenLogFile = true,
    )

    # If save directory is not already made, make it
    if !isdir(SimMetaDataDambreak.SaveLocation)
        mkpath(SimMetaDataDambreak.SaveLocation)
    end

    # How to overload and define your own viscosity model:
    # Artificial viscosity formulation.
    # using Parameters
    # using LinearAlgebra
    # struct MyTurbulenceModel <: SPHViscosity end
    # The densities of the evaluated state and their reciprocals are passed in
    # (ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹); do not read them from SimParticles.
    # @inline function SPHExampleGPU.compute_viscosity(::MyTurbulenceModel, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)
    #     @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    #     @unpack h, η² = SimKernel

    #     invd²η²   =  1.0 / (d² + η²)
    #     ρ̄ = (ρᵢ + ρⱼ) * 0.5
    #     cond = dot(vᵢⱼ, xᵢⱼ)
    #     flag = cond < 0 ? one(eltype(cond)) : zero(eltype(cond))
    #     μ = h * cond * invd²η²
    #     Π = -m₀ * (flag * (-α * c₀ * μ) / ρ̄) * ∇ᵢWᵢⱼ
    #     return 0*Π, -Π*0
    # end

    # Load in particles
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataDambreak)

    SimLogger = SimulationLogger(SimMetaDataDambreak.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataDambreak.SaveLocation)

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataDambreak,
        SimConstants = SimConstantsDambreak,
        SimKernel = SPHKernelInstance{Dimensions, FloatType}(
            WendlandC2();
            dx = SimConstantsDambreak.dx,
        ),
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimViscosity = ZeroViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        # ParticleNormalsPath  = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_GhostNodes_ThreeLayers.csv"
    )
end

"""
    dam_break_2d_polygons(; tank_width = 4.0, tank_height = 3.0,
                           water_width = 1.0, water_height = 2.0,
                           wall_thickness = 0.06)

Return the open tank and initial water column of the 2D dam-break case as
`PolyArea`s, in metres. The tank is a three-sided shell drawn along the wetted
surface with `polyline`: its floor and side walls extend `wall_thickness`
outside the fluid domain. The water starts in the left corner and is open at
the top.
"""
function dam_break_2d_polygons(;
    tank_width = 4.0,
    tank_height = 3.0,
    water_width = 1.0,
    water_height = 2.0,
    wall_thickness = 0.06,
)
    if !(0 < water_width <= tank_width) ||
       !(0 < water_height <= tank_height) ||
       !(wall_thickness > 0)
        throw(
            ArgumentError(
                "water dimensions must fit in a positive tank " * "and wall thickness",
            ),
        )
    end

    # Walking down the left wall, along the floor and up the right wall, the
    # outside of the tank is on the right.
    wetted_surface =
        [(0.0, tank_height), (0.0, 0.0), (tank_width, 0.0), (tank_width, tank_height)]
    tank = polyline(wetted_surface; thickness = wall_thickness, side = :right)
    water = rectangle((0.0, 0.0), water_width, water_height)

    return (; tank, water)
end

"""
    generate_dam_break_2d_example(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 88.14487860902641),
        water_level = nothing)

Write the tank and water polygons (`DamBreak2D_Geometry.vtkhdf`), the sampled
particles (`DamBreak2D_Dp<dx>_Particles.vtkhdf`) and the CSV input files
(`DamBreak2D_Dp<dx>_Bound.csv`, `..._Fluid.csv`) to `output_dir`.

Boundary particles start at `ρ₀`. Fluid particles start in hydrostatic
equilibrium: their density is the inverse of the solver's equation of state
for `P = ρ₀ g (water_level - y)`. `water_level = nothing` uses the highest
fluid particle. Pass the `SimConstants` of the simulation so that `ρ₀`, `g`
and `c₀` match the case.
Returns the sampled particles per region, with their densities.
"""
function generate_dam_break_2d_example(
    output_dir;
    dx = 0.02,
    SimConstants = SimulationConstants{Float64}(; dx, c₀ = 88.14487860902641),
    water_level = nothing,
)
    polygons = dam_break_2d_polygons()
    regions = [
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)

    fluid_positions = SVector{2, Float64}[]
    for r in sampled
        r.type == Fluid && append!(fluid_positions, r.positions)
    end
    level = something(water_level, maximum(last, fluid_positions))
    particles = map(sampled) do region
        density =
            region.type == Fluid ?
            hydrostatic_density(region.positions, SimConstants; water_level = level) :
            fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    prefix = joinpath(output_dir, "DamBreak2D_Dp$(dx)")
    SavePolygonVTKHDF(joinpath(output_dir, "DamBreak2D_Geometry.vtkhdf"), polygons)

    next_id = 0
    for region in particles
        next_id = write_particle_csv(
            "$(prefix)_$(region.name).csv",
            region.positions;
            density = region.density,
            first_id = next_id,
        )
    end

    (; ρ₀, c₀) = SimConstants
    positions = SVector{3, Float64}[]
    density   = Float64[]
    types     = Int8[]
    markers   = Int[]
    for (id, region) in enumerate(particles)
        append!(positions, to_3d(region.positions))
        append!(density, region.density)
        append!(types, fill(Int8(region.type), length(region.positions)))
        append!(markers, fill(id, length(region.positions)))
    end
    pressure = EquationOfStateGamma7.(density, c₀, ρ₀)
    SaveVTKHDF(
        "$(prefix)_Particles.vtkhdf",
        positions,
        ["Density", "Pressure", "Type", "GroupMarker"],
        density,
        pressure,
        types,
        markers,
    )

    return particles
end

# Including this file starts the simulation.
run_dambreak_2d()
