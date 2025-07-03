# Packages
using SPHExample
using CUDA
using Parameters
using StaticArrays
using Adapt
using Setfield
using LinearAlgebra
using BenchmarkTools
using PointNeighbors

Dimensions = 2
FloatType  = Float64
SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.5)

# Assuming SimConstantsWedge is defined somewhere else with the field `dx`
FixedBoundary = Geometry{Dimensions, FloatType}(
    CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
    GroupMarker = 1,
    Type        = Fixed,   # Using the enum value Fixed
    Motion      = nothing
)
Water = Geometry{Dimensions, FloatType}(
    CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
    GroupMarker = 2,
    Type        = Fluid,   # Using the enum value Fluid
    Motion      = nothing
)

SimulationGeometry = [FixedBoundary;Water]

# Load in particles
SimParticles = AllocateDataStructures(SimulationGeometry)
SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
    SimulationName="StillWedge", 
    SaveLocation="E:/SecondApproach/StillWedge2D_MDBC",
    SimulationTime=4.0,
    OutputTimes=0.01,
    VisualizeInParaview=true,
    ExportSingleVTKHDF=true,
    ExportGridCells=true,
    OpenLogFile=true,
    FlagOutputKernelValues=false,
    FlagLog=true,
    FlagMDBCSimple=true,
    # OutputVariables = [
    #     # "ChunkID",
    #     # "Kernel",
    #     # "KernelGradient",
    #     "Density",
    #     "Pressure",
    #     "Velocity",
    #     "Acceleration",
    #     # "BoundaryBool",
    #     # "ID",
    #     # "Type",
    #     # "GroupMarker",
    #     # "GhostPoints",
    #     # "GhostNormals",
    # ]
)
SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation; to_console=true)
CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)
SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)

mins = minimum.(eachcol([p.Position for p in SimParticles]))
maxs = maximum.(eachcol([p.Position for p in SimParticles]))



# Constants
const T     = FloatType
const ST    = Dimensions
const H     = T(SimKernel.H)
const αD    = T(SimKernel.αD)
const ϵ     = T(1e-6)
const NP    = length(SimParticles)
const NL    = 10^7

# Main Struct for Gradient Values
@with_kw struct ∇ᵢWᵢⱼStruct{T,ST}
    NL::Base.RefValue{Int}

    # Input for calculation
    xᵢⱼ     ::AbstractVector{SVector{ST,T}} = zeros(SVector{ST,T},NL[])
    xᵢⱼ²    ::AbstractVector{T}             = similar(xᵢⱼ,T,NL[])
    dᵢⱼ     ::AbstractVector{T}             = similar(xᵢⱼ,T,NL[])
    qᵢⱼ     ::AbstractVector{T}             = similar(xᵢⱼ,T,NL[])
    ∇ᵢWᵢⱼ   ::AbstractVector{SVector{ST,T}} = similar(xᵢⱼ,NL[])
end

# Helper struct to convert onto GPU
struct ∇ᵢWᵢⱼDeviceStruct{ST,T}
    xᵢⱼ     ::CuDeviceVector{SVector{ST,T}, 1}
    xᵢⱼ²    ::CuDeviceVector{T, 1}             
    dᵢⱼ     ::CuDeviceVector{T, 1}             
    qᵢⱼ     ::CuDeviceVector{T, 1}             
    ∇ᵢWᵢⱼ   ::CuDeviceVector{SVector{ST,T}, 1} 
end

# Adapt Structure for Conversion to GPU
Adapt.adapt_structure(to, s::∇ᵢWᵢⱼStruct) = ∇ᵢWᵢⱼDeviceStruct(
                                                        adapt(to, s.xᵢⱼ  ),
                                                        adapt(to, s.xᵢⱼ² ),
                                                        adapt(to, s.dᵢⱼ  ),
                                                        adapt(to, s.qᵢⱼ  ),
                                                        adapt(to, s.∇ᵢWᵢⱼ),
                                                        )


# Simple Grad Version
function GradVersion0(V::∇ᵢWᵢⱼStruct)
    for _ = 1:1000
        V.xᵢⱼ²    .= dot.(V.xᵢⱼ,V.xᵢⱼ)
        V.dᵢⱼ     .= sqrt.(V.xᵢⱼ²)
        V.qᵢⱼ     .= V.dᵢⱼ ./ H

        @. V.∇ᵢWᵢⱼ = αD *5 * (V.qᵢⱼ- 2)^3 * V.qᵢⱼ / (8H * (V.qᵢⱼ * H + ϵ)) * V.xᵢⱼ
    end

    return nothing
end

### --------------------------- Writing Our Own Custom Kernel  ---------------------------

# Actual Calculation of Gradient in Kernel
function GPU_GRAD(V)
    for _ = 1:1000
        index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        for i = index:stride:length(V.xᵢⱼ)
            xdot                  = dot(V.xᵢⱼ[i],V.xᵢⱼ[i])
            @inbounds V.xᵢⱼ²[i]  += xdot
            dist                  = sqrt(xdot)
            @inbounds V.dᵢⱼ[i]   += dist
            q                     = dist/H
            @inbounds V.qᵢⱼ[i]  += q
            wg                    = αD * 5 * (q - 2)^3 * q / (8H * (q * H + ϵ)) * V.xᵢⱼ[i]
            @inbounds V.∇ᵢWᵢⱼ[i] += wg
        end
    end
    return nothing
end

# Actual Function to Call for Kernel Gradient Calc
function GradVersion0_Kernel(V)
    kernel = @cuda launch=false GPU_GRAD(V)
    config = launch_configuration(kernel.fun)
    threads = min(length(V.xᵢⱼ), config.threads)
    blocks = cld(length(V.xᵢⱼ), threads)

    CUDA.@sync begin
        kernel(V; threads, blocks)
    end
end

### --------------------------- Benchmarking  ---------------------------
IniX         = rand(SVector{ST,T},NL)
V_CPU        = ∇ᵢWᵢⱼStruct{T,ST}(NL=Ref(NL),xᵢⱼ=IniX)
V_GPU_Simple = ∇ᵢWᵢⱼStruct{T,ST}(NL=Ref(NL),xᵢⱼ=CuArray(IniX))
V_GPU_Kernel = ∇ᵢWᵢⱼStruct{T,ST}(NL=Ref(NL),xᵢⱼ=CuArray(IniX))

@CUDA.profile GradVersion0(V_CPU)
@CUDA.profile GradVersion0(V_GPU_Simple)
@CUDA.profile GradVersion0_Kernel(V_GPU_Kernel)

if isapprox(V_CPU.∇ᵢWᵢⱼ,Array(V_GPU_Simple.∇ᵢWᵢⱼ), rtol=1e-6) & isapprox(V_CPU.∇ᵢWᵢⱼ,Array(V_GPU_Kernel.∇ᵢWᵢⱼ), rtol=1e-6)
    println("All three approaches produce the same result with rtol=1e-6.")
end

V_CPU_Bench          = @benchmark GradVersion0($V_CPU)
# I could not get sensible timings other than including @CUDA.sync here too, and not only in GradVersion0_Kernel
V_GPU_Simple_Bench   = @benchmark @CUDA.sync GradVersion0($V_GPU_Simple)
V_GPU_Kernel_Bench   = @benchmark @CUDA.sync GradVersion0_Kernel($V_GPU_Kernel)

println("V_CPU_Bench")
display(V_CPU_Bench)
println("V_GPU_Simple_Bench")
display(V_GPU_Simple_Bench)
println("V_GPU_Kernel_Bench")
display(V_GPU_Kernel_Bench)
