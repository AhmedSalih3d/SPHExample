module NoPenetrationTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const K = SPHExampleGPU.GPUKernels
const CL = SPHExampleGPU.SPHCellList

@testset "normal contact acts locally in the moving wall frame" begin
    offset = SVector(0.0,0.05)
    outside = SVector(0.0,0.03)
    inside = SVector(0.0,0.02)
    rate(separation, velocity) = K.normal_contact_rate(offset, separation, velocity, 0.05, 25.0)
    @test rate(outside, SVector(2.0,-1.0))[2] > 0
    @test iszero(rate(outside, SVector(2.0,0.0)))
    @test iszero(rate(outside, SVector(2.0,1.0)))
    @test rate(inside, zero(offset))[2] > 0
    @test iszero(rate(SVector(0.07,0.02), -offset))
    @test iszero(rate(SVector(0.0,0.05), -offset))
end

if CUDA.functional()
    @testset "normal contact GPU force and floating reaction" begin
        for kind in (Fixed,Floating), compact in (false,true)
            T = Float64
            wall = SVector(0.0,0.0)
            offset = SVector(0.0,0.05)
            speed = kind == Floating ? SVector(0.0,0.5) : zero(wall)
            floating = kind == Floating ? FloatingDetails{T}(RelativeWeight=2) : nothing
            geometry = [
                SPHGeometry{2,T}(Particles=particle_struct_array([wall],T(1000);
                    GhostPoints=[offset],GhostNormals=[offset],Velocity=[speed]),Type=kind,GroupMarker=1,Floating=floating),
                SPHGeometry{2,T}(Particles=particle_struct_array([SVector(0.0,0.03)],T(1000);
                    Velocity=[speed+SVector(0.4,-1.0)]),Type=Fluid,GroupMarker=2),
            ]
            gpu = upload_particles(AllocateDataStructures(geometry); boundary_only=compact)
            constants = SimulationConstants{T}(2;dx=0.05,c₀=25,g=0)
            kernel = SPHKernelInstance{2,T}(WendlandC2();h=0.1)
            cells = CellListWorkspace{2,T}(length(gpu))
            CL.rebuild_cell_list!(gpu,cells,kernel.H⁻¹)
            sup = GPUSupportArrays{2,T}(length(gpu); updated_mdbc=true,no_penetration=true,
                boundary_slots=gpu.BoundarySlot,boundary_count=compact ? 1 : length(gpu))
            types = Array(gpu.Type)
            fluid = only(findall(==(Fluid),types))
            solid = only(findall(==(kind),types))
            slot = compact ? 1 : solid
            data = sup.boundary_data
            offsets = zeros(SVector{2,T},length(data.active))
            offsets[slot] = offset
            copyto!(data.contact_offsets,offsets)
            speeds = zero(offsets)
            speeds[slot] = speed
            copyto!(data.velocity,speeds)
            step = HostStep(T(0.001),0.)
            launch_interactions!(sup.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,sup.∇Cᵢ,sup.∇◌rᵢ,
                gpu.Position,gpu.Density,gpu.Velocity,gpu.Type,cells.CellStart,gpu.CellID,cells.grid_dev,
                step,ZeroDensityDiffusion(),ZeroViscosity(),kernel,constants,Val(false),Val(false);boundary_data=data)
            a = Array(gpu.Acceleration)
            @test a[fluid][2] > 0
            @test a[fluid][1] == 0
            kind == Floating && @test a[fluid] ≈ -a[solid]
        end
    end
end
end
