"""GPU operators for IISPH, with frozen positions throughout each pressure solve."""
module GPUIISPH

using CUDA, StaticArrays, LinearAlgebra
using ..GPUCellGrid: thread_index, load_grid, cell_rows, row_range
using ..SimulationGeometry: Fluid, Fixed
using ..SPHKernels: Wᵢⱼ, ∇Wᵢⱼ
using ..SPHViscosityModels: Laminar
using ..GPUReductions: reduce_svector

export launch_boundary_mass!, launch_density!, launch_nonpressure!, launch_predict!,
       launch_source!, launch_pressure_acceleration!, launch_pressure_update!,
       launch_integrate!, timestep_criteria, pressure_residual

const THREADS = 128

@inline function pair_gradient(kernel, x)
    d2 = dot(x,x)
    return d2 > zero(d2) && d2 <= kernel.H² ? ∇Wᵢⱼ(kernel,sqrt(d2)*kernel.h⁻¹,x) : zero(x)
end

function boundary_mass_kernel!(mass, types, m0, n)
    i = thread_index()
    i > n && return
    T = eltype(mass)
    # Fixed walls are volume samples at the same resolution as the fluid,
    # with V_b=m0/rho0. Surface-only boundary samples are not supported here.
    # Estimating volumes from only solid neighbours would inflate V_b at the
    # fluid/solid interface and overcompress a uniform initial lattice.
    @inbounds mass[i] = types[i] == Fixed ? m0 : zero(T)
    return
end

function density_kernel!(density, position, types, mass, cells, starts, gridarg, kernel, m0, rho0, n)
    i = thread_index()
    i > n && return
    T = eltype(density)
    D = length(eltype(position))
    @inbounds if types[i] == Fluid
        grid = load_grid(gridarg)
        rho = zero(T)
        for row in cell_rows(grid)
            lo,hi = row_range(grid,starts,cells[i],row)
            for j in lo:hi
                x = SVector{D,T}(position[i]-position[j])
                d2 = dot(x,x)
                if d2 <= kernel.H²
                    rho += (types[j] == Fluid ? m0 : mass[j]) * Wᵢⱼ(kernel,sqrt(d2)*kernel.h⁻¹)
                end
            end
        end
        density[i] = rho
    else
        density[i] = rho0
    end
    return
end

function nonpressure_kernel!(acceleration, position, velocity, density, types, mass,
                             cells, starts, gridarg, kernel, constants, viscosity, n)
    i = thread_index()
    i > n && return
    T = eltype(density)
    D = length(eltype(position))
    a = zero(SVector{D,T})
    @inbounds if types[i] == Fluid
        a = SVector{D,T}(ntuple(d -> d == D ? -T(constants.g) : zero(T),Val(D)))
        if viscosity isa Laminar
            grid = load_grid(gridarg)
            for row in cell_rows(grid)
                lo,hi = row_range(grid,starts,cells[i],row)
                for j in lo:hi
                    x = SVector{D,T}(position[i]-position[j])
                    grad = pair_gradient(kernel,x)
                    mj = types[j] == Fluid ? constants.m₀ : mass[j]
                    factor = 4mj * constants.ν₀ * dot(x,grad) /
                        ((density[i]+density[j])*(dot(x,x)+kernel.η²))
                    a += factor * (velocity[i]-velocity[j])
                end
            end
        end
    end
    @inbounds acceleration[i] = a
    return
end

function predict_kernel!(adv, velocity, acceleration, types, pressure, dt, warm_start, n)
    i = thread_index()
    i > n && return
    @inbounds begin
        adv[i] = types[i] == Fluid ? velocity[i]+dt*acceleration[i] : zero(eltype(adv))
        pressure[i] = types[i] == Fluid ? eltype(pressure)(warm_start)*pressure[i] : zero(eltype(pressure))
    end
    return
end

# A(p)_i = sum_j m_j (a_i(p)-a_j(p)) dot grad(W_ij), with a_wall=0.
# Its diagonal includes wall gradients in the squared sum, but the neighbour's
# response to p_i exists only for fluid neighbours. No dt is folded into A.
function source_kernel!(diagonal, adv_density, density, position, adv_velocity, types, mass,
                        cells, starts, gridarg, kernel, m0, dt, n)
    i = thread_index()
    i > n && return
    T = eltype(density)
    D = length(eltype(position))
    @inbounds if types[i] == Fluid
        grid = load_grid(gridarg)
        gradient_sum = zero(SVector{D,T})
        gradient_square = zero(T)
        rate = zero(T)
        for row in cell_rows(grid)
            lo,hi = row_range(grid,starts,cells[i],row)
            for j in lo:hi
                x = SVector{D,T}(position[i]-position[j])
                grad = pair_gradient(kernel,x)
                mj = types[j] == Fluid ? m0 : mass[j]
                gradient_sum += mj*grad
                types[j] == Fluid && (gradient_square += m0*mj*dot(grad,grad))
                rate += mj*dot(adv_velocity[i]-adv_velocity[j],grad)
            end
        end
        diagonal[i] = -(dot(gradient_sum,gradient_sum)+gradient_square)/density[i]^2
        adv_density[i] = density[i]+dt*rate
    else
        diagonal[i] = zero(T)
        adv_density[i] = density[i]
    end
    return
end

function pressure_acceleration_kernel!(acceleration, pressure, position, density, types, mass,
                                       cells, starts, gridarg, kernel, m0, n)
    i = thread_index()
    i > n && return
    T = eltype(density)
    D = length(eltype(position))
    a = zero(SVector{D,T})
    @inbounds if types[i] == Fluid
        grid = load_grid(gridarg)
        pi_rho2 = pressure[i]/density[i]^2
        for row in cell_rows(grid)
            lo,hi = row_range(grid,starts,cells[i],row)
            for j in lo:hi
                x = SVector{D,T}(position[i]-position[j])
                grad = pair_gradient(kernel,x)
                if types[j] == Fluid
                    a -= m0*(pi_rho2+pressure[j]/density[j]^2)*grad
                else
                    a -= mass[j]*pi_rho2*grad
                end
            end
        end
    end
    @inbounds acceleration[i] = a
    return
end

function pressure_update_kernel!(next_pressure, residual, pressure, acceleration, diagonal,
                                 adv_density, position, types, mass, cells, starts, gridarg,
                                 kernel, m0, rho0, dt, relaxation, n)
    i = thread_index()
    i > n && return
    T = eltype(pressure)
    D = length(eltype(position))
    @inbounds if types[i] == Fluid
        grid = load_grid(gridarg)
        ap = zero(T)
        for row in cell_rows(grid)
            lo,hi = row_range(grid,starts,cells[i],row)
            for j in lo:hi
                x = SVector{D,T}(position[i]-position[j])
                grad = pair_gradient(kernel,x)
                mj = types[j] == Fluid ? m0 : mass[j]
                ap += mj*dot(acceleration[i]-acceleration[j],grad)
            end
        end
        density_error = adv_density[i]+dt^2*ap-rho0
        # This is the complementarity residual of the current pressure, before
        # applying the candidate update. The returned acceleration matches it.
        error = pressure[i] > zero(T) ? abs(density_error) : max(density_error,zero(T))
        residual[i] = isfinite(error) ? error/rho0 : T(Inf)
        next = diagonal[i] < zero(T) ?
            max(zero(T),pressure[i]-T(relaxation)*density_error/(dt^2*diagonal[i])) : zero(T)
        next_pressure[i] = next
        isfinite(next) || (residual[i] = T(Inf))
    else
        next_pressure[i] = zero(T)
        residual[i] = zero(T)
    end
    return
end

function integrate_kernel!(position, velocity, acceleration, types, adv, np, ap, dt, n)
    i = thread_index()
    i > n && return
    @inbounds if types[i] == Fluid
        v = adv[i]+dt*ap[i]
        velocity[i] = v
        position[i] += dt*v
        acceleration[i] = np[i]+ap[i]
    end
    return
end

@inline max_components(a,b) = max.(a,b)
@inline function criteria_map(i, velocity, old_acceleration, nonpressure, position, density, types)
    T = eltype(density)
    @inbounds begin
        valid = all(isfinite,position[i]) && all(isfinite,velocity[i]) &&
            all(isfinite,old_acceleration[i]) && all(isfinite,nonpressure[i]) &&
            isfinite(density[i]) && density[i] > zero(T)
        speed = types[i] == Fluid ? norm(velocity[i]) : zero(T)
        force = types[i] == Fluid ? max(norm(old_acceleration[i]),norm(nonpressure[i])) : zero(T)
        return SVector{4,T}(speed,force,valid ? zero(T) : one(T),zero(T))
    end
end
@inline function residual_map(i, residual)
    T = eltype(residual)
    @inbounds return SVector{4,T}(residual[i],zero(T),zero(T),zero(T))
end

function timestep_criteria(ws, gpu)
    T = eltype(gpu.Density)
    return reduce_svector(ws.reduction,criteria_map,max_components,zero(SVector{4,T}),length(gpu),
        gpu.Velocity,gpu.Acceleration,ws.nonpressure_acceleration,gpu.Position,gpu.Density,gpu.Type)
end

function pressure_residual(ws)
    T = eltype(ws.pressure)
    return first(reduce_svector(ws.reduction,residual_map,max_components,zero(SVector{4,T}),
        length(ws.residual),ws.residual))
end

function launch_boundary_mass!(ws,gpu,cl,kernel,constants)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) boundary_mass_kernel!(ws.boundary_mass,
        gpu.Type,constants.m₀,Int32(length(gpu)))
end
function launch_density!(ws,gpu,cl,kernel,constants)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) density_kernel!(gpu.Density,
        gpu.Position,gpu.Type,ws.boundary_mass,gpu.CellID,cl.CellStart,cl.grid_dev,kernel,
        constants.m₀,constants.ρ₀,Int32(length(gpu)))
end
function launch_nonpressure!(ws,gpu,cl,kernel,constants,viscosity)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) nonpressure_kernel!(ws.nonpressure_acceleration,
        gpu.Position,gpu.Velocity,gpu.Density,gpu.Type,ws.boundary_mass,gpu.CellID,cl.CellStart,
        cl.grid_dev,kernel,constants,viscosity,Int32(length(gpu)))
end
function launch_predict!(ws,gpu,dt,solver)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) predict_kernel!(ws.velocity_adv,
        gpu.Velocity,ws.nonpressure_acceleration,gpu.Type,ws.pressure,dt,solver.warm_start,Int32(length(gpu)))
end
function launch_source!(ws,gpu,cl,kernel,constants,dt)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) source_kernel!(ws.diagonal,ws.density_adv,
        gpu.Density,gpu.Position,ws.velocity_adv,gpu.Type,ws.boundary_mass,gpu.CellID,cl.CellStart,
        cl.grid_dev,kernel,constants.m₀,dt,Int32(length(gpu)))
end
function launch_pressure_acceleration!(ws,gpu,cl,kernel,constants)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) pressure_acceleration_kernel!(ws.pressure_acceleration,
        ws.pressure,gpu.Position,gpu.Density,gpu.Type,ws.boundary_mass,gpu.CellID,cl.CellStart,
        cl.grid_dev,kernel,constants.m₀,Int32(length(gpu)))
end
function launch_pressure_update!(ws,gpu,cl,kernel,constants,dt,solver)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) pressure_update_kernel!(ws.next_pressure,
        ws.residual,ws.pressure,ws.pressure_acceleration,ws.diagonal,ws.density_adv,gpu.Position,
        gpu.Type,ws.boundary_mass,gpu.CellID,cl.CellStart,cl.grid_dev,kernel,constants.m₀,
        constants.ρ₀,dt,solver.relaxation,Int32(length(gpu)))
end
function launch_integrate!(ws,gpu,dt)
    @cuda threads=THREADS blocks=cld(length(gpu),THREADS) integrate_kernel!(gpu.Position,gpu.Velocity,
        gpu.Acceleration,gpu.Type,ws.velocity_adv,ws.nonpressure_acceleration,
        ws.pressure_acceleration,dt,Int32(length(gpu)))
end

end
