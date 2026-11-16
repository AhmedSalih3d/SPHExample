"""Optional hyperbolic/parabolic velocity-divergence cleaning on the device."""
module GPUDivergenceCleaning

using CUDA, StaticArrays
using ..SimulationGeometry: Fluid
using ..SimulationEquations: EquationOfStateGamma7
using ..SimulationMetaDataConfiguration: SymplecticTimeStepping, periodic_images_enabled
using ..GPUReductions: ReductionWorkspace, launch_reduce!, block_reduce_store!, block_reduce, REDUCE_THREADS

export DivergenceCleaning

"""
    DivergenceCleaning(; kappa = 10, sigma = 1, courant = 0.1, pressure_mode = :absolute)

Experimental Fourtakas, Vacondio & Rogers (2025) velocity-divergence cleaning,
passed to `RunSimulation(SimDivergenceCleaning = ...)`. It adds `-∇ψ` to
fluid acceleration and evolves `Dψ/Dt = -c₀² div(u) - ψ/τ`, initially ψ = 0.
The difference operators use neighboring particle volumes; wall and frozen
particles have ψ = 0. This is a nonconservative cleaning operator, not a
pressure projection or an incompressibility constraint.

`1/τ = sigma * kappa * sqrt(abs(p)/ρ) / h`, including `BackgroundPressure`
in both local decay and its timestep limit. Absolute pressure magnitude extends
the paper's positive-pressure formula to negative gauge pressures without
removing local decay in rarefactions. `pressure_mode = :positive` instead uses
`max(p,0)`, a diagnostic alternative that can amplify startup disturbances in
the moving-square DBC case. Neither choice clips the actual pressure. The
propagation speed remains c₀. Each accepted-state reduction also enforces
`dt ≤ courant*h / max(c₀, maximum(local_decay_speed))` on the device.
`kappa` is not scaled automatically with c₀. At c₀ = 160 m/s in the moving-square
DBC study, kappa = 10, 40 and 160 all worsened full-transient pressure error;
central-window noise reduction alone does not justify enabling this method.

The dissipative sign follows Eq. 3.2 and the damped wave equation; the printed
plus sign on the RHS of Eq. 3.4 would amplify ψ and is not used.
See https://doi.org/10.1016/j.compfluid.2025.106638.

Supports 2D/3D Float32/Float64, standard two-stage symplectic stepping,
DBC/mDBC, prescribed motion and translated periodic queries. The first
implementation rejects single-neighbor/reference stepping, dense periodic
images (use `GPUPeriodicImages=false`), variable resolution, open boundaries,
floating bodies, timestep-provenance diagnostics and nonzero restart clocks.
`nothing` (the default) allocates no cleaning buffers and keeps the original
arithmetic. The final ψ is returned in `result.final_state.divergence_cleaning.psi`,
in the same order as `result.final_state.particles`; it is not a VTKHDF field.
"""
struct DivergenceCleaning{T<:AbstractFloat}
    kappa::T
    sigma::T
    courant::T
    absolute_pressure::Bool
    function DivergenceCleaning{T}(kappa, sigma, courant, absolute_pressure::Bool = true) where {T<:AbstractFloat}
        k, s, c = T(kappa), T(sigma), T(courant)
        all(isfinite,(k,s,c)) && k > 0 && 0 <= s <= 1 && 0 < c <= 0.5 ||
            throw(ArgumentError("DivergenceCleaning requires finite kappa > 0, 0 ≤ sigma ≤ 1 and 0 < courant ≤ 0.5"))
        new{T}(k,s,c,absolute_pressure)
    end
end

function DivergenceCleaning(; kappa = 10.,sigma = 1.,courant = 0.1,pressure_mode = :absolute)
    pressure_mode in (:absolute,:positive) || throw(ArgumentError("pressure_mode must be :absolute or :positive"))
    return DivergenceCleaning{Float64}(kappa,sigma,courant,pressure_mode === :absolute)
end

function validate_cleaning(model, meta, kernel, scheme; open_boundary = nothing, floating = false)
    model === nothing && return nothing
    model isa DivergenceCleaning || throw(ArgumentError("SimDivergenceCleaning must be DivergenceCleaning or nothing"))
    scheme isa SymplecticTimeStepping || throw(ArgumentError("DivergenceCleaning requires SymplecticTimeStepping"))
    meta.VariableResolution === nothing || throw(ArgumentError("DivergenceCleaning does not support variable resolution"))
    !periodic_images_enabled(meta,kernel) || throw(ArgumentError("DivergenceCleaning requires GPUPeriodicImages=false for periodic cases"))
    open_boundary === nothing || throw(ArgumentError("DivergenceCleaning does not support open-boundary particle activation"))
    !floating || throw(ArgumentError("DivergenceCleaning does not yet support floating-body reactions"))
    meta.GPUTimestepDiagnosticsEvery == 0 || throw(ArgumentError("DivergenceCleaning requires GPUTimestepDiagnosticsEvery=0; cleaning limiter provenance is not implemented"))
    meta.TotalTime == 0 && meta.Iteration == 0 || throw(ArgumentError("DivergenceCleaning requires a fresh simulation; restarting also requires the persistent psi field"))
    return nothing
end

struct CleaningWorkspace{T}
    model::DivergenceCleaning{T}
    psi::CuVector{T}
    half::CuVector{T}
    derivative::CuVector{T}
    signal::ReductionWorkspace{SVector{1,T}}
end

CleaningWorkspace(::Nothing, n, ::Type{T}) where {T} = nothing
function CleaningWorkspace(model::DivergenceCleaning, n, ::Type{T}) where {T}
    typed = DivergenceCleaning{T}(model.kappa,model.sigma,model.courant,model.absolute_pressure)
    return CleaningWorkspace(typed,CUDA.zeros(T,n),CUDA.zeros(T,n),CUDA.zeros(T,n),ReductionWorkspace{SVector{1,T}}(n))
end

cleaning_fields(::Nothing, half::Bool = false) = nothing
cleaning_fields(state::CleaningWorkspace, half::Bool = false) =
    (; model = state.model,psi = half ? state.half : state.psi,half = state.half,
        derivative = state.derivative,partial = state.signal.partial)

@inline function local_speed(model::DivergenceCleaning{T}, pressure, density) where {T}
    magnitude = model.absolute_pressure ? abs(T(pressure)) : max(T(pressure),zero(T))
    return model.kappa * sqrt(magnitude/density)
end
@inline cleaning_psi(::Nothing, i, type, ::Type{T}) where {T} = zero(T)
@inline cleaning_psi(fields, i, type, ::Type{T}) where {T} = type == Fluid ? (@inbounds fields.psi[i]) : zero(T)

@inline cleaning_derivative!(::Nothing, i, type, divergence_sum, pressure, density, c0, h) = nothing
@inline function cleaning_derivative!(fields, i, type, divergence_sum, pressure, density, c0, h)
    @inbounds fields.derivative[i] = type == Fluid ?
        c0*c0*divergence_sum - (fields.model.sigma*local_speed(fields.model,pressure,density)/h)*fields.psi[i] : zero(density)
    return nothing
end

@inline cleaning_predict!(::Nothing, i, type, dt) = nothing
@inline function cleaning_predict!(fields, i, type, dt)
    @inbounds fields.half[i] = type == Fluid ? fields.psi[i] + (dt/2)*fields.derivative[i] : zero(eltype(fields.psi))
    return nothing
end

@inline cleaning_commit!(::Nothing, i, type, dt) = nothing
@inline function cleaning_commit!(fields, i, type, dt)
    @inbounds fields.psi[i] = type == Fluid ? fields.psi[i] + dt*fields.derivative[i] : zero(eltype(fields.psi))
    return nothing
end

@inline signal_value(::Nothing, type, density, constants) = zero(density)
@inline function signal_value(fields, type, density, constants)
    type == Fluid || return zero(density)
    p = EquationOfStateGamma7(density,constants.c₀,constants.ρ₀,inv(constants.ρ₀),constants.BackgroundPressure)
    return local_speed(fields.model,p,density)
end
@inline signal_max(a::SVector{1,T}, b::SVector{1,T}) where {T} = SVector{1,T}(max(a[1],b[1]))
@inline store_signal!(::Nothing, speed) = nothing
@inline store_signal!(fields, speed::T) where {T} = block_reduce_store!(fields.partial,SVector{1,T}(speed),signal_max)

@inline function signal_map(i,density,types,fields,constants)
    @inbounds return SVector(signal_value(fields,types[i],density[i],constants))
end
initialize_signal!(::Nothing,density,types,constants) = nothing
function initialize_signal!(state::CleaningWorkspace{T},density,types,constants) where {T}
    fields = cleaning_fields(state)
    launch_reduce!(state.signal,signal_map,signal_max,SVector{1,T}(zero(T)),length(density),density,types,fields,constants)
    return nothing
end

@inline cleaning_timestep(::Nothing, nblocks, h, c0) = oftype(h,Inf)
@inline function cleaning_timestep(fields, nblocks, h::T, c0) where {T}
    speed = zero(T)
    b = threadIdx().x
    @inbounds while b <= nblocks
        speed = max(speed,fields.partial[b][1])
        b += blockDim().x
    end
    speed = reduce_cleaning_speed(speed)
    return fields.model.courant*h/max(c0,speed)
end
@inline reduce_cleaning_speed(speed::T) where {T} = block_reduce(SVector{1,T}(speed),signal_max)[1]

function reorder_kernel!(destination,source,permutation,types,n)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    @inbounds if i <= n
        destination[i] = types[i] == Fluid ? source[permutation[i]] : zero(eltype(source))
    end
    return nothing
end
reorder_cleaning!(::Nothing, permutation, types) = nothing
function reorder_cleaning!(state::CleaningWorkspace, permutation, types)
    n = length(state.psi)
    n == 0 && return nothing
    # Predictor storage is idle at a rebuild; preserve psi before Perm is reused.
    @cuda threads=REDUCE_THREADS blocks=cld(n,REDUCE_THREADS) reorder_kernel!(state.half,state.psi,permutation,types,Int32(n))
    copyto!(state.psi,state.half)
    return nothing
end

cleaning_configuration(configuration,::Nothing) = configuration
function cleaning_configuration(configuration,state::CleaningWorkspace)
    m = state.model
    parameters = (; method = :fourtakas_velocity_divergence,kappa = m.kappa,sigma = m.sigma,courant = m.courant,
        pressure_extension = m.absolute_pressure ? :absolute : :nonnegative,wall_psi = :zero,scalar_integration = :explicit_midpoint)
    return merge(configuration,(; effective = merge(configuration.effective,(; divergence_cleaning = parameters))))
end

end
