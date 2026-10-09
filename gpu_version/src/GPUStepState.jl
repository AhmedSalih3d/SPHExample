"""
Device resident time step state.

The size of the time step, the simulated time, the displacement bound that
decides when the cell list is rebuilt and the loop control flags live in two
small device vectors. The kernels of a step read the step size and the time
from there instead of receiving them as launch arguments. The host therefore
never waits for the reduction of the previous step before it enqueues the
next one, and the launch sequence of a step has no host visible inputs that
change from step to step, which is what allows it to be captured once as a
CUDA graph and replayed.

The control flow the host used to do per step is done by two tiny kernels
(`finish_kernel!` and `commit_kernel!` in `GPUKernels`). Every kernel of a
step first checks the stop flag: once the device decides that the cell list
must be rebuilt or that the output time is reached, the remaining launches
of the current batch are no-ops until the host reads the state back and
reacts (`SimulationLoop` in `SPHCellList`).
"""
module GPUStepState

using CUDA
using Adapt

export StepState, DeviceStep, HostStep, step_active, step_dt, step_time,
       F_DT, F_TIME, F_DX, F_TOUT, F_DISP, I_STOP, I_PHASE, I_ITER,
       I_GRID_STATUS, I_GRID_ORIGIN, I_GRID_DIMS, I_GRID_NCELLS,
       STOP_NONE, STOP_REBUILD, STOP_OUTPUT, PHASE_NEED_DT, PHASE_DT_READY,
       readback!, set_output_time!, resume_after_rebuild!, invalidate_graphs!

# Slots of the floating point state vector
const F_DT   = 1   # time step of the current step
const F_TIME = 2   # simulated time, advanced after every completed step
const F_DX   = 3   # displacement bound accumulated since the last cell list rebuild
const F_TOUT = 4   # end of the current output interval
const F_DISP = 5   # 4 * maximum displacement of the last step (host side batch size estimate)
const NUM_F  = 5

# Slots of the integer state vector
const I_STOP  = 1
const I_PHASE = 2
const I_ITER  = 3
# A conditional grid preparation publishes its header here before the existing
# timestep readback, avoiding a separate bounding-box or grid-status transfer.
const I_GRID_STATUS = 4
const I_GRID_ORIGIN = 5   # three slots; unused axes are zero
const I_GRID_DIMS   = 8   # three slots; unused axes are one
const I_GRID_NCELLS = 11
const NUM_I   = 11

const STOP_NONE    = Int32(0)
const STOP_REBUILD = Int32(1)  # the cell list must be rebuilt before the step can run
const STOP_OUTPUT  = Int32(2)  # the output time is reached, no further step is taken

const PHASE_NEED_DT  = Int32(0)  # `dt` of the next step still has to be computed
const PHASE_DT_READY = Int32(1)  # `dt` is stored, the step itself has not run yet

"""
    StepState{T}(; dt, time, dx, iteration)

Device state of the time loop plus a pinned host mirror that `readback!`
fills. `graphs` caches CUDA graphs of power-of-two step batches per set of
device pointers. The persistent particle arrays are swapped at every cell
list rebuild; the batch length is also part of the cache key.
"""
mutable struct StepState{T}
    f::CuVector{T}
    i::CuVector{Int32}
    fh::Vector{T}
    ih::Vector{Int32}
    primed::Bool        # the reduction workspace holds a valid reduction for the next `dt`
    readbacks::Int      # number of host read backs (device synchronizations of the loop)
    graphs::Dict{Tuple{UInt, Int, Int}, CUDA.CuGraphExec}
end

function StepState{T}(; dt = zero(T), time = zero(T), dx = zero(T), iteration::Integer = 0) where {T}
    fh = zeros(T, NUM_F)
    ih = zeros(Int32, NUM_I)
    fh[F_DT]   = dt
    fh[F_TIME] = time
    fh[F_DX]   = dx
    fh[F_TOUT] = time
    ih[I_ITER] = iteration
    try
        CUDA.pin(fh)
        CUDA.pin(ih)
    catch
        # pinning is an optimisation only
    end
    return StepState{T}(CuArray(fh), CuArray(ih), fh, ih, false, 0,
                        Dict{Tuple{UInt, Int, Int}, CUDA.CuGraphExec}())
end

"""
Kernel side view of a `StepState` (what `@cuda` receives when a `StepState`
is passed as an argument).
"""
struct DeviceStep{T, F <: AbstractVector{T}, I <: AbstractVector{Int32}}
    f::F
    i::I
end


Adapt.adapt_structure(to, s::StepState) = DeviceStep(adapt(to, s.f), adapt(to, s.i))
Adapt.adapt_structure(to, s::DeviceStep) = DeviceStep(adapt(to, s.f), adapt(to, s.i))

"""
Fixed step size and time passed by value, for launching the kernels of a step
in isolation (unit tests). Such a step is always active.
"""
struct HostStep{T}
    dt::T
    time::T
end

@inline step_active(s::DeviceStep) = @inbounds s.i[I_STOP] == STOP_NONE
@inline step_dt(s::DeviceStep)     = @inbounds s.f[F_DT]
@inline step_time(s::DeviceStep)   = @inbounds s.f[F_TIME]

@inline step_active(::HostStep)  = true
@inline step_dt(s::HostStep)     = s.dt
@inline step_time(s::HostStep)   = s.time

#---------------------------------------------------------------
# Host side access (every call synchronizes the device)
#---------------------------------------------------------------

"""
    readback!(state) -> state

Copy the device state into the host mirror (`state.fh`, `state.ih`). Waits
for all previously enqueued work.
"""
function readback!(s::StepState)
    copyto!(s.fh, s.f)
    copyto!(s.ih, s.i)
    s.readbacks += 1
    return s
end

# Write one slot of a device vector; the staging array outlives the copy.
function set_slot!(dev::CuVector{T}, host::Vector{T}, idx::Integer, val) where {T}
    v = convert(T, val)
    host[idx] = v
    copyto!(dev, idx, T[v], 1, 1)
    return nothing
end

"""
    set_output_time!(state, t_out)

Start a new output interval: store its end time and clear the stop flag.
"""
function set_output_time!(s::StepState, t_out)
    set_slot!(s.f, s.fh, F_TOUT, t_out)
    set_slot!(s.i, s.ih, I_STOP, STOP_NONE)
    return nothing
end

"""
    resume_after_rebuild!(state)

Continue after the host rebuilt the cell list: the displacement bound starts
from zero and the stop flag is cleared. The stored `dt` is kept (the state
that determined it was reordered, not changed).
"""
function resume_after_rebuild!(s::StepState{T}) where {T}
    set_slot!(s.f, s.fh, F_DX, zero(T))
    set_slot!(s.i, s.ih, I_STOP, STOP_NONE)
    return nothing
end

"""
Drop all cached graphs (after a device buffer they refer to was reallocated).
"""
invalidate_graphs!(s::StepState) = (empty!(s.graphs); nothing)

end # module
