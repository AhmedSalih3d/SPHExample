module StepStateTests

using Test, SPHExampleGPU, CUDA, StaticArrays
using SPHExampleGPU.GPUStepState
using SPHExampleGPU.GPUKernels: launch_finish!, launch_commit!
using SPHExampleGPU.GPUReductions: ReductionWorkspace

const STEP = SPHExampleGPU.GPUStepState

function write_state!(f, i, value)
    index = threadIdx().x
    if index <= length(f)
        f[index] = eltype(f)(value + index) / 8
    end
    if index <= length(i)
        i[index] = Int32(value - index)
    end
    return nothing
end

function check_readback(state, value, count)
    fill!(state.fh, NaN)
    fill!(state.ih, typemin(Int32))
    @test readback!(state) === state
    @test state.fh == [eltype(state.fh)(value + index) / 8 for index in eachindex(state.fh)]
    @test state.ih == [Int32(value - index) for index in eachindex(state.ih)]
    @test state.readbacks == count
end

if CUDA.functional()
    @testset "coordinated control-state readback" begin
        for T in (Float32, Float64)
            @testset "$T" begin
                state = StepState{T}(dt = T(0.125), time = T(0.5), dx = T(0.25), iteration = 7)
                @test CUDA.is_pinned(pointer(state.fh))
                @test CUDA.is_pinned(pointer(state.ih))
                @test readback!(state) === state
                @test state.fh == T[0.125, 0.5, 0.25, 0.5, 0]
                @test state.ih[STEP.I_ITER] == 7
                @test state.readbacks == 1

                stream = CuStream()
                CUDA.stream!(stream) do
                    for value in 1:20
                        @cuda threads = 32 write_state!(state.f, state.i, value)
                        check_readback(state, value, value + 1)
                    end
                    # Replayed graph writes must precede both downloads.
                    graph = CUDA.capture() do
                        @cuda threads = 32 write_state!(state.f, state.i, 73)
                    end
                    executable = CUDA.instantiate(graph)
                    for count in 22:24
                        CUDA.launch(executable)
                        check_readback(state, 73, count)
                    end
                end

                # Each array can have a different producing stream.
                CUDA.stream!(stream) do
                    fill!(state.f, T(3.5))
                end
                CUDA.stream!(CuStream()) do
                    fill!(state.i, Int32(91))
                end
                GC.gc(true)
                CUDA.stream!(CuStream()) do
                    readback!(state)
                end
                @test state.fh == fill(T(3.5), STEP.NUM_F)
                @test state.ih == fill(Int32(91), STEP.NUM_I)
                @test state.readbacks == 25

                # Reject undersized mirrors before either unsafe transfer.
                for field in (:fh, :ih)
                    invalid = StepState{T}()
                    setfield!(invalid, field, eltype(getfield(invalid, field))[])
                    @test_throws BoundsError readback!(invalid)
                    @test invalid.readbacks == 0
                end
            end
        end
    end

    @testset "readback preserves timestep and stop transitions" begin
        for T in (Float32, Float64), graph in (false, true)
            constants = SimulationConstants{T}(dx = 0.02, c₀ = 42, CFL = 0.5)
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            reduction = ReductionWorkspace{SVector{3,T}}(32)
            fill!(reduction.partial, SVector{3,T}(3, 0.5, 0.25))
            state = StepState{T}()
            set_output_time!(state, one(T))
            function step!()
                launch_finish!(state, reduction, kernel, constants)
                launch_commit!(state)
            end
            step!()
            readback!(state)
            dt = constants.CFL * min(T(0.5), kernel.h / (constants.c₀ + T(3)))
            @test state.fh[STEP.F_DT] == dt
            @test state.fh[STEP.F_TIME] == 0
            @test state.ih[STEP.I_STOP] == STOP_REBUILD
            @test state.ih[STEP.I_PHASE] == PHASE_DT_READY
            @test state.ih[STEP.I_ITER] == 0

            executable = graph ? CUDA.instantiate(CUDA.capture(step!)) : nothing
            advance!() = graph ? CUDA.launch(executable) : step!()
            advance!()
            readback!(state)
            @test state.ih[STEP.I_ITER] == 0
            @test state.fh[STEP.F_TIME] == 0

            resume_after_rebuild!(state)
            advance!()
            readback!(state)
            @test state.fh[STEP.F_DT] == dt
            @test state.fh[STEP.F_TIME] == dt
            @test state.fh[STEP.F_DX] == 0
            @test state.ih[STEP.I_ITER] == 1
            @test state.ih[STEP.I_PHASE] == PHASE_NEED_DT
            @test state.ih[STEP.I_STOP] == STOP_NONE

            set_output_time!(state, zero(T))
            advance!()
            readback!(state)
            @test state.fh[STEP.F_TIME] == dt
            @test state.ih[STEP.I_ITER] == 1
            @test state.ih[STEP.I_PHASE] == PHASE_NEED_DT
            @test state.ih[STEP.I_STOP] == STOP_OUTPUT
            @test state.readbacks == 4
        end
    end
else
    @testset "control-state readback requires CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
