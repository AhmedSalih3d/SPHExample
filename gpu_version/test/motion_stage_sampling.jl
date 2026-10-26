module MotionStageSamplingTests

using Test, SPHExampleGPU, CUDA, StaticArrays

const STEP = SPHExampleGPU.GPUStepState
const ROTATION = SPHExampleGPU.PrescribedRotation
const FORCES = SPHExampleGPU.ExternalForces
const SAMPLING = SPHExampleGPU.MotionStageSampling

if CUDA.functional()
    @testset "timestep motion samples preserve stages and stop/rebuild lifecycle" begin
        for D in (2, 3), T in (Float32, Float64), single in (false, true)
            host = (Position = [zero(SVector{D,T}), SVector{D,T}(ntuple(_ -> one(T), D))],
                ID = [8, 1001], GroupMarker = [3, 4], Type = [Moving, Fluid])
            motion = RotationMotion(3; times = [0.1, 0.3, 0.7], angles = [0.2, -0.4, 0.6])
            rotation = D == 2 ? ROTATION.prepare_rotation(motion, (), host) : nothing
            reference_rotation = D == 2 ? ROTATION.prepare_rotation(motion, (), host) : nothing
            force = ExternalForce{D,T}(4; times = [0.1, 0.3, 0.7],
                linear_acceleration = [ntuple(d -> k*d / 7, D) for k in 1:3],
                angular_acceleration = D == 2 ? [1, -2, 0] : [(1, 2, 3), (-2, 3, -1), (0, 0, 0)],
                start_time = 0.05, end_time = 0.8)
            forces = FORCES.prepare_external_forces(force, host, Val(D), T)
            reference_forces = FORCES.prepare_external_forces(force, host, Val(D), T)
            sampler = SAMPLING.MotionStageSampler(rotation, forces, Val(single))
            kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.01))
            constants = SimulationConstants{T}(D; dx = T(0.01), c₀ = T(10))
            red = ReductionWorkspace{SVector{3,T}}(2)
            fill!(red.partial, SVector{3,T}(0, 1, 0))
            snapshot() = (rotation === nothing ? nothing : Array(rotation.coefficients), Array(forces[1].sample))

            function check_samples(state)
                for (slot, fraction) in enumerate((0.0, 0.5, 1.0))
                    rotation === nothing && break
                    @cuda threads=1 blocks=1 ROTATION.rotation_coefficients_kernel!(reference_rotation, state, fraction)
                    @test Array(rotation.coefficients)[slot] == Array(reference_rotation.coefficients)[1]
                end
                for half_time in (false, true)
                    single && !half_time && continue
                    @cuda threads=1 blocks=1 FORCES.force_sample_kernel!(reference_forces[1].sample,
                        reference_forces[1].history, state, half_time)
                    slot = half_time ? 2 : 1
                    @test Array(forces[1].sample)[slot] == Array(reference_forces[1].sample)[slot]
                end
            end

            for time in T.((0, 0.05, 0.1, 0.3, 0.7, 0.8, 0.9))
                for pending in (false, true)
                    state = StepState{T}(; time, dt = T(0.13))
                    STEP.set_slot!(state.i, state.ih, STEP.I_PHASE,
                        pending ? STEP.PHASE_DT_READY : STEP.PHASE_NEED_DT)
                    launch_finish!(state, red, kernel, constants; stage_sampler = sampler)
                    readback!(state)
                    @test state.ih[STEP.I_STOP] == STEP.STOP_NONE
                    pending && @test state.fh[STEP.F_DT] == T(0.13)
                    check_samples(state)
                    for stop in (STEP.STOP_REBUILD, STEP.STOP_OUTPUT)
                        before = snapshot()
                        STEP.set_slot!(state.i, state.ih, STEP.I_STOP, stop)
                        launch_finish!(state, red, kernel, constants; stage_sampler = sampler)
                        @test snapshot() == before
                    end
                end
            end

            # A newly computed timestep can request sorting before any stage runs.
            # The resumed PHASE_DT_READY path must prepare that same timestep.
            state = StepState{T}(; time = T(0.2), dx = one(T))
            before = snapshot()
            launch_finish!(state, red, kernel, constants; stage_sampler = sampler)
            readback!(state)
            @test state.ih[STEP.I_STOP] == STEP.STOP_REBUILD
            @test snapshot() == before
            dt = state.fh[STEP.F_DT]
            STEP.resume_after_rebuild!(state)
            launch_finish!(state, red, kernel, constants; stage_sampler = sampler)
            readback!(state)
            @test state.fh[STEP.F_DT] == dt
            check_samples(state)

            # A stopped output deadline must not prepare the following step.
            state = StepState{T}(; time = T(0.3))
            STEP.set_output_time!(state, T(0.3); clip_timestep = true)
            before = snapshot()
            launch_finish!(state, red, kernel, constants; stage_sampler = sampler)
            readback!(state)
            @test state.ih[STEP.I_STOP] == STEP.STOP_OUTPUT
            @test snapshot() == before
        end
    end
end

end
