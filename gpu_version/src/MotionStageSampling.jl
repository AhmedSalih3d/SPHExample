module MotionStageSampling

using Adapt
using ..GPUStepState: step_time, step_dt
import ..GPUStepState: prepare_stage_samples!
using ..PrescribedRotation: sample_rotation!
using ..ExternalForces: ExternalForce, sample_force

struct MotionStageSampler{R,F,SingleNeighbor}
    rotation::R
    forces::F
end

function MotionStageSampler(rotation, forces, ::Val{SingleNeighbor}) where {SingleNeighbor}
    rotation === nothing && isempty(forces) && return nothing
    return MotionStageSampler{typeof(rotation),typeof(forces),SingleNeighbor}(rotation, forces)
end

function Adapt.adapt_structure(to, sampler::MotionStageSampler{R,F,SingleNeighbor}) where {R,F,SingleNeighbor}
    return MotionStageSampler(Adapt.adapt(to, sampler.rotation), Adapt.adapt(to, sampler.forces), Val(SingleNeighbor))
end

@inline sample_rotation_stages!(::Nothing, step) = nothing
@inline function sample_rotation_stages!(rotation, step)
    sample_rotation!(rotation, step, 0.0, 1)
    sample_rotation!(rotation, step, 0.5, 2)
    sample_rotation!(rotation, step, 1.0, 3)
    return nothing
end

@inline sample_force_stages!(force::ExternalForce, step, single) = nothing
@inline function sample_force_stages!(force::NamedTuple, step, ::Val{SingleNeighbor}) where {SingleNeighbor}
    time = step_time(step)
    @inbounds begin
        if !SingleNeighbor
            force.sample[1] = sample_force(force.history, time + zero(time))
        end
        force.sample[2] = sample_force(force.history, time + step_dt(step) / 2)
    end
    return nothing
end

@inline function prepare_stage_samples!(sampler::MotionStageSampler{R,F,SingleNeighbor}, step) where {R,F,SingleNeighbor}
    sample_rotation_stages!(sampler.rotation, step)
    for force in sampler.forces
        sample_force_stages!(force, step, Val(SingleNeighbor))
    end
    return nothing
end

end
