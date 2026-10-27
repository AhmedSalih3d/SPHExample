const MOTION_DURATIONS = (6.0,10.0,10.0,2.5,17.0,35.0,20.0,10.0,18.0)
const MOTION_INTERVALS = (0.01,0.02,0.02,0.01,0.05,0.05,0.02,0.02,0.02)

# The released XML reader stores duration/start attributes as Float32 before
# evaluating in Float64. Preserve those input times, including reset boundaries.
xml_time(t) = Float64(Float32(t))

function repeating_intervals(velocities, durations, start, horizon)
    intervals = Tuple{Float64,Float64,Point3}[]
    t = xml_time(start)
    while t < horizon
        for (v,d) in zip(velocities,durations)
            finish = t+xml_time(d)
            push!(intervals,(t,finish,Point3(v)))
            t = finish
            t >= horizon && break
        end
    end
    return intervals
end

function parallel_translation(tracks, horizon)
    boundaries = sort!(unique([0.0,horizon,[clamp(t,0,horizon) for track in tracks for interval in track for t in interval[1:2]]...]))
    segments = map(1:(length(boundaries)-1)) do i
        midpoint = (boundaries[i]+boundaries[i+1])/2
        velocity = zero(Point3)
        for track in tracks, (start,finish,v) in track
            start <= midpoint < finish && (velocity += v)
        end
        KinematicSegment(boundaries[i+1]-boundaries[i];velocity)
    end
    return MotionSequence(segments)
end

function masked_history(history, mask)
    return SampledTrajectory(history.times,[value .* Point3(mask) for value in history.values])
end

function case_motion(number)
    origin = (0,0,0)
    node(id;kwargs...) = PrescribedMotion(id;group_marker=id+1,center=origin,kwargs...)
    rotation(id,angles;center,axis,parent=nothing) = PrescribedMotion(id;group_marker=id+1,rotation=angles,center,axis,parent)
    angular(v,a=0) = MotionSequence([KinematicSegment(1000;velocity=deg2rad(v),acceleration=deg2rad(a))])
    if number == 1
        segments = [KinematicSegment(xml_time(d);velocity=Point3(v,0,0)) for _ in 1:2 for (v,d) in ((1,0.6),(0,0.3),(1,0.6),(0,0.3),(1,0.6),(0,0.3))]
        jumps = [Point3(i in (6,12) ? -1.8 : 0,0,0) for i in 1:12]
        return MotionSystem(node(1;translation=MotionSequence(segments;jumps)))
    elseif number == 2
        horizon = 13.0
        x = repeating_intervals([(1,0,0),(-2,0,0)],[2,1],0,horizon)
        y = repeating_intervals([(0,1,0),(0,-1,0)],[1.3,1.3],0.5,horizon)
        return MotionSystem(node(1;translation=parallel_translation((x,y),horizon)))
    elseif number == 3
        horizon = 13.0
        x = repeating_intervals([(1,0,0),(-1,0,0)],[1.5,1.5],0,horizon)
        y = repeating_intervals([(0,1,0),(0,-1,0)],[1.1,1.1],0.1,horizon)
        child = repeating_intervals([(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)],fill(0.45,4),0.2,horizon)
        return MotionSystem((node(1;translation=parallel_translation((x,y),horizon)),
            node(2;parent=1,translation=parallel_translation((child,),horizon))))
    elseif number == 4
        translation = MotionSequence([KinematicSegment(xml_time(1.411);velocity=(0,0,0),acceleration=(2,0,0)),
            KinematicSegment(1;velocity=(-2,5,0),acceleration=(0,-10,0))])
        return MotionSystem(node(1;translation))
    elseif number == 5
        return MotionSystem((rotation(3,angular(20);center=(0.5,0.5,0),axis=(0,0,-1)),
            rotation(4,angular(240);center=(0.5,0.5,0),axis=(0,0,-1))))
    elseif number == 6
        angles, center, axis = angular(-50,9),(0,0,1.85),(0,-1,0)
        parent = PrescribedMotion(1000;group_marker=nothing,rotation=angles,center,axis)
        return MotionSystem((parent,node(1;parent=1000),node(3;parent=1000),node(4;parent=1000),
            node(5;translation=CircularTrajectory(angles;reference=(1.3,-0.7,1.85),center,axis))))
    elseif number == 7
        angles = MotionSequence([SinusoidalSegment(5;amplitude=deg2rad(a),frequency=f) for _ in 1:2 for (a,f) in ((60,0.2),(75,0.4))])
        translation = MotionSequence([SinusoidalSegment(5;amplitude=(a,0,0),frequency=(f,0,0)) for _ in 1:2 for (a,f) in ((2.3,0.2),(2.55,0.4))])
        center, axis = (0,0,2.85),(0,-1,0)
        return MotionSystem((rotation(4,angles;center,axis),
            node(5;translation=CircularTrajectory(angles;reference=(0,-0.7,0.2),center,axis)),node(6;translation)))
    elseif number == 8
        history = load_translation_trajectory(joinpath(@__DIR__,"data","Motion08mov_f3.txt");value_columns=(2,3,4))
        return MotionSystem((node(200;translation=history),
            node(150;translation=TrajectoryWindow(masked_history(history,(1,1,0));end_time=8,finish_offset=(0,0,-0.02))),
            node(151;translation=masked_history(history,(1,0,1))),node(152;translation=masked_history(history,(0,1,1)))))
    else
        first_file = load_rotation_trajectory(joinpath(@__DIR__,"data","Motion09mov_deg.csv");angle_units=:degrees)
        second_file = load_rotation_trajectory(joinpath(@__DIR__,"data","Motion09mov_rad.csv");angle_units=:radians)
        # Both axes lie on the same line; reversing its direction reverses the
        # second file's signed angle. Join at 9 s without inventing a pause.
        times = [t for t in first_file.times if t < 9]
        angles = [first(trajectory_state(first_file,t)) for t in times]
        angle9 = first(trajectory_state(first_file,9))
        push!(times,9); push!(angles,angle9)
        for t in second_file.times
            0 < t < 9 || continue
            push!(times,9+t); push!(angles,angle9-first(trajectory_state(second_file,t)))
        end
        push!(times,18); push!(angles,angle9-first(trajectory_state(second_file,9)))
        return MotionSystem(rotation(1,SampledTrajectory(times,angles);center=(1,1,0.03),axis=(0,1,0)))
    end
end
