# Matching inputs for dual_motion_reference.cpp, evaluated through the public
# trajectory API. The names identify patterns, not the upstream example XMLs.
function dual_motion_case(name)
    center = (0,0,0)
    # JMatrix4's angle convention maps to our right-hand axis axisp1-axisp2.
    axis = (0,0,-1)
    linear(v,a=(0,0,0)) = MotionSequence([KinematicSegment(3;velocity=v,acceleration=a)])
    angle(v,a=0) = MotionSequence([KinematicSegment(3;velocity=v,acceleration=a)])
    sinusoidal = MotionSequence([SinusoidalSegment(3;amplitude=0.8,frequency=0.5,phase=0.3)])
    if name == "01"
        translation = MotionSequence([KinematicSegment(1;velocity=(1,0,0)),
            KinematicSegment(1;velocity=(0,0,0)),KinematicSegment(1;velocity=(0,0,1))])
        return MotionSystem(PrescribedMotion(2;center,translation))
    elseif name == "02"
        translation = MotionSequence([KinematicSegment(1;velocity=(2,0,0)),KinematicSegment(2;velocity=(0,0,1))])
        return MotionSystem(PrescribedMotion(2;center,translation))
    elseif name == "03"
        return MotionSystem((PrescribedMotion(1;group_marker=nothing,center,translation=linear((0.2,0,0.1))),
            PrescribedMotion(2;parent=1,center,axis,rotation=angle(0.4))))
    elseif name == "04"
        return MotionSystem(PrescribedMotion(2;center,translation=linear((1,0,0),(2,0,-1))))
    elseif name == "05"
        return MotionSystem(PrescribedMotion(2;center,axis,rotation=angle(0.4)))
    elseif name == "06rot"
        return MotionSystem(PrescribedMotion(2;center,axis,rotation=angle(0.3,0.2)))
    elseif name == "06cir"
        return MotionSystem(PrescribedMotion(2;center,translation=CircularTrajectory(angle(0.3,0.2);reference=(2,0,1),axis)))
    elseif name == "07rect"
        translation = MotionSequence([SinusoidalSegment(3;amplitude=(1,0.5,0.25),frequency=(0.5,0.25,0.75),phase=(0.2,0.4,0.1))])
        return MotionSystem(PrescribedMotion(2;center,translation))
    elseif name == "07rot"
        return MotionSystem(PrescribedMotion(2;center,axis,rotation=sinusoidal))
    elseif name == "07cir"
        return MotionSystem(PrescribedMotion(2;center,translation=CircularTrajectory(sinusoidal;reference=(2,0,1),axis)))
    elseif name == "08"
        return MotionSystem(PrescribedMotion(2;center,translation=SampledTrajectory([0,1,3],[(0,0,0),(2,1,-1),(3,-2,1)])))
    elseif name in ("09deg","09rad")
        return MotionSystem(PrescribedMotion(2;center,axis,rotation=SampledTrajectory([0,1,3],deg2rad.([0,45,-10]))))
    end
    error("unknown reference case $name")
end
