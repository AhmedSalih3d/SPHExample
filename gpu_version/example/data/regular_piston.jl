"""Second-order regular piston displacement and velocity, including the reference startup ramp."""
function regular_piston_state(height,period,depth;ramp_periods = 0,order = 2,gravity = 9.81)
    all(x -> isfinite(x) && x > 0,(height,period,depth,gravity)) || throw(ArgumentError("wave height, period, depth and gravity must be finite and positive"))
    isfinite(ramp_periods) && ramp_periods >= 0 || throw(ArgumentError("ramp_periods must be finite and nonnegative"))
    order in (1,2) || throw(ArgumentError("wave order must be one or two"))
    omega = 2pi/period
    k = omega^2/gravity
    for _ in 1:50
        kd = k*depth
        k -= (gravity*k*tanh(kd)-omega^2)/(gravity*tanh(kd)+gravity*k*depth*sech(kd)^2)
    end
    kd = k*depth
    transfer = 2sinh(kd)^2/(sinh(kd)*cosh(kd)+kd)
    a1 = height/(2transfer)
    a2 = order == 2 ? height^2/(32depth)*(3cosh(kd)/sinh(kd)^3-2/transfer) : 0.
    duration = ramp_periods*period
    return time -> begin
        time < 0 && return (0.,0.)
        displacement = a1*sin(omega*time)+a2*sin(2omega*time)
        velocity = a1*omega*cos(omega*time)+2a2*omega*cos(2omega*time)
        if duration > 0 && time < duration
            s = time/duration
            ramp = s^2*(6-8s+3s^2)
            derivative = 12s*(1-s)^2/duration
            return ramp*displacement,ramp*velocity+derivative*displacement
        end
        return displacement,velocity
    end
end

function regular_piston_motion(marker,::Val{D};height,period,depth,duration,ramp_periods = 0,order = 2) where D
    state = regular_piston_state(height,period,depth;ramp_periods,order)
    times = collect(0.:period/400:duration+period)
    samples = state.(times)
    vector(x) = ntuple(d -> d == 1 ? x : 0.,D)
    motion = PrescribedMotion(marker;translation = SampledTrajectory(times,vector.(first.(samples));velocities = vector.(last.(samples))),
        center = ntuple(_ -> 0.,D))
    return motion,state
end
