module TimeStepping

    export Δt

    using LinearAlgebra

    # A few time stepping controls implemented to allow for an adaptive time step
"""
    Δt(α, points, v, c₀, h, CFL)

A few time stepping controls implemented to allow for an adaptive time step:
    

"""
function Δt(α, points, v, c₀, h, CFL)
        eta2  = (0.01)h * (0.01)h
        visc  = maximum(@. abs(h * dot(v, points) / (dot(points, points) + eta2)))
        dt1   = minimum(@. sqrt(h / norm(α)))
        dt2   = h / (c₀ + visc)

        dt    = CFL * min(dt1, dt2)

        return dt
end

end