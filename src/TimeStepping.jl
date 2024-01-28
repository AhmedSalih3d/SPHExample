module TimeStepping

    export Δt

    using LinearAlgebra
    using Parameters

    # A few time stepping controls implemented to allow for an adaptive time step
    function Δt(α,points,v,SimulationConstants)

        @unpack c₀, H, CFL, η² = SimulationConstants

        visc  = maximum(@. abs(H * dot(v,points) / (dot(points,points) + η²)))
        dt1   = minimum(@. sqrt(H / norm(α)))
        dt2   = H / (c₀+visc)

        dt    = CFL*min(dt1,dt2)

        return dt
    end

end