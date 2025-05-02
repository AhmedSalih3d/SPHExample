module TimeStepping

    export Δt

    using LinearAlgebra
    using Parameters

    # Updated adaptive time stepping function without allocations
    function Δt(Position, Velocity, Acceleration, Density, Pressure, SimulationConstants, SPHKernel)
        @unpack c₀, CFL = SimulationConstants
        @unpack h, η² = SPHKernel

        κ = 10.0

        function max_visc(Velocity, Position, h, η²)
            maxval = -Inf
            for i in eachindex(Velocity, Position)
                tmp = abs(h * dot(Velocity[i], Position[i]) / (dot(Position[i], Position[i]) + η²))
                tmp > maxval && (maxval = tmp)
            end
            return maxval
        end

        visc = max_visc(Velocity, Position, h, η²)

        dt1 = Inf
        for a in Acceleration
            dt1_candidate = sqrt(h / norm(a))
            dt1 = min(dt1, dt1_candidate)
        end

        dt2 = h / (c₀ + visc)

        dt3 = Inf
        for i in eachindex(Density, Pressure)
            cᵢ = κ * sqrt(abs(Pressure[i]) / Density[i])
            local_dt = h / max(cᵢ, c₀)
            dt3 = min(dt3, local_dt)
        end

        dt = CFL * min(dt1, dt2, dt3)

        return dt
    end


end