module TimeStepping

    export Δt

    using LinearAlgebra
    using Parameters

    # A few time stepping controls implemented to allow for an adaptive time step
    function Δt(Position, Velocity, Acceleration,SimulationConstants)
        @unpack c₀, h, CFL, η² = SimulationConstants

        function max_visc(Velocity,Position,h,η²)
            maxval = -Inf
            for i in eachindex(Velocity,Position)
                 tmp = abs(h * dot(Velocity[i],Position[i]) / (dot(Position[i],Position[i]) + η²))
                 tmp > maxval && (maxval = tmp)
            end
            return maxval
         end
        
        visc = max_visc(Velocity,Position,h,η²)

        dt1 = Inf
        for Acceleration_ in Acceleration
            dt1_candidate = sqrt(h / norm(Acceleration_))
            dt1 = min(dt1, dt1_candidate)
        end


        dt2   = h / (c₀+visc)

        dt    = CFL*min(dt1,dt2)

        return dt
    end

end