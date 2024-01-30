module TimeStepping

    export Δt

    using LinearAlgebra
    using Parameters

    # A few time stepping controls implemented to allow for an adaptive time step
    function Δt(FinalResults,SimulationConstants)
        @unpack c₀, H, CFL, η²                   = SimulationConstants
        @unpack Position, Velocity, Acceleration = FinalResults

        function max_visc(Velocity,Position,h,η²)
            maxval = -Inf
            for i in eachindex(Velocity,Position)
                 tmp = abs(h * dot(Velocity[i],Position[i]) / (dot(Position[i],Position[i]) + η²))
                 tmp > maxval && (maxval = tmp)
            end
            return maxval
         end
        
        visc = max_visc(Velocity,Position,H,η²)
        
        dt1   = minimum(Acceleration) do (Acceleration_)
            sqrt(H / norm(Acceleration_))
        end

        dt2   = H / (c₀+visc)

        dt    = CFL*min(dt1,dt2)

        return dt
    end

end