# Reconstruct final stock-Dual loads independently of ComputeForces.
using SPHExampleGPU, StaticArrays, CSV
include(joinpath(@__DIR__, "dual_compare.jl"))
const C = Test06DualComparison
const S = C.SphericTest06
const OUT = joinpath(@__DIR__, "results", "coefh_reynolds")
rows = NamedTuple[]
for re in (50, 100, 150)
    final = C.dual_final_state(joinpath(@__DIR__, "runs", "coefh_reynolds", "re$(re)_dual"))
    pos = [SVector(Float64(r[Symbol("Pos.x [m]")]), Float64(r[Symbol("Pos.z [m]")])) for r in final]
    vel = [SVector(Float64(r[Symbol("Vel.x [m/s]")]), Float64(r[Symbol("Vel.z [m/s]")])) for r in final]
    rho = [Float64(r[Symbol("Rhop [kg/m^3]")]) for r in final]
    types = [Int(r.Type == 3 ? Fluid : r.Type == 1 ? Moving : Fixed) for r in final]
    markers = [r.Type == 3 ? 2 : r.Type == 1 ? 3 : 1 for r in final]
    h = Float64(Float32(1.2sqrt(2)*.025))
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h, η² = (.1h)^2)
    constants = SimulationConstants{Float64}(2; dx = .025, ρ₀ = 1000., c₀ = 160., g = 0., ν₀ = 1/re)
    loads = S.frame_loads(pos, vel, rho, types, markers, kernel, constants)
    saved = last(CSV.File(joinpath(OUT, "re$(re)_dual", "dual.csv")))
    isapprox(loads.cd_pressure, saved.cd_pressure; rtol = 1e-4) || error("Pressure extraction mismatch")
    isapprox(loads.cd_viscous, saved.cd_viscous; rtol = 1e-4) || error("Viscous extraction mismatch")
    push!(rows, (; reynolds = re, time = saved.time, tool_pressure = saved.cd_pressure,
        reconstructed_pressure = loads.cd_pressure, tool_viscous = saved.cd_viscous,
        reconstructed_viscous = loads.cd_viscous))
end
CSV.write(joinpath(OUT, "dual_force_check.csv"), rows)
foreach(println, rows)
