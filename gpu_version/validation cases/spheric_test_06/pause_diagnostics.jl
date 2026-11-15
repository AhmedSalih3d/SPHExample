using HDF5, CSV, Statistics

directory = joinpath(@__DIR__,"results","pause_diagnostic")
records = NamedTuple[]
for speed in (40,160)
    name = "re100_c$(speed)_pause"
    h5open(joinpath(@__DIR__,"runs","pause_diagnostic",name,"MovingSquare.vtkhdf"),"r") do file
        root = file["VTKHDF"]
        for frame in 1:5:length(root["Steps/Values"])
            offset,count = root["Steps/PointOffsets"][frame],root["NumberOfPoints"][frame]
            columns = offset+1:offset+count
            fluid = root["PointData/GroupMarker"][columns] .== 2
            density = Float64.(root["PointData/Density"][columns][fluid])
            velocity = Float64.(root["PointData/Velocity"][:,columns][:,fluid])
            pressure = (1000speed^2/7) .* ((density ./ 1000).^7 .- 1)
            push!(records,(; c0=speed,time=Float64(root["Steps/Values"][frame]),
                pressure_spatial_std=std(pressure; corrected=false),
                kinetic_energy=0.5*1000*0.025^2*sum(abs2,velocity)))
        end
    end
end
CSV.write(joinpath(directory,"decay.csv"),records)
