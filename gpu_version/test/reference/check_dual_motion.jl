using SPHExampleGPU, StaticArrays
include("dual_motion_cases.jl")
errors = Dict{String,Float64}()
for line in Iterators.drop(eachline(joinpath(@__DIR__,"dual_motion_reference.csv")),1)
    fields = split(line,',')
    name = fields[1]
    t = parse(Float64,fields[2])
    expected = SVector{3,Float64}(parse.(Float64,fields[3:5]))
    actual = motion_position(dual_motion_case(name),2,(2,0,1),t)
    errors[name] = max(get(errors,name,0.),maximum(abs,actual-expected))
end
foreach(pair -> println(first(pair)," ",last(pair)),sort(collect(errors);by=first))
