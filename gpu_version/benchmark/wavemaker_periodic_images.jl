# Force-only diagnosis of translated queries versus stored periodic copies.
# This intentionally excludes copy generation, sorting and integration from timings.
# Run from gpu_version/:
# julia --project=. -t 2,0 benchmark/wavemaker_periodic_images.jl [saved.vtkhdf] [1,51]
# Frame numbers are one-based. Output is written to a new output/ directory.
using SPHExampleGPU, CUDA, StaticArrays, HDF5, LinearAlgebra, Statistics, StructArrays
const CL = SPHExampleGPU.SPHCellList
const GK = SPHExampleGPU.GPUKernels
const ROOT = normpath(joinpath(@__DIR__,".."))
const DIRECTORY = mktempdir(joinpath(ROOT,"output");prefix="wavemaker_images_",cleanup=false)
const INPUT = isempty(ARGS) ? joinpath(ROOT,"output","wavemaker","Wavemaker.vtkhdf") : abspath(ARGS[1])
const FRAMES = length(ARGS)>=2 ? parse.(Int,split(ARGS[2],",")) : [1,51]
scope = Module(:WavemakerImagesAudit)
Core.eval(scope,quote
    include(path) = Base.include($scope,path)
    RunSimulation(;kwargs...) = (;kwargs...)
end)
load_expression(expr) = expr isa Expr && expr.head===:call && first(expr.args)===:run_wavemaker ? nothing : expr
case_path = joinpath(ROOT,"example","Wavemaker","WavemakerCase.jl")
case_source=read(case_path,String)
write(joinpath(DIRECTORY,"WavemakerCase.snapshot.jl"),case_source)
Base.include_string(load_expression,scope,case_source,case_path)
setup = Base.invokelatest(getproperty(scope,:run_wavemaker);simulation_time=0.1,
    visualize=false,open_log_file=false,save_location=joinpath(DIRECTORY,"setup"))

function load_host(setup,frame)
    metadata = deepcopy(setup.SimMetaData)
    metadata.TimeSteppingMode = setup.SimTimeStepping
    particles = AllocateDataStructures(setup.SimGeometry,metadata)
    time = h5open(INPUT,"r") do file
        root=file["VTKHDF"]
        count=length(particles)
        offset=Int(root["Steps/PointOffsets"][frame])
        columns=offset+1:offset+count
        particles.Position .= SVector{3,Float64}.(eachcol(root["Points"][:,columns]))
        particles.Velocity .= SVector{3,Float32}.(eachcol(root["PointData/Velocity"][:,columns]))
        particles.Density .= root["PointData/Density"][columns]
        particles.ID .= root["PointData/ID"][columns]
        particles.GroupMarker .= root["PointData/GroupMarker"][columns]
        particles.Type .= SPHExampleGPU.ParticleType.(root["PointData/Type"][columns])
        root["Steps/Values"][frame]
    end
    return particles,metadata,time
end

function make_state(setup,frame,images)
    particles,metadata,time=load_host(setup,frame)
    originals=length(particles)
    sort(particles.ID)==collect(1:originals) || error("Expected consecutive original particle IDs starting at one")
    periodicity=metadata.Periodicity
    owner_ids=Int32.(1:originals)
    owner_shifts=fill(zero(SVector{3,Float64}),originals)
    if images
        owners=Int[]
        shifts=SVector{3,Float64}[]
        H=Float64(setup.SimKernel.H)
        for i in eachindex(particles.Position)
            particles.Type[i]==Frozen && continue
            y=particles.Position[i][periodicity.axis]
            if y < periodicity.lower+H
                push!(owners,i);push!(shifts,-periodicity.translation)
            elseif y > periodicity.upper-H
                push!(owners,i);push!(shifts,periodicity.translation)
            end
        end
        names=propertynames(particles)
        fields=map(name -> vcat(getproperty(particles,name),getproperty(particles,name)[owners]),names)
        particles=StructArray(NamedTuple{names}(fields))
        for k in eachindex(owners)
            particles.Position[originals+k] += shifts[k]
            push!(owner_ids,particles.ID[originals+k])
            push!(owner_shifts,shifts[k])
            particles.ID[originals+k] = originals+k
        end
        metadata.Periodicity=nothing
        metadata.GPUPeriodicCellCoordinates=false
    end
    gpu=upload_particles(particles,metadata)
    support=GPUSupportArrays{3,Float32}(gpu,metadata)
    cells=CellListWorkspace{3,Float64}(length(particles);reach=metadata.GPUCellSubdivision,periodicity=metadata.Periodicity)
    cells.frozen_total=count(==(Frozen),particles.Type)
    CL.rebuild_cell_list!(gpu,cells,setup.SimKernel.H⁻¹;support,constants=setup.SimConstants,kernel=setup.SimKernel)
    step=HostStep(0.0002,Float64(time))
    GK.launch_pos_cells!(support.PosCells,gpu.Position,gpu.CellID,cells.grid_dev,step,setup.SimKernel)
    images || GK.launch_periodic_halo!(support.PeriodicBounds,gpu.Position,gpu.CellID,cells.grid_dev,setup.SimKernel.H,step)
    CUDA.synchronize()
    order=sortperm(Array(gpu.ID))[1:originals]
    return (;gpu,support,cells,step,time,images,originals,order,owner_ids,owner_shifts)
end

function host_search(state)
    positions=Array(state.gpu.Position)
    pc=Array(state.support.PosCells)
    pairs=state.images ? pc : GK.PeriodicCellPairSource(Float32,positions,pc,state.cells.grid)
    return (;positions,pairs,types=Array(state.gpu.Type),ids=Array(state.gpu.ID),
        cells=Array(state.gpu.CellID),starts=Array(state.cells.CellStart),
        bounds=state.images ? nothing : Array(state.support.PeriodicBounds))
end
canonical_shift(x)=ntuple(d -> iszero(x[d]) ? 0.0 : Float64(x[d]),3)
function accepted_pairs(state,host,uid,kernel)
    i=findfirst(==(uid),host.ids)
    host.types[i]==Frozen && return Dict{Tuple{Int32,NTuple{3,Float64}},SVector{3,Float32}}()
    grid=state.cells.grid
    size=SPHExampleGPU.GPUCellGrid.cell_size(grid,kernel.H)
    c=host.cells[i]
    rows=GK.neighbor_rows(state.cells.periodicity,grid,c,size,host.positions[i],host.bounds)
    result=Dict{Tuple{Int32,NTuple{3,Float64}},SVector{3,Float32}}()
    for row in rows
        lo,hi=SPHExampleGPU.GPUCellGrid.row_range(grid,host.starts,c,row)
        for j in lo:hi
            (i==j || host.types[j]==Frozen) && continue
            difference=SPHExampleGPU.GPUCellGrid.pair_vector(host.pairs,host.pairs[i],Int32(j),c+row.off,row,size)
            dot(difference,difference)<=kernel.H² || continue
            id=host.ids[j]
            key=state.images ? (state.owner_ids[id],canonical_shift(state.owner_shifts[id])) :
                (id,canonical_shift(-row.shift))
            result[key]=difference
        end
    end
    return result
end

function interaction!(state,setup)
    gpu,sup,cl,step=state.gpu,state.support,state.cells,state.step
    GK.launch_interactions!(sup.dρdtI,gpu.Acceleration,gpu.Kernel,gpu.KernelGradient,sup.∇Cᵢ,sup.∇◌rᵢ,
        gpu.Position,gpu.Density,gpu.Velocity,gpu.Type,cl.CellStart,gpu.CellID,cl.grid_dev,step,
        setup.SimDensityDiffusion,setup.SimViscosity,setup.SimKernel,setup.SimConstants,Val(false),Val(false);
        threads=128,lanes=Val(1),boundary_forces=Val(false),pos_cells=sup.PosCells,
        periodicity=cl.periodicity,periodic_bounds=state.images ? nothing : sup.PeriodicBounds,
        periodic_cell_coordinates=!state.images,viscous_dt=sup.dual_state.viscous,
        fast_math=setup.SimTimeStepping.fast_math,
        always_inline=setup.SimMetaData.GPUInteractionAlwaysInline,fastmath=setup.SimMetaData.GPUInteractionFastMath)
end
function forces(state)
    (;acceleration=Array(state.gpu.Acceleration)[state.order],rate=Array(state.support.dρdtI)[state.order],
        viscous=Array(state.support.dual_state.viscous)[state.order])
end

records=NamedTuple[]
for frame in FRAMES
    ordinary=make_state(setup,frame,false)
    images=make_state(setup,frame,true)
    interaction!(ordinary,setup);interaction!(images,setup)
    CUDA.synchronize()
    reference=forces(ordinary)
    actual=forces(images)
    difference=norm.(actual.acceleration.-reference.acceleration)
    top=partialsortperm(difference,1:10;rev=true)
    sample_ids=unique(vcat(top,collect(1:997:length(reference.rate))))
    ordinary_host=host_search(ordinary)
    images_host=host_search(images)
    mismatches=0
    vector_error=0.0
    missing_distance=Float64[]
    unmatched_gradient=0.0
    for uid in sample_ids
        expected=accepted_pairs(ordinary,ordinary_host,uid,setup.SimKernel)
        observed=accepted_pairs(images,images_host,uid,setup.SimKernel)
        missing=setdiff(keys(expected),keys(observed))
        extra=setdiff(keys(observed),keys(expected))
        mismatches += length(missing)+length(extra)
        append!(missing_distance,[norm(expected[k]) for k in missing])
        append!(missing_distance,[norm(observed[k]) for k in extra])
        for (table,indices) in ((expected,missing),(observed,extra)), key in indices
            vector=table[key]
            gradient=SPHExampleGPU.∇Wᵢⱼ(setup.SimKernel,norm(vector)*setup.SimKernel.h⁻¹,vector)
            unmatched_gradient=max(unmatched_gradient,norm(gradient))
        end
        for key in intersect(keys(expected),keys(observed))
            vector_error=max(vector_error,norm(expected[key]-observed[key]))
        end
    end
    println("IMAGES_VALIDATION ",(;frame,samples=length(sample_ids),mismatches,vector_error,unmatched_gradient,
        mismatch_distance_range=isempty(missing_distance) ? nothing : extrema(missing_distance),
        max_reference_acceleration=maximum(norm.(reference.acceleration)),
        relative_l2_acceleration=norm(difference)/norm(norm.(reference.acceleration)),
        max_reference_rate=maximum(abs.(reference.rate)),max_reference_viscous=maximum(abs.(reference.viscous))))
    for (round,state) in enumerate((ordinary,images,images,ordinary))
        wall_start=time_ns()
        seconds=CUDA.@elapsed begin
            for repeat in 1:30
                interaction!(state,setup)
            end
        end
        wall_seconds=(time_ns()-wall_start)*1e-9
        actual=forces(state)
        record=(;frame,time=state.time,round,images=state.images,particles=length(state.gpu),
            gpu_ms=1000seconds/30,wall_ms=1000wall_seconds/30,
            max_acceleration=maximum(norm.(actual.acceleration.-reference.acceleration)),
            max_rate=maximum(abs.(actual.rate.-reference.rate)),max_viscous=maximum(abs.(actual.viscous.-reference.viscous)))
        push!(records,record)
        println("IMAGES_FORCE ",record);flush(stdout)
    end
end
for frame in FRAMES, images in (false,true)
    sample=filter(r -> r.frame==frame && r.images==images,records)
    println("IMAGES_FORCE_MEDIAN ",(;frame,images,gpu_ms=median(r.gpu_ms for r in sample)))
end
open(joinpath(DIRECTORY,"results.txt"),"w") do io
    foreach(r -> println(io,r),records)
end
println("IMAGES_DIRECTORY ",DIRECTORY)
