using HDF5: HDF5, h5open
using StaticArrays: SVector
using Random

# # Random points to be written (as a vector of SVector).
# points = rand(SVector{3, Float64}, 1000)
# velocities = randn!(similar(points))

# h5open("particles.vtkhdf", "w") do io
#     # Create toplevel group /VTKHDF
#     gtop = HDF5.create_group(io, "VTKHDF")

#     # Write version of VTKHDF format as an attribute
#     HDF5.attrs(gtop)["Version"] = [2, 1]

#     # Write type of dataset ("PolyData") as an ASCII string to a "Type" attribute.
#     # This is a bit tricky because VTK/ParaView don"t support UTF-8 strings here, which is
#     # the default in HDF5.jl.
#     let s = "PolyData"
#         dtype = HDF5.datatype(s)
#         HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
#         dspace = HDF5.dataspace(s)
#         attr = HDF5.create_attribute(gtop, "Type", dtype, dspace)
#         HDF5.write_attribute(attr, dtype, s)
#     end

#     # Write points + number of points.
#     # Note that we need to reinterpret the vector of SVector onto a 3×Np matrix.
#     Np = length(points)
#     gtop["NumberOfPoints"] = [Np]
#     gtop["Points"] = reinterpret(reshape, eltype(eltype(points)), points)

#     # Write velocities as point data.
#     let g = HDF5.create_group(gtop, "PointData")
#         g["Velocity"] = reinterpret(reshape, eltype(eltype(velocities)), velocities)
#     end

#     # Create and fill Vertices group.
#     let g = HDF5.create_group(gtop, "Vertices")
#         # In our case 1 point == 1 cell.
#         g["NumberOfCells"] = [Np]
#         g["NumberOfConnectivityIds"] = [Np]
#         g["Connectivity"] = collect(0:(Np - 1))
#         g["Offsets"] = collect(0:Np)
#         close(g)
#     end

#     # Add unused PolyData types. ParaView expects this, even if they"re empty.
#     for type ∈ ("Lines", "Polygons", "Strips")
#         gempty = HDF5.create_group(gtop, type)
#         gempty["NumberOfCells"] = [0]
#         gempty["NumberOfConnectivityIds"] = [0]
#         gempty["Connectivity"] = Int[]
#         gempty["Offsets"] = [0]
#         close(gempty)
#     end
# end

    filename = "warping_spheres.vtkhdf"
    fid = h5open(filename, "w")
    root = HDF5.create_group(fid, "VTKHDF")
    HDF5.attributes(root)["Version"] = [2,0]
    # Write type of dataset ("PolyData") as an ASCII string to a "Type" attribute.
    # This is a bit tricky because VTK/ParaView don"t support UTF-8 strings here, which is
    # the default in HDF5.jl.
    let s = "PolyData"
        dtype = HDF5.datatype(s)
        HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
        dspace = HDF5.dataspace(s)
        attr = HDF5.create_attribute(root, "Type", dtype, dspace)
        HDF5.write_attribute(attr, dtype, s)
    end
    HDF5.create_dataset(root, "NumberOfPoints", Int, ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
    HDF5.create_dataset(root, "Points", Float64, ((0,3),(-0,3)), chunk=(100,3)) #-1 is equivalent to typemax(hsize_t)
    # HDF5.set_extent_dims(b, (10000,))
    # b[1:10000] = collect(1:10000)
    connectivitySets = ["Vertices", "Lines", "Polygons", "Strips"]
    for connectivitySet in connectivitySets
        connectivityGroup = HDF5.create_group(root, connectivitySet)
        HDF5.create_dataset(connectivityGroup, "NumberOfConnectivityIds" , Int, ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
        HDF5.create_dataset(connectivityGroup, "NumberOfCells"           , Int, ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
        HDF5.create_dataset(connectivityGroup, "Offsets"                 , Int, ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
        HDF5.create_dataset(connectivityGroup, "Connectivity"            , Int, ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
    end
    pointData = HDF5.create_group(root, "PointData")
    HDF5.create_dataset(pointData, "SpatioTemporalHarmonics", Float64, ((0,), (-1,)), chunk=(100,))
    steps = HDF5.create_group(root, "Steps")

    HDF5.attributes(steps)["NSteps"] = 0
    HDF5.create_dataset(steps, "Values"        , Float64 , ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
    HDF5.create_dataset(steps, "PartOffsets"   , Int     , ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
    HDF5.create_dataset(steps, "NumberOfParts" , Int     , ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
    HDF5.create_dataset(steps, "PointOffsets"  , Int     , ((0,),(-1,)), chunk=(100,)) #-1 is equivalent to typemax(hsize_t)
 
    HDF5.create_dataset(steps, "CellOffsets"            , Int     , ((0,4),(-1,4)), chunk=(100,4)) #-1 is equivalent to typemax(hsize_t)
    HDF5.create_dataset(steps, "ConnectivityIdOffsets"  , Int     , ((0,4),(-1,4)), chunk=(100,4)) #-1 is equivalent to typemax(hsize_t)
    pointDataOffsets = HDF5.create_group(steps, "PointDataOffsets")
    HDF5.create_dataset(pointDataOffsets, "SpatioTemporalHarmonics", Int, ((0,),(-1,)), chunk=(100,))
    

    ### Adding data
    # Add unused PolyData types. ParaView expects this, even if they're empty.
    for type ∈ ("Lines", "Polygons", "Strips")
        gempty = root[type]
        HDF5.delete_attribute(gempty, "NumberOfCells")
        gempty["NumberOfCells"] = [0]
        gempty["NumberOfConnectivityIds"] = [0]
        gempty["Connectivity"] = Int[]
        gempty["Offsets"] = [0]
        close(gempty)
    end

    display(fid)

    # Close file properly
    close(fid)
