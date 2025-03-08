using HDF5
using StaticArrays

fType  = Float64
idType = Int8

connectivities = ["Vertices", "Lines", "Polygons", "Strips"]


Positions = SVector{3, Float64}[
    [0.0, 0.0, 0.5],
    [0.0, 0.0, -0.5],
    [0.5, 0.0, 3.061617e-17],
    [-0.25, 0.4330127, 3.061617e-17],
    [-0.25, -0.4330127, 3.061617e-17]
]


function generate_geometry_structure(root)
    # Write version of VTKHDF format as an attribute
    HDF5.attrs(root)["Version"] = Int32.([2, 0])
    
    # Write type of dataset ("PolyData") as an ASCII string to a "Type" attribute.
    # This is a bit tricky because VTK/ParaView don't support UTF-8 strings here, which is
    # the default in HDF5.jl.
    let s = "PolyData"
        dtype = HDF5.datatype(s)
        HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
        dspace = HDF5.dataspace(s)
        attr = HDF5.create_attribute(root, "Type", dtype, dspace)
        HDF5.write_attribute(attr, dtype, s)
    end
    
    NumberOfPoints = HDF5.create_dataset(root, "NumberOfPoints" , idType , ((0,),(-1,)), chunk=(100,))
    Points         = HDF5.create_dataset(root, "Points"         , fType  , ((3,0), (3,-1)), chunk=(3,100))

    for connect in connectivities
        group = HDF5.create_group(root, connect)

        NumberOfConnectivityIds = HDF5.create_dataset(group, "NumberOfConnectivityIds", idType, ((0,),(-1,)), chunk=(100,) )
        NumberOfCells           = HDF5.create_dataset(group, "NumberOfCells", idType, ((0,),(-1,)), chunk=(100,) )
        Offsets                 = HDF5.create_dataset(group, "Offsets", idType, ((0,),(-1,)), chunk=(100,) )
        Connectivity            = HDF5.create_dataset(group, "Connectivity", idType, ((0,),(-1,)), chunk=(100,) )
    end

    pData = HDF5.create_group(root, "PointData")
    Warping = HDF5.create_dataset(pData, "Warping", fType, ((0,3),(-1,3)), chunk=(100,3))
    Normals = HDF5.create_dataset(pData, "Normals", fType, ((0,3),(-1,3)), chunk=(100,3))

    cData     = HDF5.create_group(root, "CellData")
    Materials = HDF5.create_dataset(cData, "Materials", idType, ((0,),(-1,)), chunk=(100,))

    return nothing
end

function generate_step_structure(root)
    steps = HDF5.create_group(root, "Steps")
    # HDF5.attrs(steps)["NSteps"] = 0

    NSteps, _ = HDF5.create_attribute(steps, "NSteps", Int32)

    Values = HDF5.create_dataset(steps, "Values", fType , ((0,),(-1,)), chunk=(100,))

    singleDSs = ["PartOffsets", "NumberOfParts", "PointOffsets"]
    for name in singleDSs
        HDF5.create_dataset(steps, name, idType, ((0,),(-1,)), chunk=(100,))
    end

    nTopoDSs = ["CellOffsets", "ConnectivityIdOffsets"]
    for name in nTopoDSs
        HDF5.create_dataset(steps, name, idType, ((0,4),(-1,4)), chunk=(100,4))
    end
    
    pData = HDF5.create_group(steps, "PointDataOffsets")
    Warping = HDF5.create_dataset(pData, "Warping", idType, ((0,),(-1,)), chunk=(100,))
    Normals = HDF5.create_dataset(pData, "Normals", idType, ((0,),(-1,)), chunk=(100,))

    cData     = HDF5.create_group(steps, "CellDataOffsets")
    Materials = HDF5.create_dataset(cData, "Materials", idType, ((0,),(-1,)), chunk=(100,))
end



function append_data(root, newStep, Positions)

        steps = root["Steps"]
        attrs(steps)["NSteps"] = Int32(attrs(steps)["NSteps"] + 1)

        HDF5.set_extent_dims(steps["Values"], (length(steps["Values"]) + 1,))
        steps["Values"][end] = newStep

        PointsStartIndex = size(root["Points"])[2] + 1
        PositionLength   = length(Positions)

        HDF5.set_extent_dims(root["Points"], (length(first(Positions)), size(root["Points"])[2] + PositionLength))
        root["Points"][:, PointsStartIndex:(PointsStartIndex+PositionLength-1)] = stack(Positions)

        HDF5.set_extent_dims(steps["PointOffsets"], (length(steps["PointOffsets"]) + 1,))
        steps["PointOffsets"][end] = PointsStartIndex - 1

        NumberOfPartsStartIndex = length(steps["NumberOfParts"]) + 1
        HDF5.set_extent_dims(steps["NumberOfParts"], (length(steps["NumberOfParts"]) + 1,))
        steps["NumberOfParts"][NumberOfPartsStartIndex] = 1

        PartOffsetsStartIndex = length(steps["PartOffsets"]) + 1
        PartOffsetsLength     = length(steps["PartOffsets"]) + 1
        HDF5.set_extent_dims(steps["PartOffsets"], (PartOffsetsLength,))
        steps["PartOffsets"][PartOffsetsStartIndex] = PartOffsetsLength - 1

        # steps_attr_dict = attributes(steps)
        # NSteps = steps_attr_dict["NSteps"]
        # write_attribute(NSteps,HDF5.datatype(idType), old_NSteps + 1)

        # geomOffs = []
        # if isnothing(geometryOffset)
        #     append!(geomOffs, size(root["NumberOfPoints"])[1])
        #     append!(geomOffs, 1)
        #     append!(geomOffs, size(root["Points"][1]))

        #     for connect in connectivities
        #         append!(geomOffs, size(root[connect]["Offsets"])[1] - geomOffs[1])
        #     end

        #     for connect in connectivities
        #         append!(geomOffs, size(root[connect]["Connectivity"][1]))
        #     end
        # else
        #     append!(geomOffs, steps["PartOffsets"][geometryOffset])
        #     append!(geomOffs, 1)
        #     append!(geomOffs, steps["PointOffsets"][geometryOffset])

        #     for (iC,_) in enumerate(connectivities)
        #         append!(geomOffs, steps["CellOffsets"][geometryOffset, iC])
        #     end

        #     for connect in connectivities
        #         append!(geomOffs, steps["ConnectivityIdOffsets"][geometryOffset, iC])
        #     end
        # end

end

function generate_data(root, Positions)
    generate_geometry_structure(root)
    generate_step_structure(root)

    ts = range(0,0.5,2)

    for (iT, t) in enumerate(ts)
        append_data(root, t, Positions)
    end
end


f = h5open("test.vtkhdf", "w")
root = HDF5.create_group(f, "VTKHDF")
generate_data(root, Positions)
# display(f)

close(f)