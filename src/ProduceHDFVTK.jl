module ProduceHDFVTK     

    export SaveVTKHDF, GenerateGeometryStructure, GenerateStepStructure, AppendVTKHDFData

    using HDF5

    const idType = Int64
    const fType = Float64

    function SaveVTKHDF(fid_vector, index, filepath,points, variable_names, args...)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
            io = h5open(filepath, "w")
            # Create toplevel group /VTKHDF
            gtop = HDF5.create_group(io, "VTKHDF")

            # Write version of VTKHDF format as an attribute
            HDF5.attrs(gtop)["Version"] = [2, 1]

            # Write type of dataset ("PolyData") as an ASCII string to a "Type" attribute.
            # This is a bit tricky because VTK/ParaView don't support UTF-8 strings here, which is
            # the default in HDF5.jl.
            let s = "PolyData"
                dtype = HDF5.datatype(s)
                HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
                dspace = HDF5.dataspace(s)
                attr = HDF5.create_attribute(gtop, "Type", dtype, dspace)
                HDF5.write_attribute(attr, dtype, s)
            end

            # Write points + number of points.
            # Note that we need to reinterpret the vector of SVector onto a 3×Np matrix.
            Np = length(points)
            gtop["NumberOfPoints"] = [Np]
            gtop["Points"] = reinterpret(reshape, eltype(eltype(points)), points)

            # Write velocities as point data.
            let g = HDF5.create_group(gtop, "PointData")
                for i ∈ eachindex(variable_names)
                    var_name = variable_names[i]
                    arg      = args[i]
                    g[var_name] = reinterpret(reshape, eltype(eltype(arg)), arg)
                end
            end

            # Create and fill Vertices group.
            let g = HDF5.create_group(gtop, "Vertices")
                # In our case 1 point == 1 cell.
                g["NumberOfCells"] = [Np]
                g["NumberOfConnectivityIds"] = [Np]
                g["Connectivity"] = collect(0:(Np - 1))
                g["Offsets"] = collect(0:Np)
                close(g)
            end

            # Add unused PolyData types. ParaView expects this, even if they're empty.
            for type ∈ ("Lines", "Polygons", "Strips")
                gempty = HDF5.create_group(gtop, type)
                gempty["NumberOfCells"] = [0]
                gempty["NumberOfConnectivityIds"] = [0]
                gempty["Connectivity"] = Int[]
                gempty["Offsets"] = [0]
                close(gempty)
            end

            fid_vector[index] = io
    end


    function GenerateGeometryStructure(root, variable_names, args...; chunk_size = 100, idType = Int64, fType = Float64)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
        # Write version of VTKHDF format as an attribute
        HDF5.attrs(root)["Version"] = Int32.([2, 3])
        
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
        
        NumberOfPoints = HDF5.create_dataset(root, "NumberOfPoints" , idType , ((0,),(-1,)), chunk=(chunk_size,))
        Points         = HDF5.create_dataset(root, "Points"         , fType  , ((3,0), (3,-1)), chunk=(3,chunk_size))

        connectivities = ["Vertices", "Lines", "Polygons", "Strips"]
        for connect in connectivities
            group = HDF5.create_group(root, connect)

            NumberOfConnectivityIds = HDF5.create_dataset(group, "NumberOfConnectivityIds", idType, ((0,),(-1,)), chunk=(chunk_size,) )
            NumberOfCells           = HDF5.create_dataset(group, "NumberOfCells", idType, ((0,),(-1,)), chunk=(chunk_size,) )
            Offsets                 = HDF5.create_dataset(group, "Offsets", idType, ((0,),(-1,)), chunk=(chunk_size,) )
            Connectivity            = HDF5.create_dataset(group, "Connectivity", idType, ((0,),(-1,)), chunk=(chunk_size,) )
        end

        pData = HDF5.create_group(root, "PointData")

        for i ∈ eachindex(variable_names)
            var_name = variable_names[i]
            arg      = args[i]
            arg_val_type   = eltype(eltype(arg))
            arg_val_length = length(first(arg)) > 1 ? 3 : 1

            if arg_val_length == 3
                HDF5.create_dataset(pData, var_name, arg_val_type, ((3,0),(3,-1)), chunk=(3,chunk_size))
            else
                HDF5.create_dataset(pData, var_name, arg_val_type, ((0,),(-1,)), chunk=(chunk_size,))
            end
        end

        return nothing
    end

    function GenerateStepStructure(root,  variable_names, args...; chunk_size = 1000)
        steps = HDF5.create_group(root, "Steps")
    
        NSteps, _ = HDF5.create_attribute(steps, "NSteps", Int32)
    
        Values = HDF5.create_dataset(steps, "Values", fType , ((0,),(-1,)), chunk=(chunk_size,))
    
        singleDSs = ["PartOffsets", "NumberOfParts", "PointOffsets"]
        for name in singleDSs
            HDF5.create_dataset(steps, name, idType, ((0,),(-1,)), chunk=(chunk_size,))
        end
    
        nTopoDSs = ["CellOffsets", "ConnectivityIdOffsets"]
        for name in nTopoDSs
            HDF5.create_dataset(steps, name, idType, ((4,0),(4, -1)), chunk=(4, chunk_size))
        end
        
        pData = HDF5.create_group(steps, "PointDataOffsets")


        for i ∈ eachindex(variable_names)
            var_name = variable_names[i]

            HDF5.create_dataset(pData, var_name, idType, ((0,),(-1,)), chunk=(chunk_size,))
        end
    
    end

    function AppendVTKHDFData(root, newStep, Positions, variable_names, args...)
        steps = root["Steps"]

        # To update attributes, this is the best way I've found so far
        old_NSteps = HDF5.read_attribute(steps, "NSteps")
        steps_attr_dict = attributes(steps)
        NSteps = steps_attr_dict["NSteps"]
        write_attribute(NSteps,HDF5.datatype(idType), old_NSteps + 1)

        HDF5.set_extent_dims(steps["Values"], (length(steps["Values"]) + 1,))
        steps["Values"][end] = newStep

        PointsStartIndex = size(root["Points"])[2] + 1
        PositionLength   = length(Positions)

        HDF5.set_extent_dims(root["Points"], (length(first(Positions)), size(root["Points"])[2] + PositionLength))
        root["Points"][:, PointsStartIndex:(PointsStartIndex+PositionLength-1)] = stack(Positions)

        HDF5.set_extent_dims(steps["PointOffsets"], (length(steps["PointOffsets"]) + 1,))
        steps["PointOffsets"][end] = PointsStartIndex - 1

        HDF5.set_extent_dims(root["NumberOfPoints"], (length(root["NumberOfPoints"]) + 1,))
        root["NumberOfPoints"][end] = PositionLength

        NumberOfPartsStartIndex = length(steps["NumberOfParts"]) + 1
        HDF5.set_extent_dims(steps["NumberOfParts"], (length(steps["NumberOfParts"]) + 1,))
        steps["NumberOfParts"][NumberOfPartsStartIndex] = 1

        PartOffsetsStartIndex = length(steps["PartOffsets"]) + 1
        PartOffsetsLength     = length(steps["PartOffsets"]) + 1
        HDF5.set_extent_dims(steps["PartOffsets"], (PartOffsetsLength,))
        steps["PartOffsets"][PartOffsetsStartIndex] = PartOffsetsLength - 1

        CellOffsetsStartIndex = size(steps["CellOffsets"])[2] + 1
        HDF5.set_extent_dims(steps["CellOffsets"], (4, CellOffsetsStartIndex))
        steps["CellOffsets"][:, CellOffsetsStartIndex] = zeros(4)

        ConnectivityIdOffsetsStartIndex = size(steps["ConnectivityIdOffsets"])[2] + 1
        HDF5.set_extent_dims(steps["ConnectivityIdOffsets"], (4, ConnectivityIdOffsetsStartIndex))
        steps["ConnectivityIdOffsets"][:, ConnectivityIdOffsetsStartIndex] = zeros(4)

        NumberOfPartsStartIndex = length(steps["NumberOfParts"]) + 1
        HDF5.set_extent_dims(steps["NumberOfParts"], (length(steps["NumberOfParts"]) + 1,))
        steps["NumberOfParts"][NumberOfPartsStartIndex] = 1

        for point_data_name in keys(steps["PointDataOffsets"])
            HDF5.set_extent_dims(steps["PointDataOffsets"][point_data_name], (length(steps["PointDataOffsets"][point_data_name]) + 1,))
            steps["PointDataOffsets"][point_data_name][end] = Int(PointsStartIndex - 1)
        end

        for i ∈ eachindex(variable_names)
            var_name = variable_names[i]
            arg      = args[i]
            arg_val_type   = eltype(eltype(arg))
            arg_val_length = length(first(arg)) > 1 ? 3 : 1


            if arg_val_length == 3
                HDF5.set_extent_dims(root["PointData"][var_name], (arg_val_length, size(root["PointData"][var_name], 2) + PositionLength))
                root["PointData"][var_name][:, PointsStartIndex:(PointsStartIndex + PositionLength - 1)] = stack(arg)
            else
                HDF5.set_extent_dims(root["PointData"][var_name], (length(root["PointData"][var_name]) + PositionLength,))
                root["PointData"][var_name][PointsStartIndex:(PointsStartIndex + PositionLength - 1)] = arg
            end
        end
        

        connectivities = ["Vertices", "Lines", "Polygons", "Strips"]
        for connect in connectivities
            for dataset in ["NumberOfCells", "NumberOfConnectivityIds", "Offsets", "Connectivity"]
                HDF5.set_extent_dims(root[connect][dataset], (length(root[connect][dataset]) + 1,))
                root[connect][dataset][end] = idType(0)
            end
        end
    end
end