
module ProduceHDFVTK     

    export SaveVTKHDF, GenerateGeometryStructure, GenerateStepStructure, AppendVTKHDFData, SaveCellGridVTKHDF, AppendVTKHDFGridData, SetupVTKOutput

    using HDF5
    using StaticArrays

    using ..AuxillaryFunctions: to_3d 


    const idType = Int64
    const fType = Float64

    function SaveVTKHDF(fid_vector, index, filepath,points, variable_names = String[], args...)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
            io = h5open(filepath, "w")
            # Create toplevel group /VTKHDF
            gtop = HDF5.create_group(io, "VTKHDF")

            # Write version of VTKHDF format as an attribute
            HDF5.attrs(gtop)["Version"] = [2, 3]

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


    function GenerateGeometryStructure(root, variable_names = String[], args...; chunk_size = 100, vtk_file_type = "PolyData", idType = Int64, fType = Float64)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
        # Write version of VTKHDF format as an attribute
        HDF5.attrs(root)["Version"] = Int32.([2, 3])
        
        # Write type of dataset ("PolyData") as an ASCII string to a "Type" attribute.
        # This is a bit tricky because VTK/ParaView don't support UTF-8 strings here, which is
        # the default in HDF5.jl.
        let s = vtk_file_type
            dtype = HDF5.datatype(s)
            HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
            dspace = HDF5.dataspace(s)
            attr = HDF5.create_attribute(root, "Type", dtype, dspace)
            HDF5.write_attribute(attr, dtype, s)
        end
        
        NumberOfPoints = HDF5.create_dataset(root, "NumberOfPoints" , idType , ((0,),(-1,)), chunk=(chunk_size,))
        Points         = HDF5.create_dataset(root, "Points"         , fType  , ((3,0), (3,-1)), chunk=(3,chunk_size))

        if vtk_file_type == "PolyData"
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
        elseif vtk_file_type == "UnstructuredGrid"
            HDF5.create_dataset(root, "Connectivity" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "NumberOfCells" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "NumberOfConnectivityIds" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "Offsets" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "Types" , UInt8 , ((0,),(-1,)), chunk=(chunk_size,)) #Must be UInt8
            
            FieldData = HDF5.create_group(root, "FieldData") #Currently just empty group

            CellData = HDF5.create_group(root, "CellData")
            HDF5.create_dataset(CellData, "CellData" , idType , ((0,),(-1,)), chunk=(chunk_size,))

            HDF5.create_dataset(CellData, "ChunkID" , idType , ((0,),(-1,)), chunk=(chunk_size,))
        end

        return nothing
    end

    function GenerateStepStructure(root,  variable_names = String[], args...; vtk_file_type = "PolyData", chunk_size = 1000)
        steps = HDF5.create_group(root, "Steps")
    
        NSteps, _ = HDF5.create_attribute(steps, "NSteps", Int32)
    
        Values = HDF5.create_dataset(steps, "Values", fType , ((0,),(-1,)), chunk=(chunk_size,))
    
        singleDSs = ["PartOffsets", "NumberOfParts", "PointOffsets"]
        for name in singleDSs
            HDF5.create_dataset(steps, name, idType, ((0,),(-1,)), chunk=(chunk_size,))
        end
    
        if vtk_file_type == "PolyData"
            nTopoDSs = ["CellOffsets", "ConnectivityIdOffsets"]
            for name in nTopoDSs
                HDF5.create_dataset(steps, name, idType, ((4,0),(4, -1)), chunk=(4, chunk_size))
            end
        elseif vtk_file_type == "UnstructuredGrid"
            nTopoDSs = ["CellOffsets", "ConnectivityIdOffsets"]
            for name in nTopoDSs
                HDF5.create_dataset(steps, name, idType, ((0,),(-1,)), chunk=(chunk_size,))
            end
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
        # steps["CellOffsets"][:, CellOffsetsStartIndex] = zeros(4) # When you extent dimensions, it autofills with zero values

        ConnectivityIdOffsetsStartIndex = size(steps["ConnectivityIdOffsets"])[2] + 1
        HDF5.set_extent_dims(steps["ConnectivityIdOffsets"], (4, ConnectivityIdOffsetsStartIndex))
        # steps["ConnectivityIdOffsets"][:, ConnectivityIdOffsetsStartIndex] = zeros(4) # When you extent dimensions, it autofills with zero values

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
                # root[connect][dataset][end] = idType(0) # When you extent dimensions, it autofills with zero values
            end
        end
    end

    function AppendVTKHDFGridData(root, newStep, SimKernel, UniqueCells, SimParticles)
        # Must be AbstractVector, since UniqueCells is passed in as a 'view' of a CartesianIndex array
        ExtractDimensionality(::AbstractVector{CartesianIndex{N}}) where N = N

        CartesianIndexN = ExtractDimensionality(UniqueCells)

        # Cell dimensions
        dx = dy = dz = SimKernel.H

        # Compute per-dimension minima and maxima
        if CartesianIndexN == 2
            minx, maxx = minimum(ci->ci[1], UniqueCells), maximum(ci->ci[1], UniqueCells)
            miny, maxy = minimum(ci->ci[2], UniqueCells), maximum(ci->ci[2], UniqueCells)
            # number of cells in x
            nx = maxx - minx + 1
        elseif CartesianIndexN == 3
            minx, maxx = minimum(ci->ci[1], UniqueCells), maximum(ci->ci[1], UniqueCells)
            miny, maxy = minimum(ci->ci[2], UniqueCells), maximum(ci->ci[2], UniqueCells)
            minz, maxz = minimum(ci->ci[3], UniqueCells), maximum(ci->ci[3], UniqueCells)
            # number of cells per axis
            nx = maxx - minx + 1
            ny = maxy - miny + 1
        else
            error("Dimensionality of UniqueCells must be 2 or 3, got $CartesianIndexN")
        end
    
        # Initialize lists for storing points and cells
        points = Vector{SVector{3, Float64}}()  # List to store unique SVector points
        connectivity = Int[]                    # Connectivity for each cell
        offsets      = Int[]                    # Offsets for each cell
        cell_types   = Int[]                    # Cell types (for VTK_QUAD)
        cell_data    = Int[]
    
        vtk_type::UInt8 = CartesianIndexN == 2 ? UInt8(9) : CartesianIndexN == 3 ? UInt8(12)  : error("Dimensionality of UniqueCells is wrong")   # QUAD  OR HEXADRON VTK TYPE


        push!(offsets, 0)
        # Loop through each CartesianIndex cell
        for cell in UniqueCells
            # Compute linear ID 
            id = if CartesianIndexN == 2
                # (ci2 - miny)*nx + (ci1 - minx) + 1
                (cell[2] - miny) * nx + (cell[1] - minx) + 1
            else
                # (ci3 - minz)*nx*ny + (ci2 - miny)*nx + (ci1 - minx) + 1
                (cell[3] - minz) * (nx * ny) + (cell[2] - miny) * nx + (cell[1] - minx) + 1
            end
            
            if CartesianIndexN == 2
                # Get x and y from the CartesianIndex and calculate cell center
                xi, yi = cell.I
                x_center = xi * dx
                y_center = yi * dy
        
                # Define corners individually
                corners = [
                    SVector(x_center - dx / 2, y_center - dy / 2, 0.0),
                    SVector(x_center + dx / 2, y_center - dy / 2, 0.0),
                    SVector(x_center + dx / 2, y_center + dy / 2, 0.0),
                    SVector(x_center - dx / 2, y_center + dy / 2, 0.0)
                ]
        
            elseif CartesianIndexN == 3
                # Get x, y, and z from the CartesianIndex and calculate cell center
                xi, yi, zi = cell.I 
                x_center = xi * dx
                y_center = yi * dy
                z_center = zi * dz
                
                # Calculate the 8 corners of the cell relative to the center
                corners = [
                    SVector(x_center - dx / 2, y_center - dy / 2, z_center - dz / 2),  # Bottom-front-left
                    SVector(x_center + dx / 2, y_center - dy / 2, z_center - dz / 2),  # Bottom-front-right
                    SVector(x_center + dx / 2, y_center + dy / 2, z_center - dz / 2),  # Bottom-back-right
                    SVector(x_center - dx / 2, y_center + dy / 2, z_center - dz / 2),  # Bottom-back-left
                    SVector(x_center - dx / 2, y_center - dy / 2, z_center + dz / 2),  # Top-front-left
                    SVector(x_center + dx / 2, y_center - dy / 2, z_center + dz / 2),  # Top-front-right
                    SVector(x_center + dx / 2, y_center + dy / 2, z_center + dz / 2),  # Top-back-right
                    SVector(x_center - dx / 2, y_center + dy / 2, z_center + dz / 2)   # Top-back-left
                ]
            end

            # Add each corner point and update connectivity
            n = length(points) #It is on purpose that length of points is zero, to match HDF5 0-based indexing!
            for corner in corners
                push!(points, corner)
                push!(connectivity, n)
                n += 1
            end

            # Define cell type and offsets 
            push!(offsets, length(connectivity))

            push!(cell_types, vtk_type) 

            push!(cell_data, id)
        end

        

        Positions = points

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

        ExtendedPositionLength = size(root["Points"])[2] + PositionLength
        HDF5.set_extent_dims(root["Points"], (length(first(Positions)), ExtendedPositionLength))
        root["Points"][:, PointsStartIndex:(PointsStartIndex+PositionLength-1)] = stack(Positions)

        HDF5.set_extent_dims(steps["PointOffsets"], (length(steps["PointOffsets"]) + 1,))
        steps["PointOffsets"][end] = PointsStartIndex - 1

        HDF5.set_extent_dims(steps["NumberOfParts"], (length(steps["NumberOfParts"]) + 1,))
        steps["NumberOfParts"][end] = 1

        PartOffsetsStartIndex = length(steps["PartOffsets"]) + 1
        PartOffsetsLength     = length(steps["PartOffsets"]) + 1
        HDF5.set_extent_dims(steps["PartOffsets"], (PartOffsetsLength,))
        steps["PartOffsets"][PartOffsetsStartIndex] = PartOffsetsLength - 1



        HDF5.set_extent_dims(steps["ConnectivityIdOffsets"], (length(steps["ConnectivityIdOffsets"]) + 1,))
        steps["ConnectivityIdOffsets"][end] = PointsStartIndex - 1

        ##             
        HDF5.set_extent_dims(root["NumberOfPoints"], (length(root["NumberOfPoints"]) + 1,))
        root["NumberOfPoints"][end] = PositionLength

        HDF5.set_extent_dims(root["NumberOfCells"], (length(root["NumberOfCells"]) + 1,))
        root["NumberOfCells"][end] = length(UniqueCells)

        ## Update data
        HDF5.set_extent_dims(root["Connectivity"], (ExtendedPositionLength,))
        root["Connectivity"][PointsStartIndex:end] = (PointsStartIndex:ExtendedPositionLength) .- PointsStartIndex

        HDF5.set_extent_dims(root["NumberOfConnectivityIds"], (length(root["NumberOfConnectivityIds"]) + 1,))
        root["NumberOfConnectivityIds"][end] = PositionLength



        HDF5.set_extent_dims(steps["CellOffsets"], (length(steps["CellOffsets"]) + 1,)) #For first value, it becomes 0

        if length(steps["CellOffsets"]) == 1
            LastCellOffset = 0
        else
            LastCellOffset = sum(root["NumberOfCells"][:]) - length(UniqueCells)
        end

        steps["CellOffsets"][end] = LastCellOffset

        OffsetStartIndex = length(root["Offsets"])  + 1
        OffsetsLength    = length(root["Offsets"]) + length(UniqueCells) + 1
        HDF5.set_extent_dims(root["Offsets"], (OffsetsLength,))
        root["Offsets"][OffsetStartIndex:end] = offsets

        TypesStartIndex = length(root["Types"]) + 1
        HDF5.set_extent_dims(root["Types"], (length(root["Types"]) + length(UniqueCells),))
        root["Types"][TypesStartIndex:end] = vtk_type

        CellDataStartIndex = length(root["CellData"]["CellData"]) + 1
        HDF5.set_extent_dims(root["CellData"]["CellData"], (length(root["CellData"]["CellData"]) + length(UniqueCells),))
        root["CellData"]["CellData"][CellDataStartIndex:end] = cell_data

        
        CellChunkIDIndex = length(root["CellData"]["ChunkID"]) + 1
        HDF5.set_extent_dims(root["CellData"]["ChunkID"], (length(root["CellData"]["ChunkID"]) + length(UniqueCells),))
        root["CellData"]["ChunkID"][CellChunkIDIndex:end] = SimParticles.ChunkID[1:length(cell_data)]

        return nothing
    end

    function SaveCellGridVTKHDF(FilePath, SimKernel, UniqueCells)
        # Must be AbstractVector, since UniqueCells is passed in as a 'view' of a CartesianIndex array
        ExtractDimensionality(::AbstractVector{CartesianIndex{N}}) where N = N

        CartesianIndexN = ExtractDimensionality(UniqueCells)

        # Cell dimensions
        dx = dy = dz = SimKernel.H

        # Compute per-dimension minima and maxima
        if CartesianIndexN == 2
            minx, maxx = minimum(ci->ci[1], UniqueCells), maximum(ci->ci[1], UniqueCells)
            miny, maxy = minimum(ci->ci[2], UniqueCells), maximum(ci->ci[2], UniqueCells)
            # number of cells in x
            nx = maxx - minx + 1
        elseif CartesianIndexN == 3
            minx, maxx = minimum(ci->ci[1], UniqueCells), maximum(ci->ci[1], UniqueCells)
            miny, maxy = minimum(ci->ci[2], UniqueCells), maximum(ci->ci[2], UniqueCells)
            minz, maxz = minimum(ci->ci[3], UniqueCells), maximum(ci->ci[3], UniqueCells)
            # number of cells per axis
            nx = maxx - minx + 1
            ny = maxy - miny + 1
        else
            error("Dimensionality of UniqueCells must be 2 or 3, got $CartesianIndexN")
        end    
    
        # Initialize lists for storing points and cells
        points = Vector{SVector{3, Float64}}()  # List to store unique SVector points
        connectivity = Int[]                    # Connectivity for each cell
        offsets      = Int[]                    # Offsets for each cell
        cell_types   = Int[]                    # Cell types (for VTK_QUAD)
        cell_data    = Int[]
    
        vtk_type = CartesianIndexN == 2 ? UInt8(9) : CartesianIndexN == 3 ? UInt8(12)  : error("Dimensionality of UniqueCells is wrong")   # QUAD VTK TYPE


        push!(offsets, 0)
        # Loop through each CartesianIndex cell
        for cell in UniqueCells
        # Compute linear ID 
                id = if CartesianIndexN == 2
                    # (ci2 - miny)*nx + (ci1 - minx) + 1
                    (cell[2] - miny) * nx + (cell[1] - minx) + 1
                else
                    # (ci3 - minz)*nx*ny + (ci2 - miny)*nx + (ci1 - minx) + 1
                    (cell[3] - minz) * (nx * ny) + (cell[2] - miny) * nx + (cell[1] - minx) + 1
                end
                if CartesianIndexN == 2
                # Get x and y from the CartesianIndex and calculate cell center
                xi, yi = cell.I
                x_center = xi * dx
                y_center = yi * dy
        
                # Define corners individually
                corners = [
                    SVector(x_center - dx / 2, y_center - dy / 2, 0.0),
                    SVector(x_center + dx / 2, y_center - dy / 2, 0.0),
                    SVector(x_center + dx / 2, y_center + dy / 2, 0.0),
                    SVector(x_center - dx / 2, y_center + dy / 2, 0.0)
                ]
        
            elseif CartesianIndexN == 3
                # Get x, y, and z from the CartesianIndex and calculate cell center
                xi, yi, zi = cell.I 
                x_center = xi * dx
                y_center = yi * dy
                z_center = zi * dz
                
                # Calculate the 8 corners of the cell relative to the center
                corners = [
                    SVector(x_center - dx / 2, y_center - dy / 2, z_center - dz / 2),  # Bottom-front-left
                    SVector(x_center + dx / 2, y_center - dy / 2, z_center - dz / 2),  # Bottom-front-right
                    SVector(x_center + dx / 2, y_center + dy / 2, z_center - dz / 2),  # Bottom-back-right
                    SVector(x_center - dx / 2, y_center + dy / 2, z_center - dz / 2),  # Bottom-back-left
                    SVector(x_center - dx / 2, y_center - dy / 2, z_center + dz / 2),  # Top-front-left
                    SVector(x_center + dx / 2, y_center - dy / 2, z_center + dz / 2),  # Top-front-right
                    SVector(x_center + dx / 2, y_center + dy / 2, z_center + dz / 2),  # Top-back-right
                    SVector(x_center - dx / 2, y_center + dy / 2, z_center + dz / 2)   # Top-back-left
                ]
            end

            # Add each corner point and update connectivity
            n = length(points) #It is on purpose that length of points is zero, to match HDF5 0-based indexing!
            for corner in corners
                push!(points, corner)
                push!(connectivity, n)
                n += 1
            end

            # Define cell type and offsets
            push!(offsets, length(connectivity))

            push!(cell_types, vtk_type) 

            push!(cell_data, id)
        end
    
        # Open HDF5 file for writing
        io = h5open(FilePath, "w")

        # Create top-level group "VTKHDF"
        gtop = HDF5.create_group(io, "VTKHDF")

        # Set the Version attribute
        HDF5.attrs(gtop)["Version"] = [2, 3]

        # Write Type attribute as ASCII string
        let s = "UnstructuredGrid"
            dtype = HDF5.datatype(s)
            HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
            dspace = HDF5.dataspace(s)
            attr = HDF5.create_attribute(gtop, "Type", dtype, dspace)
            HDF5.write_attribute(attr, dtype, s)
        end

        # Write Number of Points, Number of Cells, and Number of Connectivity IDs
        gtop["NumberOfPoints"]          = [length(points)]
        gtop["NumberOfCells"]           = [length(cell_types)]
        gtop["NumberOfConnectivityIds"] = [length(connectivity)]

        # Write Points
        gtop["Points"] = reinterpret(reshape, eltype(eltype(points)), points)

        # Write Connectivity, Offsets, and Types
        gtop["Connectivity"] = connectivity
        gtop["Offsets"] = offsets
        gtop["Types"] = [vtk_type for _ in cell_types]  # QUAD VTK TYPE

        # Write CellData (cell-level variables)
        let cell_group = HDF5.create_group(gtop, "CellData")
            cell_group["CellData"] = cell_data
            close(cell_group)
        end

        # Write an empty FieldData group (placeholder for additional data)
        create_group(gtop, "FieldData")
        
        # Close file
        close(io)
    end

    function SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)
        # Generate save locations
        particle_savepath = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName)
        grid_savepath = joinpath(SimMetaData.SaveLocation, "CellGrid_$(SimMetaData.SimulationName)")
    
        # File naming functions
        particle_filename = (iter) -> "$(particle_savepath)_$(lpad(iter,6,"0")).vtkhdf"
        grid_filename = (iter) -> "$(grid_savepath)_$(lpad(iter,6,"0")).vtkhdf"
        
        # Output variable names
        output_vars = ["ChunkID", "Kernel", "KernelGradient", "Density", "Pressure", "Velocity", 
                      "Acceleration", "BoundaryBool", "ID", "Type", "GroupMarker", "GhostPoints", "GhostNormals"]
    
        # Initialize storage for file handles
        file_handles = if !SimMetaData.ExportSingleVTKHDF
            # Multi-file mode: vector for particle files
            (particle_files = Vector{HDF5.File}(undef, Int(SimMetaData.SimulationTime/SimMetaData.OutputEach + 1)),
             grid_files = nothing)
        else
            # Single-file mode: handles for both files
            OutputVTKHDF = h5open("$(particle_savepath).vtkhdf", "w")
            root = HDF5.create_group(OutputVTKHDF, "VTKHDF")
            
            # Initialize particle data structure
            GenerateGeometryStructure(root, output_vars, SimParticles.ChunkID, SimParticles.Kernel, 
                                    SimParticles.KernelGradient, SimParticles.Density,
                                    SimParticles.Pressure, SimParticles.Velocity,
                                    SimParticles.Acceleration, SimParticles.BoundaryBool,
                                    SimParticles.ID, Int8.(SimParticles.Type), 
                                    SimParticles.GroupMarker, SimParticles.GhostPoints, SimParticles.GhostNormals; chunk_size=1024)
            GenerateStepStructure(root, output_vars, SimParticles.ChunkID, SimParticles.Kernel,
                                SimParticles.KernelGradient, SimParticles.Density,
                                SimParticles.Pressure, SimParticles.Velocity,
                                SimParticles.Acceleration, SimParticles.BoundaryBool,
                                SimParticles.ID, Int8.(SimParticles.Type),
                                SimParticles.GroupMarker, SimParticles.GhostPoints, SimParticles.GhostNormals)
    
            # Initialize grid file if needed
            if SimMetaData.ExportGridCells
                OutputVTKHDFGrid = h5open("$(particle_savepath)_GridCells.vtkhdf", "w")
                root_grid = HDF5.create_group(OutputVTKHDFGrid, "VTKHDF")
                GenerateGeometryStructure(root_grid; vtk_file_type="UnstructuredGrid")
                GenerateStepStructure(root_grid; vtk_file_type="UnstructuredGrid")
                
                (particle_files = OutputVTKHDF, grid_files = OutputVTKHDFGrid)
            else
                (particle_files = OutputVTKHDF, grid_files = nothing)
            end
        end
    
        # Main saving functions
        function save_particle_data(iteration)
            if Dimensions == 2
                pos = to_3d(SimParticles.Position)
                kgrad = to_3d(SimParticles.KernelGradient)
                vel = to_3d(SimParticles.Velocity)
                acc = to_3d(SimParticles.Acceleration)
                gp  = to_3d(SimParticles.GhostPoints)
                gn  = to_3d(SimParticles.GhostNormals)
            else
                pos = SimParticles.Position
                kgrad = SimParticles.KernelGradient
                vel = SimParticles.Velocity
                acc = SimParticles.Acceleration
                gp  = SimParticles.GhostPoints
                gn  = SimParticles.GhostNormals
            end
    
            if !SimMetaData.ExportSingleVTKHDF
                SaveVTKHDF(file_handles.particle_files, iteration, particle_filename(iteration), pos, output_vars,
                          SimParticles.ChunkID, SimParticles.Kernel, kgrad, SimParticles.Density, SimParticles.Pressure,
                          vel, acc, SimParticles.BoundaryBool, SimParticles.ID,
                          UInt8.(SimParticles.Type), SimParticles.GroupMarker, gp, gn)
            else
                AppendVTKHDFData(root, SimMetaData.TotalTime, pos, output_vars,
                                SimParticles.ChunkID, SimParticles.Kernel, kgrad, SimParticles.Density,
                                SimParticles.Pressure, vel, acc, SimParticles.BoundaryBool,
                                SimParticles.ID, UInt8.(SimParticles.Type), SimParticles.GroupMarker, gp, gn)
            end
        end
    
        function save_cell_grid(iteration, cells, SimParticles)
            if SimMetaData.ExportGridCells
                if !SimMetaData.ExportSingleVTKHDF
                    SaveCellGridVTKHDF(grid_filename(iteration), SimKernel, cells)
                else 
                    AppendVTKHDFGridData(root_grid, SimMetaData.TotalTime, SimKernel, cells, SimParticles)
                end
            end
        end
    
        function close_files()
            if !SimMetaData.ExportSingleVTKHDF
                # Close all particle files in multi-file mode
                for f in file_handles.particle_files
                    isopen(f) && close(f)
                end
            else
                # Close single-file handles
                isopen(file_handles.particle_files) && close(file_handles.particle_files)
                if file_handles.grid_files !== nothing
                    isopen(file_handles.grid_files) && close(file_handles.grid_files)
                end
            end
        end
    
        # Return interface functions and handles
        return (
            save_particles = save_particle_data,
            save_grid = save_cell_grid,
            close_files = close_files,
            file_handles = file_handles,  # For advanced access if needed
            variable_names = output_vars
        )
    end

end